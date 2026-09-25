"""Jev process: describes the newest snapshot in words and asks Jev what to do.

Fires a request every ~120 ms without waiting for earlier ones (up to 5 in
flight). A response is applied only if it is newer than the last applied one.
"""

import asyncio
import json
import os
import queue
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy, TypeSafeError

import navigation as N
import perception as P
import shared as S

MODEL = "jev-1.13.0"

QUESTION = {
    "action": Choice(
        instructions="What should the player do next?",
        criteria={
            "turn_left": "Rotate the view left. Use when an enemy is on the left, or when no enemy is "
                         "visible and the unexplored area is left or behind.",
            "turn_right": "Rotate the view right. Use when an enemy is on the right, or when no enemy is "
                          "visible and the unexplored area is right.",
            "move_forward": "Walk forward. Use when no enemy is visible and the unexplored area is ahead or "
                            "slightly left or slightly right.",
            "move_back": "Step backward. Use when an enemy is very close and getting closer, or when stuck.",
            "shoot": "Fire the weapon. Use when an enemy is in the crosshair and ammo is not empty.",
            "use_open_door": "Press the use key. Use only when a door is directly ahead.",
        },
    )
}


def sample(probs, temperature):
    names = [a for a in S.ACTIONS if a in probs]
    if temperature <= 0.01:
        return max(names, key=lambda a: probs[a])
    weights = [max(probs[a], 1e-9) ** (1 / temperature) for a in names]
    return random.choices(names, weights=weights)[0]


class Brain:
    def __init__(self, args, shm, jev_action, nav, snap_q, event_q, ui_q, stop):
        self.args, self.stop = args, stop
        self.nav = nav
        self.grid = None
        self.new_seen = []
        self.last_route = 0.0
        self.last_unstick = 0.0
        self.jev_action, self.snap_q, self.event_q, self.ui_q = jev_action, snap_q, event_q, ui_q
        self.ctrl, self.metas, _ = S.views(shm.buf)
        self.describer = P.Describer()
        self.seq = self.applied_seq = self.inflight = 0
        self.sent = self.stale = self.errors = self.applied = 0
        self.t0 = time.monotonic()
        self.last_error = ""
        self.latencies = deque(maxlen=300)
        self.applied_times = deque()
        self.sent_times = deque()
        self.latest = None
        self.text = ""
        self.probs = {}
        self.map = ""
        self.reflex_log = deque(maxlen=6)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(args.log_dir) / time.strftime("decisions-%Y%m%d-%H%M%S.jsonl")
        self.log_f = open(self.log_path, "a", buffering=1)

    def log(self, **rec):
        rec["ts"] = rec.pop("t", time.time())
        self.log_f.write(json.dumps(rec) + "\n")

    def to_ui(self, msg):
        try:
            self.ui_q.put_nowait(msg)
        except queue.Full:
            pass

    def drain(self):
        while True:
            try:
                self.latest = self.snap_q.get_nowait()
            except queue.Empty:
                break
            if self.grid is not None:
                self.new_seen += self.grid.mark_seen(self.latest.px, self.latest.py)
        while True:
            try:
                ev = self.event_q.get_nowait()
            except queue.Empty:
                break
            if ev["type"] == "map":
                self.map = ev["name"]
                self.describer.reset()
                self.grid, self.nav[1] = None, 0
            elif ev["type"] == "geom":
                self.grid = N.NavGrid(ev["walls"])
                self.new_seen = []
            elif ev["type"] == "reflex":
                self.reflex_log.append((time.strftime("%H:%M:%S"), ev["name"]))
                self.log(t=ev["t"], kind="reflex", reflex=True, reflex_name=ev["name"], map=self.map,
                         state_text=self.text, probabilities=None,
                         chosen="shoot" if ev["name"] == "shoot" else "stop_forward",
                         jev_action_overridden=ev["overrode"], latency_ms=0.0)

    def nav_line(self, snap, now):
        """Route to the nearest unexplored area, in words. Also shares the heading
        with the game so it can steer while walking."""
        if self.grid is None:
            return None
        if now - self.last_route > 0.3:
            self.last_route = now
            self.grid.route(snap.px, snap.py)
            seen = [self.grid.center(c) for c in self.new_seen]
            self.to_ui({"type": "nav", "path": self.grid.path[::2], "seen": seen})
            self.new_seen = []
        heading, length = self.grid.heading(snap.px, snap.py)
        if heading is None:
            self.nav[1] = 0
            return "Unexplored area: none found."
        self.nav[0], self.nav[1] = heading, 1
        dist = "close" if length < 160 else "medium" if length < 600 else "far"
        return f"Unexplored area: {N.direction_word(N.wrap(heading - snap.angle))}, {dist}."

    def dps(self, window=5.0):
        now = time.monotonic()
        while self.applied_times and now - self.applied_times[0] > window:
            self.applied_times.popleft()
        return len(self.applied_times) / window

    def median(self):
        return statistics.median(self.latencies) if self.latencies else float("nan")

    async def loop(self):
        async with AsyncTypeSafeClient(model=MODEL, retry=RetryPolicy(max_retries=0), timeout=2.0) as client:
            next_t = time.monotonic()
            last_tic = None
            last_stats = 0.0
            while not self.stop.is_set():
                next_t += self.args.interval
                self.drain()
                snap, now = self.latest, time.monotonic()
                while self.sent_times and now - self.sent_times[0] > 60:
                    self.sent_times.popleft()
                if (snap is not None and snap.tic != last_tic and self.inflight < self.args.max_inflight
                        and len(self.sent_times) < self.args.max_per_minute):
                    last_tic = snap.tic
                    self.text = self.describer.describe(snap, self.nav_line(snap, now))
                    if self.describer.stuck and self.grid is not None and now - self.last_unstick > 2:
                        self.last_unstick = now  # route around whatever we are pushing against
                        self.grid.block_ahead(snap.px, snap.py, snap.angle)
                        self.last_route = 0.0
                    self.seq += 1
                    self.inflight += 1
                    self.sent += 1
                    self.sent_times.append(now)
                    asyncio.create_task(self.ask(client, self.seq, self.text))
                if now - last_stats > 0.5:
                    last_stats = now
                    self.to_ui({"type": "stats", "dps": self.dps(), "median_ms": self.median(),
                                "inflight": self.inflight, "sent": self.sent, "stale": self.stale,
                                "errors": self.errors, "last_error": self.last_error})
                    if sys.stdout.isatty():
                        self.render()
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
            while self.inflight:  # let in-flight requests finish so they are logged
                await asyncio.sleep(0.05)

    async def ask(self, client, seq, text):
        t0 = time.monotonic()
        try:
            resp = await client.system_one(state=text, questions=QUESTION)
        except TypeSafeError as e:
            self.errors += 1
            self.last_error = type(e).__name__  # never the request or headers
            return
        finally:
            self.inflight -= 1
        latency = (time.monotonic() - t0) * 1000
        probs = dict(resp.choices["action"].probabilities)
        self.latencies.append(latency)
        fresh = seq > self.applied_seq
        chosen = None
        if fresh:
            self.applied_seq = seq
            self.applied += 1
            chosen = sample(probs, self.args.temperature)
            self.jev_action.value = S.ACTIONS.index(chosen)  # the game picks this up next tic
            self.applied_times.append(time.monotonic())
            self.probs = probs
            self.to_ui({"type": "decision", "seq": seq, "chosen": chosen, "probs": probs,
                        "latency_ms": latency, "text": text})
        else:
            self.stale += 1
        self.log(kind="jev", reflex=False, seq=seq, map=self.map, state_text=text, probabilities=probs,
                 chosen=chosen, applied=fresh, latency_ms=round(latency, 1))

    def render(self):
        m = self.metas[int(self.ctrl[0])]
        total = max(1.0, m[S.TOTAL_TICS])
        chosen = S.ACTIONS[int(m[S.JEV_IDX])]
        out = [
            f"\x1b[1mJev plays Doom\x1b[0m  model={MODEL}  {self.map}  temp={self.args.temperature}  "
            f"kills={int(m[S.KILLS])} deaths={int(m[S.DEATHS])} levels={int(m[S.LEVELS_CLEARED])}",
            f"game tics/s {m[S.TICS_PER_S]:5.1f}  loop gap p99 {m[S.GAP_P99]:3.0f} / max {m[S.GAP_MAX]:3.0f} ms"
            f"  | decisions/s {self.dps():4.1f}  median Jev latency {self.median():4.0f} ms",
            f"sent {self.sent}  stale {self.stale}  errors {self.errors} {self.last_error}  | control: "
            f"reflex {100 * m[S.REFLEX_TICS] / total:4.1f}%  jev {100 - 100 * m[S.REFLEX_TICS] / total:4.1f}%",
            "", "\x1b[1mState\x1b[0m", self.text or "(waiting for first snapshot)", "",
            "\x1b[1mJev probabilities\x1b[0m",
        ]
        for a in S.ACTIONS:
            p = self.probs.get(a, 0.0)
            out.append(f"  {a:14s} {'█' * int(p * 30):30s} {p:4.2f}{' <' if a == chosen else ''}")
        reflexes = [k for k, i in (("shoot", S.REFLEX_SHOOT), ("stop_forward", S.REFLEX_STOP)) if m[i]]
        out.append(f"\nAction: \x1b[1m{chosen}\x1b[0m" + (f"   REFLEX: {', '.join(reflexes)}" if reflexes else ""))
        out.append("Recent reflexes: " + ", ".join(f"{t} {k}" for t, k in self.reflex_log))
        sys.stdout.write("\x1b[H\x1b[J" + "\n".join(out) + "\n")
        sys.stdout.flush()


def run(args, shm, jev_action, nav, snap_q, event_q, ui_q, stop, result_q):
    os.nice(10)  # the game and UI come first; network waits don't need the CPU
    brain = Brain(args, shm, jev_action, nav, snap_q, event_q, ui_q, stop)
    try:
        asyncio.run(brain.loop())
    finally:
        result_q.put({"proc": "brain", "sent": brain.sent, "stale": brain.stale, "errors": brain.errors,
                      "median_ms": brain.median(), "applied": brain.applied,
                      "dps": brain.applied / max(1e-6, time.monotonic() - brain.t0), "log": str(brain.log_path),
                      "latencies": list(brain.latencies)})
