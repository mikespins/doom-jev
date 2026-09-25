"""Jev process: describes the newest snapshot in words and asks Jev what to do.

Fires a request every ~120 ms without waiting for earlier ones (up to 5 in
flight). A response is applied only if it is newer than the last applied one.
"""

import asyncio
import json
import math
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
                         "visible and the goal direction is left or behind.",
            "turn_right": "Rotate the view right. Use when an enemy is on the right, or when no enemy is "
                          "visible and the goal direction is right.",
            "move_forward": "Walk forward. Use when no enemy is visible and the goal direction is ahead or "
                            "slightly left or slightly right.",
            "move_back": "Step backward. Use when an enemy is very close and getting closer, or when stuck.",
            "shoot": "Fire the weapon. Use when an enemy is in the crosshair and ammo is not empty.",
            "use_open_door": "Press the use key, which opens doors and presses switches. Use when a door or "
                             "a switch is directly ahead, especially when the goal is to open it. Not for a "
                             "door that needs a key the player does not have.",
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
        self.door_try = None     # the door ahead that the player is trying to open, see check_door
        self.last_step = None    # (what the player just got done, time), for the state text
        self.items = []          # (name, x, y, kind) keys and health still on the map
        self.all_keys = set()    # key colors this map has had
        self.planning = False
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
            now = time.monotonic()
            if ev["type"] == "map":
                self.map = ev["name"]
                self.describer.reset()
                self.grid, self.nav[1] = None, 0
                self.last_step, self.items, self.all_keys = None, [], set()
            elif ev["type"] == "geom":
                self.grid = N.NavGrid(ev["walls"], ev.get("info"), ev.get("things"))
                self.new_seen = []
                self.last_route = 0.0
            elif ev["type"] == "walls" and self.grid is not None:
                asyncio.get_running_loop().run_in_executor(None, self.grid.set_walls, ev["walls"])
                self.last_route = 0.0
            elif ev["type"] == "restart":
                self.describer.reset()
                self.last_step = ("died, the level restarted", now)
                if self.grid is not None:
                    self.grid.pressed.clear()
            elif ev["type"] == "doors" and self.grid is not None:
                for door in self.grid.set_open_doors(ev["open"]):
                    cx, cy = self.grid.door_center[door]
                    if self.latest is not None and math.hypot(cx - self.latest.px, cy - self.latest.py) < 256:
                        self.grid.opened.add(door)  # opened by the player, not a monster
                        key = self.grid.door_key.get(door)
                        self.last_step = (f"opened the {key} door" if key else "opened a door", now)
                self.last_route = 0.0
            elif ev["type"] == "items":
                old = self.items
                self.items = [tuple(i) for i in ev["items"]]
                self.all_keys |= {n for n, _, _, k in self.items if k == "key"}
                gone = set(old) - set(self.items)
                snap = self.latest
                for n, x, y, k in gone:  # picked up (by the player, since monsters don't pick up)
                    if snap is not None and math.hypot(x - snap.px, y - snap.py) < 128:
                        self.last_step = (f"picked up the {n} key" if k == "key" else "picked up health", now)
                self.last_route = 0.0
            elif ev["type"] == "reflex":
                self.reflex_log.append((time.strftime("%H:%M:%S"), ev["name"]))
                self.log(t=ev["t"], kind="reflex", reflex=True, reflex_name=ev["name"], map=self.map,
                         state_text=self.text, probabilities=None,
                         chosen="shoot" if ev["name"] == "shoot" else "stop_forward",
                         jev_action_overridden=ev["overrode"], latency_ms=0.0)

    def keys(self):
        """Key colors the player holds: ones this map had that are no longer lying around."""
        return self.all_keys - {n for n, _, _, k in self.items if k == "key"}

    def check_door(self, snap, now):
        """Name the door ahead's key, and give up on a door still shut after ~3 s and 10
        presses of use. The attempt survives Jev mixing in other actions (backing off,
        turning); it ends only when the door has been out of view for 2 s. A given-up door
        (it needs a key, a switch, or opens only from the other side) is hidden from Jev and
        left out of the plan for a while, longer each time."""
        grid = self.grid
        door = None
        if snap.door_dist is not None and grid is not None:
            a = math.radians(snap.angle)
            door = grid.near_door(snap.px + math.cos(a) * snap.door_dist, snap.py + math.sin(a) * snap.door_dist)
        if door is None:
            if self.door_try and now - self.door_try["seen"] > 2:
                self.door_try = None
            return
        key = grid.door_key.get(door)
        snap.door_key = key
        if door in grid.failed:
            snap.door_dist = None
            return
        t = self.door_try
        if t is None or t["door"] != door:
            t = self.door_try = {"door": door, "start": now, "uses": 0, "seen": now}
        t["seen"] = now
        if S.ACTIONS[self.jev_action.value] == "use_open_door":
            t["uses"] += 1
        if t["uses"] >= 10 and now - t["start"] > 3:
            if key and key not in self.keys():
                what = f"tried the {key} door, it needs the {key} key"
            else:
                what = "tried a door that would not open, it may need a switch or open from the other side"
            self.last_step = (what, now)
            grid.fail_door(door)
            grid.block_ahead(snap.px, snap.py, snap.angle, seconds=10)
            self.last_route = 0.0
            snap.door_dist = None
            self.door_try = None

    def check_switch(self, snap):
        """A switch line within reach straight ahead, with nothing in front of it. When Jev
        presses use on it, count it as pressed."""
        grid = self.grid
        if grid is None or not grid.switches:
            return
        a = math.radians(snap.angle)
        dx, dy = math.cos(a) * P.USE_REACH_UNITS, math.sin(a) * P.USE_REACH_UNITS
        best = None
        for i, (x3, y3, x4, y4, _key) in enumerate(grid.switches):
            ex, ey = x4 - x3, y4 - y3
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((x3 - snap.px) * ey - (y3 - snap.py) * ex) / den
            u = ((x3 - snap.px) * dy - (y3 - snap.py) * dx) / den
            if 0 <= t <= 1 and 0 <= u <= 1 and (best is None or t < best[0]):
                best = (t, i)
        if best is None or P.ahead_distance(snap.depth) < best[0] * P.USE_REACH_UNITS - 24:
            return
        snap.switch_ahead = True
        if S.ACTIONS[self.jev_action.value] == "use_open_door" and best[1] not in grid.pressed:
            grid.pressed.add(best[1])
            self.last_step = ("pressed a switch", time.monotonic())
            self.last_route = 0.0

    async def replan(self, grid, snap):
        """Work out the checkpoints in a thread, so the route search never holds up Jev requests."""
        try:
            plan = await asyncio.to_thread(grid.plan_route, snap.px, snap.py, snap.health, self.keys(), self.items)
        finally:
            self.planning = False
        if grid is not self.grid:
            return  # the map changed meanwhile
        seen = [grid.center(c) for c in self.new_seen]
        self.new_seen = []
        self.to_ui({"type": "nav", "path": plan.path[::2], "seen": seen,
                    "target": plan.target or (grid.door_center[plan.door] if plan.door is not None else None),
                    "kind": plan.kind})

    def goal_lines(self, snap, now):
        """The current checkpoint, the ones after it and what was just done, in words, so Jev
        knows where it is in the level. Also shares the route heading with the game so it
        can steer while Jev walks."""
        grid = self.grid
        if grid is None:
            return None
        if not self.planning and now - self.last_route > 0.4:
            self.last_route, self.planning = now, True
            asyncio.create_task(self.replan(grid, snap))
        plan = grid.plan
        keys = sorted(self.keys())
        lines = []
        if plan.steps:
            lines.append("Plan: " + ", then ".join([plan.goal] + plan.steps) + ".")
        heading, length = grid.heading(snap.px, snap.py)
        if heading is None:
            self.nav[1] = 0
            lines.append(f"Current goal: {plan.goal}.")
        else:
            self.nav[0], self.nav[1] = heading, 1
            lines.append(f"Current goal: {plan.goal}. Goal direction: {self.where(snap, heading, length)}.")
        if plan.door is not None and plan.door_at < 200:
            c = grid.door_center[plan.door]
            key = grid.door_key.get(plan.door)
            ang = math.degrees(math.atan2(c[1] - snap.py, c[0] - snap.px))
            lines.append(f"On the way: open the {key + ' ' if key else ''}door, "
                         f"{self.where(snap, ang, math.hypot(c[0] - snap.px, c[1] - snap.py))}.")
        lines.append(f"Keys: {', '.join(keys) if keys else 'none'}.")
        door, center = grid.nearest_door(snap.px, snap.py)
        if door is not None:
            d = math.hypot(center[0] - snap.px, center[1] - snap.py)
            ang = math.degrees(math.atan2(center[1] - snap.py, center[0] - snap.px))
            key = grid.door_key.get(door)
            lines.append(f"Nearest door on the map: {self.where(snap, ang, d)}"
                         + (f", needs the {key} key" if key and key not in keys else "") + ".")
        if self.last_step:
            what, t = self.last_step
            lines.append(f"Last step done: {what}, {'just now' if now - t < 5 else 'a while ago'}.")
        return "\n".join(lines)

    @staticmethod
    def where(snap, heading, dist):
        return (f"{N.direction_word(N.wrap(heading - snap.angle))}, "
                f"{'close' if dist < 160 else 'medium' if dist < 600 else 'far'}")

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
                    self.check_door(snap, now)
                    self.check_switch(snap)
                    self.text = self.describer.describe(snap, self.goal_lines(snap, now))
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
