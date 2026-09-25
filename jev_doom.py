"""Jev (TypeSafe AI) plays Doom live.

Three processes share memory:
  game   ViZDoom in ASYNC_PLAYER mode at 35 tics/s; applies an action every tic
         and never waits on Jev. Reflexes live here.
  brain  describes the game in words and asks Jev every ~120 ms, overlapping
         requests; runs at lower CPU priority.
  ui     draws the game view and a live panel of Jev's decisions and key presses.
"""

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time
from multiprocessing import shared_memory

import brain
import game
import shared as S
import ui


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="campaign", choices=["campaign"] + game.ARENAS,
                    help="campaign = real Freedoom/Doom maps, advancing on exit (default)")
    ap.add_argument("--map", default=None, help="starting map, e.g. MAP05 or E1M3 (campaign only)")
    ap.add_argument("--skill", type=int, default=3, choices=range(1, 6),
                    help="1 easiest .. 3 Hurt Me Plenty .. 4 Ultra-Violence .. 5 Nightmare")
    ap.add_argument("--temperature", type=float, default=0.3, help="sampling temperature; 0 = always top choice")
    ap.add_argument("--interval", type=float, default=0.12, help="seconds between Jev requests")
    ap.add_argument("--max-inflight", type=int, default=5)
    ap.add_argument("--max-per-minute", type=int, default=1100, help="hard cap, below the 1200/min limit")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run forever)")
    ap.add_argument("--no-ui", action="store_true", help="skip the UI and show ViZDoom's own window")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()

    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        sys.exit("TYPESAFE_API_KEY is not set.\n"
                 "Create a key at https://console.typesafe.ai/ and run:\n"
                 "    export TYPESAFE_API_KEY='your-key-here'\n"
                 "then start this program again.")

    ctx = mp.get_context("fork")
    shm = shared_memory.SharedMemory(create=True, size=S.SHM_BYTES)
    try:
        S.views(shm.buf)[0][:] = (0, -1)
        jev_action = ctx.Value("i", S.ACTIONS.index("move_forward"), lock=False)
        nav = ctx.Array("d", 2, lock=False)  # route heading (degrees), valid flag
        stop = ctx.Event()
        snap_q, event_q, ui_q, result_q = ctx.Queue(8), ctx.Queue(256), ctx.Queue(256), ctx.Queue()
        procs = [ctx.Process(target=game.run, name="game",
                             args=(args, shm, jev_action, nav, snap_q, event_q, ui_q, stop, result_q)),
                 ctx.Process(target=brain.run, name="brain",
                             args=(args, shm, jev_action, nav, snap_q, event_q, ui_q, stop, result_q))]
        if not args.no_ui:
            procs.append(ctx.Process(target=ui.run, name="ui", args=(shm, ui_q, stop, result_q)))
        for p in procs:
            p.start()

        start = time.monotonic()
        try:
            while not stop.is_set() and (not args.duration or time.monotonic() - start < args.duration):
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        stop.set()
        results = {}
        for _ in procs:
            try:
                r = result_q.get(timeout=10)
                results[r["proc"]] = r
            except Exception:
                break
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        summary(results, time.monotonic() - start)
    finally:
        shm.close()
        shm.unlink()


def summary(r, elapsed):
    g, b, u = r.get("game", {}), r.get("brain", {}), r.get("ui", {})
    print(f"\n=== Jev plays Doom: {elapsed:.0f} s on {g.get('map', '?')} ===")
    if g:
        print(f"game tics/s {g['tics_per_s']:.1f}   loop gap p99 {g['gap_p99_ms']:.0f} ms, max {g['gap_max_ms']:.0f} ms"
              f" (last 30 s)")
        print(f"kills {g['kills']}   deaths {g['deaths']}   levels cleared {g['levels']}   "
              f"reflex share of tics {100 * g['reflex_share']:.1f}%")
    if u:
        print(f"UI: {u['fps']:.1f} game frames shown/s   frame time p99 {u['frame_p99_ms']:.0f} ms, "
              f"max {u['frame_max_ms']:.0f} ms (last 30 s)")
    if b:
        lat = sorted(b["latencies"])
        p90 = lat[int(len(lat) * .9)] if lat else float("nan")
        print(f"Jev: sent {b['sent']}  applied {b['applied']}  stale {b['stale']}  errors {b['errors']}   "
              f"decisions/s {b['dps']:.1f}   latency median {statistics.median(lat) if lat else float('nan'):.0f} ms"
              f", p90 {p90:.0f} ms")
        print(f"Decision log: {b['log']}")


if __name__ == "__main__":
    main()
