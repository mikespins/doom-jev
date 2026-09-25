# Jev plays Doom

The Jev decision model (TypeSafe AI, pinned to `jev-1.13.0`) plays Freedoom live through
ViZDoom. A retro UI shows the game, the keys being pressed on every tic, Jev's probabilities, a
running strip of its decisions, the exact text Jev is given, and an automap with the route code
is steering along.

## How it works

Three processes share memory, so nothing waits on a network call:

| process | file | job |
| ------- | ---- | --- |
| game | `game.py` | ViZDoom in `ASYNC_PLAYER` mode at 35 tics/s. Applies the current action every tic, runs the reflexes, and writes each frame and its button state to shared memory. |
| brain | `brain.py` | Every 120 ms, describes the newest snapshot in plain English (`perception.py`) and fires a Jev `Choice` request without waiting for earlier ones (up to 5 in flight, capped at 1,100/min). Stale responses are dropped. Runs at a lower CPU priority. |
| ui | `ui.py` | pygame window: the game view scaled up 2× with enemy boxes and an automap, plus a panel with keys, probabilities, the decision strip, the state text and stats. The keys light from the exact buttons applied to the frame on screen. |

- **Jev decides.** One Choice question: "What should the player do next?" with `turn_left`,
  `turn_right`, `move_forward`, `move_back`, `shoot` and `use_open_door`. The action is sampled
  from Jev's probabilities with a temperature (default 0.3).
- **Code does the math.** Jev only sees words:
  - position: far left … far right
  - distance: very close … far
  - health, ammo and armor levels
  - wall ahead and door ahead
  - recent damage and falling health
  - enemy movement trends
  - stuck
  - `Unexplored area: slightly left, close`
- **Navigation** (`navigation.py`) builds a coarse grid from the level's walls, remembers where
  the player has been, and runs a breadth-first search to the nearest unexplored area. When Jev
  walks forward, the game steers gently along that route. When the player is stuck, the blocked
  spot is marked off-limits for 20 s so the route goes around it.
- **Reflexes** (plain code that overrides Jev):
  - shoot when a live enemy is in the center 5% of the screen and closer than 400 units
  - stop walking forward when the depth buffer shows a wall within about 48 units
- **Levels.** On death the level restarts automatically. When a level is cleared, the game moves
  on to the next map.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export TYPESAFE_API_KEY='your-key-here'   # from https://console.typesafe.ai/
```

The key is read only from `TYPESAFE_API_KEY`. The app never prints or logs it, and it exits
with setup instructions if the variable is missing.

For the real maps, the app looks for `doom.wad`, `doom2.wad`, `freedoom2.wad` and then
`freedoom1.wad` in `/usr/share/games/doom` and a few other standard folders. If none is found,
it uses the Freedoom 2 WAD bundled with ViZDoom. On Debian/Ubuntu you can install the WADs with
`apt install freedoom`.

## Run

```sh
DISPLAY=:1 .venv/bin/python jev_doom.py                      # Freedoom campaign from MAP01, skill 3
DISPLAY=:1 .venv/bin/python jev_doom.py --skill 4            # Ultra-Violence
DISPLAY=:1 .venv/bin/python jev_doom.py --map MAP05
DISPLAY=:1 .venv/bin/python jev_doom.py --scenario defend_the_center
DISPLAY=:1 .venv/bin/python jev_doom.py --temperature 0.6
```

Press Esc or close the window to stop. Or pass `--duration N` to stop after N seconds and print
a summary: game tics/s, loop-gap stutter numbers, UI frame rate, decisions/s, and Jev latency.

| flag | default | meaning |
| ---- | ------- | ------- |
| `--scenario` | `campaign` | `campaign`, `basic`, `defend_the_center`, `defend_the_line`, `deadly_corridor` |
| `--skill` | `3` | 1–5 |
| `--map` | first map | starting map for the campaign |
| `--temperature` | `0.3` | `0` = always take Jev's top choice, `1` = sample Jev's raw distribution |
| `--interval` | `0.12` | seconds between Jev requests |
| `--max-inflight` | `5` | most requests in flight at once |
| `--max-per-minute` | `1100` | hard request cap |
| `--no-ui` | | show ViZDoom's own window instead of the UI |

## Logs

Every decision is appended to `logs/decisions-*.jsonl`:
- Jev decisions: timestamp, state text, probabilities, chosen action, latency, `reflex: false`,
  and whether the response was applied or dropped as stale.
- Reflexes: one record each time a reflex starts, with the Jev action it overrode, and
  `reflex: true`.

## Measured on a 1-vCPU VM over VNC

- Game: a steady 35 tics/s.
- Jev: about 8 decisions/s, with a median latency of about 120 ms.
- UI: about 27 fps; the VNC server takes a third of the CPU.
- Resets: a level restart takes about 95 ms. ViZDoom's `basic` scenario takes about 470 ms per
  reset, which is why it is no longer the default.
