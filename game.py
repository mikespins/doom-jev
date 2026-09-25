"""Game process: owns ViZDoom and applies an action every tic. Never waits on Jev."""

import math
import os
import queue
import time
from collections import deque
from pathlib import Path

import numpy as np
import vizdoom as vzd

import navigation as N
import perception as P
import shared as S

ARENAS = ["basic", "defend_the_center", "defend_the_line", "deadly_corridor"]


def find_wad():
    """Prefer a real Doom IWAD if one is installed, otherwise Freedoom."""
    dirs = ["/usr/share/games/doom", "/usr/share/doom", "/usr/local/share/games/doom",
            "/usr/local/share/doom", str(Path.home() / ".local/share/games/doom"), "."]
    for name in ["doom.wad", "doom2.wad", "freedoom2.wad", "freedoom1.wad"]:
        for d in dirs:
            p = Path(d) / name
            if p.is_file():
                return p
    return Path(vzd.__file__).parent / "freedoom2.wad"  # bundled with ViZDoom


def is_episodic(wad):
    return wad.name.lower() in ("doom.wad", "doom1.wad", "freedoom1.wad")


def next_map(name, episodic):
    if episodic:  # E1M1 .. E1M8 -> E2M1
        e, m = int(name[1]), int(name[3])
        return f"E{e}M{m + 1}" if m < 8 else f"E{min(e + 1, 4)}M1"
    return f"MAP{min(int(name[3:]) + 1, 32):02d}"


class Game:
    def __init__(self, args, shm, jev_action, nav, snap_q, event_q, ui_q, stop):
        self.args, self.stop = args, stop
        self.jev_action = jev_action  # shared int, written by the Jev process
        self.nav = nav                # shared [heading, valid], written by the Jev process
        self.snap_q, self.event_q, self.ui_q = snap_q, event_q, ui_q
        self.ctrl, self.metas, self.frames = S.views(shm.buf)
        self.wad = find_wad()
        self.campaign = args.scenario == "campaign"
        self.map = args.map or ("E1M1" if is_episodic(self.wad) else "MAP01")
        self.turn_vel = 0.0
        self.active_reflexes = set()
        self.kills = self.deaths = self.episodes = self.levels = 0
        self.kills_base = 0
        self.status, self.status_t = S.PLAYING, 0.0
        self.reflex_tics = self.total_tics = 0
        self.gaps = deque(maxlen=35 * 30)
        self.tics_done, self.play_time = 0, 0.0
        self.doors = None
        self.seq = 0

    def make_game(self):
        g = vzd.DoomGame()
        if self.campaign:
            g.set_doom_game_path(str(self.wad))
            g.set_doom_map(self.map)
            g.set_doom_skill(self.args.skill)
            g.set_episode_timeout(0)
            g.set_sectors_info_enabled(True)  # for door detection
        else:
            g.load_config(os.path.join(vzd.scenarios_path, f"{self.args.scenario}.cfg"))
            if self.args.scenario == "basic":
                g.set_episode_timeout(35 * 20)
        g.set_mode(vzd.Mode.ASYNC_PLAYER)
        g.set_window_visible(self.args.no_ui)  # the UI draws the frames itself
        g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        g.set_screen_format(vzd.ScreenFormat.BGRA32)
        g.set_render_hud(True)
        g.set_render_crosshair(True)
        g.set_labels_buffer_enabled(True)
        g.set_depth_buffer_enabled(True)
        g.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
                                 vzd.Button.ATTACK, vzd.Button.USE, vzd.Button.TURN_LEFT_RIGHT_DELTA])
        GV = vzd.GameVariable
        g.set_available_game_variables([GV.HEALTH, GV.ARMOR, GV.SELECTED_WEAPON_AMMO, GV.POSITION_X,
                                        GV.POSITION_Y, GV.ANGLE, GV.DAMAGE_TAKEN, GV.KILLCOUNT])
        g.init()
        return g

    def label(self):
        return f"{self.map} skill {self.args.skill}" if self.campaign else self.args.scenario

    def announce(self):
        self.doors = None
        msg = {"type": "map", "name": self.label(), "wad": self.wad.name if self.campaign else "vizdoom"}
        for q in (self.ui_q, self.event_q):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def run(self):
        g = self.make_game()
        self.announce()
        last_iter = time.monotonic()
        last_tic = g.get_episode_time()
        last_stats = time.monotonic()
        try:
            while not self.stop.is_set():
                if g.is_episode_finished():
                    self.end_episode(g)
                    last_tic, last_iter = g.get_episode_time(), time.monotonic()
                    continue
                state = g.get_state()
                if state is None:
                    g.advance_action(1)
                    continue
                self.step(g, state)

                now = time.monotonic()
                gap, last_iter = now - last_iter, now
                tic = g.get_episode_time()
                self.gaps.append(gap * 1000)
                self.play_time += gap
                self.tics_done += max(0, tic - last_tic)
                last_tic = tic
                if now - last_stats > 1.0:
                    last_stats = now
                    gs = sorted(self.gaps)
                    self.stats = (self.tics_done / max(1e-6, self.play_time),
                                  gs[int(len(gs) * .99)] if gs else 0, gs[-1] if gs else 0)
        except vzd.ViZDoomUnexpectedExitException:
            pass
        finally:
            g.close()

    stats = (0.0, 0.0, 0.0)

    def end_episode(self, g):
        self.episodes += 1
        dead = g.is_player_dead()
        if dead:
            self.deaths += 1
            self.status = S.DIED
        elif self.campaign:
            self.levels += 1
            self.status = S.LEVEL_CLEAR
            self.map = next_map(self.map, is_episodic(self.wad))
            g.set_doom_map(self.map)
        self.status_t = time.monotonic()
        self.kills_base = self.kills
        g.new_episode()
        if not dead and self.campaign:
            self.announce()
        self.turn_vel = 0.0

    def step(self, g, state):
        v = state.game_variables
        objs = P.objs_from_labels(state.labels, S.W, S.H)
        act = S.ACTIONS[self.jev_action.value]
        depth = state.depth_buffer

        door = None
        if self.campaign and state.sectors:
            if self.doors is None:
                self.doors = P.DoorFinder(state.sectors)
                geom = {"type": "geom", "walls": N.extract_walls(state.sectors)}
                for q in (self.event_q, self.ui_q):
                    try:
                        q.put(geom, timeout=0.2)
                    except queue.Full:
                        pass
            door = self.doors.distance(state.sectors, v[3], v[4], v[5])

        snap = P.Snapshot(tic=state.tic, t=time.monotonic(), health=v[0], armor=v[1], ammo=v[2],
                          px=v[3], py=v[4], angle=v[5], damage_taken=v[6], objs=objs, depth=depth,
                          move_pressed=act in ("move_forward", "move_back"), door_dist=door)
        self.kills = self.kills_base + int(v[7])
        enemies = snap.enemies()

        # Smooth turning: ease toward a target turn rate. When an enemy lies in the
        # turn direction, slow down as it nears the crosshair so we don't overshoot.
        target = 0.0
        if act in ("turn_left", "turn_right"):
            sign = -1 if act == "turn_left" else 1
            speed = S.MAX_TURN
            ahead = [e for e in enemies if (e.cx - 0.5) * sign > 0]
            if ahead:
                e = min(ahead, key=lambda e: abs(e.cx - 0.5))
                deg = math.degrees(math.atan(2 * abs(e.cx - 0.5)))  # 90° horizontal FOV
                speed = min(S.MAX_TURN, max(0.4, 0.3 * deg))
            target = sign * speed
        elif act == "move_forward" and self.nav[1]:
            # Walking: steer gently along the route to unexplored space.
            rel = N.wrap(self.nav[0] - v[5])
            if abs(rel) < 70:
                target = max(-S.MAX_TURN, min(S.MAX_TURN, -0.25 * rel))

        # Reflexes: plain code, applied instantly every tic.
        fired = set()
        fwd = act == "move_forward"
        attack = act == "shoot"
        wall = P.ahead_distance(depth)
        if snap.ammo > 0 and any(P.in_center_band(e) and snap.dist(e) < P.CLOSE for e in enemies):
            fired.add("shoot")
            attack = True
            target = 0.0  # hold aim while firing
        use = False
        if act == "use_open_door":
            use = state.tic % 2 == 0  # tap, don't hold
            if door is not None and door > P.USE_REACH_UNITS:
                fwd = True  # walk up to the door while tapping use
        if fwd and wall < P.WALL_REFLEX_UNITS:
            fired.add("stop_forward")
            fwd = False

        self.turn_vel += (target - self.turn_vel) * 0.35
        if abs(self.turn_vel) < 0.05:
            self.turn_vel = 0.0
        back = act == "move_back"
        g.set_action([float(fwd), float(back), float(attack), float(use), self.turn_vel])
        g.advance_action(1)  # returns at the next tic; the game never waits for us

        self.total_tics += 1
        self.reflex_tics += bool(fired)
        for kind in fired - self.active_reflexes:
            try:
                self.event_q.put_nowait({"type": "reflex", "name": kind, "overrode": act, "t": time.time()})
            except queue.Full:
                pass
        self.active_reflexes = fired

        self.publish(state, snap, objs, fwd, back, attack, use, fired, wall, door)
        if state.tic % 3 == 0:  # ~12 snapshots/s for the Jev process; it keeps only the newest
            small = P.Snapshot(**{**snap.__dict__, "depth": depth[::4, ::4].copy()})
            try:
                self.snap_q.put_nowait(small)
            except queue.Full:
                pass

    def publish(self, state, snap, objs, fwd, back, attack, use, fired, wall, door):
        slot = (int(self.ctrl[0]) + 1) % S.SLOTS
        m, f = self.metas[slot], self.frames[slot]
        f[:] = state.screen_buffer
        self.seq += 1
        m[S.SEQ], m[S.TIC] = self.seq, state.tic
        m[S.FWD], m[S.BACK], m[S.ATTACK], m[S.USE], m[S.TURN] = fwd, back, attack, use, self.turn_vel
        m[S.HEALTH], m[S.AMMO], m[S.ARMOR] = snap.health, snap.ammo, snap.armor
        m[S.REFLEX_SHOOT], m[S.REFLEX_STOP] = "shoot" in fired, "stop_forward" in fired
        m[S.JEV_IDX] = self.jev_action.value
        m[S.KILLS], m[S.DEATHS], m[S.EPISODES] = self.kills, self.deaths, self.episodes
        m[S.LEVELS_CLEARED] = self.levels
        m[S.POS_X], m[S.POS_Y], m[S.ANGLE] = snap.px, snap.py, snap.angle
        m[S.NAV_HEADING] = self.nav[0] if self.nav[1] else float("nan")
        m[S.TICS_PER_S], m[S.GAP_P99], m[S.GAP_MAX] = self.stats
        m[S.STATUS], m[S.STATUS_T] = self.status, self.status_t
        m[S.WALL_DIST], m[S.DOOR_DIST] = wall, -1 if door is None else door
        m[S.REFLEX_TICS], m[S.TOTAL_TICS] = self.reflex_tics, self.total_tics
        shown = sorted(objs, key=snap.dist)[:S.MAX_BOXES]
        m[S.N_BOXES] = len(shown)
        for i, o in enumerate(shown):
            enemy = P.is_enemy(o.name) and o.alive_shape
            b = S.BOXES + i * 6
            m[b:b + 6] = (o.left, o.top, o.right, o.bottom, enemy, snap.dist(o))
        self.ctrl[1] = self.seq
        self.ctrl[0] = slot


def run(args, shm, jev_action, nav, snap_q, event_q, ui_q, stop, result_q):
    game = Game(args, shm, jev_action, nav, snap_q, event_q, ui_q, stop)
    try:
        game.run()
    finally:
        stop.set()
        tps, p99, mx = game.stats
        result_q.put({"proc": "game", "tics_per_s": tps, "gap_p99_ms": p99, "gap_max_ms": mx,
                      "episodes": game.episodes, "deaths": game.deaths, "kills": game.kills,
                      "levels": game.levels, "reflex_share": game.reflex_tics / max(1, game.total_tics),
                      "map": game.label()})
