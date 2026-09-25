"""Turn raw ViZDoom state into short plain-English sentences for Jev.

Jev cannot do math, so every number (distances, screen offsets, trends,
health levels) is computed here and bucketed into words.
"""

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

# One raw depth-buffer unit is roughly this many map units (measured in the
# "basic" scenario by walking into a wall).
DEPTH_UNITS = 7.2

WALL_AHEAD_UNITS = 96  # "Wall directly ahead: yes"
WALL_REFLEX_UNITS = 48  # reflex layer stops forward motion below this

VERY_CLOSE, CLOSE, MEDIUM = 150, 400, 800  # enemy distance buckets (map units)

MONSTERS = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre",
    "LostSoul", "Cacodemon", "HellKnight", "BaronOfHell", "Arachnotron",
    "PainElemental", "Revenant", "Fatso", "Archvile", "SpiderMastermind",
    "Cyberdemon", "WolfensteinSS", "ScriptedMarine",
}
IGNORED = {"DoomPlayer", "BulletPuff", "Blood", "TeleportFog", "ItemFog",
           "DoomImpBall", "CacodemonBall", "BaronBall", "Rocket", "PlasmaBall",
           "ArachnotronPlasma", "RevenantTracer", "FatShot", "RevenantTracerSmoke"}


def is_ignored(name):
    return name in IGNORED or name.startswith("Dead")


def is_enemy(name):
    return name in MONSTERS or (name.endswith("Vzd") and "Marine" in name)


@dataclass
class Obj:
    """One labelled thing on screen, in screen fractions and map units."""
    oid: int
    name: str
    left: float    # 0..1 screen x of bbox left edge
    right: float   # 0..1 screen x of bbox right edge
    cx: float      # 0..1 screen x of bbox center
    top: float     # 0..1 screen y of bbox top
    bottom: float  # 0..1 screen y of bbox bottom
    w_px: int
    h_px: int
    x: float       # map position
    y: float

    @property
    def alive_shape(self):
        # Corpses keep their class name but lie flat; standing monsters don't.
        return self.h_px >= 0.7 * self.w_px


@dataclass
class Snapshot:
    tic: int
    t: float
    health: float
    armor: float
    ammo: float
    px: float
    py: float
    angle: float
    damage_taken: float
    objs: list
    depth: np.ndarray
    move_pressed: bool
    door_dist: float | None = None  # map units to a closed door straight ahead
    door_key: str | None = None     # the key color that door needs, set by the Jev process
    switch_ahead: bool = False      # a switch within reach straight ahead, set by the Jev process

    def enemies(self):
        return [o for o in self.objs if is_enemy(o.name) and o.alive_shape]

    def items(self):
        return [o for o in self.objs if not is_enemy(o.name) and not is_ignored(o.name)]

    def dist(self, o):
        return math.hypot(o.x - self.px, o.y - self.py)


def objs_from_labels(labels, width, height):
    out = []
    for l in labels:
        if is_ignored(l.object_name):
            continue
        left, right = l.x / width, (l.x + l.width) / width
        out.append(Obj(l.object_id, l.object_name, left, right, (left + right) / 2,
                       l.y / height, (l.y + l.height) / height,
                       l.width, l.height, l.object_position_x, l.object_position_y))
    return out


def ahead_distance(depth):
    """Map-unit distance to whatever is in the middle of the view."""
    h, w = depth.shape
    step = max(1, w // 160)
    region = depth[int(h * .42):int(h * .58):step, int(w * .45):int(w * .55):step]
    return float(np.percentile(region, 10)) * DEPTH_UNITS


def open_side(depth):
    h, w = depth.shape
    step = max(1, w // 106)
    band = depth[int(h * .4):int(h * .6):step, ::step].astype(np.float32)
    thirds = np.array_split(band, 3, axis=1)
    left, mid, right = (float(t.mean()) for t in thirds)
    best = max((left, "left"), (mid, "ahead"), (right, "right"))
    return best[1]


def position_word(cx):
    if cx < 0.2:
        return "far left"
    if cx < 0.4:
        return "left"
    if cx <= 0.6:
        return "center"
    if cx <= 0.8:
        return "right"
    return "far right"


def distance_word(d):
    if d < VERY_CLOSE:
        return "very close"
    if d < CLOSE:
        return "close"
    if d < MEDIUM:
        return "medium"
    return "far"


def health_word(h):
    return ("dead" if h <= 0 else "critical" if h < 25 else "low" if h < 50
            else "medium" if h < 80 else "high")


def ammo_word(a):
    return "empty" if a <= 0 else "low" if a < 10 else "some" if a < 30 else "plenty"


def armor_word(a):
    return "none" if a <= 0 else "some" if a < 50 else "good"


def in_center_band(o, half_width=0.025):
    """True if the enemy's bbox overlaps the central ~5% of the screen."""
    return o.left <= 0.5 + half_width and o.right >= 0.5 - half_width


DOOR_SEE_UNITS = 200  # "Door ahead: yes" within this distance
USE_REACH_UNITS = 80  # Doom's use range is 64 units from the player's edge


class DoorFinder:
    """Finds a closed door straight ahead by casting a ray against the map's lines.

    A closed door in Doom is a sector whose ceiling has come down to its floor.
    Line geometry is static, so it is turned into numpy arrays once per map;
    each call only re-reads sector heights.
    """

    def __init__(self, sectors, doors):
        self.doors = doors  # sector indices that are real doors (see navigation.door_sectors)
        segs, owner, blocking = [], [], []
        for i, sec in enumerate(sectors):
            for ln in sec.lines:
                segs.append((ln.x1, ln.y1, ln.x2, ln.y2))
                owner.append(i)
                blocking.append(ln.is_blocking)
        self.segs = np.array(segs, dtype=np.float64).reshape(-1, 4)
        self.owner = np.array(owner, dtype=np.int64)
        self.blocking = np.array(blocking, dtype=bool)

    def distance(self, sectors, px, py, angle, reach=DOOR_SEE_UNITS):
        if not len(self.segs):
            return None
        closed_sec = np.array([i in self.doors and s.ceiling_height - s.floor_height < 8
                               for i, s in enumerate(sectors)])
        closed = closed_sec[self.owner]
        a = math.radians(angle)
        dx, dy = math.cos(a) * reach, math.sin(a) * reach
        x3, y3, x4, y4 = self.segs.T
        ex, ey = x4 - x3, y4 - y3
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x3 - px) * ey - (y3 - py) * ex) / den
            u = ((x3 - px) * dy - (y3 - py) * dx) / den
        hit = (np.abs(den) > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & (closed | self.blocking)
        if not hit.any():
            return None
        t_hit = np.where(hit, t, np.inf)
        best = t_hit.min()
        # The first thing the ray meets must be a door, not a wall in front of it.
        if closed[(t_hit - best) < 1e-6].any():
            return float(best * reach)
        return None


class Describer:
    """Keeps a short history of snapshots so it can describe trends."""

    def __init__(self):
        self.history = deque()  # snapshots from the last few seconds
        self.stuck = False
        self.stuck_until = 0.0

    def reset(self):
        self.history.clear()
        self.stuck_until = 0.0

    def _past(self, age):
        """Most recent snapshot at least `age` seconds old."""
        now = self.history[-1].t
        for s in reversed(self.history):
            if now - s.t >= age:
                return s
        return None

    def describe(self, snap, nav_line=None):
        if self.history and snap.tic < self.history[-1].tic:
            self.reset()  # new episode
        self.history.append(snap)
        while self.history and snap.t - self.history[0].t > 4.0:
            self.history.popleft()

        lines = [
            f"Health: {health_word(snap.health)}. Armor: {armor_word(snap.armor)}. "
            f"Ammo: {ammo_word(snap.ammo)}."
        ]

        old = self._past(2.0)
        took_damage = old is not None and snap.damage_taken > old.damage_taken
        hp_old = self._past(3.0) or (self.history[0] if self.history else snap)
        lines.append(f"Took damage recently: {'yes' if took_damage else 'no'}. "
                     f"Health dropping: {'yes' if snap.health < hp_old.health else 'no'}.")

        enemies = sorted(snap.enemies(), key=snap.dist)
        prev = self._past(0.4)
        prev_by_id = {o.oid: o for o in prev.enemies()} if prev else {}
        if not enemies:
            lines.append("Enemies visible: none.")
        else:
            lines.append(f"Enemies visible: {len(enemies)}.")
            for i, e in enumerate(enemies[:4], 1):
                d = snap.dist(e)
                parts = [f"Enemy {i}: {e.name}, {position_word(e.cx)}, {distance_word(d)}"]
                p = prev_by_id.get(e.oid)
                if p is not None:
                    dx = e.cx - p.cx
                    if abs(dx) > 0.03:
                        parts.append("moving right" if dx > 0 else "moving left")
                    dd = d - prev.dist(p)
                    if abs(dd) > 20:
                        parts.append("getting closer" if dd < 0 else "getting farther")
                lines.append(", ".join(parts) + ".")
            nearest = enemies[0]
            lines.append(f"Nearest enemy is in the crosshair: "
                         f"{'yes' if in_center_band(nearest, 0.04) else 'no'}.")

        items = sorted(snap.items(), key=snap.dist)[:2]
        if items:
            lines.append("Items: " + "; ".join(
                f"{o.name} {position_word(o.cx)} {distance_word(snap.dist(o))}" for o in items) + ".")

        wall = ahead_distance(snap.depth) < WALL_AHEAD_UNITS
        door = "no" if snap.door_dist is None else f"yes, it needs the {snap.door_key} key" if snap.door_key else "yes"
        lines.append(f"Wall directly ahead: {'yes' if wall else 'no'}. Door ahead: {door}. "
                     f"Switch ahead: {'yes' if snap.switch_ahead else 'no'}. "
                     f"Most open space: {open_side(snap.depth)}.")

        if nav_line:
            lines.append(nav_line)
        if self._stuck(snap):
            self.stuck_until = snap.t + 1.5  # hold it, so it doesn't flicker as the player backs off
        self.stuck = snap.t < self.stuck_until
        lines.append(f"Stuck: {'yes' if self.stuck else 'no'}.")
        return "\n".join(lines)

    def _stuck(self, snap):
        """Trying to move for ~3 s without really going anywhere."""
        old = self._past(3.0)
        if old is None:
            return False
        window = [s for s in self.history if s.t >= old.t]
        # 30%, not most of the time: flipping between turning and walking into the same
        # wall is stuck too.
        trying = sum(s.move_pressed for s in window) >= 0.3 * len(window)
        return trying and math.hypot(snap.px - old.px, snap.py - old.py) < 32
