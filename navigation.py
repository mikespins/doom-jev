"""Map memory and route finding, all in code.

The game process turns the level's lines into plain wall segments once per map.
The Jev process keeps a coarse grid of the level: which cells are walls, which
cells the player has already seen, and a breadth-first route to the nearest
unexplored cell. Jev only ever sees the result in words ("Unexplored area:
slightly left, close").
"""

import math
import time
from collections import deque

import numpy as np

CELL = 16          # map units per grid cell
SEEN_RADIUS = 96   # cells within this distance of the player count as explored
STEP_HEIGHT = 24   # Doom's maximum step-up
MIN_GAP = 56       # player height; a smaller opening blocks unless it is a door


def door_sectors(sectors):
    """Indices of closed doors: sectors whose ceiling has come down to a floor that is
    level with the rooms around it. A closed sector whose floor rose up to meet its
    ceiling is a pillar or a lift, so it is a wall."""
    nbrs, by_key = {}, {}
    for i, sec in enumerate(sectors):
        for ln in sec.lines:
            a, b = (ln.x1, ln.y1), (ln.x2, ln.y2)
            by_key.setdefault((min(a, b), max(a, b)), set()).add(i)
    for secs in by_key.values():
        for i in secs:
            nbrs.setdefault(i, set()).update(secs - {i})
    out = set()
    for i, sec in enumerate(sectors):
        if sec.ceiling_height - sec.floor_height < 8 and nbrs.get(i):
            if sec.floor_height <= min(sectors[j].floor_height for j in nbrs[i]) + STEP_HEIGHT:
                out.add(i)
    return out


def extract_walls(sectors):
    """Classify every line of the map, once per map (runs in the game process).

    Returns a list of (x1, y1, x2, y2, kind, door) with kind "wall" or "door";
    door is the index of the closed door sector (-1 for walls). Two-sided lines
    between sectors of similar height are open and left out.
    """
    doors = door_sectors(sectors)
    by_key = {}
    for i, sec in enumerate(sectors):
        for ln in sec.lines:
            a, b = (ln.x1, ln.y1), (ln.x2, ln.y2)
            key = (min(a, b), max(a, b))
            entry = by_key.setdefault(key, {"secs": set(), "blocking": False})
            entry["secs"].add(i)
            entry["blocking"] |= bool(ln.is_blocking)
    out = []
    for (a, b), e in by_key.items():
        secs = [sectors[i] for i in e["secs"]]
        closed = sorted(e["secs"] & doors)
        solid = any(sectors[i].ceiling_height - sectors[i].floor_height < 8 for i in e["secs"])
        door = -1
        if len(secs) < 2:
            kind = "wall"  # one-sided, including a door's side jambs
        elif closed:
            kind, door = "door", closed[0]
        elif e["blocking"] or solid:
            kind = "wall"
        else:
            s1, s2 = secs[0], secs[1]
            gap = min(s1.ceiling_height, s2.ceiling_height) - max(s1.floor_height, s2.floor_height)
            kind = "wall" if abs(s1.floor_height - s2.floor_height) > STEP_HEIGHT or gap < MIN_GAP else None
        if kind:
            out.append((a[0], a[1], b[0], b[1], kind, door))
    return out


def door_open(sector):
    """A door counts as open once the player fits under it."""
    return sector.ceiling_height - sector.floor_height >= MIN_GAP


def direction_word(rel):
    """rel: degrees to turn, positive = left (Doom angles grow counter-clockwise)."""
    a = abs(rel)
    side = "left" if rel > 0 else "right"
    if a < 20:
        return "ahead"
    if a < 60:
        return f"slightly {side}"
    if a < 135:
        return side
    return "behind"


def wrap(deg):
    return (deg + 180) % 360 - 180


class NavGrid:
    def __init__(self, walls):
        xs = [w[0] for w in walls] + [w[2] for w in walls]
        ys = [w[1] for w in walls] + [w[3] for w in walls]
        self.x0, self.y0 = min(xs) - 64, min(ys) - 64
        self.w = int((max(xs) + 64 - self.x0) // CELL) + 1
        self.h = int((max(ys) + 64 - self.y0) // CELL) + 1
        self.walls = np.zeros((self.h, self.w), dtype=bool)
        self.door_cells = {}    # door sector -> cells its closed faces cover
        self.door_center = {}   # door sector -> map point in the middle of the door
        faces = {}
        for x1, y1, x2, y2, kind, door in walls:
            n = max(2, int(math.hypot(x2 - x1, y2 - y1) // 4))
            cells = {self.cell(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t) for t in np.linspace(0, 1, n)}
            if kind == "door":
                self.door_cells.setdefault(door, set()).update(cells)
                faces.setdefault(door, []).append(((x1 + x2) / 2, (y1 + y2) / 2))
            else:
                for cx, cy in cells:
                    self.walls[cy, cx] = True
        for door, mids in faces.items():
            self.door_center[door] = (sum(m[0] for m in mids) / len(mids), sum(m[1] for m in mids) / len(mids))
        self.open_doors = set()
        self.opened = set()        # doors the player has opened at least once
        self.failed = {}           # door -> expiry time, for doors that would not open
        self.seen = np.zeros_like(self.walls)
        self.temp_blocked = {}  # cell -> expiry time, for things we got stuck on
        self.path = []
        self.goal = None        # ("explore", None) or ("door", door sector)
        self._rebuild()

    def _rebuild(self):
        """Closed doors block like walls. `tight` is the raw grid; `blocked` also keeps
        routes a player-width away from walls so steering doesn't scrape along them.
        Many "closed" sectors are lifts or props, so doors are only ever a goal to walk
        up to and open, never something to route through."""
        self.tight = self.walls.copy()
        self.door_at = {}  # cell -> closed door sector
        for door, cells in self.door_cells.items():
            if door in self.open_doors:
                continue
            for cx, cy in cells:
                self.tight[cy, cx] = True
                self.door_at[(cx, cy)] = door
        b = self.tight.copy()
        b[1:, :] |= self.tight[:-1, :]
        b[:-1, :] |= self.tight[1:, :]
        b[:, 1:] |= self.tight[:, :-1]
        b[:, :-1] |= self.tight[:, 1:]
        self.blocked = b

    def set_open_doors(self, open_doors):
        """Door sectors that are open right now. Returns the ones that just opened."""
        open_doors = set(open_doors) & set(self.door_cells)
        newly = open_doors - self.open_doors
        if open_doors != self.open_doors:
            self.open_doors = open_doors
            self._rebuild()
        return newly

    def near_door(self, x, y, radius=128):
        """The closest door sector within radius of a map point, or None."""
        best = min(self.door_center.items(), key=lambda d: math.hypot(d[1][0] - x, d[1][1] - y), default=None)
        if best and math.hypot(best[1][0] - x, best[1][1] - y) < radius:
            return best[0]
        return None

    def fail_door(self, door, seconds=30):
        self.failed[door] = time.monotonic() + seconds

    def cell(self, x, y):
        cx = min(self.w - 1, max(0, int((x - self.x0) // CELL)))
        cy = min(self.h - 1, max(0, int((y - self.y0) // CELL)))
        return cx, cy

    def center(self, c):
        return self.x0 + (c[0] + 0.5) * CELL, self.y0 + (c[1] + 0.5) * CELL

    def mark_seen(self, x, y):
        """Mark cells around the player as explored. Returns the newly seen cells."""
        cx, cy = self.cell(x, y)
        r = SEEN_RADIUS // CELL
        ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
        disk = xs * xs + ys * ys <= r * r
        y0, y1 = max(0, cy - r), min(self.h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(self.w, cx + r + 1)
        patch = disk[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
        region = self.seen[y0:y1, x0:x1]
        new = patch & ~region
        region |= patch
        nys, nxs = np.nonzero(new)
        return [(int(x + x0), int(y + y0)) for y, x in zip(nys, nxs)]

    def block_ahead(self, x, y, angle, seconds=20):
        """We got stuck here: treat the spot in front of us as a wall for a while."""
        a = math.radians(angle)
        until = time.monotonic() + seconds
        for d in (16, 32):
            self.temp_blocked[self.cell(x + math.cos(a) * d, y + math.sin(a) * d)] = until

    def route(self, x, y):
        """Pick the current goal and a route to it, in order of preference:
        1. the nearest unexplored area, keeping clear of walls;
        2. the nearest unexplored area through tight spots, or a door never opened;
        3. any door that isn't known to be stuck (to go back through).
        """
        now = time.monotonic()
        self.temp_blocked = {c: t for c, t in self.temp_blocked.items() if t > now}
        self.failed = {d: t for d, t in self.failed.items() if t > now}
        usable = set(self.door_at.values()) - set(self.failed)
        fresh = usable - self.opened
        for blocked, explore, doors in ((self.blocked, True, ()), (self.tight, True, fresh),
                                        (self.tight, False, usable)):
            found = self._bfs(x, y, blocked, explore, doors)
            if found:
                self.goal, cells = found
                self.path = [self.center(c) for c in cells]
                return self.path
        self.goal, self.path = None, []
        return None

    def _bfs(self, x, y, blocked, explore, doors, max_cells=40000):
        """Breadth-first search to the nearest unexplored open cell (if explore) or a
        cell next to one of the given closed doors."""
        start = self.cell(x, y)
        prev = {start: None}
        q = deque([start])
        seen, w, h, door_at, temp = self.seen, self.w, self.h, self.door_at, self.temp_blocked
        while q and len(prev) < max_cells:
            c = q.popleft()
            if explore and not seen[c[1], c[0]]:
                goal = ("explore", None)
                break
            cx, cy = c
            nbrs = ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1))
            door = next((door_at[n] for n in nbrs if door_at.get(n) in doors), None) if doors else None
            if door is not None:
                goal = ("door", door)
                break
            for n in nbrs:
                if 0 <= n[0] < w and 0 <= n[1] < h and n not in prev and not blocked[n[1], n[0]] and n not in temp:
                    prev[n] = c
                    q.append(n)
        else:
            return None
        path = []
        while c is not None:
            path.append(c)
            c = prev[c]
        path.reverse()
        return goal, path

    def clear(self, x1, y1, x2, y2):
        """True if a straight walk between two map points crosses no wall cell."""
        n = max(1, int(math.hypot(x2 - x1, y2 - y1) // 8))
        for i in range(1, n + 1):
            cx, cy = self.cell(x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n)
            if self.tight[cy, cx]:
                return False
        return True

    def nearest_door(self, x, y):
        """The closest closed door that isn't known to be stuck: (door sector, center)."""
        doors = [(d, c) for d, c in self.door_center.items() if d not in self.open_doors and d not in self.failed]
        return min(doors, key=lambda d: math.hypot(d[1][0] - x, d[1][1] - y), default=(None, None))

    def heading(self, x, y, lookahead=96):
        """Map angle toward a point ~lookahead units along the route, and route length.
        Near the end of a route to a door, aim at the middle of the door itself."""
        if not self.path:
            return None, None
        length = len(self.path) * CELL
        target = self.path[0]
        for p in self.path:
            if not self.clear(x, y, p[0], p[1]):
                break  # aim at the last route point we can walk straight to
            target = p
            if math.hypot(p[0] - x, p[1] - y) >= lookahead:
                break
        else:
            if self.goal and self.goal[0] == "door":
                target = self.door_center[self.goal[1]]
        return math.degrees(math.atan2(target[1] - y, target[0] - x)), length
