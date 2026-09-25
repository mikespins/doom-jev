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


def extract_walls(sectors):
    """Classify every line of the map, once per map (runs in the game process).

    Returns a list of (x1, y1, x2, y2, kind) with kind "wall" or "door".
    Two-sided lines between sectors of similar height are open and left out.
    """
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
        door = any(s.ceiling_height - s.floor_height < 8 for s in secs)
        if door:
            kind = "door"
        elif e["blocking"] or len(secs) < 2:
            kind = "wall"
        else:
            s1, s2 = secs[0], secs[1]
            gap = min(s1.ceiling_height, s2.ceiling_height) - max(s1.floor_height, s2.floor_height)
            kind = "wall" if abs(s1.floor_height - s2.floor_height) > STEP_HEIGHT or gap < MIN_GAP else None
        if kind:
            out.append((a[0], a[1], b[0], b[1], kind))
    return out


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
        self.blocked = np.zeros((self.h, self.w), dtype=bool)
        for x1, y1, x2, y2, kind in walls:
            if kind != "wall":
                continue  # closed doors stay passable: Jev can open them
            n = max(2, int(math.hypot(x2 - x1, y2 - y1) // 4))
            for t in np.linspace(0, 1, n):
                cx, cy = self.cell(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
                self.blocked[cy, cx] = True
        self.seen = np.zeros_like(self.blocked)
        self.temp_blocked = {}  # cell -> expiry time, for things we got stuck on
        self.path = []

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

    def route(self, x, y, max_cells=40000):
        """Breadth-first search to the nearest unexplored open cell."""
        now = time.monotonic()
        self.temp_blocked = {c: t for c, t in self.temp_blocked.items() if t > now}
        start = self.cell(x, y)
        prev = {start: None}
        q = deque([start])
        goal = None
        blocked, seen, w, h = self.blocked, self.seen, self.w, self.h
        while q and len(prev) < max_cells:
            c = q.popleft()
            if not seen[c[1], c[0]]:
                goal = c
                break
            cx, cy = c
            for n in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (0 <= n[0] < w and 0 <= n[1] < h and n not in prev and not blocked[n[1], n[0]]
                        and n not in self.temp_blocked):
                    prev[n] = c
                    q.append(n)
        if goal is None:
            self.path = []
            return None
        path = []
        c = goal
        while c is not None:
            path.append(c)
            c = prev[c]
        path.reverse()
        self.path = [self.center(c) for c in path]
        return self.path

    def heading(self, x, y, lookahead=96):
        """Map angle toward a point ~lookahead units along the route, and route length."""
        if not self.path:
            return None, None
        length = len(self.path) * CELL
        target = self.path[-1]
        for p in self.path:
            if math.hypot(p[0] - x, p[1] - y) >= lookahead:
                target = p
                break
        return math.degrees(math.atan2(target[1] - y, target[0] - x)), length
