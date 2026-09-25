"""Map memory, checkpoints and route finding, all in code.

The game process turns the level's lines into plain wall segments once per map,
and reads which doors need keys, the switches and the exit from the WAD. The Jev
process keeps a coarse grid of the level (walls, doors, what the player has seen)
and a plan: an ordered list of checkpoints worked out from that map data, such as
"get the blue key, then open the blue door, then reach the exit". Jev only ever
sees the result in words; it still decides every action.
"""

import math
import time
from array import array
from dataclasses import dataclass, field

import numpy as np

CELL = 16          # map units per grid cell
CLEARANCE = 16     # Doom's player is a 32x32 square: a walkable cell's center keeps this far from walls
REACH = 40         # how close counts as having reached a line: a switch, the exit
SEEN_RADIUS = 96   # cells within this distance of the player, in line of sight, count as explored
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


def extract_walls(sectors, doors=None):
    """Classify every line of the map, once per map (runs in the game process).

    Returns a list of (x1, y1, x2, y2, kind, door) with kind "wall" or "door";
    door is the index of the closed door sector (-1 for walls). Two-sided lines
    between sectors of similar height are open and left out.
    """
    if doors is None:
        doors = door_sectors(sectors)  # pass the map-start set to keep door ids stable
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


@dataclass
class Plan:
    """What the player should be doing, worked out from the map. Jev sees it in words."""
    kind: str                 # "health", "exit", "key", "explore", "switch" or "none"
    goal: str                 # e.g. "get the blue key"
    steps: list = field(default_factory=list)  # the checkpoints after this one, in order
    target: tuple | None = None                # map point of the goal, if it is a thing
    path: list = field(default_factory=list)   # map points from the player to the goal
    door: int | None = None                    # the closed door the route crosses first
    door_at: float = 0.0                       # route distance to that door
    switch: int | None = None                  # index into switches, for "switch" goals


def _raster(x1, y1, x2, y2, cell):
    n = max(2, int(math.hypot(x2 - x1, y2 - y1) // 4))
    return {cell(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t) for t in np.linspace(0, 1, n)}


def _sight_rays(r):
    """For every cell offset in a disk of radius r, the cells a straight line to it crosses."""
    offs, samples = [], []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            n = max(abs(dx), abs(dy)) * 2
            pts = {(round(dx * i / n), round(dy * i / n)) for i in range(1, n)} - {(0, 0), (dx, dy)}
            offs.append((dx, dy))
            samples.append(sorted(pts))
    s = max(1, max(len(p) for p in samples))
    sx = np.zeros((len(offs), s), dtype=np.int64)
    sy = np.zeros_like(sx)
    valid = np.zeros(sx.shape, dtype=bool)
    for i, pts in enumerate(samples):
        for j, (px, py) in enumerate(pts):
            sx[i, j], sy[i, j], valid[i, j] = px, py, True
    o = np.array(offs, dtype=np.int64)
    return o[:, 0], o[:, 1], sx, sy, valid


class NavGrid:
    def __init__(self, walls, info=None, things=()):
        """walls from extract_walls; info from wad.read_map (None: every closed door is a
        plain door, and there are no known keys, switches or exits); things: solid
        decorations as (x, y, radius), like the pillars the player used to get stuck on."""
        xs = [w[0] for w in walls] + [w[2] for w in walls]
        ys = [w[1] for w in walls] + [w[3] for w in walls]
        self.x0, self.y0 = min(xs) - 64, min(ys) - 64
        self.w = int((max(xs) + 64 - self.x0) // CELL) + 1
        self.h = int((max(ys) + 64 - self.y0) // CELL) + 1
        self.things = list(things)
        self._geometry(walls)
        info = info or {}
        doors = info.get("doors")
        # Doors that open with the use key, and the key each needs. Closed sectors that only
        # a switch or a walk-over line opens are walls until they open.
        self.manual = set(self.door_cells) if doors is None else {d for d in self.door_cells if d in doors}
        self.door_key = {d: (doors or {}).get(d) for d in self.door_cells}
        self.switches = [tuple(s) for s in info.get("switches", [])]
        self.switch_cells = [self.near(*s[:4], REACH) for s in self.switches]
        exits = [tuple(e) for e in info.get("exits", [])]
        exits = [e for e in exits if not e[5]] or exits  # secret exits only if there is no other
        self.exits = exits
        self.exit_cells = set()
        for e in exits:
            self.exit_cells |= self.near(*e[:4], REACH)
        self.exit_point = ((exits[0][0] + exits[0][2]) / 2, (exits[0][1] + exits[0][3]) / 2) if exits else None
        self.open_doors = set()
        self.opened = set()        # doors the player has opened at least once
        self.failed = {}           # door -> expiry time, for doors that would not open
        self.fails = {}            # door -> how many times it would not open, for backoff
        self.pressed = set()       # switches the player has pressed
        self.seen = np.zeros_like(self.walls)
        self.temp_blocked = {}  # cell -> expiry time, for things we got stuck on
        self.plan = Plan("none", "look around")
        self.rays = _sight_rays(SEEN_RADIUS // CELL)
        self._rebuild()

    def _geometry(self, walls):
        """walls: the raw lines, for line of sight. near_walls: cells too close to a wall for
        the player to stand in, from exact distances, so a real 48-unit corridor stays open
        while a 16-unit gap beside a door jamb doesn't."""
        walls_arr = np.zeros((self.h, self.w), dtype=bool)
        near_walls = np.zeros_like(walls_arr)
        door_cells = {}         # door sector -> cells its closed faces cover
        door_near = {}          # door sector -> cells too close to its closed faces to stand in
        door_center = {}        # door sector -> map point in the middle of the door
        faces = {}
        for x1, y1, x2, y2, kind, door in walls:
            cells = _raster(x1, y1, x2, y2, self.cell)
            near = self.near(x1, y1, x2, y2, CLEARANCE, square=True)
            if kind == "door":
                door_cells.setdefault(door, set()).update(cells)
                door_near.setdefault(door, set()).update(near)
                faces.setdefault(door, []).append(((x1 + x2) / 2, (y1 + y2) / 2))
            else:
                for cx, cy in cells:
                    walls_arr[cy, cx] = True
                for cx, cy in near:
                    near_walls[cy, cx] = True
        for x, y, r in self.things:  # the player's box and the thing's box must not overlap
            cx0, cy0 = self.cell(x - r - CLEARANCE, y - r - CLEARANCE)
            cx1, cy1 = self.cell(x + r + CLEARANCE, y + r + CLEARANCE)
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    ccx, ccy = self.center((cx, cy))
                    if max(abs(ccx - x), abs(ccy - y)) < r + CLEARANCE:
                        near_walls[cy, cx] = True
        for door, mids in faces.items():
            door_center[door] = (sum(m[0] for m in mids) / len(mids), sum(m[1] for m in mids) / len(mids))
        self.walls, self.near_walls = walls_arr, near_walls
        self.door_cells, self.door_near, self.door_center = door_cells, door_near, door_center

    def set_walls(self, walls):
        """Lifts, stairs and floors moved: redo the geometry, keeping everything learned."""
        self._geometry(walls)
        self._rebuild()

    def near(self, x1, y1, x2, y2, r, square=False):
        """Cells whose centers are within r map units of a line segment. square: measure
        like Doom's collision box (a player centered there would touch the line)."""
        cx0, cy0 = self.cell(min(x1, x2) - r, min(y1, y2) - r)
        cx1, cy1 = self.cell(max(x1, x2) + r, max(y1, y2) + r)
        gx, gy = np.meshgrid(np.arange(cx0, cx1 + 1), np.arange(cy0, cy1 + 1))
        px, py = self.x0 + (gx + 0.5) * CELL, self.y0 + (gy + 0.5) * CELL
        if square:  # sample the line every 2 units; box distance to the nearest sample
            t = np.linspace(0, 1, max(2, int(math.hypot(x2 - x1, y2 - y1) // 2) + 1))
            sx, sy = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            d = np.maximum(np.abs(px[..., None] - sx), np.abs(py[..., None] - sy)).min(axis=-1)
        else:
            ex, ey = x2 - x1, y2 - y1
            ll = ex * ex + ey * ey
            t = np.clip(((px - x1) * ex + (py - y1) * ey) / ll, 0, 1) if ll else np.zeros_like(px)
            d = np.hypot(px - (x1 + t * ex), py - (y1 + t * ey))
        m = d < r
        return set(zip(gx[m].tolist(), gy[m].tolist()))

    def _rebuild(self):
        """Closed doors block like walls: `tight` (raw lines, for line of sight) and `blocked`
        (too close to stand in, for routes). Arrays are replaced, never edited, so a plan
        running in a thread stays consistent."""
        tight = self.walls.copy()
        blocked = self.near_walls.copy()
        door_at = {}  # cell -> closed door sector
        for door, cells in self.door_cells.items():
            if door in self.open_doors:
                continue
            for cx, cy in cells:
                tight[cy, cx] = True
                door_at[(cx, cy)] = door
            for cx, cy in self.door_near[door]:
                blocked[cy, cx] = True
        self.tight, self.blocked, self.door_at = tight, blocked, door_at

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

    def fail_door(self, door):
        """A door that would not open: leave it alone for a while, longer each time."""
        self.fails[door] = self.fails.get(door, 0) + 1
        self.failed[door] = time.monotonic() + 30 * 2 ** min(3, self.fails[door] - 1)

    def cell(self, x, y):
        cx = min(self.w - 1, max(0, int((x - self.x0) // CELL)))
        cy = min(self.h - 1, max(0, int((y - self.y0) // CELL)))
        return cx, cy

    def center(self, c):
        return self.x0 + (c[0] + 0.5) * CELL, self.y0 + (c[1] + 0.5) * CELL

    def mark_seen(self, x, y):
        """Mark cells around the player that are in line of sight as explored, so a room
        on the other side of a wall doesn't count as visited. Returns the newly seen cells."""
        cx, cy = self.cell(x, y)
        ox, oy, sx, sy, valid = self.rays
        h, w, tight = self.h, self.w, self.tight
        tx, ty = cx + ox, cy + oy
        inside = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        wall = tight[np.clip(cy + sy, 0, h - 1), np.clip(cx + sx, 0, w - 1)] & valid
        vis = inside & ~wall.any(axis=1)
        tx, ty = tx[vis], ty[vis]
        new = ~self.seen[ty, tx]
        self.seen[ty, tx] = True
        return [(int(a), int(b)) for a, b in zip(tx[new], ty[new])]

    def block_ahead(self, x, y, angle, seconds=20):
        """We got stuck here: treat the spot in front of us as a wall for a while."""
        a = math.radians(angle)
        until = time.monotonic() + seconds
        for d in (16, 32):
            self.temp_blocked[self.cell(x + math.cos(a) * d, y + math.sin(a) * d)] = until

    def usable_doors(self, keys):
        """Closed doors the player can open right now."""
        return {d for d in set(self.door_at.values()) & self.manual
                if d not in self.failed and (self.door_key[d] is None or self.door_key[d] in keys)}

    def locked_doors(self, keys):
        """Closed doors that need a key the player doesn't have, by color."""
        out = {}
        for d in set(self.door_at.values()) & self.manual:
            k = self.door_key[d]
            if k and k not in keys:
                out.setdefault(k, []).append(d)
        return out

    def _passable(self, doors):
        """Cells the player can stand in, as a flat bytearray, with the given closed doors
        counted as open (the player can open them on the way)."""
        pas = ~self.near_walls
        closed = set(self.door_at.values()) - set(doors)
        for d in closed:
            for cx, cy in self.door_near[d]:
                pas[cy, cx] = False
        for cx, cy in tuple(self.temp_blocked):
            pas[cy, cx] = False
        pas[0, :] = pas[-1, :] = pas[:, 0] = pas[:, -1] = False  # so neighbours stay in range
        return bytearray(pas.astype(np.uint8).tobytes())

    def _flood(self, pas, start):
        """Breadth-first search over the whole reachable area: (prev, cells in BFS order)."""
        w = self.w
        prev = array("i", [-2]) * len(pas)
        prev[start] = -1
        order = [start]
        i = 0
        while i < len(order):
            c = order[i]
            i += 1
            for n in (c + 1, c - 1, c + w, c - w):
                if pas[n] and prev[n] == -2:
                    prev[n] = c
                    order.append(n)
        return prev, order

    def _path(self, prev, goal):
        out = []
        while goal >= 0:
            out.append(goal)
            goal = prev[goal]
        out.reverse()
        return out

    def plan_route(self, x, y, health, keys, items):
        """Work out the current checkpoint and the ones after it, and a route to it.

        keys: colors held. items: (name, x, y, kind) still on the map, kind "key" or
        "health"; only ones in explored cells count as known. In order of preference:
        1. health, when low and some is known and reachable;
        2. the exit, when it can be reached by opening doors the player can open;
        3. a key a locked door needs, when its spot has been seen;
        4. the nearest unexplored area, keeping clear of walls, then through tight spots
           and doors the player can open;
        5. a switch not yet pressed.
        """
        now = time.monotonic()
        # dict() takes a snapshot in one step: the Jev loop may add entries while this runs.
        self.temp_blocked = {c: t for c, t in dict(self.temp_blocked).items() if t > now}
        self.failed = {d: t for d, t in dict(self.failed).items() if t > now}
        w = self.w
        flat = lambda c: c[1] * w + c[0]
        start = flat(self.cell(x, y))
        seen = self.seen.ravel()
        usable = self.usable_doors(keys)
        locked = self.locked_doors(keys)
        # Two searches: the area reachable without opening a door (explored first), and
        # everything reachable by opening the doors the player can open.
        near_prev, near_order = self._flood(self._passable(()), start)
        prev, order = self._flood(self._passable(usable), start)
        reach = set(order)

        def route(c):
            return self._path(prev, c)

        def nearest(cells):
            cells = {flat(c) for c in cells} & reach
            return next((c for c in order if c in cells), None) if cells else None

        known = [(n, ix, iy, k) for n, ix, iy, k in items if seen[flat(self.cell(ix, iy))]]
        missing = sorted(set(locked) - set(keys))   # key colors locked doors still need
        exit_cell = nearest(self.exit_cells)
        after = []
        if exit_cell is None:
            after = [f"get the {k} key and open the {k} door" for k in missing] + (["reach the exit"] if self.exits else [])

        plan = None
        if health < 40:
            spots = {self.cell(ix, iy): (ix, iy) for n, ix, iy, k in known if k == "health"}
            c = nearest(spots)
            if c is not None:
                plan = Plan("health", "pick up health", ["reach the exit"] if exit_cell is not None else after,
                            target=spots[(c % w, c // w)], path=route(c))
        if plan is None and exit_cell is not None:
            plan = Plan("exit", "reach the exit", [], target=self.exit_point, path=route(exit_cell))
        if plan is None:
            for n, ix, iy, k in known:
                if k != "key" or n not in missing:
                    continue
                c = nearest({self.cell(ix, iy)})
                if c is not None:
                    rest = [f"open the {n} door"] + [s for s in after if n not in s]
                    plan = Plan("key", f"get the {n} key", rest, target=(ix, iy), path=route(c))
                    break
        if plan is None:
            c = next((c for c in near_order if not seen[c]), None)
            p = self._path(near_prev, c) if c is not None else None
            if p is None:  # nothing left here: go through a door
                c = next((c for c in order if not seen[c]), None)
                p = route(c) if c is not None else None
            if p is not None:
                unknown = [k for k in missing if not any(n == k and kk == "key" for n, _, _, kk in known)]
                steps = [f"find the {k} key" for k in unknown[:1]] + (after or ["find the exit"])
                plan = Plan("explore", "explore", steps, path=p)
        if plan is None:
            for i, cells in enumerate(self.switch_cells):
                key = self.switches[i][4]
                if i in self.pressed or (key and key not in keys):
                    continue
                c = nearest(cells)
                if c is not None:
                    s = self.switches[i]
                    plan = Plan("switch", "press the switch", after, target=((s[0] + s[2]) / 2, (s[1] + s[3]) / 2),
                                path=route(c), switch=i)
                    break
        if plan is None:
            plan = Plan("none", "look around for a way on", after)

        # The first closed door the route crosses: Jev has to open it on the way.
        cells = plan.path
        plan.path = [self.center((c % w, c // w)) for c in cells]
        door_at = self.door_at
        for i, c in enumerate(cells):
            d = door_at.get((c % w, c // w))
            if d is not None:
                plan.door, plan.door_at = d, i * CELL
                break
        self.plan = plan
        return plan

    def clear(self, x1, y1, x2, y2):
        """True if a straight walk between two map points keeps a player's width from every
        wall (ignoring the first 16 units, in case the player is already brushing one)."""
        d = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(d // 8))
        for i in range(1, n + 1):
            if d * i / n < 16:
                continue
            cx, cy = self.cell(x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n)
            if self.blocked[cy, cx]:
                return False
        return True

    def nearest_door(self, x, y):
        """The closest closed door that opens with the use key: (door sector, center)."""
        doors = [(d, c) for d, c in self.door_center.items()
                 if d in self.manual and d not in self.open_doors and d not in self.failed]
        return min(doors, key=lambda d: math.hypot(d[1][0] - x, d[1][1] - y), default=(None, None))

    def heading(self, x, y, lookahead=96):
        """Map angle toward the farthest point along the route (up to lookahead) that can
        be walked to in a straight line, and the route length. Close to a door on the way
        aim at the middle of the door; close to the goal, at the goal itself."""
        plan = self.plan
        if not plan.path:
            return None, None
        length = len(plan.path) * CELL
        if plan.door is not None and plan.door_at < lookahead:
            target = self.door_center[plan.door]
        elif plan.target and length < lookahead:
            target = plan.target
        else:
            target = plan.path[0]
            for p in plan.path:
                if not self.clear(x, y, p[0], p[1]):
                    break
                target = p
                if math.hypot(p[0] - x, p[1] - y) >= lookahead:
                    break
        return math.degrees(math.atan2(target[1] - y, target[0] - x)), length
