"""UI process: the game view plus a retro panel showing Jev's decisions live.

Keys light up from the exact buttons the game applied to the frame being shown,
so there is no lag between what you see pressed and what the player does.
"""

import math
import os
import queue
import time
from collections import deque

import numpy as np

import shared as S

WIN_W, WIN_H = 1900, 1000
VIEW_X, VIEW_Y, VIEW_W, VIEW_H = 12, 20, 1280, 960
PX, PW = VIEW_X + VIEW_W + 20, WIN_W - (VIEW_X + VIEW_W + 20) - 12  # panel x and width

BG = (14, 12, 12)
PANEL = (34, 30, 28)
EDGE_LIGHT = (96, 88, 80)
EDGE_DARK = (8, 6, 6)
TEXT = (220, 210, 190)
DIM = (130, 120, 108)
DOOM_RED = (230, 40, 30)
DOOM_YELLOW = (250, 210, 60)
JEV_GREEN = (80, 255, 90)
REFLEX_ORANGE = (255, 120, 20)

ACTION_COLORS = {
    "turn_left": (80, 160, 255), "turn_right": (170, 110, 255), "move_forward": (80, 230, 100),
    "move_back": (60, 200, 190), "shoot": (255, 60, 50), "use_open_door": (250, 210, 60),
}
ACTION_LABELS = {
    "turn_left": "TURN LEFT", "turn_right": "TURN RIGHT", "move_forward": "FORWARD",
    "move_back": "BACK", "shoot": "SHOOT", "use_open_door": "OPEN DOOR",
}


FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


class Text:
    """Chunky pixel text: a small bitmap-rendered font scaled up nearest-neighbour."""

    def __init__(self, pg):
        self.pg = pg
        bold = FONT_DIR + "DejaVuSansMono-Bold.ttf"
        regular = FONT_DIR + "DejaVuSansMono.ttf"
        self.pix_font = pg.font.Font(bold if os.path.exists(bold) else None, 9)
        self.small = pg.font.Font(bold if os.path.exists(bold) else None, 11)
        self.mono = pg.font.Font(regular if os.path.exists(regular) else None, 15)
        self.cache = {}

    def _get(self, key, make):
        s = self.cache.get(key)
        if s is None:
            if len(self.cache) > 3000:
                self.cache.clear()
            s = self.cache[key] = make()
        return s

    def pix(self, text, color, scale=2):
        def make():
            if scale == 1:
                return self.small.render(text.upper(), False, color)
            s = self.pix_font.render(text.upper(), False, color)
            return self.pg.transform.scale(s, (s.get_width() * scale, s.get_height() * scale))
        return self._get(("p", text, color, scale), make)

    def body(self, text, color):
        return self._get(("m", text, color), lambda: self.mono.render(text, False, color))


def bevel(pg, surf, rect, fill=PANEL, width=3):
    pg.draw.rect(surf, fill, rect)
    x, y, w, h = rect
    for i in range(width):
        pg.draw.line(surf, EDGE_LIGHT, (x + i, y + i), (x + w - 1 - i, y + i))
        pg.draw.line(surf, EDGE_LIGHT, (x + i, y + i), (x + i, y + h - 1 - i))
        pg.draw.line(surf, EDGE_DARK, (x + i, y + h - 1 - i), (x + w - 1 - i, y + h - 1 - i))
        pg.draw.line(surf, EDGE_DARK, (x + w - 1 - i, y + i), (x + w - 1 - i, y + h - 1 - i))


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Automap:
    """Low-poly overhead map drawn over the game view: walls, doors, explored
    area, the route code is steering along, and the player arrow."""
    SCALE = 5.0  # map units per pixel
    KEY_COLORS = {"blue": (70, 120, 255), "red": (240, 60, 60), "yellow": (250, 230, 40)}
    GOAL_COLORS = {"exit": (80, 230, 110), "key": (90, 160, 255), "health": (255, 255, 255),
                   "switch": (80, 220, 230)}
    SIZE = 300

    def __init__(self, pg, walls, info=None, things=()):
        import navigation as N
        self.pg, self.N = pg, N
        xs = [w[0] for w in walls] + [w[2] for w in walls]
        ys = [w[1] for w in walls] + [w[3] for w in walls]
        self.x0, self.y1 = min(xs) - 32, max(ys) + 32
        w = int((max(xs) + 32 - self.x0) / self.SCALE) + 1
        h = int((self.y1 - (min(ys) - 32)) / self.SCALE) + 1
        self.lines = pg.Surface((w, h), pg.SRCALPHA)
        self.seen = pg.Surface((w, h), pg.SRCALPHA)
        self.info, self.things = info or {}, things or ()
        self.draw_lines(walls)
        self.path, self.target, self.kind = [], None, None
        self.box = pg.Surface((self.SIZE, self.SIZE), pg.SRCALPHA)

    def draw_lines(self, walls):
        """Walls, doors (colored by the key they need), solid decorations and the exit."""
        pg = self.pg
        self.lines.fill((0, 0, 0, 0))
        keys = self.info.get("doors") or {}
        for x1, y1, x2, y2, kind, door in walls:
            col = (190, 180, 165) if kind != "door" else self.KEY_COLORS.get(keys.get(door), (250, 210, 60))
            pg.draw.line(self.lines, col, self.px(x1, y1), self.px(x2, y2), 2 if kind == "door" else 1)
        for x, y, r in self.things:  # pillars, lamps, barrels
            pg.draw.rect(self.lines, (190, 180, 165), (*self.px(x - r, y + r), 2 * r / self.SCALE, 2 * r / self.SCALE), 1)
        for x1, y1, x2, y2, *_ in self.info.get("exits", []):
            pg.draw.line(self.lines, (80, 230, 110), self.px(x1, y1), self.px(x2, y2), 3)

    def px(self, x, y):
        return (x - self.x0) / self.SCALE, (self.y1 - y) / self.SCALE

    def update(self, path, seen, target=None, kind=None):
        self.path, self.target, self.kind = path, target, kind
        c = max(2, int(self.N.CELL / self.SCALE) + 1)
        half = self.N.CELL / 2
        for x, y in seen:  # explored cell centers, in map units
            self.pg.draw.rect(self.seen, (40, 110, 50, 150), (*self.px(x - half, y + half), c, c))

    def draw(self, scr, x, y, angle, t):
        pg, box, half = self.pg, self.box, self.SIZE // 2
        box.fill((0, 0, 0, 170))
        px, py = self.px(x, y)
        off = (half - px, half - py)
        box.blit(self.seen, off)
        box.blit(self.lines, off)
        if len(self.path) > 1:
            pts = [(self.px(*p)[0] + off[0], self.px(*p)[1] + off[1]) for p in self.path]
            pg.draw.lines(box, (255, 60, 50), False, pts, 2)
        if self.target:  # the current checkpoint: a key, the exit, health, a switch, or a door on the way
            tx, ty = self.px(*self.target)
            pg.draw.circle(box, self.GOAL_COLORS.get(self.kind, (250, 210, 60)), (tx + off[0], ty + off[1]), 6 + 2 * math.sin(time.monotonic() * 6), 2)
        a = math.radians(angle)
        tip = (half + math.cos(a) * 10, half - math.sin(a) * 10)
        l = (half + math.cos(a + 2.5) * 7, half - math.sin(a + 2.5) * 7)
        r = (half + math.cos(a - 2.5) * 7, half - math.sin(a - 2.5) * 7)
        pg.draw.polygon(box, (255, 255, 255), [tip, l, r])
        X, Y = VIEW_X + VIEW_W - self.SIZE - 12, VIEW_Y + 12
        scr.blit(box, (X, Y))
        pg.draw.rect(scr, (96, 88, 80), (X - 2, Y - 2, self.SIZE + 4, self.SIZE + 4), 2)
        scr.blit(t.pix("AUTOMAP  RED = ROUTE", (190, 180, 165), 1), (X + 6, Y + self.SIZE - 18))


VIEW_RECT = (VIEW_X, VIEW_Y, VIEW_W, VIEW_H)
HEADER_RECT = (PX, 60, PW, 40)
KEYS_RECT = (PX, 124, PW, 212)
DECISION_RECT = (PX, 348, PW, 200)
TAPE_RECT = (PX, 580, PW, 70)
STATE_RECT = (PX, 680, PW, 196)
STATS_RECT = (PX, 908, PW, 80)


class UI:
    def __init__(self, pg, shm, ui_q, stop):
        self.pg, self.ui_q, self.stop = pg, ui_q, stop
        self.ctrl, self.metas, self.frames = S.views(shm.buf)
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
        pg.display.init()
        pg.font.init()
        pg.display.set_caption("JEV PLAYS DOOM")
        self.screen = pg.display.set_mode((WIN_W, WIN_H))
        self.t = Text(pg)
        # Frames arrive in the window's own pixel layout, so scaling needs no conversion.
        self.src = pg.Surface((S.W, S.H), 0, self.screen)
        self.src_px = np.asarray(self.src.get_view("2")).T.view(np.uint8).reshape(S.H, S.W, 4)
        self.view = pg.Surface((VIEW_W, VIEW_H), 0, self.screen)
        self.map_name = ""
        self.decision = None
        self.stats = {}
        self.tape = deque()  # (time, action)
        self.reflex_tape = deque()
        self.last_reflex = (0, 0)
        self.static = self.build_static()
        self.last_seq = -1
        self.meta = np.zeros(S.META_LEN)
        self.frame_ms = deque(maxlen=35 * 30)
        self.shown = 0
        self.automap = None

    # --- static chrome ---------------------------------------------------
    def build_static(self):
        pg, t = self.pg, self.t
        s = pg.Surface((WIN_W, WIN_H))
        s.fill(BG)
        for y in range(0, WIN_H, 4):  # faint scanlines
            pg.draw.line(s, (18, 16, 16), (0, y), (WIN_W, y))
        bevel(pg, s, (VIEW_X - 6, VIEW_Y - 6, VIEW_W + 12, VIEW_H + 12), fill=(0, 0, 0), width=4)
        s.blit(t.pix("JEV PLAYS DOOM", DOOM_RED, 4), (PX, 14))
        for y, title in ((104, "KEYS PRESSED"), (348, "JEV DECISION"), (560, "LIVE DECISIONS"),
                         (660, "FED TO JEV"), (888, "STATS")):
            s.blit(t.pix(title, DOOM_YELLOW, 2), (PX, y))
        bevel(pg, s, (PX, 124, PW, 212))
        bevel(pg, s, (PX, 368, PW, 180))
        bevel(pg, s, (PX, 580, PW, 70))
        bevel(pg, s, (PX, 680, PW, 196))
        bevel(pg, s, (PX, 908, PW, 80))
        return s

    # --- data ------------------------------------------------------------
    def drain(self):
        now = time.monotonic()
        while True:
            try:
                msg = self.ui_q.get_nowait()
            except queue.Empty:
                break
            if msg["type"] == "decision":
                self.decision = msg
                self.tape.append((now, msg["chosen"]))
            elif msg["type"] == "stats":
                self.stats = msg
            elif msg["type"] == "map":
                self.map_name = msg["name"]
                self.automap = None
            elif msg["type"] == "geom":
                self.automap = Automap(self.pg, msg["walls"], msg.get("info"), msg.get("things"))
            elif msg["type"] == "walls" and self.automap:
                self.automap.draw_lines(msg["walls"])
            elif msg["type"] == "nav" and self.automap:
                self.automap.update(msg["path"], msg["seen"], msg.get("target"), msg.get("kind"))
        while self.tape and now - self.tape[0][0] > 7:
            self.tape.popleft()
        while self.reflex_tape and now - self.reflex_tape[0][0] > 7:
            self.reflex_tape.popleft()

    def grab_frame(self):
        slot = int(self.ctrl[0])
        seq = int(self.ctrl[1])
        if seq == self.last_seq:
            return False
        self.last_seq = seq
        self.shown += 1
        self.meta = self.metas[slot].copy()
        self.src_px[:] = self.frames[slot]
        self.pg.transform.scale(self.src, (VIEW_W, VIEW_H), self.view)
        m = self.meta
        if m[S.REFLEX_SHOOT] or m[S.REFLEX_STOP]:
            now = time.monotonic()
            if not self.reflex_tape or now - self.reflex_tape[-1][0] > 0.05:
                self.reflex_tape.append((now, "shoot" if m[S.REFLEX_SHOOT] else "stop"))
        return True

    # --- drawing ---------------------------------------------------------
    def draw_view(self):
        pg, t, m, scr = self.pg, self.t, self.meta, self.screen
        scr.blit(self.view, (VIEW_X, VIEW_Y))
        # What Jev is told about: boxes around labelled things, colored by distance.
        for i in range(int(m[S.N_BOXES])):
            l, tp, r, b, enemy, dist = m[S.BOXES + i * 6: S.BOXES + i * 6 + 6]
            x0, y0 = VIEW_X + l * VIEW_W, VIEW_Y + tp * VIEW_H
            x1, y1 = VIEW_X + r * VIEW_W, VIEW_Y + b * VIEW_H
            if enemy:
                word, col = (("VERY CLOSE", DOOM_RED) if dist < 150 else ("CLOSE", REFLEX_ORANGE) if dist < 400
                             else ("MEDIUM", DOOM_YELLOW) if dist < 800 else ("FAR", DIM))
            else:
                word, col = "ITEM", (90, 200, 220)
            c = max(6, min(18, (x1 - x0) / 4))
            for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                pg.draw.line(scr, col, (px, py), (px + dx * c, py), 3)
                pg.draw.line(scr, col, (px, py), (px, py + dy * c), 3)
            scr.blit(t.pix(word, col, 2), (x0, max(VIEW_Y, y0 - 20)))
        # Reflex zone: the center 5% of the screen.
        cx = VIEW_X + VIEW_W // 2
        for dx in (-VIEW_W * 0.025, VIEW_W * 0.025):
            pg.draw.line(scr, (255, 255, 255), (cx + dx, VIEW_Y + VIEW_H * .44), (cx + dx, VIEW_Y + VIEW_H * .47), 2)
            pg.draw.line(scr, (255, 255, 255), (cx + dx, VIEW_Y + VIEW_H * .53), (cx + dx, VIEW_Y + VIEW_H * .56), 2)
        if self.automap and not np.isnan(m[S.POS_X]):
            self.automap.draw(scr, m[S.POS_X], m[S.POS_Y], m[S.ANGLE], t)
        banners = []
        if m[S.REFLEX_SHOOT]:
            banners.append(("REFLEX: FIRE", REFLEX_ORANGE))
        if m[S.REFLEX_STOP]:
            banners.append(("REFLEX: WALL STOP", REFLEX_ORANGE))
        for i, (txt, col) in enumerate(banners):
            s = t.pix(txt, col, 3)
            scr.blit(s, (VIEW_X + (VIEW_W - s.get_width()) // 2, VIEW_Y + 24 + i * 34))
        age = time.monotonic() - m[S.STATUS_T]
        if m[S.STATUS] != S.PLAYING and age < 1.8:
            txt, col = ("YOU DIED", DOOM_RED) if m[S.STATUS] == S.DIED else ("LEVEL CLEAR", JEV_GREEN)
            s = t.pix(txt, col, 8)
            s.set_alpha(int(255 * max(0.0, 1 - age / 1.8)))
            scr.blit(s, (VIEW_X + (VIEW_W - s.get_width()) // 2, VIEW_Y + VIEW_H // 3))

    def keycap(self, x, y, w, h, label, lit, color, glyph=None, blocked=False):
        pg, scr = self.pg, self.screen
        press = 6 * lit
        face = mix((70, 64, 58), color, lit)
        skirt = mix((36, 32, 30), tuple(c // 2 for c in color), lit)
        pg.draw.polygon(scr, (10, 8, 8), [(x, y + 10), (x + w, y + 10), (x + w, y + h), (x, y + h)])
        top = y + press
        pg.draw.polygon(scr, skirt, [(x + 4, top + h - 22), (x + w - 4, top + h - 22),
                                     (x + w, y + h), (x, y + h)])
        pg.draw.polygon(scr, face, [(x + 10, top), (x + w - 10, top), (x + w - 4, top + h - 22),
                                    (x + 4, top + h - 22)])
        pg.draw.line(scr, mix(face, (255, 255, 255), 0.35), (x + 10, top), (x + w - 10, top), 2)
        cx, cy = x + w // 2, top + (h - 22) // 2
        ink = (20, 16, 14) if lit > 0.5 else TEXT
        if glyph:
            k = 11
            cy -= 8
            pts = {"up": [(cx, cy - k), (cx - k, cy + k // 2), (cx + k, cy + k // 2)],
                   "down": [(cx, cy + k), (cx - k, cy - k // 2), (cx + k, cy - k // 2)],
                   "left": [(cx - k, cy), (cx + k // 2, cy - k), (cx + k // 2, cy + k)],
                   "right": [(cx + k, cy), (cx - k // 2, cy - k), (cx - k // 2, cy + k)]}[glyph]
            pg.draw.polygon(scr, ink, pts)
            cap = self.t.pix({"up": "FWD", "down": "BACK", "left": "TURN L", "right": "TURN R"}[glyph], ink, 1)
            scr.blit(cap, (cx - cap.get_width() // 2, cy + 16))
        else:
            s = self.t.pix(label, ink, 2)
            scr.blit(s, (cx - s.get_width() // 2, cy - s.get_height() // 2))
        if blocked:
            pg.draw.line(scr, DOOM_RED, (x + 12, top + 6), (x + w - 12, top + h - 28), 4)
            pg.draw.line(scr, DOOM_RED, (x + w - 12, top + 6), (x + 12, top + h - 28), 4)

    def overlay_active(self):
        return self.meta[S.STATUS] != S.PLAYING and time.monotonic() - self.meta[S.STATUS_T] < 2.0

    def draw_keys(self):
        m, t, scr = self.meta, self.t, self.screen
        turn = m[S.TURN] / S.MAX_TURN
        kw, kh, x0, y0 = 88, 78, PX + 20, 140
        jev = S.ACTIONS[int(m[S.JEV_IDX])]
        wants_fwd = jev == "move_forward"
        self.keycap(x0 + kw + 8, y0, kw, kh, "", float(m[S.FWD]), JEV_GREEN, "up",
                    blocked=bool(m[S.REFLEX_STOP]) and wants_fwd)
        y1 = y0 + kh + 8
        self.keycap(x0, y1, kw, kh, "", min(1.0, max(0.0, -turn) * 1.4), JEV_GREEN, "left")
        self.keycap(x0 + kw + 8, y1, kw, kh, "", float(m[S.BACK]), JEV_GREEN, "down")
        self.keycap(x0 + 2 * (kw + 8), y1, kw, kh, "", min(1.0, max(0.0, turn) * 1.4), JEV_GREEN, "right")
        xr = x0 + 3 * (kw + 8) + 24
        fire_col = REFLEX_ORANGE if m[S.REFLEX_SHOOT] else JEV_GREEN
        self.keycap(xr, y0, PX + PW - 20 - xr, kh, "FIRE", float(m[S.ATTACK]), fire_col)
        self.keycap(xr, y1, PX + PW - 20 - xr, kh, "USE", float(m[S.USE] or jev == "use_open_door"), JEV_GREEN)
        scr.blit(t.pix("GREEN = JEV   ORANGE = REFLEX   RED X = BLOCKED", DIM, 1), (PX + 20, 314))

    def draw_decision(self):
        t, scr, pg = self.t, self.screen, self.pg
        d = self.decision
        probs = d["probs"] if d else {}
        chosen = d["chosen"] if d else None
        y = 384
        for a in S.ACTIONS:
            p = probs.get(a, 0.0)
            col = ACTION_COLORS[a]
            on = a == chosen
            scr.blit(t.pix((">" if on else " ") + ACTION_LABELS[a], col if on else TEXT, 2), (PX + 14, y + 3))
            bx, blocks = PX + 210, 22
            for i in range(blocks):
                filled = i < round(p * blocks)
                pg.draw.rect(scr, col if filled else (52, 46, 42), (bx + i * 14, y + 2, 11, 18))
            scr.blit(t.pix(f"{int(round(p * 100)):3d}%", TEXT, 2), (bx + blocks * 14 + 6, y + 3))
            y += 26
        if d:
            scr.blit(t.pix(f"LATENCY {d['latency_ms']:.0f} MS", DIM, 1), (PX + PW - 130, 352))

    def draw_tape(self):
        pg, scr, now = self.pg, self.screen, time.monotonic()
        x0, x1, y = PX + 12, PX + PW - 12, 590
        span = x1 - x0
        items = list(self.tape)
        for i, (ts, a) in enumerate(items):
            end = items[i + 1][0] if i + 1 < len(items) else now
            xa = x1 - (now - ts) / 6.0 * span
            xb = x1 - (now - end) / 6.0 * span
            if xb < x0:
                continue
            pg.draw.rect(scr, ACTION_COLORS[a], (max(x0, xa), y, max(2, xb - max(x0, xa) - 1), 30))
        for ts, kind in self.reflex_tape:
            xa = x1 - (now - ts) / 6.0 * span
            if xa >= x0:
                pg.draw.rect(scr, REFLEX_ORANGE, (xa, y + 36, 3, 12))
        scr.blit(self.t.pix("NOW", DIM, 1), (x1 - 22, y + 38))
        scr.blit(self.t.pix("-6S", DIM, 1), (x0, y + 38))

    def draw_state(self):
        d = self.decision
        text = d["text"] if d else "waiting for first decision..."
        y, max_chars = 692, (PW - 28) // 9
        for line in text.split("\n"):
            while line and y < 866:
                cut = line if len(line) <= max_chars else line[:line.rfind(" ", 0, max_chars)]
                col = TEXT
                if ": yes" in cut or "very close" in cut:
                    col = DOOM_YELLOW
                self.screen.blit(self.t.body(cut, col), (PX + 14, y))
                y += 18
                line = line[len(cut):].lstrip()

    def draw_stats(self):
        m, st, t, scr = self.meta, self.stats, self.t, self.screen
        total = max(1.0, m[S.TOTAL_TICS])
        cells = [
            ("DECISIONS/S", f"{st.get('dps', 0):.1f}"), ("JEV MEDIAN", f"{st.get('median_ms', 0):.0f}MS"),
            ("IN FLIGHT", f"{st.get('inflight', 0)}"), ("GAME FPS", f"{m[S.TICS_PER_S]:.0f}"),
            ("REFLEX", f"{100 * m[S.REFLEX_TICS] / total:.0f}%"), ("KILLS", f"{int(m[S.KILLS])}"),
            ("DEATHS", f"{int(m[S.DEATHS])}"), ("LEVELS", f"{int(m[S.LEVELS_CLEARED])}"),
        ]
        cw = (PW - 20) // 4
        for i, (k, v) in enumerate(cells):
            x, y = PX + 12 + (i % 4) * cw, 916 + (i // 4) * 36
            scr.blit(t.pix(k, DIM, 1), (x, y))
            scr.blit(t.pix(v, DOOM_YELLOW, 2), (x, y + 10))
        sub = f"JEV-1.13.0  {self.map_name}".upper()
        scr.blit(t.pix(sub, TEXT, 2), (PX, 62))
        hp = f"HEALTH {int(m[S.HEALTH])}  AMMO {int(m[S.AMMO])}  ARMOR {int(m[S.ARMOR])}"
        scr.blit(t.pix(hp, DOOM_RED, 2), (PX, 82))

    def run(self):
        pg = self.pg
        last = last_slow = time.monotonic()
        shown_decision, first = None, True
        while not self.stop.is_set():
            # Pace on the game: draw as soon as a new frame arrives, so each game
            # frame is shown once (no beat between two 35 Hz clocks).
            deadline = time.monotonic() + 0.05
            while int(self.ctrl[1]) == self.last_seq and time.monotonic() < deadline:
                time.sleep(0.001)
            now = time.monotonic()
            self.frame_ms.append((now - last) * 1000)
            last = now
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                    self.stop.set()
            self.drain()
            fresh = self.grab_frame()
            self.screen.blit(self.static, (0, 0))
            self.draw_view()
            self.draw_keys()
            self.draw_decision()
            self.draw_tape()
            self.draw_state()
            self.draw_stats()
            # Only push what changed: the full window is too slow to flip every frame.
            dirty = [KEYS_RECT, TAPE_RECT]
            if fresh or self.overlay_active():
                dirty.append(VIEW_RECT)
            if self.decision is not shown_decision:
                shown_decision = self.decision
                dirty += [DECISION_RECT, STATE_RECT]
            if now - last_slow > 0.25:
                last_slow = now
                dirty += [HEADER_RECT, STATS_RECT]
            if first:
                first = False
                pg.display.flip()
            else:
                pg.display.update(dirty)
        pg.quit()


def run(shm, ui_q, stop, result_q):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame as pg
    ui = None
    t0 = time.monotonic()
    try:
        ui = UI(pg, shm, ui_q, stop)
        ui.run()
    finally:
        stop.set()
        ms = sorted(ui.frame_ms) if ui else []
        result_q.put({"proc": "ui", "fps": (ui.shown if ui else 0) / max(1e-6, time.monotonic() - t0),
                      "frame_p99_ms": ms[int(len(ms) * .99)] if ms else 0, "frame_max_ms": ms[-1] if ms else 0})
