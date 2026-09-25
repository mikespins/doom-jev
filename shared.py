"""Shared-memory layout between the game, Jev and UI processes.

The game process writes every tic into one of three slots (frame + metadata),
then publishes the slot index. The UI reads the newest slot, so the keys it
lights up are exactly the buttons applied to the frame it is showing.
"""

import numpy as np

W, H = 640, 480  # game render resolution; the UI scales it up 2x
FRAME_BYTES = W * H * 4  # BGRA32, the same byte order as an X11 32-bit window
META_LEN = 128
SLOTS = 3
CTRL_BYTES = 16
SLOT_BYTES = META_LEN * 8 + FRAME_BYTES
SHM_BYTES = CTRL_BYTES + SLOTS * SLOT_BYTES

ACTIONS = ["turn_left", "turn_right", "move_forward", "move_back", "shoot", "use_open_door"]
MAX_TURN = 5.0  # degrees per tic at full turn speed
MAX_BOXES = 10

# Metadata fields (float64 indices)
SEQ, TIC, FWD, BACK, ATTACK, USE, TURN = 0, 1, 2, 3, 4, 5, 6
HEALTH, AMMO, ARMOR = 7, 8, 9
REFLEX_SHOOT, REFLEX_STOP, JEV_IDX = 10, 11, 12
KILLS, DEATHS, TICS_PER_S, GAP_P99, GAP_MAX = 13, 14, 15, 16, 17
STATUS, STATUS_T = 18, 19          # status: 0 playing, 1 died, 2 level clear
N_BOXES, BOXES = 21, 22            # boxes: MAX_BOXES x (left, top, right, bottom, is_enemy, dist)
WALL_DIST, DOOR_DIST = 90, 91      # door -1 when none
EPISODES, REFLEX_TICS, TOTAL_TICS = 92, 93, 94
LEVELS_CLEARED = 95
POS_X, POS_Y, ANGLE, NAV_HEADING = 96, 97, 98, 99  # NAV_HEADING is NaN when there is no route

PLAYING, DIED, LEVEL_CLEAR = 0, 1, 2


def views(buf):
    """Numpy views into the shared buffer: ctrl, [meta per slot], [frame per slot]."""
    ctrl = np.ndarray((2,), dtype=np.int64, buffer=buf, offset=0)
    metas, frames = [], []
    for i in range(SLOTS):
        off = CTRL_BYTES + i * SLOT_BYTES
        metas.append(np.ndarray((META_LEN,), dtype=np.float64, buffer=buf, offset=off))
        frames.append(np.ndarray((H, W, 4), dtype=np.uint8, buffer=buf, offset=off + META_LEN * 8))
    return ctrl, metas, frames
