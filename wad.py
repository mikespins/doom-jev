"""Read what ViZDoom doesn't expose straight from the map's WAD lumps: which doors
open with the use key (and which key they need), which lines are switches, and
where the exit is. Plain map data, the same for every level; nothing is scripted.
"""

import struct

KEY_DOORS = {26: "blue", 32: "blue", 27: "yellow", 34: "yellow", 28: "red", 33: "red"}
MANUAL_DOORS = {1, 31, 117, 118} | set(KEY_DOORS)       # opened by pressing use on them
KEY_SWITCHES = {99: "blue", 133: "blue", 134: "red", 135: "red", 136: "yellow", 137: "yellow"}
SWITCHES = {7, 9, 14, 15, 18, 20, 21, 23, 29, 41, 49, 50, 55, 71, 101, 102, 103, 111, 112, 113,
            122, 127, 131, 140, 42, 43, 45, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 114, 115,
            116, 123, 132, 138, 139} | set(KEY_SWITCHES)
EXITS = {11: ("switch", False), 51: ("switch", True), 52: ("walk", False), 124: ("walk", True)}

# ViZDoom object names of the keys, by color.
KEY_NAMES = {"BlueCard": "blue", "BlueSkull": "blue", "YellowCard": "yellow", "YellowSkull": "yellow",
             "RedCard": "red", "RedSkull": "red"}
HEALTH_NAMES = {"Stimpack", "Medikit", "Soulsphere", "Megasphere"}

# Solid decorations and their collision radius (Doom's collision is a square this size).
# They aren't map lines, so the route has to know about them separately.
SOLID = {name: 16 for name in (
    "TechPillar", "TechLamp", "TechLamp2", "Column", "TallGreenColumn", "ShortGreenColumn",
    "TallRedColumn", "ShortRedColumn", "SkullColumn", "HeartColumn", "EvilEye", "FloatingSkull",
    "TorchTree", "BlueTorch", "GreenTorch", "RedTorch", "ShortBlueTorch", "ShortGreenTorch",
    "ShortRedTorch", "Candelabra", "Stalagtite", "BurningBarrel", "HeadsOnAStick", "HeadOnAStick",
    "HeadCandles", "DeadStick", "LiveStick", "Meat2", "Meat3", "Meat4", "Meat5", "HangTSkull",
    "HangTLookingUp", "HangTLookingDown", "HangTNoBrain", "HangTSkullSolid", "HangTNoGuts",
    "CommanderKeen", "BloodyTwitch", "NonsolidMeat2")}
SOLID.update({"BigTree": 32, "ExplosiveBarrel": 10})
del SOLID["NonsolidMeat2"]


def _lumps(path):
    with open(path, "rb") as f:
        data = f.read()
    n, off = struct.unpack_from("<4xii", data, 0)
    dirs = [struct.unpack_from("<ii8s", data, off + 16 * i) for i in range(n)]
    return data, [(name.rstrip(b"\0").decode("ascii", "replace"), pos, size) for pos, size, name in dirs]


def read_map(path, name):
    """Returns {"doors": {sector: key or None}, "switches": [(x1, y1, x2, y2, key)],
    "exits": [(x1, y1, x2, y2, kind, secret)]} for one map, or None if not found."""
    data, lumps = _lumps(path)
    idx = next((i for i, l in enumerate(lumps) if l[0] == name.upper()), None)
    if idx is None:
        return None
    lump = {}
    for lname, pos, size in lumps[idx + 1:idx + 12]:
        if lname in ("THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SECTORS") and lname not in lump:
            lump[lname] = data[pos:pos + size]
    if "BEHAVIOR" in (l[0] for l in lumps[idx + 1:idx + 12]):
        return None  # Hexen-format map: different line layout
    verts = list(struct.iter_unpack("<hh", lump["VERTEXES"]))
    sides = [s[5] for s in struct.iter_unpack("<hh8s8s8sh", lump["SIDEDEFS"])]
    tags = [s[6] for s in struct.iter_unpack("<hh8s8shhh", lump["SECTORS"])]
    out = {"doors": {}, "switches": [], "exits": []}
    for v1, v2, _flags, special, tag, right, left in struct.iter_unpack("<hhhhhhh", lump["LINEDEFS"]):
        (x1, y1), (x2, y2) = verts[v1], verts[v2]
        if special in MANUAL_DOORS and left != -1:
            out["doors"][sides[left]] = KEY_DOORS.get(special)
        if special in SWITCHES:
            key = KEY_SWITCHES.get(special)
            out["switches"].append((x1, y1, x2, y2, key))
            if key:  # the doors this switch opens need its key too
                for s, t in enumerate(tags):
                    if t == tag and tag:
                        out["doors"].setdefault(s, key)
        if special in EXITS:
            out["exits"].append((x1, y1, x2, y2, *EXITS[special]))
    return out
