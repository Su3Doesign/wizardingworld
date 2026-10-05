"""world - the single source of truth for the layout of the valley, the castle crag and everything around it.

Frame: metres, Z up, +X east, +Y north (Blender frame; Unreal gets (100x, -100y, 100z) cm, see export_fbx.py).
The surface of the Black Lake is z = 0; the castle stands on its crag at z = 80.

Layout (after the plan references: castle on a crag over the lake's north shore, Hogsmeade to the north, the
Forbidden Forest east, the Quidditch pitch north-west, the station and the boats on the lake's east shore):

    * the castle crag: the tip of the grounds plateau, cut off by cliffs on its west (the river gorge), south (the lake)
      and east (the ravine)
    * the river gorge (west): a deep granite gorge with the river at the bottom, entering the lake at lake level
      (reference 1: castle above a forested gorge, river below)
    * the ravine (east): a hanging valley whose stream drops ~45 m over the cliff into the lake
      (reference 2: waterfall under the castle); the viaduct crosses it
    * the grounds plateau (north / east): lawns, gates, the pitch, the hut, the greenhouses, the willow
    * mountains all round (hydraulically eroded), forested up to the tree line, rock above

H(x, y) is the terrain height field; the hero rock region round the crag is re-meshed from a signed distance field
(crag.py) that adds the 3D granite structure (blocks, joints, ledges, overhangs) on top of H.
"""
from __future__ import annotations

import math
import os

import numpy as np

import fastnoise as fn

LAKE_Z = 0.0
CASTLE_Z = 80.0
EXTENT = 4600.0                 # terrain covers [-EXTENT, EXTENT]^2
TREE_LINE = 520.0
ERODE_FILE = os.environ.get("WW_ERODE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "erosion.npz"))

# ----------------------------------------------------------------------------------------------- helpers


def catmull(points, step=4.0, closed=False):
    """Catmull-Rom resampling of a control polyline to roughly `step` metre spacing."""
    P = np.asarray(points, np.float64)
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(math.ceil(np.linalg.norm(p2 - p1) / step)))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        a = 2 * p1
        b = p2 - p0
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3
        out.append(0.5 * (a + b * t + c * t * t + d * t * t * t))
    if not closed:
        out.append(P[-2][None])
    return np.vstack(out)


def sstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tiered(over, widths, slopes):
    """Piecewise-linear cliff profile: height gained at horizontal distance `over` from the foot, through segments of
    the given widths (arrays or scalars) and slopes; the last slope continues indefinitely."""
    over = np.asarray(over, np.float64)
    h = np.zeros_like(over)
    x0 = np.zeros_like(over)
    for w, k in zip(widths, slopes[:-1]):
        h += k * np.clip(over - x0, 0, w)
        x0 = x0 + w
    h += slopes[-1] * np.clip(over - x0, 0, None)
    return h


def points_in_polygon(x, y, poly):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    inside = np.zeros(x.shape, bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cond = (y0 > y) != (y1 > y)
        xint = x0 + (y - y0) * (x1 - x0) / ((y1 - y0) if abs(y1 - y0) > 1e-12 else 1e-12)
        inside ^= cond & (x < xint)
    return inside


# ----------------------------------------------------------------------------------------------- layout

# ---- the castle (layout after the Legacy castle map: map pixel (mx, my) -> world ((mx - 1000) * 0.24, (1100 - my) * 0.24))
# The castle stands on two rocks 80 m above the lake, joined by bridges over the water that threads between them:
#   * the north-west rock (the mainland side): the Astronomy wing, the Transfiguration court, the Library, the Bell Tower,
#     the greenhouses, Divination / Potions and the Viaduct Entrance
#   * the south-east rock: the Viaduct court, the Entrance Hall, the Great Hall (by the east arm of the lake), the Grand
#     Staircase Tower, the Quad, Ravenclaw / Gryffindor, the Hospital wing, the Faculty tower and the South Wing with the
#     Clock Tower running down to the main lake
# The lake's east arm runs north along the castle; its north-east bay opens into the inlet that threads the castle
# (under the grand viaduct, the stone bridge and the suspension bridge) to the west basin (the Map Chamber's rock),
# which drains south into the main lake.  The boats come up the east arm into the bay: the boathouse sits at the tip of
# the spur on the bay's north side; its entry stairs climb the spur, and the stream from the north-east hills falls
# through their arches into the bay.
CASTLE_REGION = catmull([
    (-172, -60), (-170, 60), (-158, 150), (-110, 182), (-20, 192), (40, 196), (60, 176), (64, 130), (150, 92), (186, 70),
    (188, 0), (184, -110), (150, -210), (-60, -214), (-160, -170),
], step=2.0, closed=True)
CLIFF_MARGIN = (12.0, 14.0)      # plateau rim to the water's edge: narrow water (inlet, basin) / the open lake

# west river gorge: river centre line from the north-west hills to the lake (the last points are under water)
GORGE_PATH = catmull([
    (-2300, 3000), (-1750, 2250), (-1250, 1650), (-860, 1150), (-560, 760), (-380, 470), (-262, 250), (-205, 80),
    (-200, -40), (-212, -150), (-245, -255), (-290, -380),
], step=2.0)
_GORGE_BED_Y = np.array([3000, 2250, 1650, 1150, 760, 470, 250, 80, -40, -150, -255, -380], np.float64)
_GORGE_BED_Z = np.array([260, 175, 122, 92, 70, 50, 30, 15, 7, 2.2, -2.0, -8.0], np.float64)

# east ravine -> the stream from the north-east hills: it crosses the grounds and reaches the rim of the bay's north
# face, where it drops down a cleft through the entry stairs into the bay (FALL_BED).  Bed 2-3 m below the natural ground.
RAVINE_PATH = catmull([
    (320, 1400), (240, 1060), (180, 780), (140, 560), (116, 400), (114, 300), (134, 262), (148, 240), (150, 228),
], step=2.0)
# near the castle the stream has cut a gorge ~25 m deep into the grounds (the covered bridge spans it)
_RAVINE_BED_Y = np.array([1400, 1060, 780, 560, 400, 330, 300, 262, 240, 228], np.float64)
_RAVINE_BED_Z = np.array([96.0, 89.0, 84.0, 80.0, 75.0, 63.0, 58.0, 54.0, 52.0, 51.0], np.float64)
RAVINE_GAP = 0.0

# the entry stairs: three flights up the bay's north face from the boathouse quay to the castle's north-east corner,
# joined by landing bastions.  The stream's cleft cuts down the face; every flight bridges it on an arch and the stream
# falls on the rock banks between them - seen from the boats, the waterfall drops through the stairs.
# Flights run east-west:  (x_start, x_end, y, z_start, z_end);  landings / the quay:  (x0, x1, y0, y1, z, open side)
STAIR_FLIGHTS = [(228.0, 92.0, 188.5, 2.5, 25.0), (92.0, 206.0, 201.5, 27.0, 47.0), (206.0, 70.0, 214.5, 49.0, 77.5)]
STAIR_W = 5.0                    # the stair between its walls
STAIR_WALL = 1.0                 # wall thickness (the outer wall stands on a battered retaining wall)
STAIR_BAND = 4.0                 # half-width of a flight's bench (the stair, its walls, a margin)
STAIR_LANDINGS = [(222.0, 252.0, 176.0, 193.0, 1.2, "W"), (82.0, 96.0, 184.5, 205.5, 26.0, "E"),
                  (202.0, 216.0, 197.5, 218.5, 48.0, "W")]
FALL_X = 150.0                   # the stream's cleft
FALL_W = 7.0                     # cleft width at the bed
# the stream bed down the cleft (y, z): out of the gorge into a pool under the top flight, a fall, under the middle
# flight, a fall, under the bottom flight, the last fall into the bay
FALL_BED = np.array([(244.0, 52.0), (228.0, 51.0), (219.5, 50.5), (209.0, 49.5), (207.4, 31.5), (196.6, 30.0), (195.0, 9.0),
                     (184.6, 8.5), (183.2, -1.5), (170.0, -4.0)], np.float64)
WATERFALL_LIP = (FALL_X, 209.0, 49.5)
WATERFALL_FOOT = (FALL_X, 183.0, 0.0)

# the boathouse spur: a rocky ridge from the north-west rock's north-east corner down into the bay
SPUR_PATH = catmull([(50, 222), (100, 224), (150, 220), (200, 208), (244, 192)], step=2.0)
SPUR_Z = (78.0, 3.0)             # crest height at the root / at the tip

# Black Lake: the main lake (south) with the east arm and the bay, counter-clockwise
LAKE_OUTLINE = catmull([
    (-290, -330), (-200, -224), (-150, -205), (-60, -226), (10, -212), (52, -170), (58, -138), (96, -132), (140, -140),
    (178, -128), (196, -40), (200, 40), (196, 84), (150, 102), (100, 104), (74, 112), (72, 150), (76, 180), (120, 186),
    (170, 182), (222, 170), (258, 172), (270, 210), (320, 300), (430, 400), (580, 470), (720, 470), (800, 380), (830, 200),
    (810, 0),
    (770, -190), (760, -420), (980, -560), (1180, -760), (1320, -1080), (1340, -1450), (1180, -1800), (850, -2050),
    (300, -2180), (-300, -2160), (-850, -1980), (-1250, -1650), (-1450, -1250), (-1400, -850), (-1150, -560),
    (-820, -420), (-520, -370),
], step=3.0, closed=True)
# the inlet / channel through the castle: from the bay, under the grand viaduct, the stone bridge and the suspension
# bridge, to the west basin
CHANNEL_PATH = catmull([(86, 130), (50, 112), (24, 74), (2, 46), (-14, 34), (-26, 12), (-34, -12), (-54, -30), (-80, -42)],
                       step=2.0)
CHANNEL_W = (52.0, 20.0)         # width at the bay end / further in
WEST_BASIN = catmull([
    (-66, -24), (-50, -40), (-48, -78), (-60, -108), (-88, -122), (-120, -120), (-140, -98), (-144, -60), (-130, -28),
    (-104, -16),
], step=2.0, closed=True)
BASIN_OUTLET = catmull([(-122, -110), (-128, -150), (-138, -190), (-150, -215)], step=2.0)
BASIN_OUTLET_W = 18.0

ISLANDS = [  # (x, y, radius, height above the lake, name)
    (-470, -700, 44.0, 9.0, "WhiteTombIsland"),
    (380, -980, 26.0, 7.0, "Islet_A"),
    (-900, -1250, 34.0, 12.0, "Islet_B"),
    (700, -1500, 18.0, 5.0, "Islet_C"),
]

# grounds / landmarks (x, y)
GATES = (95.0, 600.0)
ROAD_PATH = catmull([(50, 222), (80, 420), (95, 600), (70, 880), (0, 1200), (-80, 1500), (-110, 1760), (-120, 2050)], step=3.0)
HOGSMEADE = (-130.0, 1800.0)
PITCH = (-125.0, 475.0)            # Quidditch stadium centre: on the lawn beside the castle, north-west of the greenhouses
PITCH_YAW = -14.0                  # deg: its long axis turned from north towards the castle (axis direction (sin, cos))
PITCH_SIZE = (67.0, 168.0)         # the stands' inner oval: full width, full length (m)
FLYING_LAWN = (-10.0, 390.0)       # the flying lesson: rows of brooms on the lawn between the greenhouses and the stadium
FLYING_LAWN_YAW = -36.5            # deg: the rows' direction (they point at the stadium)
HUT = (520.0, 560.0)               # the gamekeeper's hut on the north shore of the east arm, by the forest
WILLOW = (300.0, 430.0)
GREENHOUSES = (-40.0, 270.0)      # the outer greenhouses north of the castle
STONE_CIRCLE = (190.0, 330.0)      # on the grounds above the stream
VIADUCT = ((10.0, 132.0), (46.0, 110.0), (70.0, 68.0))   # the grand viaduct over the inlet's mouth: Viaduct Entrance -> bend -> Viaduct court
STONE_BRIDGE = ((-36.0, 40.0), (6.0, 22.0))
SUSPENSION_BRIDGE = ((-58.0, -2.0), (-14.0, -24.0))
STACKS = [(-100.0, -66.0, 13.0, 64.0, "MapChamberRock")]   # sea stacks: (x, y, radius, top z, name)
STATION = (1010.0, -640.0)
BOATHOUSE = (240.0, 170.0)          # its water gate faces south into the bay (the boats' landing)
BOATHOUSE_LEN = 26.0
BOAT_DOCK_STATION = (930.0, -600.0)

FOREST_CENTRES = [(1300, 600, 1500.0), (900, 1600, 1000.0), (-1100, 300, 900.0), (-700, -150, 500.0)]
VALLEY_CORRIDOR = catmull([(-300, -200), (0, 300), (-60, 1100), (-130, 1800), (-200, 2300)], step=5.0)

_PF = {}


def pathfield(name):
    """Cached fastnoise.PathField per layout path."""
    if name not in _PF:
        spec = {
            "region": (CASTLE_REGION, 0.5, True), "lake": (LAKE_OUTLINE, 1.0, True), "gorge": (GORGE_PATH, 0.5, False),
            "channel": (CHANNEL_PATH, 0.5, False), "basin": (WEST_BASIN, 0.5, True), "outlet": (BASIN_OUTLET, 0.5, False),
            "spur": (SPUR_PATH, 0.5, False),
            "ravine": (RAVINE_PATH, 0.5, False), "road": (ROAD_PATH, 0.5, False), "corridor": (VALLEY_CORRIDOR, 4.0, False),
        }[name]
        _PF[name] = fn.PathField(*spec)
    return _PF[name]


def _bed(s, ys, zs, path):
    s_along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    z_along = np.interp(-path[:, 1], -ys, zs)
    return np.interp(s, s_along, z_along)


def gorge_bed(s):
    return _bed(s, _GORGE_BED_Y, _GORGE_BED_Z, GORGE_PATH)


def ravine_bed(s):
    return _bed(s, _RAVINE_BED_Y, _RAVINE_BED_Z, RAVINE_PATH)


def gorge_bed_at_y(y):
    return np.interp(-np.asarray(y, np.float64), -_GORGE_BED_Y, _GORGE_BED_Z)


# ----------------------------------------------------------------------------------------------- terrain


class Fields:
    """Terrain height H and the masks used by the scatter / materials, at arbitrary points (x, y)."""

    def __init__(self, x, y, erosion=True):
        self.x = np.asarray(x, np.float64).ravel()
        self.y = np.asarray(y, np.float64).ravel()
        self.erosion = erosion
        self._c = {}

    def _get(self, key, fn_):
        if key not in self._c:
            self._c[key] = fn_()
        return self._c[key]

    def crag(self):
        """Signed distance to the castle plateaus (negative on them): the castle region minus the water and a cliff margin
        (so the inlet and the basins split it into the north-west and the south-east rocks)."""
        def _c():
            open_d, narrow_d = self.lake_parts()
            reg = fn.polygon_signed(pathfield("region"), self.x, self.y)
            return np.maximum.reduce([reg, CLIFF_MARGIN[1] - open_d, CLIFF_MARGIN[0] - narrow_d])
        return self._get("crag", _c)

    def lake_parts(self):
        """Signed distances (negative in the water) to the open lake (main lake, east arm, bay) and to the narrow water
        (the inlet / channel, the west basin and its outlet)."""
        def _l():
            main = fn.polygon_signed(pathfield("lake"), self.x, self.y)
            cd_, cs_, _ = pathfield("channel").query(self.x, self.y)
            w = CHANNEL_W[0] + (CHANNEL_W[1] - CHANNEL_W[0]) * sstep(0.0, 90.0, cs_)
            basin = fn.polygon_signed(pathfield("basin"), self.x, self.y)
            od, _, _ = pathfield("outlet").query(self.x, self.y)
            return main, np.minimum.reduce([cd_ - w / 2, basin, od - BASIN_OUTLET_W / 2])
        return self._get("lakeparts", _l)

    def lake(self):
        return self._get("lake", lambda: np.minimum(*self.lake_parts()))

    def gorge(self):
        return self._get("gorge", lambda: pathfield("gorge").query(self.x, self.y))

    def ravine(self):
        return self._get("ravine", lambda: pathfield("ravine").query(self.x, self.y))

    def road(self):
        return self._get("road", lambda: pathfield("road").query(self.x, self.y)[0])

    def n(self, scale, octaves=4, seed=0, gain=0.5):
        return fn.fbm2(self.x, self.y, scale, octaves, seed, gain)

    # -- the height field
    def analytic(self):
        return self._get("A", self._analytic)

    def _analytic(self):
        x, y = self.x, self.y
        # 1. grounds plateau: ~78 m round the castle, rising gently north and towards the hills
        g = 77.0 + 0.020 * np.clip(y - 220, 0, None) + 3.0 * self.n(260.0, 4, 11) + 1.0 * self.n(55.0, 3, 12)
        # 2. mountains: their amplitude grows with the distance from the valley floor
        dv = self.valley_distance()
        amp = sstep(60.0, 2000.0, dv)
        wx = 380.0 * self.n(1900.0, 3, 21)
        wy = 380.0 * self.n(1900.0, 3, 25)
        rid = fn.ridged2(x + wx, y + wy, 2100.0, 7, 22, 0.5, 2.0)
        mtn = 1250.0 * amp ** 1.25 * (0.18 + 0.82 * rid) + 130.0 * amp * self.n(800.0, 5, 23)
        hills = 85.0 * sstep(40.0, 650.0, dv) * (0.55 + 0.45 * self.n(420.0, 4, 24))
        h = g + mtn + hills
        # 3. the castle plateau: flat at 80 m inside the outline, blending into the grounds over ~20 m
        cd = self.crag()
        top = CASTLE_Z
        w = sstep(22.0, 0.0, cd)
        h = h * (1 - w) + top * w
        # 4. river gorge: floor + walls. The east wall (crag side, left of the downstream direction) is a sheer granite
        #    face beside the castle; the west wall is a steep forested slope; upstream the gorge opens into a valley.
        dg, sg, side = self.gorge()
        bed = gorge_bed(sg)
        floor_w = 6.0 + 4.0 * (0.5 + 0.5 * self.n(90.0, 3, 31))
        near = sstep(560.0, 80.0, np.hypot(x + 210.0, y - 20.0))
        open_up = sstep(500.0, 1500.0, sg)
        steep_e = (0.9 + 3.6 * near) * (1.0 - 0.75 * open_up) + 0.12
        steep_w = (0.7 + 0.55 * near) * (1.0 - 0.7 * open_up) + 0.10
        steep = np.where(side > 0, steep_e, steep_w) * (1.0 + 0.18 * self.n(60.0, 3, 32))
        over = np.clip(dg - floor_w, 0, None)
        # crag side: forested talus apron -> lower cliff band -> vegetated ledge (sometimes absent) -> upper cliff
        n1, n2, n3 = self.n(70.0, 3, 33), self.n(55.0, 3, 34), self.n(90.0, 3, 35)
        t_w = (4.0 + 8.0 * (0.5 + 0.5 * n1)) * near
        z_led = 26.0 + 14.0 * (0.5 + 0.5 * n2)                         # ledge height above the river bed
        l_w = np.clip(6.0 + 16.0 * n3, 0.0, 17.0) * near
        k_c = np.maximum(steep, 0.5)
        c1_w = np.clip(z_led - 0.62 * t_w, 0, None) / k_c
        crag_wall = bed + tiered(over, [t_w, c1_w, l_w], [0.62, k_c, 0.32, k_c * 1.15])
        plain_wall = bed + over * steep + 0.0025 * over ** 2
        wall = np.where(side > 0, crag_wall * near + plain_wall * (1 - near), plain_wall)
        h = np.minimum(h, wall)

        # 4b. the boathouse spur: a rocky ridge from the castle's north-east corner down into the bay (the entry stairs
        #     climb it; the stream cuts through its root, step 5, and falls into the bay)
        dsp, ssp, _ = pathfield("spur").query(x, y)
        L_sp = pathfield("spur").length
        crest = SPUR_Z[0] + (SPUR_Z[1] - SPUR_Z[0]) * sstep(0.0, L_sp, ssp) ** 2.2 + 2.0 * self.n(30.0, 3, 47)
        spur = crest - 1.25 * np.clip(dsp - 5.0, 0, None) - 0.004 * np.clip(dsp - 5.0, 0, None) ** 2
        h = np.where(dsp < 90.0, np.maximum(h, spur), h)

        # 5. east ravine -> the stream: a shallow valley across the grounds, then a narrow cleft through the root of the
        #    spur, ending at the lip above the bay (the waterfall)
        dr, sr, rside = self.ravine()
        rbed = ravine_bed(sr)
        L_r = pathfield("ravine").length
        gapw = sstep(RAVINE_GAP, RAVINE_GAP - 120.0, L_r - sr)
        rfloor = 4.0 + 3.0 * (0.5 + 0.5 * self.n(70.0, 3, 41)) + 24.0 * gapw
        rnear = sstep(420.0, 60.0, L_r - sr)
        rsteep = (0.35 + 2.6 * rnear) * (1.0 + 0.2 * self.n(50.0, 3, 42))
        keep = sstep(L_r + 10.0, L_r - 1.0, sr)
        rover = np.clip(dr - rfloor, 0, None)
        r1, r2 = self.n(50.0, 3, 43), self.n(65.0, 3, 44)
        rz = 12.0 + 8.0 * (0.5 + 0.5 * r1)
        rl = np.clip(3.0 + 10.0 * r2, 0.0, 10.0) * rnear
        rk = np.maximum(rsteep, 0.35)
        floor_rise = gapw * (0.17 * np.clip(np.minimum(dr, rfloor) - 4.0, 0, None) + 1.2 * self.n(18.0, 3, 45))
        rwall = rbed + floor_rise + tiered(rover, [rz / rk, rl], [rk, 0.35, rk * 1.1]) + 0.004 * rover ** 2
        h = h * (1 - keep) + np.minimum(h, rwall) * keep
        # 6. the lake basin: sheer cliffs below the crag and the headlands round it, shelving beaches elsewhere
        ld = self.lake()
        depth = 3.0 + 42.0 * sstep(0.0, 650.0, -ld) + 5.0 * self.n(300.0, 3, 51)
        bottom = LAKE_Z - depth
        cliffy = sstep(170.0, 30.0, cd)                                  # the crag and ~150 m either side of it
        shore_k = 0.10 + 0.20 * (0.5 + 0.5 * self.n(400.0, 3, 52))
        inside = LAKE_Z - 1.0 + ld * (0.30 + 2.5 * cliffy)               # ld < 0 inside: shelving down from the shore
        beach = LAKE_Z - 1.0 + ld * shore_k + 0.0025 * ld * ld           # rising onto the land (limited reach)
        m1, m2, m3 = self.n(60.0, 3, 53), self.n(75.0, 3, 54), self.n(110.0, 3, 55)
        z1 = 30.0 + 16.0 * (0.5 + 0.5 * m1)                              # first ledge above the water
        lw = np.clip(4.0 + 15.0 * m2, 0.0, 16.0)
        boulder = 2.5 + 3.0 * (0.5 + 0.5 * m3)
        cliff = LAKE_Z - 1.0 + tiered(ld, [boulder, z1 / 5.2, lw], [0.45, 5.2, 0.30, 5.8])
        # the inlet and the basins between the castle rocks: sheer walls straight out of the water
        open_d, narrow_d = self.lake_parts()
        narrow = sstep(4.0, -4.0, narrow_d - open_d)
        sheer = LAKE_Z - 1.0 + tiered(ld, [1.2 + 1.5 * (0.5 + 0.5 * m3)], [0.5, 7.5 + 2.0 * m1])
        cliff = cliff * (1 - narrow) + sheer * narrow
        outside = cliff * cliffy + beach * (1 - cliffy)
        lake_h = np.where(ld < 0, np.maximum(bottom, inside), outside)
        h = np.minimum(h, lake_h)
        # 6b. the castle plateau wins inside its outline: the cliffs start exactly at the rim (buildings stand on rock)
        h = np.where(cd < 0, np.maximum(h, CASTLE_Z - 0.6 * sstep(-30.0, 0.0, cd)), h)
        # 7. islands (only ever raise the lake bed locally)
        for ix, iy, ir, ih, _ in ISLANDS:
            d = np.hypot(x - ix, y - iy) * (1 + 0.22 * fn.fbm2(x, y, 40.0, 2, int(abs(ix)) % 97))
            isl = LAKE_Z + ih * (1 - sstep(0.0, ir, d)) ** 0.7 - 1.5 - 40.0 * sstep(ir, ir * 2.2, d)
            h = np.maximum(h, isl)
        # 7b. sea stacks (the Map Chamber's rock in the west basin): steep pillars out of the water
        for sx, sy, sr_, sz, _ in STACKS:
            d = np.hypot(x - sx, y - sy) * (1 + 0.12 * fn.fbm2(x, y, 25.0, 2, 77))
            h = np.maximum(h, sz - 7.0 * np.clip(d - sr_, 0, None) - 0.5 * np.clip(d - sr_ * 0.6, 0, None))
        # 8. flatten the pitch, Hogsmeade, the station shelf, the greenhouse lawn; soften the road
        h = self._flatten(h, PITCH, (PITCH_SIZE[0] * 0.65 + 14, PITCH_SIZE[1] * 0.58 + 14), 40.0, yaw=PITCH_YAW)
        h = self._flatten(h, HOGSMEADE, (260.0, 330.0), 140.0)
        h = self._flatten(h, STATION, (95.0, 42.0), 60.0, target=LAKE_Z + 3.2)
        h = self._flatten(h, GREENHOUSES, (65.0, 45.0), 35.0)
        h = self._flatten(h, HUT, (25.0, 25.0), 30.0)
        h = h - 0.35 * sstep(5.0, 1.5, self.road())
        # 9. the bay's north face shaped for the entry stairs, the stream's cleft, the boathouse cove
        return stair_face(x, y, h)

    def erosion_weight(self):
        """Where the eroded grid may change the analytic terrain (mountains, not the authored valley floor)."""
        return self._get("ew", lambda: sstep(150.0, 900.0, self.valley_distance()))

    def height(self):
        def _h():
            h = self.analytic()
            if self.erosion and os.path.isfile(ERODE_FILE):
                z = np.load(ERODE_FILE)
                d = fn.bicubic(z["diff"], float(z["x0"]), float(z["y0"]), float(z["cell"]), self.x, self.y)
                h = h + d * self.erosion_weight()
            return h
        return self._get("H", _h)

    def valley_distance(self):
        def _v():
            x, y = self.x, self.y
            ld = self.lake()
            dc, _, _ = pathfield("corridor").query(x, y)
            dcor = dc - 520.0
            east = np.hypot(x - 650.0, (y - 450.0) / 1.5) - 520.0
            west = np.hypot(x + 650.0, (y - 300.0) / 1.6) - 300.0
            d = np.minimum.reduce([ld - 150.0, dcor, east, west])
            d = d + 120.0 * self.n(600.0, 3, 61)
            return np.clip(d, 0, None)
        return self._get("dv", _v)

    def _flatten(self, h, c, half, blend, target=None, yaw=0.0):
        x, y = self.x, self.y
        sa, ca = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        u = (x - c[0]) * ca - (y - c[1]) * sa                  # across / along the (turned) long axis
        v = (x - c[0]) * sa + (y - c[1]) * ca
        r = np.hypot(u / half[0], v / half[1])
        w = sstep(1.0 + blend / max(half), 1.0, r)
        if target is None:
            target = flat_target(c)
        return h * (1 - w) + target * w

    # -- masks (0..1)
    def forest(self):
        """Tree density: forests on the hills and gorge sides; none on lawns, roads, the lake, the castle plateau or above the tree line."""
        def _f():
            x, y = self.x, self.y
            h = self.height()
            f = np.zeros_like(x)
            for cx, cy, r in FOREST_CENTRES:
                f = np.maximum(f, sstep(r * 1.15, r * 0.55, np.hypot(x - cx, y - cy)))
            f = np.maximum(f, sstep(60.0, 420.0, self.valley_distance()))
            dg, sg, _ = self.gorge()
            f = np.maximum(f, sstep(150.0, 25.0, dg) * sstep(1.5, 5.0, h - gorge_bed(sg)))
            dr, sr, _ = self.ravine()
            f = np.maximum(f, 0.85 * sstep(100.0, 12.0, dr) * sstep(1.0, 3.5, h - ravine_bed(sr)))
            f = f * sstep(-0.2, 0.35, self.n(230.0, 4, 71) + 0.22)
            f = f * sstep(TREE_LINE + 60, TREE_LINE - 90, h)
            f = f * sstep(1.5, 6.0, h - LAKE_Z) * sstep(-6.0, 6.0, self.lake())
            f = f * sstep(-6.0, 18.0, self.crag())
            lawn = sstep(470.0, 330.0, np.hypot((x - 40.0) / 1.25, (y - 360.0)))
            f = f * (1 - lawn)
            for c, rr in ((PITCH, 160.0), (HOGSMEADE, 390.0), (STATION, 130.0), (GREENHOUSES, 95.0), (HUT, 60.0), (WILLOW, 50.0),
                          (STONE_CIRCLE, 45.0)):
                f = f * sstep(rr * 0.7, rr, np.hypot(x - c[0], y - c[1]))
            f = f * sstep(7.0, 20.0, self.road())
            # no trees in the stream, nor on the floor of the gap under the viaduct (the arches stay visible)
            L_r = pathfield("ravine").length
            f = f * sstep(4.0, 9.0, dr) * (1.0 - sstep(140.0, 60.0, L_r - sr) * sstep(60.0, 20.0, dr))
            return np.clip(f, 0, 1)
        return self._get("forest", _f)


def stair_elements():
    """The stair's platforms as boxes with a deck height linear along x: (x0, x1, y0, y1, z_at_x0, z_at_x1, lowest)."""
    B = STAIR_BAND
    out = []
    for i, (xa, xb, yc, za, zb) in enumerate(STAIR_FLIGHTS):
        x0, x1 = min(xa, xb), max(xa, xb)
        z0, z1 = (za, zb) if xa < xb else (zb, za)
        out.append((x0, x1, yc - B, yc + B, z0, z1, i == 0))
    for k, (x0, x1, y0, y1, z, _) in enumerate(STAIR_LANDINGS):
        out.append((x0, x1, y0, y1, z, z, k < 2))
    return out


def stair_deck(x, e):
    x0, x1, _, _, z0, z1, _ = e
    return z0 + (z1 - z0) * np.clip((x - x0) / max(x1 - x0, 1e-6), 0.0, 1.0)


def stair_distance(x, y):
    """Distance (m) to the nearest stair platform (0 on them)."""
    d = np.full(np.shape(x), 1e9)
    for x0, x1, y0, y1, *_ in stair_elements():
        dx = np.clip(np.maximum(x0 - x, x - x1), 0, None)
        dy = np.clip(np.maximum(y0 - y, y - y1), 0, None)
        d = np.minimum(d, np.hypot(dx, dy))
    return d


def fall_bed(y):
    return np.interp(-np.asarray(y, np.float64), -FALL_BED[:, 0], FALL_BED[:, 1])


def stair_face(x, y, h):
    """Step 9 of the height field.  Every stair platform gets a bench: fills raise the ground up to it (steep rock below,
    very steep on the lake side of the lowest ones), cuts lower the ground to it (steep rock above) - so the flights sit
    on benches stacked up the face with rock banks between them; the benches are then levelled exactly.  Then the
    stream's cleft is cut down the face (it passes under the flights) and the boathouse cove is dug out of the shore."""
    zone = (x > 30.0) & (x < 280.0) & (y > 140.0) & (y < 260.0)
    if not zone.any():
        return h
    h = h.copy()
    xs, ys, hs = x[zone], y[zone], h[zone]
    E = stair_elements()
    n = 0.8 * fn.fbm2(xs, ys, 9.0, 3, 91)
    for e in E:
        x0, x1, y0, y1, _, _, lowest = e
        deck = stair_deck(xs, e) - 0.6
        dx = np.clip(np.maximum(x0 - xs, xs - x1), 0, None)
        ds = np.clip(y0 - ys, 0, None)
        dn = np.clip(ys - y1, 0, None)
        hs = np.maximum(hs, deck - 3.0 * (dx + np.clip(dn - 3.0, 0, None)) - (8.0 if lowest else 3.0) * ds + n)
    for e in E:
        x0, x1, y0, y1, _, _, lowest = e
        deck = stair_deck(xs, e) - 0.6
        dx = np.clip(np.maximum(x0 - xs, xs - x1), 0, None)
        ds = np.clip(y0 - ys, 0, None)
        dn = np.clip(ys - y1, 0, None)
        if lowest:          # the lake side of the lowest platforms drops away: their outer walls stand on the rock / water
            cut = np.where(ds > 0, np.maximum(deck - 0.9 - 6.0 * ds, LAKE_Z - 4.0) + 4.0 * dx, deck + 4.0 * (dx + dn))
        else:
            cut = deck + 4.0 * (dx + dn + ds)
        hs = np.minimum(hs, cut + np.abs(n))
    for e in E:                                                    # level the benches
        x0, x1, y0, y1 = e[:4]
        on = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        hs = np.where(on, stair_deck(xs, e) - 0.6, hs)
    # the cleft: the stream's bed down the face, steep rock walls
    dxf = np.clip(np.abs(xs - FALL_X) - FALL_W / 2, 0, None)
    cleft = fall_bed(ys) + 4.0 * dxf + 0.6 * np.abs(n)
    hs = np.where((ys > 172.0) & (ys < 250.0), np.minimum(hs, cleft), hs)
    # the boathouse cove: the slip under the boathouse and the water before its gate
    bx, by = BOATHOUSE
    dcx = np.clip(np.abs(xs - bx) - 10.0, 0, None)
    dcy = np.clip(np.maximum((by - BOATHOUSE_LEN / 2 - 14.0) - ys, ys - (by + 2.0)), 0, None)
    hs = np.minimum(hs, LAKE_Z - 3.5 + 3.0 * (dcx + dcy))
    h[zone] = hs
    return h


def stair_mask(x, y):
    """1 on and near the stair benches and in the cleft (no granite displacement there), 0 elsewhere."""
    d = stair_distance(x, y)
    m = sstep(3.0, 0.6, d)
    near = (y > 172.0) & (y < 250.0)
    m = np.maximum(m, np.where(near, sstep(8.0, 4.0, np.abs(x - FALL_X)), 0.0))
    bx, by = BOATHOUSE
    m = np.maximum(m, sstep(6.0, 1.0, np.hypot(np.clip(np.abs(x - bx) - 10.0, 0, None),
                                               np.clip(np.abs(y - by) - BOATHOUSE_LEN / 2, 0, None))))
    return m


_FLAT = {}


def flat_target(c):
    """Height a flattened area is levelled to: the analytic height at its centre (cached, deterministic)."""
    key = (round(c[0], 2), round(c[1], 2))
    if key not in _FLAT:
        F = Fields([c[0]], [c[1]], erosion=False)
        # evaluate without the flattening itself (avoid recursion): compute the analytic field minus step 8
        _FLAT[key] = _analytic_no_flatten(F)[0]
    return _FLAT[key]


def _analytic_no_flatten(F):
    saved = Fields._flatten
    try:
        Fields._flatten = lambda self, h, c, half, blend, target=None, yaw=0.0: h
        return F._analytic()
    finally:
        Fields._flatten = saved


def ground(x, y):
    """Terrain height at points (analytic + erosion; the granite displacement of the core is not included)."""
    return Fields(np.atleast_1d(x), np.atleast_1d(y)).height()


def crag_region():
    """Box (m) of the hero rock region re-meshed from a signed distance field by crag.py."""
    return dict(x=(-330.0, 320.0), y=(-262.0, 250.0), z=(-26.0, 118.0))


# ----------------------------------------------------------------------------------------------- erosion pre-pass


def build_erosion(out=ERODE_FILE, n=1536, drops=1_400_000, seed=7):
    """Erode the mountains once on a coarse grid and store (eroded - analytic) for bicubic lookup."""
    import time

    t = time.time()
    xs = np.linspace(-EXTENT, EXTENT, n)
    cell = float(xs[1] - xs[0])
    X, Y = np.meshgrid(xs, xs)                       # rows = y ascending
    F = Fields(X, Y, erosion=False)
    A = F.analytic().reshape(n, n)
    W = F.erosion_weight().reshape(n, n)
    ii = np.arange(n)
    edge = np.minimum(ii, n - 1 - ii).astype(np.float64)
    W = W * sstep(4.0, 60.0, np.minimum(edge[:, None], edge[None, :]))       # no erosion at the map border (droplets leave there)
    print(f"  analytic grid {n}x{n} ({cell:.1f} m) in {time.time() - t:.1f}s", flush=True)
    H = A.copy()
    H = fn.erode(H, drops, seed, cell, W, 0.05, 5.0, 0.01, 0.22, 0.32, 0.012, 9.0, 80, 2)
    print(f"  hydraulic erosion: {drops:,} droplets in {time.time() - t:.1f}s", flush=True)
    H = fn.thermal(H, 12, cell * 0.95, 0.5, W)
    D = np.clip(H - A, -140.0, 90.0).astype(np.float32)
    from scipy import ndimage
    D = ndimage.gaussian_filter(D, 0.6).astype(np.float32)                # remove single-cell pits / spikes
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, diff=D, x0=float(xs[0]), y0=float(xs[0]), cell=cell)
    print(f"  erosion grid -> {out}: diff {D.min():.1f} .. {D.max():.1f} m ({time.time() - t:.1f}s)", flush=True)


def hillshade(path, n=900, extent=EXTENT, center=(0.0, 0.0), erosion=True):
    from PIL import Image

    xs = np.linspace(center[0] - extent, center[0] + extent, n)
    ys = np.linspace(center[1] + extent, center[1] - extent, n)
    X, Y = np.meshgrid(xs, ys)
    F = Fields(X, Y, erosion=erosion)
    H = F.height().reshape(n, n)
    cell = xs[1] - xs[0]
    gy, gx = np.gradient(H, -cell, cell)
    nl = np.sqrt(1 + gx * gx + gy * gy)
    shade = np.clip(0.5 + 0.5 * (-gx * 0.55 + gy * 0.55 + 0.6) / nl, 0, 1)
    hn = np.clip(H / 1200.0, 0, 1)[..., None]
    col = np.array([0.42, 0.50, 0.30]) * (1 - hn) + np.array([0.80, 0.78, 0.74]) * hn
    fo = F.forest().reshape(n, n)[..., None]
    col = col * (1 - 0.55 * fo) + np.array([0.06, 0.20, 0.08]) * 0.55 * fo
    col = np.where((H < LAKE_Z)[..., None], np.array([0.10, 0.22, 0.33]) * (0.6 + 0.4 * np.clip(1 + H[..., None] / 40, 0, 1)), col)
    img = (np.clip(col * (0.25 + 1.1 * shade[..., None]), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    return H


if __name__ == "__main__":
    import sys
    import time

    cmd = sys.argv[1]
    if cmd == "erode":
        build_erosion()
    elif cmd == "map":
        t = time.time()
        ext = float(sys.argv[3]) if len(sys.argv) > 3 else EXTENT
        cx = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        cy = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        H = hillshade(sys.argv[2], 900, ext, (cx, cy))
        print(f"map in {time.time() - t:.1f}s, height {H.min():.1f} .. {H.max():.1f}")
