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

# castle plateau outline (z = 80 inside), counter-clockwise
CRAG_OUTLINE = catmull([
    (-150, -150), (-60, -166), (40, -163), (128, -148), (176, -98), (190, -10), (192, 80), (182, 160), (120, 210),
    (20, 220), (-80, 208), (-145, 160), (-166, 70), (-168, -40),
], step=2.0, closed=True)

# west river gorge: river centre line from the north-west hills to the lake (the last points are under water)
GORGE_PATH = catmull([
    (-2300, 3000), (-1750, 2250), (-1250, 1650), (-860, 1150), (-560, 760), (-380, 470), (-262, 250), (-205, 80),
    (-200, -40), (-212, -150), (-245, -255), (-290, -380),
], step=2.0)
_GORGE_BED_Y = np.array([3000, 2250, 1650, 1150, 760, 470, 250, 80, -40, -150, -255, -380], np.float64)
_GORGE_BED_Z = np.array([260, 175, 122, 92, 70, 50, 30, 15, 7, 2.2, -2.0, -8.0], np.float64)

# east ravine: stream from the eastern hills to the lip -> waterfall into the lake
RAVINE_PATH = catmull([
    (1100, 1500), (820, 1000), (600, 680), (420, 400), (296, 190), (232, 40), (220, -60), (218, -136),
], step=2.0)
_RAVINE_BED_Y = np.array([1500, 1000, 680, 400, 190, 40, -60, -136], np.float64)
_RAVINE_BED_Z = np.array([175, 122, 90, 71, 59, 51, 46.5, 45], np.float64)
WATERFALL_LIP = (218.0, -136.0, 45.0)
WATERFALL_FOOT = (218.0, -168.0, 0.0)

# Black Lake shoreline (counter-clockwise); the north shore follows the crag foot
LAKE_OUTLINE = catmull([
    (-290, -330), (-195, -205), (-110, -183), (-20, -182), (70, -178), (150, -168), (214, -160), (300, -195), (420, -250),
    (560, -310), (760, -420), (980, -560), (1180, -760), (1320, -1080), (1340, -1450), (1180, -1800), (850, -2050),
    (300, -2180), (-300, -2160), (-850, -1980), (-1250, -1650), (-1450, -1250), (-1400, -850), (-1150, -560),
    (-820, -420), (-520, -370),
], step=3.0, closed=True)

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
PITCH = (-170.0, 840.0)            # Quidditch pitch centre; long axis north-south
PITCH_SIZE = (75.0, 170.0)         # stand oval: full width, full length (m)
HUT = (470.0, 470.0)               # the gamekeeper's hut at the forest edge
WILLOW = (370.0, 300.0)
GREENHOUSES = (390.0, 140.0)
STONE_CIRCLE = (372.0, -62.0)      # the viaduct lands here
VIADUCT = ((186.0, -40.0), (352.0, -54.0))
STATION = (1010.0, -640.0)
BOATHOUSE = (-10.0, -180.0)
BOAT_DOCK_STATION = (930.0, -600.0)

FOREST_CENTRES = [(1300, 600, 1500.0), (900, 1600, 1000.0), (-1100, 300, 900.0), (-700, -150, 500.0)]
VALLEY_CORRIDOR = catmull([(-300, -200), (0, 300), (-60, 1100), (-130, 1800), (-200, 2300)], step=5.0)

_PF = {}


def pathfield(name):
    """Cached fastnoise.PathField per layout path."""
    if name not in _PF:
        spec = {
            "crag": (CRAG_OUTLINE, 0.5, True), "lake": (LAKE_OUTLINE, 1.0, True), "gorge": (GORGE_PATH, 0.5, False),
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
        return self._get("crag", lambda: fn.polygon_signed(pathfield("crag"), self.x, self.y))

    def lake(self):
        return self._get("lake", lambda: fn.polygon_signed(pathfield("lake"), self.x, self.y))

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

        # 5. east ravine: a narrow cleft between the castle crag and the eastern headland near its lip (the waterfall),
        #    opening upstream into a shallow stream valley
        dr, sr, rside = self.ravine()
        rbed = ravine_bed(sr)
        L_r = pathfield("ravine").length
        rfloor = 4.0 + 3.0 * (0.5 + 0.5 * self.n(70.0, 3, 41))
        rnear = sstep(420.0, 60.0, L_r - sr)
        rsteep = (0.35 + 2.6 * rnear) * (1.0 + 0.2 * self.n(50.0, 3, 42))
        keep = sstep(L_r + 10.0, L_r - 1.0, sr)
        rover = np.clip(dr - rfloor, 0, None)
        r1, r2 = self.n(50.0, 3, 43), self.n(65.0, 3, 44)
        rz = 12.0 + 8.0 * (0.5 + 0.5 * r1)
        rl = np.clip(3.0 + 10.0 * r2, 0.0, 10.0) * rnear
        rk = np.maximum(rsteep, 0.35)
        rwall = rbed + tiered(rover, [rz / rk, rl], [rk, 0.35, rk * 1.1]) + 0.004 * rover ** 2
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
        # 8. flatten the pitch, Hogsmeade, the station shelf, the greenhouse lawn; soften the road
        h = self._flatten(h, PITCH, (PITCH_SIZE[0] * 0.65 + 14, PITCH_SIZE[1] * 0.58 + 14), 40.0)
        h = self._flatten(h, HOGSMEADE, (260.0, 330.0), 140.0)
        h = self._flatten(h, STATION, (95.0, 42.0), 60.0, target=LAKE_Z + 3.2)
        h = self._flatten(h, GREENHOUSES, (65.0, 45.0), 35.0)
        h = self._flatten(h, HUT, (25.0, 25.0), 30.0)
        h = h - 0.35 * sstep(5.0, 1.5, self.road())
        return h

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

    def _flatten(self, h, c, half, blend, target=None):
        x, y = self.x, self.y
        r = np.hypot((x - c[0]) / half[0], (y - c[1]) / half[1])
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
            return np.clip(f, 0, 1)
        return self._get("forest", _f)


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
        Fields._flatten = lambda self, h, c, half, blend, target=None: h
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
