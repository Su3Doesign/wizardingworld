"""castle_plan - the castle as data: every hall, range, court front, gatehouse, wall, stair and (recursive) tower, placed on
the two crags of world.py.  The massing block-out (castle_massing.py) and the detailed build (castle.py) both read it.

Composition, after the references (the night poster with the boats, the day poster with the stair, the original ref 2
with the viaduct and the waterfall, the studio model, the Legacy map's skyline for the stepped towers) - seen from the
lake (south), west -> east:

    * the Great Hall along the south cliff, its long buttressed side (pinnacles, two tiers of lancets, steep roof with
      lucarnes, a fleche) towards the lake, the west gable flanked by two spired octagonal turrets
    * the Grand Staircase Tower behind the hall's east end: the huge round tower with a corbelled crown and the tall
      cone studded with lucarnes; the Headmaster's Tower (stepped stages, a crown of spires) against its north-east side
    * the squat round tower at the cliff edge in front of it; the walled stair down the cliff to the boathouse
    * the arcaded court front, the Entrance Hall's gable behind it, a needle-spired tower
    * the grand viaduct on tall slender arches across the gap between the crags (the waterfall falls below it)
    * on the east crag the dense cluster: the Clock Tower (clock stage, open belfry, spire), the Astronomy Tower (the
      tallest: body, corbelled gallery, upper drum, a crown of turrets round a central spire), spired round towers,
      the square tower with corner turrets, the chapel with the rose window
    * inside: small courts (the Quad, the viaduct court, the clock court, ...) packed round with ranges and towers -
      a dense roofscape from above, no empty lawns

Towers within towers: a Tower is a stack of Stages (each round / octagonal / square, set back from the one below,
topped by a band: string course, corbel table + parapet, corbelled gallery with balustrade, crenellations or an open
arcaded lantern), a Roof (cone, octagonal spire, pyramid, helm, ogee, flat), crown towers standing on its top and
attached turrets (stair turrets, corbelled bartizans) - each of those is a Tower again.

Units: metres; x east, y north, z absolute (the crag tops are at Z = 80).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import world as W

Z = W.CASTLE_Z


# ---------------------------------------------------------------------------------------------------------- primitives
@dataclass
class Stage:
    h: float                               # height of the stage (m)
    size: float                            # radius (round / octagon) or half-width (square)
    band: str = "string"                   # top: string | corbel | gallery | crenel | arcade | none
    windows: int = 0                       # windows per floor (round / octagon) or per side (square); 0 = automatic
    floor_h: float = 5.2
    shape: str | None = None               # None = the tower's shape
    gables: bool = False                   # square stages: a steep gable (wimperg) with a window on every face
    clock: bool = False                    # clock faces (on the faces given by Tower.faces)
    depth: float | None = None             # square stages: half-depth if not square


@dataclass
class Roof:
    kind: str = "cone"                     # cone | spire (octagonal) | pyramid | helm | ogee | flat | dome
    h: float = 10.0
    lucarnes: int = 0                      # rows of lucarnes (small gabled dormers) on cones / spires
    finial: float = 3.0
    flare: float = 0.15                    # eave flare of cones / spires


@dataclass
class Tower:
    name: str
    at: tuple                              # (x, y) of the axis
    stages: list
    roof: Roof
    shape: str = "round"                   # round | octagon | square
    base: float = Z                        # z where the first stage starts
    foot: float | None = None              # foundation bottom (None = down to the rock)
    crown: list = field(default_factory=list)      # [(dx, dy, Tower)] standing on the top of the last stage
    attached: list = field(default_factory=list)   # [(angle_deg, inset, Tower)] against the side (stair turrets, bartizans)
    faces: tuple = (0.0,)                  # square towers: yaw of the front (deg); clock faces on these directions
    yaw: float = 0.0
    role: str = ""                         # landmark name for the docs / the shots

    def top(self):
        return self.base + sum(s.h for s in self.stages)

    def summit(self):
        """Highest point: roof + finial, or the tallest crown / attached piece (with their resolved bases)."""
        return max(b + sum(s.h for s in t.stages) + t.roof.h + t.roof.finial for t, _, b in walk(self))


@dataclass
class Hall:
    """A great hall: long buttressed walls with traceried lancets, pinnacles, parapet, steep roof, gable ends."""
    name: str
    p0: tuple
    p1: tuple                              # axis end points (x, y)
    width: float
    wall: float                            # eave height above the base
    bays: int
    pitch: float = 58.0
    base: float = Z
    tiers: int = 2                         # rows of windows per bay
    fleche: float = 0.0                    # slender spire on the ridge (height; 0 = none)
    lucarnes: int = 1                      # rows of lucarnes per roof side
    gable_turrets: tuple = (True, False)   # spired octagonal turrets flanking the gable at p0 / p1
    turret_h: float = 0.0                  # their height above the base (0 = 1.9 x wall)
    end_window: tuple = (True, False)      # great traceried window in the gable at p0 / p1
    parapet: str = "pierced"               # pierced | crenel | plain


@dataclass
class Range:
    """A wing: storeys of windows, a steep roof with dormers and chimneys (or a parapeted flat roof)."""
    name: str
    p0: tuple
    p1: tuple
    depth: float
    wall: float
    base: float = Z
    roof: str = "gable"                    # gable | hip | flat
    pitch: float = 52.0
    floor_h: float = 4.6
    dormers: bool = True
    chimneys: bool = True
    parapet: str = "none"                  # none | crenel | pierced
    ends: tuple = ("gable", "gable")       # gable | hip | none (abuts something)
    win: str = "lancet"                    # lancet | square | pair


@dataclass
class Arcade:
    """An arcaded front: an open loggia of tall arches with a storey and a balustraded terrace above."""
    name: str
    p0: tuple
    p1: tuple
    depth: float
    h: float
    arches: int
    base: float = Z
    upper: float = 6.0                     # height of the storey above the arcade


@dataclass
class Gatehouse:
    name: str
    at: tuple
    w: float
    d: float
    h: float
    yaw: float = 0.0                       # passage direction (deg, 0 = along x)
    turret_r: float = 3.0
    turret_h: float = 0.0
    base: float = Z


@dataclass
class Wall:
    """Curtain wall / terrace wall along a polyline (crenellated, machicolated on corbels)."""
    name: str
    path: list
    h: float
    thick: float = 2.4
    base: float = Z
    crenel: bool = True
    machicolations: bool = False


@dataclass
class Viaduct:
    """The grand viaduct: a polyline (it bends at a pier tower), tiers of arches down to the water or the rock (the
    lower tier wider and taller), pier buttresses, an arcaded gallery on the deck."""
    name: str
    path: list                             # [(x, y), ...] - a bend at every inner point (a pier tower there)
    deck: float
    width: float
    span: float = 13.0                     # upper-tier arch span (m); the lower tier spans two of them
    tiers: int = 2
    pier: float = 3.0
    parapet: str = "arcaded"               # arcaded | pierced | plain


@dataclass
class ArchBridge:
    """A stone bridge of one or a few big arches (the stone bridge over the inlet)."""
    name: str
    p0: tuple
    p1: tuple
    deck: float
    width: float = 6.0
    arches: int = 1


@dataclass
class SuspensionBridge:
    """A covered timber walkway hung from chains between two stone pylon towers."""
    name: str
    p0: tuple
    p1: tuple
    deck: float
    width: float = 4.0
    pylon_h: float = 18.0


@dataclass
class Glasshouse:
    """Greenhouses: iron-and-glass halls on a stone plinth."""
    name: str
    p0: tuple
    p1: tuple
    width: float
    h: float = 9.0
    base: float = Z


@dataclass
class Stair:
    """The entry stairs: straight flights along x (x_start, x_end, y, z_start, z_end) on benches up the cliff, landing
    bastions (x0, x1, y0, y1, z, side the flights leave from), a pointed arch where a flight bridges the stream's cleft."""
    name: str
    flights: list
    landings: list
    width: float = 5.0
    wall_t: float = 1.0
    cleft_x: float | None = None


@dataclass
class CoveredBridge:
    """A covered timber bridge (after the illustrated map and the Legacy key art): a gallery with planked lower walls, a
    band of open bays and a slate roof, on stone piers in the gorge and timber trestles on the banks."""
    name: str
    p0: tuple
    p1: tuple
    z0: float                              # deck at p0 / p1
    z1: float
    width: float = 4.4


@dataclass
class CliffWalls:
    """The castle's deep foundations: battered, buttressed walls down the cliff faces along the plateau rims, rows of
    small windows low on the rock, crenellated terrace parapets where no building stands at the rim, towers rooted
    deep on the cliff at the rocks' corners."""
    name: str
    depth: tuple = (5.0, 16.0)            # wall depth below the plateau: range along the rim (m); the rock swallows the foot
    towers: int = 12


@dataclass
class Boathouse:
    name: str
    at: tuple
    yaw: float
    length: float
    width: float
    base: float


# ------------------------------------------------------------------------------------------------------ tower helpers
def R(h, r, band="string", **kw):
    return Stage(h, r, band, **kw)


def round_tower(name, at, r, h, roof_h, *, base=Z, band="corbel", lucarnes=0, roof="cone", stages=None, crown=(), attached=(),
                role="", finial=3.0, **kw):
    st = stages if stages is not None else [Stage(h, r, band, **kw)]
    return Tower(name, at, st, Roof(roof, roof_h, lucarnes, finial), "round", base, crown=list(crown), attached=list(attached),
                 role=role)


def stair_turret(name, r, h, roof_h, base=Z):
    return Tower(name, (0.0, 0.0), [Stage(h, r, "corbel", floor_h=4.0)], Roof("cone", roof_h, 0, 1.6), "round", base)


def pinnacle_turret(name, r, h, roof_h):
    """Slender corner turret for crowns (octagonal, spired)."""
    return Tower(name, (0.0, 0.0), [Stage(h, r, "string", floor_h=3.2)], Roof("spire", roof_h, 0, 1.2), "octagon")


def corners(half, inset=0.0):
    a = half - inset
    return [(a, a), (-a, a), (-a, -a), (a, -a)]


# ---------------------------------------------------------------------------------------------------------- the plan
def quad(name, x0, x1, y0, y1, depth, wall, *, skip=(), roof="gable", parapets=("none",) * 4, walls=None, cross_x=(), cross_y=(),
         turrets=True):
    """Four ranges round a court (outer rectangle x0..x1, y0..y1): south, east, north, west; `skip` drops sides that
    something else closes; cross_x / cross_y add ranges across the court (splitting it into smaller courts); stair
    turrets stand in the court's inner corners."""
    d = depth / 2
    sides = {"S": ((x0, y0 + d), (x1, y0 + d)), "E": ((x1 - d, y0 + depth), (x1 - d, y1 - depth)),
             "N": ((x1, y1 - d), (x0, y1 - d)), "W": ((x0 + d, y1 - depth), (x0 + d, y0 + depth))}
    out, towers = [], []
    for k, (side, (a, b)) in enumerate(sides.items()):
        if side in skip:
            continue
        w = wall if walls is None else walls[k]
        out.append(Range(f"{name}_{side}", a, b, depth, w, roof=roof, parapet=parapets[k], ends=("none", "none")))
    for i, x in enumerate(cross_x):
        out.append(Range(f"{name}_X{i}", (x, y0 + depth), (x, y1 - depth), depth * 0.8, wall - 2.0, ends=("none", "none")))
    for i, y in enumerate(cross_y):
        out.append(Range(f"{name}_Y{i}", (x0 + depth, y), (x1 - depth, y), depth * 0.8, wall - 2.0, ends=("none", "none")))
    if turrets:
        for j, (cx, cy) in enumerate(((x0 + depth, y0 + depth), (x1 - depth, y0 + depth), (x1 - depth, y1 - depth), (x0 + depth, y1 - depth))):
            if (j == 0 and ("S" in skip or "W" in skip)) or (j == 1 and ("S" in skip or "E" in skip)) \
                    or (j == 2 and ("N" in skip or "E" in skip)) or (j == 3 and ("N" in skip or "W" in skip)):
                continue
            towers.append(Tower(f"{name}_T{j}", (cx, cy), [Stage(wall + 10.0, 3.4, "corbel", floor_h=4.2)],
                                Roof("cone", 12.0, 0, 2.0), "round"))
    return out, towers


def quad_r(name, centre, half_w, half_d, angle, depth, wall, *, skip=(), parapets=("none",) * 4, walls=None, turrets=True,
           cross=()):
    """A quadrangle rotated by `angle` (deg): four ranges round a court; sides S (local -y), E, N, W.  `cross`: local x
    positions of ranges across the court."""
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    cx, cy = centre

    def P(u, v):
        return (cx + u * ca - v * sa, cy + u * sa + v * ca)
    d = depth / 2
    sides = {"S": (P(-half_w, -half_d + d), P(half_w, -half_d + d)), "E": (P(half_w - d, -half_d + depth), P(half_w - d, half_d - depth)),
             "N": (P(half_w, half_d - d), P(-half_w, half_d - d)), "W": (P(-half_w + d, half_d - depth), P(-half_w + d, -half_d + depth))}
    out, towers = [], []
    for k, (side, (a, b)) in enumerate(sides.items()):
        if side in skip:
            continue
        w = wall if walls is None else walls[k]
        out.append(Range(f"{name}_{side}", a, b, depth, w, parapet=parapets[k], ends=("none", "none")))
    for i, u in enumerate(cross):
        out.append(Range(f"{name}_X{i}", P(u, -half_d + depth), P(u, half_d - depth), depth * 0.8, wall - 2.0, ends=("none", "none")))
    if turrets:
        for j, (u, v) in enumerate(((-half_w + depth, -half_d + depth), (half_w - depth, -half_d + depth), (half_w - depth, half_d - depth),
                                    (-half_w + depth, half_d - depth))):
            sides_j = (("S", "W"), ("S", "E"), ("N", "E"), ("N", "W"))[j]
            if any(sd in skip for sd in sides_j):
                continue
            towers.append(Tower(f"{name}_T{j}", P(u, v), [Stage(wall + 10.0, 3.4, "corbel", floor_h=4.2)], Roof("cone", 12.0, 0, 2.0),
                                "round"))
    return out, towers


def plan():
    """Layout after the Legacy castle map (world = ((mx - 1000) * 0.24, (1100 - my) * 0.24)); buildings in the film style of
    the posters.  Seen from the boats in the east arm (looking west): the Great Hall's long side on the left, the Grand
    Staircase Tower behind its north end, the Viaduct court, the grand viaduct over the inlet, the north-west rock's
    towers on the right, the boathouse spur with the entry stairs and the waterfall in front."""
    T = []          # towers
    B = []          # halls, ranges, arcades, gatehouses, walls, bridges, stairs, boathouse

    def Q(*args, **kw):
        r, t = quad_r(*args, **kw)
        B.extend(r)
        T.extend(t)

    # ======================================================================= the south-east rock (the island)
    # --- the Great Hall: north-south along the east arm, its long buttressed side to the lake (the boats' view)
    B.append(Hall("GreatHall", (150.0, -22.0), (150.0, -100.0), 30.0, 34.0, 10, pitch=58.0, tiers=2, fleche=22.0, lucarnes=2,
                  gable_turrets=(False, True), turret_h=66.0, end_window=(False, True)))
    B.append(Range("GreatHallWestAisle", (126.0, -30.0), (126.0, -96.0), 14.0, 22.0, ends=("none", "hip"), parapet="crenel"))
    # --- the Entrance Hall and the Reception Hall between the Viaduct court and the Grand Staircase
    B.append(Hall("EntranceHall", (86.0, 0.0), (168.0, 0.0), 22.0, 32.0, 7, pitch=56.0, tiers=1, lucarnes=1,
                  gable_turrets=(False, True), turret_h=52.0, end_window=(False, True), parapet="crenel"))
    B.append(Range("ReceptionHall", (72.0, -20.0), (118.0, -20.0), 18.0, 28.0, ends=("none", "none")))
    B.append(Range("Kitchens", (70.0, -48.0), (98.0, -48.0), 18.0, 22.0, ends=("hip", "none"), parapet="crenel"))
    # --- the Grand Staircase Tower (the great round tower with the tall cone) + the stepped Headmaster's Tower
    hm_top = [(0.0, 0.0, Tower("HeadmasterCrown", (0, 0), [Stage(10.0, 4.0, "string", floor_h=3.0)],
                               Roof("spire", 20.0, 1, 3.0), "octagon"))]
    hm_top += [(dx, dy, pinnacle_turret("HMPin", 1.2, 7.0, 8.0)) for dx, dy in corners(5.2, 0.6)]
    headmaster = Tower("HeadmasterTower", (0.0, 0.0),
                       [Stage(58.0, 6.8, "string"), Stage(18.0, 6.8, "gallery", gables=True), Stage(14.0, 5.2, "corbel")],
                       Roof("flat", 0.0, 0, 0.0), "square", crown=hm_top, role="the Headmaster's Tower")
    T.append(Tower("GrandStaircaseTower", (56.0, -28.0),
                   [Stage(30.0, 16.5, "string", floor_h=6.0), Stage(32.0, 16.5, "corbel", floor_h=6.0)],
                   Roof("cone", 54.0, 3, 6.0, 0.18), "round",
                   attached=[(25.0, 0.32, headmaster), (200.0, 0.5, stair_turret("GSTStair", 3.4, 72.0, 12.0))],
                   role="the Grand Staircase Tower"))
    # --- the Viaduct court: the quadrangle where the grand viaduct lands
    Q("ViaductCourt", (104.0, 46.0), 34.0, 36.0, 0.0, 20.0, 30.0, walls=(30.0, 28.0, 32.0, 28.0), parapets=("none", "crenel", "crenel", "none"))
    T.append(round_tower("ViaductCourtNE", (138.0, 82.0), 7.5, 50.0, 22.0, band="corbel", lucarnes=1))
    T.append(round_tower("ViaductCourtNW", (72.0, 80.0), 6.0, 44.0, 20.0, band="corbel"))
    T.append(round_tower("ViaductCourtSE", (166.0, 12.0), 6.0, 46.0, 20.0, band="corbel"))
    # --- the Ravenclaw tower: great round tower, open arcaded belvedere under a needle spire
    T.append(Tower("RavenclawTower", (-2.0, 12.0),
                   [Stage(52.0, 12.5, "gallery"), Stage(11.0, 10.5, "arcade", windows=10), Stage(3.0, 10.5, "corbel")],
                   Roof("cone", 44.0, 2, 4.0, 0.1), "round", role="the Ravenclaw tower (arcaded belvedere, needle spire)"))
    B.append(Range("SlytherinRange", (14.0, 28.0), (52.0, 18.0), 18.0, 26.0, ends=("none", "hip")))
    # --- the Quad
    Q("Quad", (14.0, -32.0), 26.0, 21.0, -8.0, 16.0, 28.0, walls=(28.0, 26.0, 30.0, 28.0))
    # --- the Hospital wing, the Faculty tower, Gryffindor
    B.append(Range("HospitalWing", (6.0, -54.0), (10.0, -100.0), 22.0, 28.0, ends=("none", "none"), parapet="crenel"))
    T.append(round_tower("FacultyTower", (36.0, -78.0), 8.0, 54.0, 24.0, band="corbel", lucarnes=1, role="the Faculty tower"))
    B.append(Range("FacultyRange", (20.0, -64.0), (52.0, -92.0), 16.0, 24.0, ends=("none", "hip")))
    T.append(round_tower("GryffindorTower", (-34.0, -66.0), 11.0, 70.0, 30.0, band="corbel", lucarnes=1,
                         attached=[(140.0, 0.45, stair_turret("GTStair", 3.0, 60.0, 11.0))], role="Gryffindor tower"))
    B.append(Range("GryffindorRange", (-24.0, -48.0), (4.0, -46.0), 18.0, 26.0, ends=("hip", "none")))
    # --- Hufflepuff (mostly underground): a garden terrace with its round pod turrets over the south-east cliff
    B.append(Range("HufflepuffRange", (36.0, -110.0), (100.0, -104.0), 14.0, 16.0, ends=("hip", "hip"), parapet="crenel"))
    for px, py in ((50.0, -122.0), (72.0, -124.0), (94.0, -120.0)):
        T.append(round_tower(f"Pod{int(px)}", (px, py), 4.6, 12.0, 6.0, band="crenel", base=Z - 4.0))
    # --- the South Wing: the Clock Tower and the Clock Tower court, slanting south-south-west to the lake
    clock = Tower("ClockTower", (-16.0, -114.0),
                  [Stage(42.0, 8.0, "string"), Stage(24.0, 8.0, "corbel", clock=True),
                   Stage(13.0, 6.8, "arcade", windows=3), Stage(2.5, 6.8, "crenel")],
                  Roof("spire", 32.0, 1, 3.5), "square", faces=(-90.0, 0.0), yaw=-25.0,
                  crown=[(dx, dy, pinnacle_turret("ClockPin", 1.1, 6.0, 9.0)) for dx, dy in corners(6.8, 0.8)],
                  role="the Clock Tower")
    T.append(clock)
    B.append(Range("SouthWingNorth", (-2.0, -100.0), (-30.0, -112.0), 22.0, 28.0, ends=("none", "none")))
    Q("ClockCourt", (-34.0, -146.0), 22.0, 26.0, -25.0, 14.0, 24.0, walls=(24.0, 22.0, 26.0, 22.0),
      parapets=("crenel", "none", "none", "crenel"))
    T.append(round_tower("SouthTipW", (-60.0, -168.0), 5.0, 30.0, 14.0, band="corbel", base=Z - 2.0))
    T.append(round_tower("SouthTipE", (-30.0, -182.0), 5.0, 30.0, 14.0, band="corbel", base=Z - 2.0))

    # ======================================================================= the north-west rock (the mainland side)
    # --- the Library and North Hall block with the Bell Tower at its north-west corner
    Q("LibraryBlock", (-70.0, 122.0), 44.0, 24.0, -21.0, 20.0, 32.0, walls=(30.0, 32.0, 32.0, 30.0),
      parapets=("none", "none", "crenel", "none"))
    T.append(Tower("BellTower", (-100.0, 138.0), [Stage(44.0, 8.0, "string"), Stage(14.0, 8.0, "corbel", gables=True)],
                   Roof("spire", 24.0, 0, 3.0), "square", yaw=-21.0,
                   crown=[(dx, dy, pinnacle_turret("BTPin", 1.4, 10.0, 10.0)) for dx, dy in corners(8.0, 0.7)]
                   + [(dx * 0.5, dy * 0.5, pinnacle_turret("BTPin2", 0.9, 6.0, 7.0)) for dx, dy in corners(8.0)],
                   role="the Bell Tower"))
    # --- the Transfiguration court
    Q("TransfigurationCourt", (-68.0, 60.0), 40.0, 30.0, -20.0, 20.0, 28.0, skip=("N",), walls=(28.0, 30.0, 28.0, 30.0))
    # --- the Astronomy wing along the west basin: the Astronomy Tower (the tallest), Charms, the DADA tower, the West Tower
    B.append(Range("AstronomyWing", (-138.0, 36.0), (-54.0, 16.0), 24.0, 30.0, ends=("none", "none"), parapet="crenel"))
    B.append(Range("FigsWing", (-120.0, 4.0), (-62.0, -6.0), 18.0, 24.0, ends=("hip", "hip")))
    astro_crown = [(0.0, 0.0, Tower("AstroLantern", (0, 0), [Stage(15.0, 5.4, "corbel", floor_h=4.0)], Roof("cone", 28.0, 1, 3.5),
                                    "round"))]
    for ang in (45, 135, 225, 315):
        astro_crown.append((7.0 * math.cos(math.radians(ang)), 7.0 * math.sin(math.radians(ang)),
                            pinnacle_turret("AstroPin", 1.4, 10.0, 11.0)))
    T.append(Tower("AstronomyTower", (-98.0, 20.0),
                   [Stage(42.0, 11.0, "string"), Stage(30.0, 11.0, "gallery"), Stage(26.0, 9.0, "corbel"), Stage(3.0, 9.0, "crenel")],
                   Roof("flat", 0.0, 0, 0.0), "round", crown=astro_crown,
                   attached=[(250.0, 0.4, stair_turret("AstroStair", 3.0, 80.0, 13.0))],
                   role="the Astronomy Tower (the tallest)"))
    T.append(Tower("DADATower", (-48.0, 22.0), [Stage(50.0, 7.5, "string"), Stage(14.0, 6.5, "corbel", gables=True)],
                   Roof("spire", 30.0, 1, 3.0), "square", yaw=-12.0,
                   crown=[(dx, dy, pinnacle_turret("DADAPin", 1.0, 5.0, 7.0)) for dx, dy in corners(6.5, 0.7)],
                   role="the Defence Against the Dark Arts tower"))
    T.append(round_tower("CharmsTower", (-76.0, 26.0), 6.0, 50.0, 22.0, band="corbel"))
    T.append(Tower("WestTower", (-144.0, 42.0), [Stage(46.0, 11.0, "string"), Stage(4.0, 11.0, "crenel")],
                   Roof("spire", 44.0, 1, 3.0, 0.08), "octagon",
                   crown=[(10.4 * math.cos(math.radians(ang)), 10.4 * math.sin(math.radians(ang)),
                           pinnacle_turret("WTPin", 1.1, 4.0, 7.0)) for ang in range(0, 360, 90)], role="the West Tower"))
    # --- along the inlet: the Central Hall tower, the Arithmancy tower, Divination, the Viaduct Entrance, Potions
    B.append(Range("CentralRange", (-26.0, 120.0), (0.0, 92.0), 22.0, 30.0, ends=("none", "none")))
    T.append(Tower("CentralHallTower", (-34.0, 80.0), [Stage(54.0, 9.0, "string"), Stage(16.0, 9.0, "corbel", gables=True)],
                   Roof("helm", 22.0, 0, 3.0), "square", yaw=-20.0,
                   crown=[(dx, dy, pinnacle_turret("CTPin", 1.2, 5.0, 7.0)) for dx, dy in corners(9.0, 0.9)],
                   role="the Central Hall tower"))
    T.append(round_tower("ArithmancyTower", (-36.0, 100.0), 8.6, 56.0, 26.0, band="gallery", lucarnes=1))
    T.append(Tower("DivinationTower", (32.0, 156.0), [Stage(54.0, 8.0, "gallery"), Stage(12.0, 6.6, "corbel")],
                   Roof("cone", 28.0, 1, 3.0), "round", role="the Divination tower"))
    B.append(Gatehouse("ViaductEntrance", (-2.0, 138.0), 18.0, 16.0, 30.0, yaw=-30.0, turret_r=3.4, turret_h=42.0))
    B.append(Range("PotionsRange", (-8.0, 66.0), (12.0, 40.0), 18.0, 26.0, ends=("hip", "hip"), parapet="crenel"))
    # --- the north: the annex's greenhouses and walls, the north gate to the grounds
    B.append(Glasshouse("Greenhouse1", (-30.0, 150.0), (20.0, 168.0), 16.0, 10.0))
    B.append(Glasshouse("Greenhouse2", (-6.0, 136.0), (6.0, 182.0), 14.0, 12.0))
    B.append(Wall("AnnexWall", [(-112.0, 168.0), (-60.0, 192.0), (0.0, 196.0), (40.0, 188.0), (62.0, 172.0), (64.0, 204.0)], 9.0,
                  crenel=True))
    B.append(Gatehouse("NorthGate", (-40.0, 196.0), 20.0, 16.0, 26.0, yaw=90.0, turret_r=6.0, turret_h=38.0))
    T.append(round_tower("NorthWestTower", (-140.0, 150.0), 7.0, 50.0, 22.0, band="corbel"))

    # ======================================================================= the bridges over the inlet
    B.append(Viaduct("GrandViaduct", list(W.VIADUCT), Z - 1.0, 8.0, span=12.5, tiers=2, pier=3.0, parapet="arcaded"))
    T.append(Tower("ViaductPierTower", W.VIADUCT[1], [Stage(Z + 12.0, 6.0, "corbel", floor_h=6.0)], Roof("cone", 14.0, 0, 2.0),
                   "round", base=-2.0, role="the tower at the viaduct's bend"))
    B.append(ArchBridge("StoneBridge", W.STONE_BRIDGE[0], W.STONE_BRIDGE[1], Z - 2.0, 6.0, 2))
    B.append(SuspensionBridge("SuspensionBridge", W.SUSPENSION_BRIDGE[0], W.SUSPENSION_BRIDGE[1], Z - 6.0, 4.0, 16.0))

    # ======================================================================= the boathouse, the entry stairs, the Map Chamber
    B.append(Boathouse("Boathouse", W.BOATHOUSE, 90.0, W.BOATHOUSE_LEN, 13.0, 1.2))
    # the entry stairs climb the bay's north face in three flights joined by landing bastions; the stream falls through
    # the arch under each flight (world.py shapes the face, the benches and the cleft to match)
    B.append(Stair("EntryStairs", list(W.STAIR_FLIGHTS), list(W.STAIR_LANDINGS), W.STAIR_W, W.STAIR_WALL, W.FALL_X))
    T.append(round_tower("StairHeadTower", (62.0, 209.0), 4.6, 20.0, 13.0, band="corbel", base=77.0,
                         role="the tower at the head of the entry stairs"))
    # ======================================================================= the back: the covered bridge over the stream's gorge
    # from a gate tower at the castle's north-east corner (walled to the stair head) across the gorge to the grounds and
    # the stone circle - as in the films, where the wooden bridge leads to the stone circle and on down to the hut
    B.append(Wall("BridgeCourtWall", [(62.0, 213.6), (72.0, 236.0), (92.0, 256.0)], 8.0, crenel=True))
    T.append(Tower("BridgeGate", (98.0, 261.0), [Stage(15.0, 6.0, "string", floor_h=5.0), Stage(8.0, 6.0, "corbel", gables=True)],
                   Roof("helm", 14.0, 0, 2.5), "square", base=77.0, yaw=25.0,
                   crown=[(dx, dy, pinnacle_turret("BGPin", 0.9, 4.0, 6.0)) for dx, dy in corners(6.0, 0.6)],
                   role="the gate of the covered bridge"))
    B.append(CoveredBridge("CoveredBridge", (104.0, 264.0), (151.0, 286.0), 79.0, 79.0, 4.4))
    T.append(round_tower("BridgeEastTower", (155.5, 288.0), 3.6, 10.0, 9.0, band="corbel", base=77.0,
                         role="the far end of the covered bridge"))
    # ======================================================================= the deep foundations down the cliffs
    B.append(CliffWalls("CliffWalls"))
    T.append(round_tower("MapChamber", (STACKS_MAP[0], STACKS_MAP[1]), 11.0, 16.0, 10.0, band="corbel", base=STACKS_MAP[2] - 2.0,
                         roof="dome", role="the Map Chamber on its sea stack"))
    B.append(ArchBridge("MapChamberBridge", (-90.0, -46.0), (-82.0, -12.0), STACKS_MAP[2] + 2.0, 3.5, 1))
    return T, B


STACKS_MAP = (W.STACKS[0][0], W.STACKS[0][1], W.STACKS[0][3])


def walk(t, origin=(0.0, 0.0, None)):
    """Yield (tower, (x, y), base) for a tower and all its crown / attached towers, resolving relative placement."""
    x, y = t.at[0] + origin[0], t.at[1] + origin[1]
    base = t.base if origin[2] is None else origin[2]
    yield t, (x, y), base
    top = base + sum(s.h for s in t.stages)
    for dx, dy, c in t.crown:
        yield from walk(c, (x + dx - c.at[0], y + dy - c.at[1], top))
    r0 = t.stages[0].size
    for ang, inset, c in t.attached:
        a = math.radians(ang)
        rc = c.stages[0].size
        d = r0 + rc * (1.0 - 2.0 * inset)
        if t.shape == "square":
            d = r0 / max(abs(math.cos(a)), abs(math.sin(a)), 1e-6) * 0.98 + rc * (1.0 - 2.0 * inset)
        cbase = c.base if c.base != Z else base
        yield from walk(c, (x + d * math.cos(a) - c.at[0], y + d * math.sin(a) - c.at[1], cbase))


if __name__ == "__main__":
    T, B = plan()
    n = sum(1 for t in T for _ in walk(t))
    print(f"{len(T)} towers ({n} with crowns / turrets), {len(B)} other buildings")
    for t in sorted(T, key=lambda t: -t.summit())[:12]:
        print(f"  {t.name:22s} summit {t.summit():6.1f} m  ({t.summit() - Z:5.1f} above the crag)  {t.role}")
