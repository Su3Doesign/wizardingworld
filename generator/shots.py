"""shots - the camera placements for the main events + the lighting presets (single source of truth: the scatter cuts
sight-line clearings for them, the Cycles previews render them, scene.json carries them to the Unreal builder).

Positions are Blender metres (z absolute); 'above' cameras are placed at a height above the ground computed at build
time.  A shot with several keys is animated (seconds); 'preset' picks the lighting / atmosphere set in Unreal.
"""
from __future__ import annotations

import math

import numpy as np

PRESETS = {
    # sun (or moon) elevation / azimuth (deg, azimuth clockwise from north), intensities in Unreal units (lux-ish with
    # auto exposure), fog, windows (0 = dark, 1 = lit), sky / cloud settings
    "mist": dict(sun_elev=16.0, sun_az=118.0, sun_lux=6.5, sun_color=(255, 236, 214), sky=1.0, fog_density=0.045,
                 fog_falloff=0.010, fog_height=40.0, vol_fog=True, mist_banks=True, windows=0.0, clouds=0.85, exposure=0.3,
                 tint=(0.96, 1.0, 1.0), saturation=0.92),
    "day": dict(sun_elev=42.0, sun_az=205.0, sun_lux=10.0, sun_color=(255, 245, 230), sky=1.0, fog_density=0.012,
                fog_falloff=0.006, fog_height=0.0, vol_fog=True, mist_banks=False, windows=0.0, clouds=0.55, exposure=0.0,
                tint=(1.0, 1.0, 1.0), saturation=1.0),
    # sunset: a midsummer Highland sunset in the north-west.  The mountain ring rises 12-20 deg above the castle, so the
    # sun is already behind the ridge: the valley is in shadow, the sky glows behind the castle (ref 4's silhouette)
    "sunset": dict(sun_elev=3.5, sun_az=316.0, sun_lux=8.0, sun_color=(255, 168, 112), sky=1.0, fog_density=0.009,
                   fog_falloff=0.010, fog_height=10.0, vol_fog=True, mist_banks=True, windows=0.55, clouds=0.6, exposure=0.6,
                   tint=(1.0, 0.96, 0.92), saturation=1.05),
    # night, after the night castle reference: a bright blue moonlit night (day-for-night), clouds lit by the moon, blue
    # mist in the valleys, warm windows and lanterns
    "night": dict(sun_elev=24.0, sun_az=340.0, sun_lux=0.8, sun_color=(165, 192, 255), sky=0.2, fog_density=0.02,
                  fog_falloff=0.012, fog_height=5.0, vol_fog=True, mist_banks=True, windows=1.0, clouds=0.6, exposure=1.2,
                  tint=(0.88, 0.95, 1.08), saturation=1.0, moon=True, stars=True, fog_color=(0.03, 0.05, 0.11)),
    "dusk": dict(sun_elev=0.8, sun_az=300.0, sun_lux=3.0, sun_color=(255, 150, 120), sky=0.6, fog_density=0.025,
                 fog_falloff=0.010, fog_height=8.0, vol_fog=True, mist_banks=True, windows=0.8, clouds=0.6, exposure=1.0,
                 tint=(0.98, 0.98, 1.02), saturation=1.0),
}

# aspect: portrait 9:16 (2160x3840), tall 2:3 (2560x3840), landscape 16:9 (3840x2160)
RES = {"portrait": (2160, 3840), "tall": (2560, 3840), "landscape": (3840, 2160), "square": (3000, 3000)}

SHOTS = {
    "CAM_01_Gorge_Mist": dict(
        title="Ref 1: the castle above the gorge, misty morning", aspect="portrait", lens=24.0, fstop=8.0, preset="mist",
        subject_low=30.0,
        keys=[dict(t=0.0, loc=(-300.0, -30.0, ("above", 30.0)), look_at=(-140.0, 40.0, 98.0)),
              dict(t=10.0, loc=(-296.0, -10.0, ("above", 33.0)), look_at=(-140.0, 44.0, 100.0))]),
    "CAM_02_Boats_Night": dict(
        title="Ref 2 / the night poster: the first-years cross the east arm by lantern light towards the boathouse", aspect="tall",
        lens=28.0, fstop=4.0, preset="night",
        keys=[dict(t=0.0, loc=(440.0, -190.0, 3.4), look_at=(100.0, 40.0, 100.0)),
              dict(t=12.0, loc=(426.0, -140.0, 3.6), look_at=(100.0, 50.0, 100.0))]),
    "CAM_03_Lake_Sunset": dict(
        title="Ref 4: the castle silhouetted against the sunset, from the lake", aspect="landscape", lens=35.0, fstop=8.0,
        preset="sunset", keys=[dict(t=0.0, loc=(338.0, -433.0, 7.0), look_at=(0.0, 0.0, 84.0))]),
    "CAM_04_Viaduct_Walk": dict(
        title="Walking the grand viaduct over the inlet into the Viaduct court", aspect="landscape", lens=24.0, fstop=5.6,
        preset="day",
        keys=[dict(t=0.0, loc=(20.0, 126.5, 81.6), look_at=(62.0, 84.0, 90.0)),            # out of the Viaduct Entrance
              dict(t=12.0, loc=(40.0, 114.0, 80.75), look_at=(72.0, 62.0, 92.0))]),        # deck 79 m
    "CAM_05_Quidditch_Aerial": dict(
        title="Over the Quidditch stadium towards the castle", aspect="landscape", lens=28.0, fstop=8.0, preset="day",
        keys=[dict(t=0.0, loc=(-360.0, 1060.0, 205.0), look_at=(-40.0, 140.0, 105.0)),
              dict(t=10.0, loc=(-300.0, 960.0, 185.0), look_at=(-40.0, 140.0, 105.0))]),
    "CAM_06_ForestEdge_Hut": dict(
        title="From the edge of the forest by the hut, dusk", aspect="landscape", lens=30.0, fstop=4.0, preset="dusk",
        keys=[dict(t=0.0, loc=(512.0, 512.0, ("above", 2.2)), look_at=(30.0, 20.0, 120.0))]),
    "CAM_07_Astronomy_Tower": dict(
        title="From the top of the Astronomy Tower over the lake, night", aspect="landscape", lens=20.0, fstop=8.0,
        preset="night", keys=[dict(t=0.0, loc=(-98.0, 13.5, 182.8), look_at=(60.0, -700.0, 20.0))]),
    "CAM_08_Station_Night": dict(
        title="From the station across the lake to the lit castle", aspect="landscape", lens=55.0, fstop=5.6, preset="night",
        keys=[dict(t=0.0, loc=(985.0, -655.0, 7.5), look_at=(0.0, -40.0, 112.0))]),
    "CAM_09_Grand_Orbit": dict(
        title="Grand orbit (24 s)", aspect="landscape", lens=24.0, fstop=11.0, preset="day", orbit=dict(center=(10.0, 20.0, 105.0),
                                                                                                       radius=720.0, z=260.0,
                                                                                                       a0=-120.0, a1=60.0,
                                                                                                       seconds=24.0)),
    "CAM_10_Cliff_Moss": dict(
        title="Detail: moss, ledges and the deep walls on the gorge face", aspect="portrait", lens=40.0, fstop=4.0,
        preset="mist", keys=[dict(t=0.0, loc=(-250.0, 10.0, 52.0), look_at=(-175.0, 30.0, 60.0))]),
    "CAM_11_Gorge_River": dict(
        title="From the river up the gorge to the castle", aspect="portrait", lens=22.0, fstop=8.0, preset="mist",
        keys=[dict(t=0.0, loc=(-215.0, -170.0, 7.0), look_at=(-150.0, 20.0, 100.0))]),
    "CAM_12_Plan_Top": dict(
        title="Site plan (top view)", aspect="square", lens=60.0, fstop=16.0, preset="day",
        keys=[dict(t=0.0, loc=(0.0, 200.0, 4200.0), look_at=(0.0, 201.0, 0.0))]),
    "CAM_13_Night_Cliff": dict(
        title="The night castle reference: the castle on its cliff over the lake, deep walls and lit windows, moonlight",
        aspect="landscape", lens=35.0, fstop=5.6, preset="night",
        keys=[dict(t=0.0, loc=(-470.0, -420.0, 45.0), look_at=(-70.0, -30.0, 105.0))]),
    "CAM_14_Boathouse_Stairs": dict(
        title="The studio model's view: the entry stairs climbing the cliff, the waterfall through their arches, dusk",
        aspect="landscape", lens=24.0, fstop=5.6, preset="dusk",
        keys=[dict(t=0.0, loc=(255.0, 95.0, 5.0), look_at=(135.0, 200.0, 45.0))]),
    "CAM_15_Covered_Bridge": dict(
        title="The covered bridge over the stream's gorge, from the grounds, misty morning", aspect="landscape", lens=28.0,
        fstop=8.0, preset="mist",
        keys=[dict(t=0.0, loc=(113.0, 362.0, ("above", 8.0)), look_at=(128.0, 270.0, 76.0))]),
}


def orbit_keys(o, step=2.0):
    keys = []
    n = int(o["seconds"] / step)
    for i in range(n + 1):
        t = i * step
        a = math.radians(o["a0"] + (o["a1"] - o["a0"]) * i / n)
        cx, cy, cz = o["center"]
        keys.append(dict(t=t, loc=(cx + o["radius"] * math.cos(a), cy + o["radius"] * math.sin(a), o["z"]), look_at=(cx, cy, cz)))
    return keys


def resolve(ground_fn):
    """Absolute camera keys (resolving ('above', h) heights with ground_fn(x, y))."""
    out = {}
    for name, s in SHOTS.items():
        keys = orbit_keys(s["orbit"]) if "orbit" in s else s["keys"]
        rk = []
        for k in keys:
            x, y, z = k["loc"]
            if isinstance(z, tuple):
                z = float(ground_fn(x, y)) + z[1]
            rk.append(dict(t=float(k["t"]), loc=[float(x), float(y), float(z)], look_at=[float(v) for v in k["look_at"]]))
        if len(rk) == 1:
            rk.append(dict(t=1.0 / 24.0, loc=rk[0]["loc"], look_at=rk[0]["look_at"]))
        out[name] = dict(title=s["title"], aspect=s["aspect"], res=list(RES[s["aspect"]]), lens=s["lens"], fstop=s["fstop"],
                         preset=s["preset"], sequence="LS_" + name[4:], keys=rk, subject_low=s.get("subject_low"))
    return out


def clearings(resolved, half_angle=34.0):
    """Sight lines: (camera position, view direction, target point, half angle).  The scatter removes trees whose crown
    rises into the line of sight from the camera to the low part of the subject (so treetops lower down the slope stay
    in the foreground, as in the references)."""
    out = []
    for name, s in resolved.items():
        if name == "CAM_12_Plan_Top":
            continue
        for k in s["keys"]:
            p = np.array(k["loc"])
            tgt = np.array(k["look_at"], np.float64)
            low = tgt.copy()
            low[2] = s.get("subject_low") if s.get("subject_low") is not None else tgt[2] - 40.0
            d = tgt - p
            d /= np.linalg.norm(d)
            out.append((p, d, low, half_angle))
    return out


MOON_DIR = (348.0, 24.0)          # azimuth, elevation (deg) of the moon in the night preset
MOON_DIAMETER_DEG = 7.0           # a cinematic moon (ref 2), ~14x the real one


def moon_position(dist=20000.0):
    az, el = (math.radians(v) for v in MOON_DIR)
    d = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    return d * dist, 2 * dist * math.tan(math.radians(MOON_DIAMETER_DEG / 2))
