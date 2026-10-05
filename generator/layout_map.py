"""layout_map - an annotated plan of the world (hillshade + forests + water + landmarks + every shot's camera and view
direction) for the README:  python layout_map.py ../docs/layout_map.png"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import shots as SH
import world as W


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main(out, n=1600, half=1250.0, center=(150.0, 120.0)):
    xs = np.linspace(center[0] - half, center[0] + half, n)
    ys = np.linspace(center[1] + half, center[1] - half, n)
    X, Y = np.meshgrid(xs, ys)
    F = W.Fields(X, Y)
    H = F.height().reshape(n, n)
    cell = xs[1] - xs[0]
    gy, gx = np.gradient(H, -cell, cell)
    nl = np.sqrt(1 + gx * gx + gy * gy)
    shade = np.clip(0.5 + 0.5 * (-gx * 0.55 + gy * 0.55 + 0.6) / nl, 0, 1)
    hn = np.clip(H / 900.0, 0, 1)[..., None]
    col = np.array([0.50, 0.56, 0.36]) * (1 - hn) + np.array([0.84, 0.82, 0.78]) * hn
    fo = F.forest().reshape(n, n)[..., None]
    col = col * (1 - 0.6 * fo) + np.array([0.10, 0.25, 0.12]) * 0.6 * fo
    water = (H < W.LAKE_Z)[..., None]
    col = np.where(water, np.array([0.16, 0.30, 0.42]) * (0.7 + 0.3 * np.clip(1 + H[..., None] / 40, 0, 1)), col)
    img = Image.fromarray((np.clip(col * (0.3 + 1.0 * shade[..., None]), 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    def px(x, y):
        return ((x - (center[0] - half)) / (2 * half) * n, ((center[1] + half) - y) / (2 * half) * n)

    # rivers
    for path, wcol in ((W.GORGE_PATH, (60, 120, 170, 255)), (W.RAVINE_PATH, (60, 120, 170, 255)), (W.ROAD_PATH, (150, 120, 80, 255))):
        pts = [px(*p) for p in path[::3]]
        d.line(pts, fill=wcol, width=4 if wcol[0] < 100 else 3)
    # the castle's two rocks (the plateau rims), the stairs, the bridges
    from skimage import measure

    cd = F.crag().reshape(n, n)
    for C in measure.find_contours(cd, 0.0):
        if len(C) < 20:
            continue
        pts = [(c[1], c[0]) for c in C[::2]]
        d.line(pts + [pts[0]], fill=(250, 235, 200, 255), width=3)
    for xa, xb, yc, _, _ in W.STAIR_FLIGHTS:
        d.line([px(xa, yc), px(xb, yc)], fill=(250, 235, 200, 255), width=3)
    d.line([px(*q) for q in W.VIADUCT], fill=(250, 235, 200, 255), width=4)
    # the Quidditch stadium's oval
    ya = math.radians(W.PITCH_YAW)
    ex, ey = np.array([math.cos(ya), -math.sin(ya)]), np.array([math.sin(ya), math.cos(ya)])
    t = np.linspace(0, 2 * math.pi, 73)
    ov = [px(*(np.array(W.PITCH) + ex * W.PITCH_SIZE[0] / 2 * math.cos(a) + ey * W.PITCH_SIZE[1] / 2 * math.sin(a))) for a in t]
    d.line(ov, fill=(250, 235, 200, 255), width=3)
    f1, f2, f3 = font(30), font(22), font(18)
    # cameras: position, view cone, number
    resolved = SH.resolve(lambda x, y: float(W.ground(x, y)[0]))
    for i, (name, s) in enumerate(resolved.items()):
        if name == "CAM_12_Plan_Top":
            continue
        k = s["keys"][0]
        p = np.array(k["loc"][:2])
        t = np.array(k["look_at"][:2])
        dv = t - p
        L = np.linalg.norm(dv)
        if L < 1:
            continue
        dv /= L
        sensor = 36.0
        res = s["res"]
        hfov = 2 * math.atan((sensor / 2 if res[0] >= res[1] else sensor * res[0] / res[1] / 2) / s["lens"])
        rng_ = min(L * 0.8, 300.0)
        a0 = math.atan2(dv[1], dv[0])
        cone = [px(*p)] + [px(*(p + rng_ * np.array([math.cos(a0 + da), math.sin(a0 + da)]))) for da in np.linspace(-hfov / 2, hfov / 2, 12)]
        colr = {"mist": (190, 220, 235), "night": (120, 140, 255), "sunset": (255, 160, 90), "day": (255, 240, 140), "dusk": (220, 150, 220)}[s["preset"]]
        d.polygon(cone, fill=(*colr, 38), outline=(*colr, 200))
        if len(s["keys"]) > 2:
            d.line([px(*kk["loc"][:2]) for kk in s["keys"]], fill=(*colr, 255), width=3)
        X0, Y0 = px(*p)
        d.ellipse([X0 - 9, Y0 - 9, X0 + 9, Y0 + 9], fill=(20, 20, 20, 255), outline=(*colr, 255), width=3)
        num = name.split("_")[1]
        d.text((X0, Y0), num, fill=(255, 255, 255, 255), font=font(13), anchor="mm")
    labels = [((-30, -40), "CASTLE"), ((-330, 330), "river gorge"), ((175, 520), "the stream"),
              ((360, -60), "east arm"), ((190, 128), "the bay"), ((-30, 345), "covered bridge"), ((-120, 140), "grand viaduct"),
              ((W.PITCH[0] - 150, W.PITCH[1] + 40), "Quidditch stadium"), ((W.FLYING_LAWN[0], W.FLYING_LAWN[1] + 48), "flying lawn"),
              (W.HUT, "hut"), (W.WILLOW, "willow"), ((W.GREENHOUSES[0] - 60, W.GREENHOUSES[1] + 20), "greenhouses"),
              ((W.STONE_CIRCLE[0] + 60, W.STONE_CIRCLE[1] + 10), "stone circle"), (W.GATES, "gates"), (W.STATION, "station"),
              ((W.BOATHOUSE[0] + 80, W.BOATHOUSE[1] + 10), "boathouse"), ((W.FALL_X + 140, 222), "stairs + waterfall"),
              ((380, -880), "THE BLACK LAKE"), (W.ISLANDS[0][:2], "tomb island"), ((900, 700), "the forest"), ((-130, 1300), "to Hogsmeade")]
    for (x, y), t in labels:
        X0, Y0 = px(x, y)
        fnt = f1 if t.isupper() else f2
        d.text((X0 + 1, Y0 + 1), t, fill=(0, 0, 0, 200), font=fnt, anchor="mm")
        d.text((X0, Y0), t, fill=(255, 250, 235, 255), font=fnt, anchor="mm")
    # legend
    lines = [f"{name.split('_')[1]}  {name[7:].replace('_', ' ')}  ({s['preset']}, {s['lens']:.0f} mm, {s['aspect']})" for name, s in resolved.items()]
    lx, ly = 20, n - 30 - 24 * len(lines)
    d.rectangle([10, ly - 40, 640, n - 15], fill=(0, 0, 0, 150))
    d.text((lx, ly - 30), "SHOTS (camera, view cone coloured by lighting preset)", fill=(255, 255, 255, 255), font=f3)
    for i, ln in enumerate(lines):
        d.text((lx, ly + 24 * i), ln, fill=(235, 235, 235, 255), font=f3)
    # scale bar + north arrow
    bx, by = n - 330, n - 50
    d.line([(bx, by), (bx + 500 / (2 * half) * n, by)], fill=(255, 255, 255, 255), width=5)
    d.text((bx, by - 22), "500 m", fill=(255, 255, 255, 255), font=f3)
    d.polygon([(n - 60, 40), (n - 75, 85), (n - 45, 85)], fill=(255, 255, 255, 255))
    d.text((n - 60, 100), "N", fill=(255, 255, 255, 255), font=f2, anchor="mm")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    img.convert("RGB").save(out, quality=92)
    print("layout map ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../docs/layout_map.jpg")
