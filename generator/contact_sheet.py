"""contact_sheet - all the shot previews on one page (justified rows, labelled with shot, preset and lens):
    python contact_sheet.py ../Previews ../Previews/contact_sheet.jpg"""
from __future__ import annotations

import glob
import os
import sys

from PIL import Image, ImageDraw

import shots as SH
from layout_map import font


def main(src, out, width=2400, row_h=430, gap=12, pad=24):
    paths = sorted(p for p in glob.glob(os.path.join(src, "CAM_*.png")))
    if not paths:
        raise SystemExit(f"no CAM_*.png in {src}")
    ims = [(os.path.splitext(os.path.basename(p))[0], Image.open(p).convert("RGB")) for p in paths]
    inner = width - 2 * pad
    rows, cur = [], []
    for item in ims:
        cur.append(item)
        aspect = sum(im.width / im.height for _, im in cur)
        if aspect * row_h + gap * (len(cur) - 1) >= inner:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    # scale each row to the full width; the last row keeps the target height if it is short
    layout, y = [], pad + 70
    for r in rows:
        aspect = sum(im.width / im.height for _, im in r)
        h = (inner - gap * (len(r) - 1)) / aspect
        if r is rows[-1] and h > row_h:
            h = row_h
        x = pad
        for name, im in r:
            w = im.width / im.height * h
            layout.append((name, im, int(round(x)), int(round(y)), int(round(w)), int(round(h))))
            x += w + gap
        y += h + gap
    sheet = Image.new("RGB", (width, int(round(y - gap + pad))), (16, 17, 19))
    d = ImageDraw.Draw(sheet, "RGBA")
    d.text((pad, pad + 6), f"The Castle on the Crag: Cycles previews of the {len(ims)} shots (not Unreal renders)", fill=(235, 235, 235),
           font=font(30))
    f = font(19)
    for name, im, x, y0, w, h in layout:
        sheet.paste(im.resize((w, h), Image.LANCZOS), (x, y0))
        s = SH.SHOTS.get(name, {})
        label = f"{name.split('_')[1]}  {name[7:].replace('_', ' ')}"
        if s:
            label += f"  ·  {s['preset']}, {s['lens']:.0f} mm"
        tw = d.textlength(label, font=f)
        d.rectangle([x, y0 + h - 34, x + min(w, tw + 20), y0 + h], fill=(0, 0, 0, 150))
        d.text((x + 10, y0 + h - 17), label, fill=(245, 245, 240), font=f, anchor="lm")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sheet.save(out, quality=88)
    print(f"contact sheet ({len(layout)} shots, {sheet.width}x{sheet.height}) ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../Previews", sys.argv[2] if len(sys.argv) > 2 else "../Previews/contact_sheet.jpg")
