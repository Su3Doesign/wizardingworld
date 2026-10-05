"""Minimal binary-FBX reader used to verify what the Blender exporter really wrote:
global axis / unit settings, vertex extents, normals, UV layers and material names.
"""
from __future__ import annotations

import struct
import sys
import zlib

import numpy as np


class Node:
    def __init__(self, name):
        self.name = name
        self.props = []
        self.children = []

    def find(self, name):
        return [c for c in self.children if c.name == name]

    def first(self, name):
        r = self.find(name)
        return r[0] if r else None

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def _read_prop(f, data, pos):
    t = chr(data[pos])
    pos += 1
    if t == "Y":
        return struct.unpack_from("<h", data, pos)[0], pos + 2
    if t == "C":
        return bool(data[pos]), pos + 1
    if t == "I":
        return struct.unpack_from("<i", data, pos)[0], pos + 4
    if t == "F":
        return struct.unpack_from("<f", data, pos)[0], pos + 4
    if t == "D":
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if t == "L":
        return struct.unpack_from("<q", data, pos)[0], pos + 8
    if t in "fdlib":
        n, enc, clen = struct.unpack_from("<III", data, pos)
        pos += 12
        raw = data[pos:pos + clen]
        pos += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        dt = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[t]
        return np.frombuffer(raw, dtype=dt, count=n), pos
    if t == "S":
        n = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return data[pos:pos + n].decode("utf-8", "replace"), pos + n
    if t == "R":
        n = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return data[pos:pos + n], pos + n
    raise ValueError(f"unknown FBX property type {t!r}")


def parse(path):
    data = open(path, "rb").read()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError("not a binary FBX")
    version = struct.unpack_from("<I", data, 23)[0]
    pos = 27
    wide = version >= 7500
    root = Node("root")

    def read_node(pos):
        if wide:
            end, nprops, plen, nlen = struct.unpack_from("<QQQB", data, pos)
            pos += 25
        else:
            end, nprops, plen, nlen = struct.unpack_from("<IIIB", data, pos)
            pos += 13
        if end == 0:
            return None, pos
        name = data[pos:pos + nlen].decode()
        pos += nlen
        node = Node(name)
        for _ in range(nprops):
            v, pos = _read_prop(None, data, pos)
            node.props.append(v)
        while pos < end:
            ch, pos = read_node(pos)
            if ch is None:
                break
            node.children.append(ch)
        return node, end

    while pos < len(data) - 200:
        n, pos2 = read_node(pos)
        if n is None:
            break
        root.children.append(n)
        pos = pos2
    root.version = version
    return root


def summarize(path):
    root = parse(path)
    out = {"version": root.version}
    gs = root.first("GlobalSettings")
    if gs:
        p70 = gs.first("Properties70")
        d = {}
        for p in p70.children:
            if p.name == "P" and p.props:
                d[p.props[0]] = p.props[4] if len(p.props) > 4 else None
        out["global"] = {k: d.get(k) for k in ("UpAxis", "UpAxisSign", "FrontAxis", "FrontAxisSign", "CoordAxis", "CoordAxisSign", "UnitScaleFactor", "OriginalUnitScaleFactor")}
    objs = root.first("Objects")
    geoms = []
    mats = []
    for g in objs.find("Geometry"):
        v = g.first("Vertices").props[0].reshape(-1, 3)
        idx = g.first("PolygonVertexIndex").props[0]
        info = {"verts": len(v), "polys": int((idx < 0).sum()), "min": v.min(0).round(3).tolist(), "max": v.max(0).round(3).tolist()}
        nl = g.find("LayerElementNormal")
        info["normals"] = [(l.first("MappingInformationType").props[0], len(l.first("Normals").props[0]) // 3) for l in nl]
        uvs = g.find("LayerElementUV")
        info["uv_layers"] = [(u.first("Name").props[0] if u.first("Name") else "", len(u.first("UV").props[0]) // 2) for u in uvs]
        info["colors"] = len(g.find("LayerElementColor"))
        mm = g.first("LayerElementMaterial")
        if mm:
            info["material_mapping"] = mm.first("MappingInformationType").props[0]
            ids = mm.first("Materials").props[0]
            info["material_ids_used"] = sorted(set(int(i) for i in ids))
        geoms.append(info)
    for m in objs.find("Material"):
        mats.append(m.props[1].split("\x00")[0])
    out["geometry"] = geoms
    out["materials"] = mats
    models = []
    for m in objs.find("Model"):
        d = {"name": m.props[1].split("\x00")[0]}
        p70 = m.first("Properties70")
        if p70:
            for p in p70.children:
                if p.name == "P" and p.props and p.props[0] in ("Lcl Translation", "Lcl Rotation", "Lcl Scaling", "PreRotation", "RotationActive"):
                    d[p.props[0]] = [round(float(x), 4) for x in p.props[4:7]] if len(p.props) >= 7 else p.props[4:]
        models.append(d)
    out["models"] = models
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(summarize(sys.argv[1]), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
