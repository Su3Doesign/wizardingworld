"""preview - Cycles renders of the whole world from every camera of shots.py (NOT Unreal renders: path-traced previews of
the same meshes, instances, textures, cameras and lighting presets; Lumen / Unreal's fog and clouds will differ).

    python preview.py GEO_DIR TEX_DIR OUT_DIR [--shots CAM_01_Gorge_Mist,CAM_02_Boats_Night] [--scale 0.5] [--samples 48]

Materials mirror the Unreal builder: triplanar granite with height-blended moss on the terrain (masks from the UV
channels), castle stone with grime / moss, course-aligned slate, lit leaded windows at night, two-sided foliage with
translucency, reflective water.  Atmosphere: a compositor fog driven by the depth and the height (position pass).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

import bpy
import mathutils

import blender_util as bu
import instances as ins
import shots as SH
import world as W
from export_world import INSTANCE_SETS, MAT_NAMES
from meshkit import Mesh

TEX = None


# ----------------------------------------------------------------------------------------------- node helpers
class NB:
    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.n = self.nt.nodes
        self.l = self.nt.links
        self.out = self.n.new("ShaderNodeOutputMaterial")
        self.bsdf = self.n.new("ShaderNodeBsdfPrincipled")
        self.l.new(self.bsdf.outputs[0], self.out.inputs[0])

    def link(self, a, b):
        self.l.new(a, b)

    def img(self, name, cs="sRGB", vec=None, proj="FLAT", blend=0.25, interp="Linear"):
        path = f"{TEX}/{name}"
        im = bpy.data.images.load(path, check_existing=True)
        im.colorspace_settings.name = cs
        t = self.n.new("ShaderNodeTexImage")
        t.image = im
        t.projection = proj
        t.projection_blend = blend
        t.interpolation = interp
        if vec is not None:
            self.l.new(vec, t.inputs[0])
        return t

    def coord(self, kind="Object", scale=1.0, uv=None):
        if uv:
            u = self.n.new("ShaderNodeUVMap")
            u.uv_map = uv
            src = u.outputs[0]
        else:
            tc = self.n.new("ShaderNodeTexCoord")
            src = tc.outputs[kind]
        if scale != 1.0:
            mp = self.n.new("ShaderNodeVectorMath")
            mp.operation = "SCALE"
            self.l.new(src, mp.inputs[0])
            mp.inputs["Scale"].default_value = scale
            return mp.outputs[0]
        return src

    def uvsep(self, uv):
        u = self.n.new("ShaderNodeUVMap")
        u.uv_map = uv
        s = self.n.new("ShaderNodeSeparateXYZ")
        self.l.new(u.outputs[0], s.inputs[0])
        return s.outputs[0], s.outputs[1]

    def sep(self, col):
        s = self.n.new("ShaderNodeSeparateColor")
        self.l.new(col, s.inputs[0])
        return s.outputs[0], s.outputs[1], s.outputs[2]

    def math(self, op, a, b=None, clamp=False):
        m = self.n.new("ShaderNodeMath")
        m.operation = op
        m.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.l.new(v, m.inputs[i])
        return m.outputs[0]

    def mix(self, fac, a, b, blend="MIX", kind="RGBA"):
        m = self.n.new("ShaderNodeMix")
        m.data_type = kind
        m.blend_type = blend
        ia, ib, io = (6, 7, 2) if kind == "RGBA" else (4, 5, 1)
        if isinstance(fac, (int, float)):
            m.inputs[0].default_value = fac
        else:
            self.l.new(fac, m.inputs[0])
        for idx, v in ((ia, a), (ib, b)):
            if isinstance(v, tuple):
                m.inputs[idx].default_value = v if kind == "VECTOR" else (*v[:3], 1.0)
            else:
                self.l.new(v, m.inputs[idx])
        return m.outputs[io]

    def smooth(self, x, lo, hi):
        mr = self.n.new("ShaderNodeMapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        mr.clamp = True
        mr.inputs["From Min"].default_value = lo
        mr.inputs["From Max"].default_value = hi
        self.l.new(x, mr.inputs[0])
        return mr.outputs[0]

    def normal(self, col, strength=1.0):
        nm = self.n.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = strength
        self.l.new(col, nm.inputs["Color"])
        return nm.outputs[0]

    def geo_up(self):
        g = self.n.new("ShaderNodeNewGeometry")
        s = self.n.new("ShaderNodeSeparateXYZ")
        self.l.new(g.outputs["Normal"], s.inputs[0])
        return s.outputs[2]

    def set(self, **kw):
        for k, v in kw.items():
            sock = self.bsdf.inputs[k]
            if isinstance(v, (int, float, tuple)):
                sock.default_value = v if not isinstance(v, tuple) or len(v) == len(sock.default_value) else (*v, 1.0)
            else:
                self.l.new(v, sock)


def tset(nb, name, vec, proj="FLAT", blend=0.25):
    """BC / N / ORH triple of a texture set."""
    bc = nb.img(f"T_{name}_BC.jpg", "sRGB", vec, proj, blend)
    nr = nb.img(f"T_{name}_N.png", "Non-Color", vec, proj, blend)
    orh = nb.img(f"T_{name}_ORH.png", "Non-Color", vec, proj, blend)
    return bc.outputs[0], nr.outputs[0], orh.outputs[0]


# ----------------------------------------------------------------------------------------------- materials
def m_terrain(name="M_Terrain"):
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.coord("Object")
    gro = b.coord("Object", 1 / 2.0)
    g_bc, g_n, g_orh = tset(b, "Grass", gro)
    f_bc, f_n, f_orh = tset(b, "ForestFloor", gro)
    s_bc, s_n, s_orh = tset(b, "Shore", gro)
    d_bc, d_n, d_orh = tset(b, "Dirt", gro)
    rk = b.coord("Object", 1 / 4.0)
    r_bc, r_n, r_orh = tset(b, "Granite", rk, "BOX", 0.3)
    ms = b.coord("Object", 1 / 1.2)
    m_bc, m_n, m_orh = tset(b, "Moss", ms, "BOX", 0.3)
    mac = b.img("T_Macro.png", "Non-Color", b.coord("Object", 1 / 40.0)).outputs[0]
    mac_r, mac_g, mac_b = b.sep(mac)
    rock, moss = b.uvsep("UV1")
    forest, shore = b.uvsep("UV2")
    road, ao = b.uvsep("UV3")
    up = b.geo_up()
    # ground: grass -> forest floor -> shore -> road
    c = b.mix(b.smooth(forest, 0.25, 0.75), g_bc, f_bc)
    c = b.mix(b.smooth(shore, 0.2, 0.8), c, s_bc)
    c = b.mix(b.smooth(road, 0.3, 0.8), c, d_bc)
    nrm = b.mix(b.smooth(forest, 0.25, 0.75), g_n, f_n, kind="RGBA")
    # rock where the mask says so and on anything steep
    steep = b.smooth(up, 0.78, 0.55)
    rk_f = b.math("MAXIMUM", b.smooth(rock, 0.35, 0.7), steep)
    rock_c = b.mix(1.0, r_bc, (0.86, 0.90, 0.95), blend="MULTIPLY")                     # cool grey granite
    c = b.mix(rk_f, c, b.mix(b.math("MULTIPLY", mac_r, 0.35), rock_c, (0.30, 0.29, 0.26), blend="MULTIPLY"))
    nrm = b.mix(rk_f, nrm, r_n)
    # moss on the rock: height blend (mask + up-facing + moss canopy height - rock height)
    _, _, r_h = b.sep(r_orh)
    _, _, m_h = b.sep(m_orh)
    mv = b.math("ADD", b.math("MULTIPLY", moss, 1.6), b.math("MULTIPLY", b.math("SUBTRACT", m_h, r_h), 0.6))
    mv = b.math("ADD", mv, b.math("MULTIPLY", b.math("SUBTRACT", mac_b, 0.5), 0.5))
    mf = b.math("MULTIPLY", b.smooth(mv, 0.40, 0.75), b.smooth(up, 0.05, 0.5))
    c = b.mix(mf, c, m_bc)
    nrm = b.mix(mf, nrm, m_n)
    # ambient occlusion
    c = b.mix(1.0, c, b.mix(b.math("POWER", ao, 2.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), blend="MULTIPLY")
    rough_r, _, _ = b.sep(r_orh)
    b.set(**{"Base Color": c, "Roughness": 0.88, "Normal": b.normal(nrm, 1.0)})
    return m


def m_boulder(name="M_Boulder"):
    m = bpy.data.materials.new(name)
    b = NB(m)
    rk = b.coord("Object", 1 / 2.5)
    r_bc, r_n, r_orh = tset(b, "Granite", rk, "BOX", 0.3)
    m_bc, m_n, _ = tset(b, "Moss", b.coord("Object", 1 / 0.8), "BOX", 0.3)
    up = b.geo_up()
    mf = b.smooth(up, 0.35, 0.8)
    b.set(**{"Base Color": b.mix(mf, r_bc, m_bc), "Roughness": 0.85, "Normal": b.normal(b.mix(mf, r_n, m_n), 1.0)})
    return m


def m_stone(name, tex, tile=2.0, moss_k=1.0):
    m = bpy.data.materials.new(name)
    b = NB(m)
    v = b.coord("Object", 1 / tile)
    bc, nr, orh = tset(b, tex, v, "BOX", 0.2)
    m_bc, m_n, _ = tset(b, "Moss", b.coord("Object", 1 / 1.0), "BOX", 0.2)
    grime, moss = b.uvsep("UV1")
    mac = b.img("T_Macro.png", "Non-Color", b.coord("Object", 1 / 30.0)).outputs[0]
    mr, mg, _ = b.sep(mac)
    c = b.mix(b.math("MULTIPLY", grime, 0.65), bc, b.mix(1.0, bc, (0.32, 0.31, 0.26), blend="MULTIPLY"))
    c = b.mix(b.math("MULTIPLY", mr, 0.3), c, b.mix(1.0, c, (0.80, 0.78, 0.74), blend="MULTIPLY"))
    mf = b.math("MULTIPLY", b.smooth(moss, 0.55, 0.95), moss_k)
    c = b.mix(mf, c, m_bc)
    b.set(**{"Base Color": c, "Roughness": 0.84, "Normal": b.normal(b.mix(mf, nr, m_n), 1.0)})
    return m


def m_slate(name="M_Slate"):
    m = bpy.data.materials.new(name)
    b = NB(m)
    v = b.coord(uv="UV0", scale=1 / 2.52)
    bc, nr, orh = tset(b, "Slate", v)
    grime, moss = b.uvsep("UV1")
    m_bc, _, _ = tset(b, "Moss", b.coord("Object", 1 / 1.0), "BOX", 0.2)
    rr, _, _ = b.sep(orh)
    c = b.mix(b.smooth(moss, 0.5, 0.9), bc, m_bc)
    b.set(**{"Base Color": c, "Roughness": b.math("MULTIPLY_ADD", rr, 0.5) if False else b.math("ADD", b.math("MULTIPLY", rr, 0.4), 0.55),
             "Normal": b.normal(nr, 1.0), "Specular IOR Level": 0.35})
    return m


def m_simple(name, tex=None, color=(0.5, 0.5, 0.5), rough=0.8, metal=0.0, uv_scale=1 / 2.0, obj=True):
    m = bpy.data.materials.new(name)
    b = NB(m)
    if tex:
        v = b.coord("Object", uv_scale) if obj else b.coord(uv="UV0", scale=uv_scale)
        bc, nr, orh = tset(b, tex, v, "BOX" if obj else "FLAT", 0.2)
        b.set(**{"Base Color": bc, "Normal": b.normal(nr, 1.0)})
    else:
        b.set(**{"Base Color": color})
    b.set(Roughness=rough, Metallic=metal)
    return m


def m_moss_clump(name="M_MossClump"):
    """Moss cushions, as the Unreal material: the moss set with a per-instance tint between a cool and a warm green."""
    m = bpy.data.materials.new(name)
    b = NB(m)
    bc, nr, orh = tset(b, "Moss", b.coord("Object", 1 / 0.25), "BOX", 0.2)
    oi = b.n.new("ShaderNodeObjectInfo")
    tint = b.mix(oi.outputs["Random"], (0.78, 0.88, 0.70), (1.15, 1.08, 0.74))
    b.set(**{"Base Color": b.mix(1.0, bc, tint, blend="MULTIPLY"), "Normal": b.normal(nr, 1.0), "Roughness": 0.9})
    return m


def m_glass(name="M_Glass", night=0.0):
    m = bpy.data.materials.new(name)
    b = NB(m)
    v = b.coord(uv="UV0", scale=1.0)
    bc = b.img("T_Glass_BC.jpg", "sRGB", v).outputs[0]
    orh = b.img("T_Glass_ORH.png", "Non-Color", v).outputs[0]
    _, _, came = b.sep(orh)
    rnd, lit = b.uvsep("UV2")
    glow = b.math("MULTIPLY", b.math("MULTIPLY", lit, b.math("SUBTRACT", 1.0, came)), 9.0 * night)
    glow.node.name = "GLOW"
    # per-window intensity and warmth from UV2.x, as in the Unreal material
    glow = b.math("MULTIPLY", glow, b.math("ADD", b.math("MULTIPLY", rnd, 0.6), 0.6))
    warm = b.mix(rnd, (1.0, 0.50, 0.18), (1.0, 0.68, 0.36))
    b.set(**{"Base Color": bc, "Roughness": 0.08, "Emission Color": warm, "Emission Strength": glow})
    return m


def m_glass_inst(name="M_GlassInst", night=0.0):
    """Glass of the instanced window modules, as the Unreal material: lit / brightness / warmth per instance."""
    m = bpy.data.materials.new(name)
    b = NB(m)
    v = b.coord(uv="UV0", scale=1.0)
    bc = b.img("T_Glass_BC.jpg", "sRGB", v).outputs[0]
    orh = b.img("T_Glass_ORH.png", "Non-Color", v).outputs[0]
    _, _, came = b.sep(orh)
    rnd = b.n.new("ShaderNodeObjectInfo").outputs["Random"]
    lit = b.smooth(rnd, 0.30, 0.33)
    rnd2 = b.math("FRACT", b.math("MULTIPLY", rnd, 7.31))
    glow = b.math("MULTIPLY", b.math("MULTIPLY", lit, b.math("SUBTRACT", 1.0, came)), 9.0 * night)
    glow.node.name = "GLOW"
    glow = b.math("MULTIPLY", glow, b.math("ADD", b.math("MULTIPLY", rnd2, 0.6), 0.6))
    warm = b.mix(rnd2, (1.0, 0.50, 0.18), (1.0, 0.68, 0.36))
    b.set(**{"Base Color": bc, "Roughness": 0.08, "Emission Color": warm, "Emission Strength": glow})
    return m


def m_cloth(name="M_Cloth"):
    m = bpy.data.materials.new(name)
    b = NB(m)
    pick, _ = b.uvsep("UV2")
    ramp = b.n.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.45, 0.04, 0.03, 1)
    els[1].position, els[1].color = 0.25, (0.03, 0.22, 0.08, 1)
    e = els.new(0.5)
    e.color = (0.04, 0.08, 0.35, 1)
    e = els.new(0.75)
    e.color = (0.65, 0.45, 0.04, 1)
    b.link(pick, ramp.inputs[0])
    b.set(**{"Base Color": ramp.outputs[0], "Roughness": 0.9})
    return m


def m_foliage(name, tex, tint_a, tint_b, translucency=0.35, obj_uv=False, nrm=None):
    m = bpy.data.materials.new(name)
    b = NB(m)
    v = b.coord(uv="UV0") if not obj_uv else b.coord("Object", 1.0)
    bc = b.img(tex, "sRGB", v).outputs[0]
    tint, sun = b.uvsep("UV1")
    rnd = b.n.new("ShaderNodeObjectInfo")
    r = b.math("ADD", b.math("MULTIPLY", tint, 0.6), b.math("MULTIPLY", rnd.outputs["Random"], 0.4))
    tcol = b.mix(r, tint_a, tint_b)
    c = b.mix(1.0, bc, tcol, blend="MULTIPLY")
    c = b.mix(1.0, c, b.mix(b.math("ADD", b.math("MULTIPLY", sun, 0.55), 0.45), (0, 0, 0), (1, 1, 1)), blend="MULTIPLY")
    b.set(**{"Base Color": c, "Roughness": 0.55})
    if nrm:
        b.set(Normal=b.normal(b.img(nrm, "Non-Color", v).outputs[0], 0.8))
    # translucency: mix in a translucent BSDF (light glowing through the needles / leaves)
    tr = b.n.new("ShaderNodeBsdfTranslucent")
    b.link(c, tr.inputs["Color"])
    mx = b.n.new("ShaderNodeMixShader")
    mx.inputs[0].default_value = translucency
    b.l.remove(b.out.inputs[0].links[0])
    b.link(b.bsdf.outputs[0], mx.inputs[1])
    b.link(tr.outputs[0], mx.inputs[2])
    b.link(mx.outputs[0], b.out.inputs[0])
    return m


def m_water(name, color=(0.006, 0.014, 0.016), rough=0.03, flow=False, foam=0.0):
    m = bpy.data.materials.new(name)
    b = NB(m)
    v1 = b.coord("Object", 1 / 9.0) if not flow else b.coord(uv="UV0", scale=1.0)
    v2 = b.coord("Object", 1 / 2.3) if not flow else b.coord(uv="UV0", scale=3.0)
    n1 = b.img("T_Water_N.png", "Non-Color", v1).outputs[0]
    n2 = b.img("T_Water_Ripple_N.png", "Non-Color", v2).outputs[0]
    n = b.mix(0.4, n1, n2)
    c = color
    if foam > 0:
        fm = b.img("T_Foam.png", "Non-Color", b.coord(uv="UV0", scale=0.7)).outputs[0]
        f, _, _ = b.sep(fm)
        c = b.mix(b.math("MULTIPLY", b.smooth(f, 0.55, 0.9), foam), color, (0.75, 0.78, 0.78))
    b.set(**{"Base Color": c, "Roughness": rough, "Normal": b.normal(n, 0.35 if not flow else 0.6), "Specular IOR Level": 0.5})
    return m


def m_fall(name="M_Fall"):
    m = bpy.data.materials.new(name)
    b = NB(m)
    v = b.coord(uv="UV0", scale=1.0)
    bc = b.img("T_Fall_BC.jpg", "sRGB", v).outputs[0]
    a = b.img("T_Fall_A.png", "Non-Color", v).outputs[0]
    ar, _, _ = b.sep(a)
    b.set(**{"Base Color": bc, "Roughness": 0.35, "Alpha": ar, "Emission Color": (0.6, 0.65, 0.65), "Emission Strength": 0.05})
    return m


def m_mist(name="M_Mist"):
    """The spray dome at the foot of the waterfall, as the Unreal material: thin (0.22 x foam), the silhouette faded out
    (x (1 - fresnel)), so it reads as a cloud of spray rather than a shell."""
    m = bpy.data.materials.new(name)
    b = NB(m)
    foam, _, _ = b.sep(b.img("T_Foam.png", "Non-Color", b.coord("Object", 1 / 18.0)).outputs[0])
    lw = b.n.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5                       # Facing = 1 - |N.V|
    edge = b.math("SUBTRACT", 1.0, b.math("POWER", lw.outputs["Facing"], 1.5))
    b.set(**{"Base Color": (0.85, 0.88, 0.9), "Roughness": 1.0, "Alpha": b.math("MULTIPLY", b.math("MULTIPLY", foam, 0.22), edge)})
    return m


def m_emit(name, color, strength):
    m = bpy.data.materials.new(name)
    b = NB(m)
    b.set(**{"Base Color": color, "Emission Color": color, "Emission Strength": strength})
    return m


def build_materials(night=0.0):
    M = {}
    M["M_Terrain"] = m_terrain()
    M["M_Boulder"] = m_boulder()
    M["M_CastleStone"] = m_stone("M_CastleStone", "CastleStone", 2.0, 1.0)
    M["M_Trim"] = m_stone("M_Trim", "Trim", 2.0, 0.6)
    M["M_Slate"] = m_slate()
    M["M_Lead"] = m_simple("M_Lead", "Lead", rough=0.45, metal=0.4)
    M["M_Glass"] = m_glass(night=night)
    M["M_GlassInst"] = m_glass_inst(night=night)
    M["M_ClockFace"] = m_emit("M_ClockFace", (1.0, 0.93, 0.78), 14.0 * night + 0.3)
    M["M_Wood"] = m_simple("M_Wood", "Wood", rough=0.8)
    M["M_Cloth"] = m_cloth()
    M["M_Marble"] = m_simple("M_Marble", None, (0.75, 0.74, 0.70), 0.35)
    M["M_Water"] = m_water("M_Water")
    M["M_River"] = m_water("M_River", (0.02, 0.035, 0.03), 0.08, flow=True, foam=0.7)
    M["M_Fall"] = m_fall()
    M["M_Mist"] = m_mist()
    M["M_Lantern"] = m_emit("M_Lantern", (1.0, 0.6, 0.25), 60.0 * night + 0.5)
    M["M_Bark"] = m_simple("M_Bark", "Bark", rough=0.9, obj=False, uv_scale=1 / 1.5)
    M["M_BirchBark"] = m_simple("M_BirchBark", "BirchBark", rough=0.7, obj=False, uv_scale=1 / 1.5)
    M["M_DeadWood"] = m_simple("M_DeadWood", None, (0.30, 0.28, 0.25), 0.85)
    M["M_PineBark"] = m_simple("M_PineBark", None, (0.42, 0.20, 0.10), 0.8)
    M["M_Needles"] = m_foliage("M_Needles", "T_Needles_BC.jpg", (0.75, 0.85, 0.72), (1.15, 1.05, 0.85), 0.3, nrm="T_Needles_N.png")
    M["M_Leaf"] = m_foliage("M_Leaf", "T_Leaf_BC.jpg", (0.8, 0.95, 0.7), (1.2, 1.1, 0.75), 0.4)
    M["M_MossClump"] = m_moss_clump()
    M["M_Fern3D"] = m_foliage("M_Fern3D", "T_FernPinna_BC.jpg", (0.8, 0.95, 0.75), (1.1, 1.05, 0.85), 0.35)
    M["M_Grass3D"] = m_foliage("M_Grass3D", "T_GrassBlade_BC.jpg", (0.85, 0.95, 0.75), (1.1, 1.05, 0.8), 0.3)
    return M


# ----------------------------------------------------------------------------------------------- scene
def add_mesh(path, M, name=None):
    m = Mesh.load(path)
    ids = sorted(set(int(i) for i in m.mat))
    remap = {old: new for new, old in enumerate(ids)}
    m.mat = np.array([remap[int(i)] for i in m.mat], np.int16)
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("lib_boulder"):
        ids = [1]
    mats = [M.get(MAT_NAMES.get(i, "M_Terrain"), M["M_Terrain"]) for i in ids]
    smooth = None if base.startswith(("terrain_", "water_", "lib_spruce", "lib_birch", "lib_pine", "lib_willow", "lib_snag",
                                      "lib_fern", "lib_grass", "lib_moss")) else 35.0
    ob = bu.add_object(name or base, m, mats, normals_angle=smooth)
    return ob


def add_instances(set_path, lib_obs, scale=1.0, rng=None):
    d = ins.load_set(set_path)
    sname = os.path.splitext(os.path.basename(set_path))[0]
    made = []
    for v in sorted(set(int(x) for x in d["variant"])):
        lib = INSTANCE_SETS[sname](v)
        if lib not in lib_obs:
            continue
        sel = d["variant"] == v
        if scale < 1.0:
            sel &= (np.arange(len(sel)) * 0.6180339887) % 1.0 < scale
        pos = d["pos"][sel].astype(np.float64)
        if not len(pos):
            continue
        me = bpy.data.meshes.new(f"pts_{sname}_{v}")
        me.vertices.add(len(pos))
        me.vertices.foreach_set("co", pos.astype(np.float32).ravel())
        a = me.attributes.new("rot", "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", ins.euler_xyz(d["R"][sel]).astype(np.float32).ravel())
        s = me.attributes.new("scl", "FLOAT_VECTOR", "POINT")
        s.data.foreach_set("vector", d["scale"][sel].astype(np.float32).ravel())
        ob = bpy.data.objects.new(f"inst_{sname}_{v}", me)
        bpy.context.scene.collection.objects.link(ob)
        mod = ob.modifiers.new("gn", "NODES")
        mod.node_group = _instancer(lib_obs[lib])
        made.append(ob)
    return made


def _instancer(lib_ob):
    name = f"inst_{lib_ob.name}"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    iop = ng.nodes.new("GeometryNodeInstanceOnPoints")
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs[0].default_value = lib_ob
    oi.transform_space = "ORIGINAL"
    ra = ng.nodes.new("GeometryNodeInputNamedAttribute")
    ra.data_type = "FLOAT_VECTOR"
    ra.inputs[0].default_value = "rot"
    sa = ng.nodes.new("GeometryNodeInputNamedAttribute")
    sa.data_type = "FLOAT_VECTOR"
    sa.inputs[0].default_value = "scl"
    ng.links.new(gi.outputs[0], iop.inputs["Points"])
    ng.links.new(oi.outputs["Geometry"], iop.inputs["Instance"])
    ng.links.new(ra.outputs[0], iop.inputs["Rotation"])
    ng.links.new(sa.outputs[0], iop.inputs["Scale"])
    ng.links.new(iop.outputs[0], go.inputs[0])
    return ng


def setup_world(preset, stars_path):
    sc = bpy.context.scene
    w = bpy.data.worlds.get("W") or bpy.data.worlds.new("W")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(bg.outputs[0], out.inputs[0])
    P = SH.PRESETS[preset]
    if P.get("stars"):
        night_sky(nt, bg, P, stars_path)
    else:
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(max(P["sun_elev"], -2.0))
        sky.sun_rotation = math.radians(90.0 - P["sun_az"])
        sky.altitude = 300
        sky.air_density = 1.0
        sky.dust_density = 3.0 if preset in ("mist", "sunset", "dusk") else 1.2
        sky.sun_disc = True
        nt.links.new(sky.outputs[0], bg.inputs[0])
        bg.inputs["Strength"].default_value = 0.22 * P["sky"]
    sc.world = w
    # Daylight: the sky texture's own sun disc is the sun.  Its sun : sky ratio and its colour at a low elevation are
    # physical, as with Unreal's sky atmosphere, and the exposure is metered per shot (auto_exposure).  A lamp on top
    # would double the sunlight.  Night: a moon lamp under the star dome.
    for ob in [o for o in bpy.data.objects if o.name.startswith("SunLamp")]:
        bpy.data.objects.remove(ob)
    if P.get("moon"):
        el, az = math.radians(P["sun_elev"]), math.radians(P["sun_az"])
        d = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
        bu.add_sun(d, strength=1.1, angle_deg=0.55, color=tuple(c / 255.0 for c in P["sun_color"]), name="SunLamp")


def night_sky(nt, bg, P, stars_path):
    """The night sky of the night castle reference: a deep blue gradient brightening to the horizon, soft clouds lit by
    the moon (brighter towards it), a halo round the moon, stars between the clouds."""
    L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord")
    dirn = tc.outputs["Generated"]
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(dirn, sep.inputs[0])

    def maprange(x, a, b):
        m = nt.nodes.new("ShaderNodeMapRange")
        m.interpolation_type = "SMOOTHSTEP"
        m.clamp = True
        m.inputs["From Min"].default_value = a
        m.inputs["From Max"].default_value = b
        L(x, m.inputs[0])
        return m.outputs[0]

    def mixc(fac, a, b):
        m = nt.nodes.new("ShaderNodeMix")
        m.data_type = "RGBA"
        for sock, v in ((m.inputs[6], a), (m.inputs[7], b)):
            if isinstance(v, tuple):
                sock.default_value = (*v, 1.0)
            else:
                L(v, sock)
        if isinstance(fac, float):
            m.inputs[0].default_value = fac
        else:
            L(fac, m.inputs[0])
        return m.outputs[2]

    def math_(op, a, b=None):
        m = nt.nodes.new("ShaderNodeMath")
        m.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L(v, m.inputs[i])
        return m.outputs[0]
    up = maprange(sep.outputs[2], -0.02, 0.6)
    grad = mixc(up, (0.105, 0.16, 0.30), (0.012, 0.026, 0.07))
    # the moon's direction (from the moon lamp's elevation / azimuth)
    el, az = math.radians(P["sun_elev"]), math.radians(P["sun_az"])
    md = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    L(dirn, dot.inputs[0])
    dot.inputs[1].default_value = md
    near = math_("POWER", math_("MAXIMUM", dot.outputs["Value"], 0.0), 5.0)
    halo = math_("POWER", math_("MAXIMUM", dot.outputs["Value"], 0.0), 160.0)
    # clouds: fbm on the direction, flattened towards the horizon, thinning out low down
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.4, 2.4, 7.0)
    L(dirn, mp.inputs[0])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 1.7
    nz.inputs["Detail"].default_value = 9.0
    nz.inputs["Roughness"].default_value = 0.62
    L(mp.outputs[0], nz.inputs["Vector"])
    cov = maprange(nz.outputs["Fac"], 0.47, 0.70)
    cloud = math_("MULTIPLY", cov, maprange(sep.outputs[2], 0.0, 0.12))
    cloud = math_("MULTIPLY", cloud, 0.9)
    ccol = mixc(near, (0.085, 0.11, 0.17), (0.55, 0.62, 0.76))
    sky = mixc(cloud, grad, ccol)
    # stars between the clouds, above the horizon
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(stars_path, check_existing=True)
    st_k = math_("MULTIPLY", math_("SUBTRACT", 1.0, cloud), maprange(sep.outputs[2], 0.02, 0.25))
    stars = nt.nodes.new("ShaderNodeMix")
    stars.data_type = "RGBA"
    stars.blend_type = "ADD"
    L(st_k, stars.inputs[0])
    L(sky, stars.inputs[6])
    L(env.outputs[0], stars.inputs[7])
    halo_add = nt.nodes.new("ShaderNodeMix")
    halo_add.data_type = "RGBA"
    halo_add.blend_type = "ADD"
    L(math_("MULTIPLY", halo, 0.6), halo_add.inputs[0])
    L(stars.outputs[2], halo_add.inputs[6])
    halo_add.inputs[7].default_value = (0.75, 0.82, 1.0, 1.0)
    L(halo_add.outputs[2], bg.inputs[0])
    bg.inputs["Strength"].default_value = 1.0


# exposure bias over the metered middle grey, per preset (the mist is airy, the sunset a silhouette, dusk is dim)
EV_BIAS = {"mist": 0.4, "day": 0.0, "sunset": -0.6, "dusk": -0.8, "night": -1.1}


def auto_exposure(preset, out_dir, long=192):
    """Metered exposure (as Unreal's auto exposure): a small, quick render of the composited frame; the log-average
    luminance of its 5-95 % luminance range is brought to middle grey (0.18), plus the preset's bias."""
    import tempfile

    sc = bpy.context.scene
    r = sc.render
    keep = (r.resolution_x, r.resolution_y, sc.cycles.samples, sc.cycles.use_denoising, r.filepath, r.image_settings.file_format,
            r.image_settings.color_depth, sc.view_settings.exposure)
    s = long / max(r.resolution_x, r.resolution_y)
    r.resolution_x, r.resolution_y = max(16, int(r.resolution_x * s)), max(16, int(r.resolution_y * s))
    sc.cycles.samples, sc.cycles.use_denoising = 6, False
    sc.view_settings.exposure = 0.0
    r.image_settings.file_format, r.image_settings.color_depth = "OPEN_EXR", "32"
    path = os.path.join(tempfile.mkdtemp(dir=out_dir), "meter.exr")
    r.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:], np.float32).reshape(-1, 4)[:, :3]
    bpy.data.images.remove(img)
    os.remove(path)
    os.rmdir(os.path.dirname(path))
    (r.resolution_x, r.resolution_y, sc.cycles.samples, sc.cycles.use_denoising, r.filepath, r.image_settings.file_format,
     r.image_settings.color_depth, _) = keep
    lum = np.maximum(px @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-5)
    lo, hi = np.percentile(lum, [5, 95])
    sel = (lum >= lo) & (lum <= hi)
    avg = float(np.exp(np.log(lum[sel]).mean()))
    ev = float(np.clip(math.log2(0.18 / avg) + EV_BIAS.get(preset, 0.0), -8.0, 8.0))
    sc.view_settings.exposure = ev
    return ev


def setup_compositor(preset, cam_z):
    """Fog: depth-based aerial perspective + height-limited ground mist, tinted per preset."""
    sc = bpy.context.scene
    sc.use_nodes = True
    vl = sc.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_position = True
    nt = sc.node_tree
    nt.nodes.clear()
    P = SH.PRESETS[preset]
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    # fog amount = (1 - exp(-depth * k)) * height falloff
    k = P["fog_density"] * 0.012
    mul = nt.nodes.new("CompositorNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = -k
    nt.links.new(rl.outputs["Depth"], mul.inputs[0])
    ex = nt.nodes.new("CompositorNodeMath")
    ex.operation = "EXPONENT"
    nt.links.new(mul.outputs[0], ex.inputs[0])
    inv = nt.nodes.new("CompositorNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(ex.outputs[0], inv.inputs[1])
    # height factor from the position pass (z): more fog low down (valley mist)
    sp = nt.nodes.new("CompositorNodeSeparateXYZ")
    nt.links.new(rl.outputs["Position"], sp.inputs[0])
    hz = nt.nodes.new("CompositorNodeMath")
    hz.operation = "SUBTRACT"
    nt.links.new(sp.outputs["Z"], hz.inputs[0])
    hz.inputs[1].default_value = P.get("fog_height", 0.0)
    hs = nt.nodes.new("CompositorNodeMath")
    hs.operation = "MULTIPLY"
    hs.inputs[1].default_value = -1.0 / (60.0 if P.get("mist_banks") else 400.0)
    nt.links.new(hz.outputs[0], hs.inputs[0])
    he = nt.nodes.new("CompositorNodeMath")
    he.operation = "EXPONENT"
    nt.links.new(hs.outputs[0], he.inputs[0])
    hc = nt.nodes.new("CompositorNodeMath")
    hc.operation = "MINIMUM"
    hc.inputs[1].default_value = 1.0
    nt.links.new(he.outputs[0], hc.inputs[0])
    hmix = nt.nodes.new("CompositorNodeMath")
    hmix.operation = "MULTIPLY_ADD"
    nt.links.new(hc.outputs[0], hmix.inputs[0])
    hmix.inputs[1].default_value = 0.65
    hmix.inputs[2].default_value = 0.35
    amt = nt.nodes.new("CompositorNodeMath")
    amt.operation = "MULTIPLY"
    amt.use_clamp = True
    nt.links.new(inv.outputs[0], amt.inputs[0])
    nt.links.new(hmix.outputs[0], amt.inputs[1])
    # the sky keeps its gradient, sun glow and stars (the sky texture already carries the atmosphere): background
    # pixels (depth 1e10) get a light constant haze instead of the full fog colour
    is_sky = nt.nodes.new("CompositorNodeMath")
    is_sky.operation = "GREATER_THAN"
    is_sky.inputs[1].default_value = 1.0e6
    nt.links.new(rl.outputs["Depth"], is_sky.inputs[0])
    dsk = nt.nodes.new("CompositorNodeMath")
    dsk.operation = "SUBTRACT"
    dsk.inputs[0].default_value = {"mist": 0.8, "day": 0.25, "sunset": 0.12, "night": 0.3, "dusk": 0.2}[preset]
    nt.links.new(amt.outputs[0], dsk.inputs[1])
    dsk2 = nt.nodes.new("CompositorNodeMath")
    dsk2.operation = "MULTIPLY"
    nt.links.new(dsk.outputs[0], dsk2.inputs[0])
    nt.links.new(is_sky.outputs[0], dsk2.inputs[1])
    amt_ = amt
    amt = nt.nodes.new("CompositorNodeMath")
    amt.operation = "ADD"
    amt.use_clamp = True
    nt.links.new(amt_.outputs[0], amt.inputs[0])
    nt.links.new(dsk2.outputs[0], amt.inputs[1])
    fogc = {"mist": (0.62, 0.68, 0.70), "day": (0.62, 0.70, 0.82), "sunset": (0.95, 0.62, 0.45), "night": (0.050, 0.080, 0.150),
            "dusk": (0.45, 0.40, 0.48)}[preset]
    mix = nt.nodes.new("CompositorNodeMixRGB")
    nt.links.new(amt.outputs[0], mix.inputs[0])
    nt.links.new(rl.outputs["Image"], mix.inputs[1])
    mix.inputs[2].default_value = (*fogc, 1.0)
    glare = nt.nodes.new("CompositorNodeGlare")
    for k_, v_ in (("glare_type", "FOG_GLOW"), ("quality", "MEDIUM"), ("threshold", 1.2), ("size", 8)):
        if hasattr(glare, k_):
            try:
                setattr(glare, k_, v_)
            except Exception:
                pass
    nt.links.new(mix.outputs[0], glare.inputs[0])
    nt.links.new(glare.outputs[0], comp.inputs[0])


def moon_disc(tex_dir, on):
    ob = bpy.data.objects.get("MoonDisc")
    if ob is None:
        pos, dia = SH.moon_position()
        bpy.ops.mesh.primitive_circle_add(vertices=64, radius=dia / 2, fill_type="NGON", location=tuple(pos))
        ob = bpy.context.object
        ob.name = "MoonDisc"
        d = mathutils.Vector(tuple(-pos)).normalized()
        ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.cube_project()
        bpy.ops.object.mode_set(mode="OBJECT")
        m = bpy.data.materials.new("M_Moon")
        b = NB(m)
        tc = b.n.new("ShaderNodeTexCoord")
        mp = b.n.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1.0 / dia, 1.0 / dia, 1.0)
        mp.inputs["Location"].default_value = (0.5, 0.5, 0.0)
        b.link(tc.outputs["Object"], mp.inputs[0])
        t = b.img("T_Moon.jpg", "sRGB", mp.outputs[0])
        t.extension = "CLIP"
        em = b.n.new("ShaderNodeEmission")
        b.link(t.outputs[0], em.inputs[0])
        em.inputs["Strength"].default_value = 6.0
        b.l.remove(b.out.inputs[0].links[0])
        b.link(em.outputs[0], b.out.inputs[0])
        ob.data.materials.append(m)
    ob.hide_render = not on


def night_lights(boat_pos, on):
    for ob in [o for o in bpy.data.objects if o.name.startswith("LanternLight")]:
        bpy.data.objects.remove(ob)
    if not on:
        return
    for i, p in enumerate(boat_pos):
        ld = bpy.data.lights.new(f"LanternLight{i}", "POINT")
        ld.energy = 120.0
        ld.color = (1.0, 0.6, 0.28)
        ld.shadow_soft_size = 0.15
        ob = bpy.data.objects.new(f"LanternLight{i}", ld)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = p


def main():
    global TEX
    ap = argparse.ArgumentParser()
    ap.add_argument("geo")
    ap.add_argument("tex")
    ap.add_argument("out")
    ap.add_argument("--shots", default="")
    ap.add_argument("--scale", type=float, default=1.0, help="instance density for the previews (0..1)")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--long", type=int, default=1280, help="long side of the preview images (px)")
    a = ap.parse_args()
    TEX = os.path.abspath(a.tex)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    bu.reset_scene()
    sc = bpy.context.scene
    bu.setup_cycles(samples=a.samples, res=(a.long, a.long), max_bounces=5, clamp=8.0)
    sc.render.use_persistent_data = True
    sc.cycles.texture_limit_render = "2048"
    resolved = SH.resolve(lambda x, y: float(W.ground(x, y)[0]))
    names = [s for s in a.shots.split(",") if s] or list(resolved)
    M_day = build_materials(0.0)
    # meshes
    for p in sorted(glob.glob(f"{a.geo}/*.npz")):
        base = os.path.basename(p)
        if base.startswith(("core_columns", "core_top", "sky_")):          # the Unreal night sky (the preview has its own)
            continue
        add_mesh(p, M_day)
    lib_obs = {}
    for p in sorted(glob.glob(f"{a.geo}/lib/lib_*.npz")):
        ob = add_mesh(p, M_day)
        ob.hide_render = True
        ob.hide_viewport = True
        lib_obs[os.path.splitext(os.path.basename(p))[0]] = ob
    for p in sorted(glob.glob(f"{a.geo}/instances/*.npz")):
        add_instances(p, lib_obs, scale=a.scale)
    boats = ins.load_set(f"{a.geo}/instances/boats.npz")
    lantern_pos = [tuple(p + R @ np.array([2.12, 0.0, 0.98])) for p, R in zip(boats["pos"], boats["R"])]
    if os.path.isfile(f"{a.geo}/instances/castle_lantern.npz"):
        cl = ins.load_set(f"{a.geo}/instances/castle_lantern.npz")
        lantern_pos += [tuple(p + R @ (np.array([0.0, 0.0, 1.65]) * sc_)) for p, R, sc_ in zip(cl["pos"], cl["R"], cl["scale"])]
    print(f"scene built in {time.time() - t0:.0f}s", flush=True)
    glass = bpy.data.materials["M_Glass"]
    lantern = bpy.data.materials["M_Lantern"]
    for name in names:
        s = resolved[name]
        P = SH.PRESETS[s["preset"]]
        night = P["windows"]
        # window glow / lanterns for this preset
        glass.node_tree.nodes["GLOW"].inputs[1].default_value = 9.0 * night
        bpy.data.materials["M_GlassInst"].node_tree.nodes["GLOW"].inputs[1].default_value = 9.0 * night
        clock = bpy.data.materials["M_ClockFace"].node_tree.nodes["Principled BSDF"]
        clock.inputs["Emission Strength"].default_value = 14.0 * night + 0.3
        lantern.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 25.0 * night + 0.5
        night_lights(lantern_pos, P.get("moon", False))
        moon_disc(TEX, P.get("moon", False))
        setup_world(s["preset"], f"{TEX}/T_Stars.jpg")
        k = s["keys"][0]
        for ob in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(ob)
        res = s["res"]
        long = a.long
        rx, ry = (long, int(long * res[1] / res[0])) if res[0] >= res[1] else (int(long * res[0] / res[1]), long)
        sc.render.resolution_x, sc.render.resolution_y = rx, ry
        sensor = 36.0
        cam = bu.add_camera(k["loc"], k["look_at"], lens=s["lens"], sensor=sensor, name=name)
        cam.data.sensor_fit = "HORIZONTAL" if res[0] >= res[1] else "VERTICAL"
        if res[0] < res[1]:
            cam.data.sensor_height = 36.0
        cam.data.clip_end = 30000.0
        cam.data.clip_start = 0.3
        setup_compositor(s["preset"], k["loc"][2])
        t = time.time()
        sc.view_settings.look = "Medium High Contrast"          # closer to Unreal's filmic tone curve than AgX's base
        auto_exposure(s["preset"], a.out)
        bu.render_to(f"{a.out}/{name}.png")
        print(f"rendered {name} ({rx}x{ry}, preset {s['preset']}, exposure {sc.view_settings.exposure:+.2f}) in {time.time() - t:.0f}s",
              flush=True)
    print(f"all previews in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
