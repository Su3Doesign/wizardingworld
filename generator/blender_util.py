"""Helpers for turning meshkit meshes into Blender objects and rendering them with Cycles (CPU)."""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

import bpy
import mathutils

from meshkit import Mesh


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0
    return sc


def to_bpy_mesh(name, M: Mesh, normals_angle=35.0, materials=None, uv_names=("UV0", "UV1", "UV2", "UV3")):
    """Build a bpy Mesh datablock from a meshkit Mesh (custom split normals + UV layers)."""
    me = bpy.data.meshes.new(name)
    nv, nf = M.nv, M.nf
    me.vertices.add(nv)
    me.vertices.foreach_set("co", M.V.astype(np.float32).ravel())
    me.loops.add(3 * nf)
    me.loops.foreach_set("vertex_index", M.F.astype(np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, 3 * nf, 3, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(nf, 3, dtype=np.int32))
    except Exception:
        pass
    me.update(calc_edges=True)
    me.polygons.foreach_set("material_index", M.mat.astype(np.int32))
    for attr, uname in zip(("uv0", "uv1", "uv2", "uv3"), uv_names):
        a = getattr(M, attr)
        if a is None:
            continue
        layer = me.uv_layers.new(name=uname)
        layer.data.foreach_set("uv", np.nan_to_num(a).astype(np.float32).reshape(-1))
    me.polygons.foreach_set("use_smooth", np.ones(nf, dtype=bool))
    if normals_angle is not None:
        N = M.corner_normals(normals_angle).astype(np.float32).reshape(-1, 3)
        me.normals_split_custom_set(N)
    me.update()
    if materials:
        for mat in materials:
            me.materials.append(mat)
    return me


def add_object(name, M: Mesh, materials=None, collection=None, normals_angle=35.0):
    me = to_bpy_mesh(name, M, normals_angle=normals_angle, materials=materials)
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def clay_material(color=(0.55, 0.52, 0.47), rough=0.85):
    m = bpy.data.materials.new("clay")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    return m


def setup_cycles(samples=24, denoise=True, threads=0, res=(960, 540), adaptive=True, clamp=10.0, max_bounces=6):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = adaptive
    cy.adaptive_threshold = 0.02
    cy.use_denoising = denoise
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    cy.max_bounces = max_bounces
    cy.diffuse_bounces = min(4, max_bounces)
    cy.glossy_bounces = 4
    cy.transmission_bounces = 8
    cy.transparent_max_bounces = 16
    cy.volume_bounces = 2
    cy.sample_clamp_indirect = clamp
    cy.volume_step_rate = 1.0
    sc.render.threads_mode = "AUTO" if threads == 0 else "FIXED"
    if threads:
        sc.render.threads = threads
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX" if "AgX" in [v.identifier for v in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    return sc


def look_at(obj, target):
    d = mathutils.Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_camera(loc, target, lens=35.0, sensor=36.0, name="Cam", dof_target=None, fstop=None):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.sensor_width = sensor
    cam.sensor_fit = "AUTO"
    cam.clip_start = 0.05
    cam.clip_end = 500
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    if fstop:
        cam.dof.use_dof = True
        cam.dof.aperture_fstop = fstop
        cam.dof.focus_distance = (mathutils.Vector(dof_target) - mathutils.Vector(loc)).length if dof_target else 10.0
    bpy.context.scene.camera = ob
    return ob


def add_sun(direction_from, strength=4.0, angle_deg=0.6, color=(1.0, 0.93, 0.8), name="Sun"):
    """direction_from: vector pointing from the scene *towards* the sun."""
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = strength
    ld.angle = math.radians(angle_deg)
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    d = mathutils.Vector(direction_from)
    d.normalize()
    # sun shines along -Z of the object; we want -Z == -d, so +Z == d
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    return ob


def world_color(color=(0.3, 0.4, 0.55), strength=0.6):
    w = bpy.data.worlds.new("World")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1)
    bg.inputs["Strength"].default_value = strength
    bpy.context.scene.world = w
    return w


def setup_compositor(glare_mix=-0.6, glare_threshold=1.0, glare_size=8, quality="HIGH", strength=0.45, saturation=1.0, value=1.0):
    """Fog-glow bloom (the soft halo round bright shafts and sun-lit foam).  Returns True when the node setup worked.
    Blender 4.5 exposes the Glare settings both as legacy node properties and as sockets; set whatever exists."""
    sc = bpy.context.scene
    try:
        sc.use_nodes = True
        nt = sc.node_tree
        nt.nodes.clear()
        rl = nt.nodes.new("CompositorNodeRLayers")
        gl = nt.nodes.new("CompositorNodeGlare")
        for k, v in (("glare_type", "FOG_GLOW"), ("quality", quality), ("threshold", glare_threshold), ("size", glare_size), ("mix", glare_mix)):
            if hasattr(gl, k):
                try:
                    setattr(gl, k, v)
                except Exception:
                    pass
        for k, v in (("Threshold", glare_threshold), ("Strength", strength)):
            if k in gl.inputs:
                gl.inputs[k].default_value = v
        co = nt.nodes.new("CompositorNodeComposite")
        nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
        last = gl
        if abs(saturation - 1.0) > 1e-3 or abs(value - 1.0) > 1e-3:
            hs = nt.nodes.new("CompositorNodeHueSat")
            hs.inputs["Saturation"].default_value = saturation
            hs.inputs["Value"].default_value = value
            nt.links.new(gl.outputs["Image"], hs.inputs["Image"])
            last = hs
        nt.links.new(last.outputs["Image"], co.inputs["Image"])
        return True
    except Exception as e:                                         # noqa: BLE001
        print("compositor setup failed:", e)
        return False


def render_to(path):
    sc = bpy.context.scene
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - t
