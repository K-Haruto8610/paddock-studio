"""Fixed-angle renders of the horse parts (side, front 3/4, top) for docs/STYLE_GUIDE.md's visual check.

    python blender/run_in_blender.py blender/preview.py out=<dir> [focus=head|horse] [engine=EEVEE|WORKBENCH]
"""
import math, os
import bpy
from mathutils import Vector

out = ARGS["out"]
focus = ARGS.get("focus", "head")
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y = 720, 600
sc.render.image_settings.file_format = "PNG"
# STYLE_GUIDE asks for ACES; Blender ships no ACES transform without an OCIO config, so use the closest filmic one here.
for vt in ("ACES", "Filmic", "AgX", "Standard"):
    try:
        sc.view_settings.view_transform = vt
        break
    except TypeError:
        continue
sc.render.engine = "BLENDER_EEVEE" if ARGS.get("engine", "EEVEE") == "EEVEE" else "BLENDER_WORKBENCH"

# neutral world + a sun and a soft fill, shared by every part
w = sc.world or bpy.data.worlds.new("World")
sc.world = w
w.use_nodes = True
bg = w.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.62, 0.66, 0.72, 1)
bg.inputs["Strength"].default_value = 0.55
for o in [o for o in bpy.data.objects if o.name.startswith("preview_")]:
    bpy.data.objects.remove(o, do_unlink=True)
sun = bpy.data.objects.new("preview_sun", bpy.data.lights.new("preview_sun", "SUN"))
sun.data.energy, sun.data.angle = 2.2, math.radians(8)
sun.rotation_euler = (math.radians(55), math.radians(15), math.radians(35))
sc.collection.objects.link(sun)
fill = bpy.data.objects.new("preview_fill", bpy.data.lights.new("preview_fill", "AREA"))
fill.data.energy, fill.data.size = 250, 6
fill.location = (-3, -4, 3)
sc.collection.objects.link(fill)
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0)) if False else None
floor = bpy.data.objects.new("preview_floor", bpy.data.meshes.new("preview_floor"))
floor.data.from_pydata([(-15, -15, 0), (15, -15, 0), (15, 15, 0), (-15, 15, 0)], [], [(0, 1, 2, 3)])
fm = bpy.data.materials.get("preview_floor") or bpy.data.materials.new("preview_floor")
fm.use_nodes = True
fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.37, 0.34, 1)
floor.data.materials.append(fm)
sc.collection.objects.link(floor)

cam = bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam"))
sc.collection.objects.link(cam)
sc.camera = cam

if focus == "head":
    target, dist = Vector((1.25, 0, 2.15)), 6.5
else:
    target, dist = Vector((0.4, 0, 1.4)), 12
cam.data.lens = 85 if focus == "head" else 70
views = {
    "side": Vector((0, -1, 0.06)),
    "front34": Vector((0.8, -0.7, 0.25)),
    "top": Vector((0.0, -0.02, 1)),
}
if ARGS.get("under"):
    views["under34"] = Vector((0.7, -0.6, -0.55))
for name, d in views.items():
    d = d.normalized()
    floor.hide_render = d.z < 0          # the floor would hide a view from below
    cam.location = target + d * dist
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(out, name + ".png")
    bpy.ops.render.render(write_still=True)
print("rendered", list(views), "->", out, "engine", sc.render.engine, "view", sc.view_settings.view_transform)
