"""Export the plush horse (every object named horse_*) to public/models/horse.glb for the app (app/blenderHorse.ts).

    python blender/run_in_blender.py blender/export.py [out=public/models/horse.glb]

glTF is +Y up, so Blender (X, Y, Z) arrives as (X, Z, -Y): +X forward, Y up, Z the horse's right, the app's own frame.
No materials are exported: the app colours and fabrics every part itself, by the object's name. Only the straps carry UVs
(the body, hood and ears get planar UVs in the app, as the code-built ones always did) and `_cover` (strap on hood = 1).
"""
import os
import bpy

out = ARGS.get("out", "public/models/horse.glb")
if not os.path.isabs(out):
    out = os.path.join(ARGS["root"], out)
os.makedirs(os.path.dirname(out), exist_ok=True)

for o in bpy.context.scene.objects:
    o.select_set(o.name.startswith("horse_"))
bpy.context.view_layer.objects.active = bpy.data.objects["horse_body"]
bpy.ops.export_scene.gltf(
    filepath=out, export_format="GLB", use_selection=True, export_yup=True,
    export_materials="NONE", export_apply=True, export_normals=True, export_texcoords=True,
    export_attributes=True, use_mesh_edges=True, export_cameras=False, export_lights=False, export_extras=False,
)
print("exported", out, os.path.getsize(out), "bytes")
