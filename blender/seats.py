"""A stadium seat for the grandstands (docs/STYLE_GUIDE.md: 建物・スタンド are realistic, with small bevels, no plastic shine).

The app (app/raceLandscape.ts) sets ~3,000 of these in rows on the stand's concrete terraces as one InstancedMesh, so a seat
is kept to a small budget: 72 triangles (a moulded bucket shell, drawn double-sided, on a bracket). Colours are the app's own.

  stand_seat   one seat. App frame (glTF): origin on the floor under the middle of the seat's front edge, the seat faces -z
               (toward the track when the stand's front is -z), y up, metres. Attribute `_shade`: 1 on the moulded shell,
               0.35 on the steel bracket (the app multiplies the seat colour by it).

    python blender/run_in_blender.py blender/seats.py            # build stand_seat in the "stand" collection
    python blender/run_in_blender.py blender/seats.py export=1   # and write public/models/stand.glb
"""
import math
import os
import bmesh
import bpy
from mathutils import Vector

WIDTH = 0.46            # shell width; seats are set 0.5 m apart
ACROSS = 4              # quads across the shell
SHOW_AT = (-4.0, 0.0)   # where the seat sits in the Blender scene, clear of the horse (not exported: the app uses the mesh)


def _app_to_blender(x, y, z):
    return Vector((x, -z, y))


def _profile():
    """Side section of the shell, front lip to the top of the back, as (depth, height); depth + is toward the back."""
    pts = [(-0.02, 0.405), (0.0, 0.43), (0.08, 0.44), (0.20, 0.435), (0.30, 0.45), (0.345, 0.50), (0.37, 0.60), (0.39, 0.72), (0.405, 0.82)]
    return pts


def seat_mesh():
    bm = bmesh.new()
    shade = bm.verts.layers.float.new("_shade")
    prof = _profile()
    grid = []
    for i, (d, h) in enumerate(prof):
        row = []
        back = min(1.0, max(0.0, (h - 0.45) / 0.3))                   # 0 on the pan, 1 up the back
        for j in range(ACROSS + 1):
            u = j / ACROSS * 2 - 1                                     # -1 .. 1 across
            # bucket: the pan's sides rise a little, the back's sides wrap forward
            x = u * WIDTH / 2 * (1 - 0.06 * back)
            y = h + 0.022 * u * u * (1 - back)
            z = d - 0.045 * u * u * back                               # app z: depth runs to +z, so the seat faces -z
            v = bm.verts.new(_app_to_blender(x, y, z))
            v[shade] = 1.0
            row.append(v)
        grid.append(row)
    for i in range(len(prof) - 1):
        for j in range(ACROSS):
            bm.faces.new((grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]))
    # bracket: a steel blade from the floor to the underside of the pan, open at the bottom (8 triangles)
    w, f, b, top = 0.02, 0.08, 0.30, 0.425
    corners = [(-w, 0, f), (w, 0, f), (w, 0, b), (-w, 0, b), (-w, top, f), (w, top, f), (w, top, b), (-w, top, b)]
    bv = [bm.verts.new(_app_to_blender(*c)) for c in corners]
    for v in bv: v[shade] = 0.35
    for quad in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([bv[k] for k in quad])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:ACROSS * (len(prof) - 1)])
    me = bpy.data.meshes.new("stand_seat")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons: p.use_smooth = True
    # _shade as a point attribute the glTF exporter writes out (custom attributes start with "_")
    return me


def build_seat():
    old = bpy.data.objects.get("stand_seat")
    if old:
        mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if mesh.users == 0: bpy.data.meshes.remove(mesh)
    col = bpy.data.collections.get("stand") or bpy.data.collections.new("stand")
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    obj = bpy.data.objects.new("stand_seat", seat_mesh())
    obj.location = (SHOW_AT[0], SHOW_AT[1], 0)
    mat = bpy.data.materials.get("MAT_seat") or bpy.data.materials.new("MAT_seat")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.2, 0.29, 0.28, 1)
    bsdf.inputs["Roughness"].default_value = 0.75
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def export_stand(out="public/models/stand.glb"):
    if not os.path.isabs(out):
        out = os.path.join(ARGS["root"], out)
    for o in bpy.context.scene.objects:
        o.select_set(o.name.startswith("stand_"))
    bpy.context.view_layer.objects.active = bpy.data.objects["stand_seat"]
    bpy.ops.export_scene.gltf(
        filepath=out, export_format="GLB", use_selection=True, export_yup=True,
        export_materials="NONE", export_apply=True, export_normals=True, export_texcoords=False,
        export_attributes=True, export_cameras=False, export_lights=False, export_extras=False,
    )
    return out, os.path.getsize(out)


seat = build_seat()
print("stand_seat tris:", sum(len(p.vertices) - 2 for p in seat.data.polygons))
if ARGS.get("export") == "1":
    print("exported", *export_stand(ARGS.get("out", "public/models/stand.glb")))
