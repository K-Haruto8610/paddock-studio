"""Saddle cloth (ゼッケン) for the plush horse: a drape over the back and the two number panels on its sides.

  horse_saddle           the cloth: a 10 x 24 grid draped over the trunk from the right flank over the back to the left
  horse_saddle_print_L/R the number panels, a 12 x 12 grid on each side, standing 8 mm off the cloth (uv 0..1, the app prints on them)

Ported from clothPoint() and the PlaneGeometry grids in plush() of app/plushHorse.ts: the same lines on the same trunk
(`trunk_np`: torso and legs, no neck or head). The cloth is baked at the minimum standoff, 25 mm; the app lifts it
radially to clear the fur (pile length + 12 mm) and recomputes the normals.

Library: horse.py calls build_saddle().
"""
SADDLE_STANDOFF = 0.025
PRINT_LIFT = 0.008


def trunk(x, y, z):
    return float(trunk_np(np.float64(x), np.float64(y), np.float64(z)))


def cloth_point(x, t, lift=0.0):
    """The point at angle t around the back's axis (y = 1.22, z = 0), on the trunk plus the standoff and `lift`."""
    lo, hi = 0.0, 1.6
    for _ in range(16):
        r = (lo + hi) / 2
        if trunk(x, 1.22 + r * math.cos(t), r * math.sin(t)) < 0: lo = r
        else: hi = r
    r = (lo + hi) / 2 + SADDLE_STANDOFF + lift
    return (x, 1.22 + r * math.cos(t), r * math.sin(t))


def plane_grid(name, width, height, nx, ny, place):
    """A PlaneGeometry(width, height, nx, ny) with every vertex put where place(x, y) says; same vertex order, faces and uv as three.js (once exported)."""
    verts, uvs, faces = [], [], []
    for iy in range(ny + 1):
        y = iy * height / ny - height / 2
        for ix in range(nx + 1):
            x = ix * width / nx - width / 2
            verts.append(to_blender(*place(x, -y)))
            uvs.append((ix / nx, iy / ny))        # glTF flips v on export, so this reaches the app as three.js's own 1 - iy / ny
    stride = nx + 1
    for iy in range(ny):
        for ix in range(nx):
            a, b, c, d = ix + stride * iy, ix + stride * (iy + 1), ix + 1 + stride * (iy + 1), ix + 1 + stride * iy
            faces.append((a, b, c, d))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    uv = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li, vi in zip(poly.loop_indices, poly.vertices):
            uv.data[li].uv = uvs[vi]
    me.update()
    return me


def build_saddle():
    for o in [o for o in bpy.data.objects if o.name.startswith("horse_saddle")]:
        bpy.data.objects.remove(o, do_unlink=True)
    objs = {}
    cloth = plane_grid("horse_saddle", 1.08, 2.8, 10, 24, lambda x, y: cloth_point(x - 0.65, y / 1.4 * 1.72))
    objs["horse_saddle"] = cloth
    for side in (-1, 1):
        tag = "L" if side < 0 else "R"
        objs["horse_saddle_print_" + tag] = plane_grid(
            "horse_saddle_print_" + tag, 1, 1, 12, 12,
            lambda x, y, side=side: cloth_point(-0.65 + x * 0.9 * side, side * (1.25 - y * 0.85), PRINT_LIFT))
    cloth_mat = material("MAT_saddle", "#234b98", 0.95, sheen=0.4, sheen_roughness=0.6, grain=0.15)
    for name, me in objs.items():
        o = bpy.data.objects.new(name, me)
        o.data.materials.append(cloth_mat)
        smooth(o)
        link(o)
    return {k: sum(len(p.vertices) - 2 for p in v.polygons) for k, v in objs.items()}


if ARGS.get("run_saddle") == "1":
    print(build_saddle())
