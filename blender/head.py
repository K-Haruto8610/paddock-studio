"""Horse head for the plush horse (docs/STYLE_GUIDE.md). Run with blender/run_in_blender.py.

Objects (all parented to `horse_head`, which sits at the neck pivot so the head can nod about its origin):
  horse_head       shell of the face: forehead + muzzle, closed, smooth quads. Slots: MAT_coat
  horse_ear_L/R    cupped ears (double-layer cloth). Slots: MAT_ear (hood colour, or the coat bare-headed)
  horse_eye_L/R    bead eyes. Slots: MAT_eye
  horse_nostril_L/R small dark ovals. Slots: MAT_nostril

Frame: metres, +X forward, +Z up, ground Z=0, left of the horse is +Y. In glTF (+Y up) that is x, z, -y.
The shape is the same signed-distance head as app/plushHorse.ts (faceDistance, SAVED_HEAD), sampled onto a quad sphere,
so it fills the same space as the head the code builds today.
"""
import math
import bmesh
import bpy
from mathutils import Matrix, Vector

# --- the head of app/plushHorse.ts, in its own units (HEAD_ORIGIN 0.67/1.91, scale 1.18) -------------------------
HEAD_X, HEAD_Y, HEAD_S = 0.67, 1.91, 1.18
PIVOT = (0.55, 1.6)                       # HEAD_PIVOT: inside the neck, the head nods about it
WIDTH, NOSE_SIZE, NOSE_TAPER, FORE_SLOPE, FACE_BLEND = 0.95, 0.25, 0.5, 0.15, 0.15
EAR_S = 1.18
CAGE = int(ARGS.get("cage", 16))          # quads along each edge of each of the 6 cube faces


def lerp(a, b, t): return a + (b - a) * t
def clamp(x, lo, hi): return max(lo, min(hi, x))
def smoothstep(x, e0, e1):
    t = clamp((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def smooth_union(a, b, k):
    if k <= 0: return min(a, b)
    h = max(k - abs(a - b), 0) / k
    return min(a, b) - h * h * k * 0.25


def superellipsoid(x, y, z, rx, ry, rz, p):
    return ((abs(x / rx) ** p + abs(y / ry) ** p + abs(z / rz) ** p) ** (1 / p) - 1) * min(rx, ry, rz)


def face_distance_app(px, py, pz):
    """Exact port of faceDistance() in app/plushHorse.ts with SAVED_HEAD: the head the app drew before the Blender work.
    px forward, py up, pz sideways (world metres). Negative inside the head."""
    power = 2 / 0.82
    hx, hy, hz = (px - HEAD_X) / HEAD_S, (py - HEAD_Y) / HEAD_S, pz / (HEAD_S * WIDTH)

    def rounded(x, y, z, sx, sy, sz):
        t = abs(x / sx) ** power + abs(y / sy) ** power + abs(z / sz) ** power
        return (t ** (1 / power) - 1) * min(sx, sy, sz)

    old = rounded(hx - 0.24, hy - 0.15, hz, 0.72, 0.68, 0.61)
    t = clamp((hx + (hy - 0.15) * -0.33) / 1.1089, 0, 1)
    sloped = math.sqrt((hx - t) ** 2 + (hy - (0.15 - 0.33 * t)) ** 2 + hz ** 2) - (0.59 - 0.32 * t)
    forehead = lerp(old, sloped, FORE_SLOPE)
    nx, ny = hx - 0.79, hy + 0.24
    rx, ry = nx * math.cos(0.13) - ny * math.sin(0.13), nx * math.sin(0.13) + ny * math.cos(0.13)
    taper = 1 - NOSE_TAPER * 0.4 * smoothstep(hx, 0.5, 1.35)
    nose = rounded(rx, ry, hz / taper, lerp(0.59, 0.44, NOSE_SIZE), lerp(0.47, 0.3, NOSE_SIZE), lerp(0.59, 0.4, NOSE_SIZE))
    return smooth_union(forehead, nose, 0.2 * FACE_BLEND) * HEAD_S * WIDTH


def face_distance_plush(px, py, pz):
    """A reshaped variant (shape=plush): a round dome and a short, wide muzzle that narrows to a round nose, melted
    together with a soft dip under the cheek. Kept for comparison; the owner prefers the app's own head."""
    dome = superellipsoid(px - 0.95, py - 2.12, pz, 0.80, 0.74, 0.66, 2.3)
    mx, my = px - 1.66, py - 1.62
    c, s = math.cos(0.18), math.sin(0.18)
    taper = 1 - 0.34 * smoothstep(mx, -0.15, 0.42)          # the muzzle narrows toward the nose
    muzzle = superellipsoid(mx * c - my * s, (mx * s + my * c) / (1 - 0.12 * smoothstep(mx, 0.0, 0.42)), pz / taper, 0.40, 0.45, 0.47, 2.5)
    # A light jaw under both: the throat line keeps a soft dip between cheek and chin, with no hard step.
    jaw = superellipsoid(px - 1.36, py - 1.60, pz, 0.56, 0.36, 0.46, 2.4)
    face = smooth_union(dome, muzzle, 0.14)
    return smooth_union(face, jaw, 0.16)


SHAPE = ARGS.get("shape", "app")
face_distance = face_distance_app if SHAPE == "app" else face_distance_plush
UNIFIED = ARGS.get("unified") == "1"     # the head is part of the body shell (body.py), not a shell of its own


def surface_distance(px, py, pz):
    """The surface the eyes, ears and nostrils sit on: the head alone here; body.py swaps in the whole body."""
    return face_distance(px, py, pz)

# Where the small parts sit, per shape. "app" is the head-local placement app/plushHorse.ts uses.
if SHAPE == "app":
    CENTER = (HEAD_X + 0.59, HEAD_Y - 0.06)                                   # inside forehead and muzzle both
    EAR = dict(at=(HEAD_X - 0.06 * HEAD_S, HEAD_Y + 0.58 * HEAD_S, 0.3 * WIDTH * HEAD_S), tilt=0.2, wid=0.18, pow=0.8, narrow=0.35, sx=1.0, sy=1.0, sz=1.0)
    EYE = dict(at=(HEAD_X + 0.47 * HEAD_S, HEAD_Y + 0.26 * HEAD_S), radii=(0.11, 0.08, 0.14), turn=0.0)
    NOSTRIL = dict(front=False, at=(HEAD_X + 1.1 * HEAD_S, HEAD_Y - 0.13 * HEAD_S), radii=(0.032, 0.024, 0.046))
else:
    CENTER = (1.35, 1.9)
    EAR = dict(at=(0.86, None, 0.36), tilt=0.12, wid=0.19, pow=0.55, narrow=0.12, sx=1.0, sy=1.4, sz=0.55)
    EYE = dict(at=(1.36, 2.05), radii=(0.075, 0.055, 0.09), turn=0.42)
    NOSTRIL = dict(front=True, at=(None, 1.64, 0.15), radii=(0.024, 0.032, 0.05))


def surface_z(x, y):
    lo, hi = 0.0, 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        if surface_distance(x, y, mid) < 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2


# --- scene bookkeeping -------------------------------------------------------------------------------------------
def to_blender(x, y, z):
    """(forward, up, sideways) -> Blender (X, Y, Z). Sideways + is the horse's right, which is -Y."""
    return Vector((x, -z, y))


def wipe():
    for o in [o for o in bpy.data.objects if o.name.startswith("horse_") and o.name != "horse_body"]:
        bpy.data.objects.remove(o, do_unlink=True)
    for m in [m for m in bpy.data.meshes if m.name.startswith("horse_") and m.users == 0]:
        bpy.data.meshes.remove(m)


def material(name, color, roughness, sheen=0.0, sheen_roughness=0.5, grain=0.0, coat=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    c = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    bsdf.inputs["Base Color"].default_value = (*c, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Sheen Weight"].default_value = sheen
    bsdf.inputs["Sheen Roughness"].default_value = sheen_roughness
    bsdf.inputs["Sheen Tint"].default_value = (1, 0.97, 0.92, 1)
    bsdf.inputs["Coat Weight"].default_value = coat
    if grain:
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 260
        noise.inputs["Detail"].default_value = 6
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = grain
        bump.inputs["Distance"].default_value = 0.01
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def smooth(obj):
    me = obj.data
    try:
        me.shade_smooth()
    except AttributeError:
        for p in me.polygons: p.use_smooth = True


def link(obj, parent=None, collection="horse"):
    col = bpy.data.collections.get(collection) or bpy.data.collections.new(collection)
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    col.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


# --- the shell ---------------------------------------------------------------------------------------------------
def head_shell(pivot):
    n = CAGE
    cx, cy = CENTER          # inside both the forehead and the muzzle
    bm = bmesh.new()
    verts = {}

    def vert(a, b, c):
        key = (a, b, c)
        if key not in verts:
            d = Vector((a, b, c)).normalized()
            lo, hi = 0.0, 2.2
            for _ in range(36):
                mid = (lo + hi) / 2
                if face_distance(cx + d.x * mid, cy + d.y * mid, d.z * mid) < 0: lo = mid
                else: hi = mid
            r = (lo + hi) / 2
            p = to_blender(cx + d.x * r, cy + d.y * r, d.z * r) - pivot
            verts[key] = bm.verts.new(p)
        return verts[key]

    for axis in range(3):
        for sign in (-1, 1):
            for i in range(n):
                for j in range(n):
                    corners = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        u, v = i + di - n // 2, j + dj - n // 2
                        # n is even: lattice coordinates run -n/2 .. n/2
                        coord = [0, 0, 0]
                        coord[axis] = sign * (n // 2)
                        coord[(axis + 1) % 3], coord[(axis + 2) % 3] = u, v
                        corners.append(vert(*coord))
                    try:
                        bm.faces.new(corners)
                    except ValueError:
                        pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("horse_head")
    bm.to_mesh(me)
    bm.free()
    return me


def ear_mesh(side):
    """Port of earGeometry() in app/plushHorse.ts (head units, scaled by HEAD_S), tilted outward; EAR sets the variant."""
    rows, cols = 10, 8
    bm = bmesh.new()
    layers = []
    for layer in range(2):
        grid = []
        for row in range(rows + 1):
            t = row / rows
            width = EAR["wid"] * math.sin(math.pi * t) ** EAR["pow"] * (1 - t * EAR["narrow"])
            line = []
            for col in range(cols + 1):
                u = col / cols * 2 - 1
                bowl = math.sin(math.pi * t) * (0.08 * u * u - 0.065 * (1 - u * u))
                thick = 0.018 + 0.02 * math.sin(math.pi * t) * (1 - u * u)
                x = bowl + (thick if layer == 0 else -thick)
                y, z = t * 0.66, u * width
                a = side * EAR["tilt"]                      # rotation.x in the code: sideways tilt
                y, z = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)
                line.append(bm.verts.new(Vector((x * EAR_S * EAR["sx"], -z * EAR_S * EAR["sy"], y * EAR_S * EAR["sz"]))))   # Blender: up = Z, sideways = -Y
            grid.append(line)
        layers.append(grid)
    for layer in range(2):
        for r in range(rows):
            for c in range(cols):
                g = layers[layer]
                try: bm.faces.new((g[r][c], g[r][c + 1], g[r + 1][c + 1], g[r + 1][c]))
                except ValueError: pass
    for r in range(rows):
        for c in (0, cols):
            try: bm.faces.new((layers[0][r][c], layers[0][r + 1][c], layers[1][r + 1][c], layers[1][r][c]))
            except ValueError: pass
    for c in range(cols):
        for r in (0, rows):
            try: bm.faces.new((layers[0][r][c], layers[1][r][c], layers[1][r][c + 1], layers[0][r][c + 1]))
            except ValueError: pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("horse_ear_" + ("L" if side < 0 else "R"))
    bm.to_mesh(me)
    bm.free()
    return me


def ellipsoid(name, radii, segs=24, rings=14):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=1.0)
    for v in bm.verts:
        v.co.x *= radii[0]; v.co.y *= radii[1]; v.co.z *= radii[2]
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def build():
    wipe()
    # keep the user's default cube out of the previews
    cube = bpy.data.objects.get("Cube")
    if cube: bpy.data.objects.remove(cube, do_unlink=True)

    coat = material("MAT_coat", "#b8773c", 0.95, sheen=0.65, sheen_roughness=0.6, grain=0.25)
    ear_mat = material("MAT_ear", "#b8773c", 0.95, sheen=0.65, sheen_roughness=0.6, grain=0.25)
    eye_mat = material("MAT_eye", "#151416", 0.35)
    nose_mat = material("MAT_nostril", "#29272a", 0.8)

    pivot = to_blender(PIVOT[0], PIVOT[1], 0)
    if UNIFIED:                                        # the head group: an empty at the neck pivot the parts hang from
        head = bpy.data.objects.new("horse_head", None)
        head.location = pivot
    else:
        head = bpy.data.objects.new("horse_head", head_shell(pivot))
        head.location = pivot
        head.data.materials.append(coat)
        smooth(head)
    link(head)

    parts = {}
    for side in (-1, 1):                              # side -1 = the horse's left = +Y
        tag = "L" if side < 0 else "R"
        # ears
        ex, ey, ez = EAR["at"]
        ez *= side
        if ey is None:                                  # sit on the dome, sunk a little
            lo, hi = 1.5, 3.4
            for _ in range(40):
                mid = (lo + hi) / 2
                if surface_distance(ex, mid, ez) < 0: lo = mid
                else: hi = mid
            ey = lo - 0.07
        ear = bpy.data.objects.new("horse_ear_" + tag, ear_mesh(side))
        ear.location = to_blender(ex, ey, ez)
        ear.data.materials.append(ear_mat)
        smooth(ear)
        link(ear, head)
        ear.matrix_parent_inverse = Matrix.Translation(-pivot)
        # eyes: on the face, standing proud like a sewn bead
        ex, ey = EYE["at"]
        ez = surface_z(ex, ey)
        eye = bpy.data.objects.new("horse_eye_" + tag, ellipsoid("horse_eye_" + tag, EYE["radii"], 12, 8))
        eye.location = to_blender(ex, ey, side * (ez - 0.004))
        eye.rotation_euler.z = side * EYE["turn"]
        smooth(eye)
        eye.data.materials.append(eye_mat)
        link(eye, head)
        eye.matrix_parent_inverse = Matrix.Translation(-pivot)
        # nostrils: on the side of the nose (app), or on its front face (plush)
        if NOSTRIL["front"]:
            ny, nz = NOSTRIL["at"][1], side * NOSTRIL["at"][2]
            lo, hi = 1.2, 3.0
            for _ in range(40):
                mid = (lo + hi) / 2
                if surface_distance(mid, ny, nz) < 0: lo = mid
                else: hi = mid
            nx = lo - 0.004
        else:
            nx, ny = NOSTRIL["at"]
            nz = side * (surface_z(nx, ny) - 0.004)
        nos = bpy.data.objects.new("horse_nostril_" + tag, ellipsoid("horse_nostril_" + tag, NOSTRIL["radii"], 10, 6))
        nos.location = to_blender(nx, ny, nz)
        smooth(nos)
        nos.data.materials.append(nose_mat)
        link(nos, head)
        nos.matrix_parent_inverse = Matrix.Translation(-pivot)
    return head


if ARGS.get("lib") != "1":
    head = build()
    print("built", head.name, "tris:", sum(len(p.vertices) - 2 for p in head.data.polygons) if head.data else 0)
