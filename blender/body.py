"""Body of the plush horse: torso, chest, neck, head AND legs as ONE seamless shell (docs/STYLE_GUIDE.md).

The app has always drawn the head and body as one stuffed cloth skin (`bodyDistance` in app/plushHorse.ts: the torso, the
chest that leans up into the neck, and the face, melted together), nodding the head by bending that skin about the neck.
So the Blender body is one mesh too: `horse_body`. The legs are fused into it as well (owner's decision 2026-11-19); the app
swings them by vertex position (legRig), as it does today.

  horse_body   torso + neck + head + 4 legs, one closed quad mesh. Slots: MAT_coat; MAT_hoof below 0.25 m (the socks).
               The head nods about (0.55, 1.6): the app derives the nod and gait weights from vertex positions.

The shell is sampled from the same signed-distance field as the app (surface nets: regular, smooth quads), with the same
SAVED_HEAD settings. Run through blender/horse.py, which then hangs the head parts on it.

    python blender/run_in_blender.py blender/body.py [cell=0.07]
"""
import numpy as np

ARGS["lib"], ARGS["unified"] = "1", "1"
if "head_shell" not in globals():
    exec(open(ARGS["root"] + "/blender/head.py", encoding="utf-8").read())

CELL = float(ARGS.get("cell", 0.07))                 # grid spacing of the surface-nets sampling, metres
CHEST_SLOPE = 0.25                                   # SAVED_HEAD.chestSlope
LEG_X = (-0.82, 0.64)                                # hind, fore leg positions (LEGS in the app)
LEG_WIDTH = 0.8                                      # DEFAULT_LEG_WIDTH


# --- bodyDistance of app/plushHorse.ts, vectorised ------------------------------------------------
def _ss(x, e0, e1):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _su(a, b, k):
    if k <= 0: return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def _blend(py, a, b):
    s = np.maximum(0.32 * _ss(py, 0.22, 0.65), 1e-9)
    h = np.maximum(s - np.abs(a - b), 0) / s
    return np.minimum(a, b) - h * h * s * 0.25


def face_np(px, py, pz):
    power = 2 / 0.82
    hx, hy, hz = (px - HEAD_X) / HEAD_S, (py - HEAD_Y) / HEAD_S, pz / (HEAD_S * WIDTH)

    def rounded(x, y, z, sx, sy, sz):
        t = np.abs(x / sx) ** power + np.abs(y / sy) ** power + np.abs(z / sz) ** power
        return (t ** (1 / power) - 1) * min(sx, sy, sz)

    old = rounded(hx - 0.24, hy - 0.15, hz, 0.72, 0.68, 0.61)
    t = np.clip((hx + (hy - 0.15) * -0.33) / 1.1089, 0, 1)
    sloped = np.sqrt((hx - t) ** 2 + (hy - (0.15 - 0.33 * t)) ** 2 + hz ** 2) - (0.59 - 0.32 * t)
    forehead = old + (sloped - old) * FORE_SLOPE
    nx, ny = hx - 0.79, hy + 0.24
    rx, ry = nx * math.cos(0.13) - ny * math.sin(0.13), nx * math.sin(0.13) + ny * math.cos(0.13)
    taper = 1 - NOSE_TAPER * 0.4 * _ss(hx, 0.5, 1.35)
    nose = rounded(rx, ry, hz / taper, lerp(0.59, 0.44, NOSE_SIZE), lerp(0.47, 0.3, NOSE_SIZE), lerp(0.59, 0.4, NOSE_SIZE))
    return _su(forehead, nose, 0.2 * FACE_BLEND) * HEAD_S * WIDTH


def body_np(px, py, pz):
    power = 2 / 0.82
    d = ((np.abs((px + 0.15) / 1.22) ** power + np.abs((py - 1.22) / 0.87) ** power + np.abs(pz / 0.8) ** power) ** (1 / power) - 1) * 0.8
    feet = np.full(np.shape(d), np.inf)
    for lx in LEG_X:
        for lz in (-0.43, 0.43):
            # circular section with softly rounded ends; the upper end sinks into the belly
            radial = np.hypot(px - lx, pz - lz) - (0.35 * LEG_WIDTH - 0.06)
            vertical = np.abs(py - 0.59) - (0.59 - 0.06)
            foot = np.hypot(np.maximum(radial, 0), np.maximum(vertical, 0)) + np.minimum(np.maximum(radial, vertical), 0) - 0.06
            feet = np.minimum(feet, foot)
    torso = _blend(py, d, feet)
    face = face_np(px, py, pz)
    lean = np.clip(((px - 0.55) * 0.38 + (py - 1.25) * 0.65) / 0.5669, 0, 1)
    chest = (np.hypot(np.hypot(px - (0.55 + 0.38 * lean), py - (1.25 + 0.65 * lean)), pz / WIDTH) - (0.57 - 0.08 * lean)) * WIDTH
    base = _su(torso, face, 0.32 * FACE_BLEND)
    joined = _blend(py, _blend(py, torso, chest), face)
    return base + (joined - base) * CHEST_SLOPE


def surface_distance(px, py, pz):                    # replaces head.py's: eyes, ears, hood and straps sit on the body
    return float(body_np(np.float64(px), np.float64(py), np.float64(pz)))


# --- surface nets ------------------------------------------------------------------------------------------------
def surface_nets(f, lo, hi, h):
    xs, ys, zs = (np.arange(lo[i], hi[i] + h, h) for i in range(3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    F = f(X, Y, Z)
    nx, ny, nz = F.shape
    inside = F < 0
    cs = (nx - 1, ny - 1, nz - 1)
    sl = lambda o: (slice(o[0], nx - 1 + o[0]), slice(o[1], ny - 1 + o[1]), slice(o[2], nz - 1 + o[2]))
    offs = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    total = sum(inside[sl(o)].astype(int) for o in offs)
    active = (total > 0) & (total < 8)
    idx = -np.ones(cs, dtype=np.int64)
    idx[active] = np.arange(int(active.sum()))
    acc, cnt = np.zeros(cs + (3,)), np.zeros(cs)
    pts = np.stack([X, Y, Z], axis=-1)
    for a in offs:
        for b in offs:
            if sum(abs(a[i] - b[i]) for i in range(3)) != 1 or a > b: continue
            Fa, Fb = F[sl(a)], F[sl(b)]
            cross = ((Fa < 0) != (Fb < 0)) & active
            t = np.where(cross, Fa / np.where(Fa == Fb, 1, Fa - Fb), 0)
            pa, pb = pts[sl(a)], pts[sl(b)]
            acc += np.where(cross[..., None], pa + (pb - pa) * t[..., None], 0)
            cnt += cross
    verts = acc[active] / cnt[active][:, None]

    quads = []

    def emit(Fa, Fb, c00, c10, c11, c01):
        m = (Fa < 0) != (Fb < 0)
        q = np.stack([c00[m], c10[m], c11[m], c01[m]], axis=1)
        flip = ~(Fa[m] < 0)
        q[flip] = q[flip][:, ::-1]
        quads.append(q)

    I = idx
    emit(F[:-1, 1:-1, 1:-1], F[1:, 1:-1, 1:-1], I[:, 0:ny - 2, 0:nz - 2], I[:, 1:ny - 1, 0:nz - 2], I[:, 1:ny - 1, 1:nz - 1], I[:, 0:ny - 2, 1:nz - 1])
    emit(F[1:-1, :-1, 1:-1], F[1:-1, 1:, 1:-1], I[0:nx - 2, :, 0:nz - 2], I[0:nx - 2, :, 1:nz - 1], I[1:nx - 1, :, 1:nz - 1], I[1:nx - 1, :, 0:nz - 2])
    emit(F[1:-1, 1:-1, :-1], F[1:-1, 1:-1, 1:], I[0:nx - 2, 0:ny - 2, :], I[1:nx - 1, 0:ny - 2, :], I[1:nx - 1, 1:ny - 1, :], I[0:nx - 2, 1:ny - 1, :])
    return verts, np.concatenate(quads)


def build_body():
    old = bpy.data.objects.get("horse_body")
    if old: bpy.data.objects.remove(old, do_unlink=True)
    verts, quads = surface_nets(body_np, (-1.75, -0.2, -1.0), (2.45, 3.1, 1.0), CELL)
    for _ in range(3):                               # settle each vertex onto the field's zero level
        e = 1e-3
        g = np.stack([body_np(verts[:, 0] + e, verts[:, 1], verts[:, 2]) - body_np(verts[:, 0] - e, verts[:, 1], verts[:, 2]),
                      body_np(verts[:, 0], verts[:, 1] + e, verts[:, 2]) - body_np(verts[:, 0], verts[:, 1] - e, verts[:, 2]),
                      body_np(verts[:, 0], verts[:, 1], verts[:, 2] + e) - body_np(verts[:, 0], verts[:, 1], verts[:, 2] - e)], axis=1) / (2 * e)
        d = body_np(verts[:, 0], verts[:, 1], verts[:, 2])
        verts = verts - g * (d / np.maximum((g * g).sum(axis=1), 1e-9))[:, None]
    blender_verts = [(float(x), float(-z), float(y)) for x, y, z in verts]     # (fwd, up, side) -> Blender
    me = bpy.data.meshes.new("horse_body")
    me.from_pydata(blender_verts, [], quads.tolist())
    me.update()
    obj = bpy.data.objects.new("horse_body", me)
    obj.data.materials.append(bpy.data.materials.get("MAT_coat") or material("MAT_coat", "#b8773c", 0.95, sheen=0.65, sheen_roughness=0.6, grain=0.25))
    obj.data.materials.append(material("MAT_hoof", "#29272a", 0.95, sheen=0.4, sheen_roughness=0.6, grain=0.2))
    for poly in me.polygons:                         # the app's socks: dark below 0.25 m (smoothstep 0.23 - 0.27)
        poly.material_index = 1 if poly.center.z < 0.25 else 0
    smooth(obj)
    link(obj)
    return obj


if ARGS.get("run_body") == "1":
    body = build_body()
    print("body tris:", sum(len(p.vertices) - 2 for p in body.data.polygons), "verts:", len(body.data.vertices))
