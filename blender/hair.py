"""Guide hairs for the plush horse's mane and tail (docs/STYLE_GUIDE.md).

The mane and tail are real hairs: tens of thousands of thin tubes that sway in a vertex shader (app/plushHorse.ts). They are
not meshes to model by hand, so Blender holds the GUIDES the app grows them around:

  horse_mane_guides        one polyline per lock, from the crest root outward (17 points), for a bare head
  horse_mane_guides_menko  the same, grown over the menko (locks that meet the hood lie on it)
  horse_tail_spine         the tail's centre line from the dock (37 points); the app wraps its hairs around it

They are loose-edge meshes with a per-vertex `_strand` index; points run root to tip. Edit them in Blender (move points, delete or
add locks) and re-export; the app scatters its hairs around whatever it finds.

Ported from growGuide(), the crest sampling in maneGeometry() and growStrand() in app/plushHorse.ts (the drawn outline, the ears,
the 'outside the shell' pushing), with the same SAVED_HEAD. Locks differ from the code's random ones because the code seeds
differently; the method is the same.

Library: horse.py calls build_hair(). Needs body.py and menko.py.
"""
import random as _random

MANE_SIDE = [
    (3.1, 0.24, 0.58), (3.04, 0.12, 0.63), (2.95, 0.04, 0.63), (2.9, -0.03, 0.62), (2.8, -0.09, 0.6),
    (2.7, -0.13, 0.62), (2.6, -0.16, 0.6), (2.5, -0.17, 0.52), (2.4, -0.2, 0.46), (2.3, -0.25, 0.41),
    (2.2, -0.34, 0.35), (2.1, -0.5, 0.27), (2.02, -0.52, 0.12),
]
MANE_BACK = [
    (3.12, -0.08, 0.12), (3.07, -0.22, 0.26), (3.01, -0.28, 0.32), (2.95, -0.33, 0.37), (2.65, -0.33, 0.37),
    (2.45, -0.31, 0.35), (2.35, -0.28, 0.32), (2.25, -0.24, 0.28), (2.19, -0.2, 0.24), (2.13, -0.12, 0.16), (2.03, -0.03, 0.07),
]
MANE_SPILL, MANE_SIDE_SPILL = 0.05, 0.2
MANE_LOCKS = int(ARGS.get("mane_locks", 150))
MANE_STEPS = 16
TAIL_ROOT = (-1.22, 1.38, 0.0)


def outline_span(table, y):
    if y > table[0][0] or y < table[-1][0]: return None
    for i in range(len(table) - 1):
        ya, a0, a1 = table[i]
        yb, b0, b1 = table[i + 1]
        if yb <= y <= ya:
            t = (ya - y) / (ya - yb)
            return (lerp(a0, b0, t), lerp(a1, b1, t))
    return None


def S_body(x, y, z): return float(body_np(np.float64(x), np.float64(y), np.float64(z)))
def S_trunk(x, y, z): return float(trunk_np(np.float64(x), np.float64(y), np.float64(z)))


def S_hood(x, y, z):
    """hoodDistance(): the cloth shell cut to the hood's outline, the eye holes and the head's volume."""
    return smooth_max(S_body(x, y, z) - HOOD_T, hood_outline(x, y, z), 0.035 * HEAD_S)


def S_ear(x, y, z):
    """The ears as thin ellipsoids, so hair parts around them (the same ellipsoids the code uses)."""
    hx, hy, hz = (x - HEAD_X) / HEAD_S, (y - HEAD_Y) / HEAD_S, z / HEAD_S
    s = -1 if hz < 0 else 1
    a = -s * 0.2
    ly, lz = hy - 0.58, hz - s * 0.3 * WIDTH
    ry, rz = ly * math.cos(a) - lz * math.sin(a), ly * math.sin(a) + lz * math.cos(a)
    return (math.sqrt(((hx + 0.06) / 0.07) ** 2 + ((ry - 0.33) / 0.36) ** 2 + (rz / 0.17) ** 2) - 1) * 0.07 * HEAD_S


def gradient(f, p, e=0.01):
    g = Vector((f(p.x + e, p.y, p.z) - f(p.x - e, p.y, p.z), f(p.x, p.y + e, p.z) - f(p.x, p.y - e, p.z), f(p.x, p.y, p.z + e) - f(p.x, p.y, p.z - e)))
    return g.normalized() if g.length > 0 else g


def mane_guides(menko, count=MANE_LOCKS, seed=9157):
    rnd = _random.Random(seed)
    top, base = MANE_SIDE[0][0], 2.0
    skin = (lambda x, y, z: min(S_body(x, y, z), S_hood(x, y, z))) if menko else S_body
    solid = lambda x, y, z: min(skin(x, y, z), S_ear(x, y, z))

    def inside(p, tol=0.0, sideways=None):
        sideways = tol if sideways is None else sideways
        sx, bz = outline_span(MANE_SIDE, p.y), outline_span(MANE_BACK, p.y)
        return bool(sx and bz and sx[0] - tol <= p.x <= sx[1] + tol and bz[0] - sideways <= p.z <= bz[1] + sideways)

    def push_out(p, clearance):
        for _ in range(4):
            d = solid(p.x, p.y, p.z)
            if d >= clearance: return
            p += gradient(solid, p) * (clearance - d)

    def surface_below(x, z):
        lo, hi = 1.0, top + 0.3
        if skin(x, hi, z) < 0 or skin(x, lo, z) > 0: return None
        for _ in range(22):
            y = (lo + hi) / 2
            if skin(x, y, z) < 0: lo = y
            else: hi = y
        return Vector((x, (lo + hi) / 2, z))

    backward, sideways_v = Vector((-1, 0, 0)), Vector((0, 0, 1))

    def grow(root, lean, fan, curl, drape):
        n = gradient(skin, root)
        if drape:
            v = (n * 0.35 + backward * 0.4 + sideways_v * (-1 if fan < 0 else 1)).normalized()
            bend, step = Vector((-0.25, -1, 0)) * 0.22, 0.45 / MANE_STEPS
        else:
            v = (n + backward * lean + sideways_v * fan).normalized()
            bend, step = Vector((-0.8, -0.45, 0)) * curl, 0.8 / MANE_STEPS
        pts, p = [root.copy()], root.copy()
        for _ in range(MANE_STEPS):
            v += bend
            if not drape: v.z *= 0.97
            v.normalize()
            p += v * step
            push_out(p, 0.012)
            if not drape:
                bz = outline_span(MANE_BACK, p.y)
                if bz: p.z = clamp(p.z, bz[0], bz[1])
            if not inside(p, MANE_SPILL if drape else 0.012, MANE_SIDE_SPILL if drape else 0.012): break
            pts.append(p.copy())
        return pts

    # The crest climbs steeply from the withers to the poll: space roots evenly along it, not along x.
    crest_x, crest_len, prev, total = [], [], None, 0.0
    for i in range(111):
        at = surface_below(-0.5 + i * 0.01, 0)
        if not at: continue
        if prev is not None: total += (at - prev).length
        prev = at
        crest_x.append(at.x); crest_len.append(total)

    def crest_at(u):
        target = u * crest_len[-1]
        i = 1
        while i < len(crest_len) - 1 and crest_len[i] < target: i += 1
        t = (target - crest_len[i - 1]) / max(crest_len[i] - crest_len[i - 1], 1e-9)
        return lerp(crest_x[i - 1], crest_x[i], t)

    guides = []
    for _ in range(6000):
        if len(guides) >= count: break
        x = crest_at(rnd.random())
        poll = smoothstep(x, 0.15, 0.5)
        half = lerp(0.15, 0.2, poll)
        across = rnd.random() * 2 - 1
        surface = surface_below(x, across * half)
        if surface is None or not inside(surface, 0.03): continue
        root = surface - gradient(skin, surface) * 0.01
        drape = x > 0.05 and abs(across) > 0.6 and rnd.random() < 0.3
        fan = (across + (rnd.random() - 0.5) * 0.3) * lerp(0.9, 0.7, poll)
        sweep = lerp(0.35 + rnd.random() * 0.9, rnd.random() * 0.55, poll)
        curl = lerp(0.05, 0.008, poll) * (0.6 + rnd.random() * 0.8)
        guide = []
        for attempt in range(5):
            if len(guide) >= 6: break
            guide = grow(root, sweep * (1 - attempt / 5), fan, curl, drape)
        if len(guide) < 4: continue
        guides.append(guide)
    return guides


def tail_spine():
    """growStrand() for the tail's centre line, in the horse frame (the dock sits at TAIL_ROOT)."""
    solid = lambda x, y, z: S_trunk(x + TAIL_ROOT[0], y + TAIL_ROOT[1], z + TAIL_ROOT[2])
    gravity = Vector((-0.04, -1, 0)).normalized()
    droop, clearance, length, segments = 0.085, 0.12, 1.5, 36
    p, v = Vector((-0.15, 0, 0)), Vector((-1, 0.4, 0)).normalized()
    points = [p.copy()]
    for i in range(1, segments + 1):
        prev = p.copy()
        v = (v + gravity * droop).normalized()
        p = p + v * (length / segments)
        gap = clearance * min(1, i / 2)
        touched = False
        for _ in range(2):
            d = solid(p.x, p.y, p.z)
            if d >= gap: break
            n = gradient(solid, p)
            if n.length == 0: break
            p = p + n * (gap - d)
            touched = True
        if touched: v = (p - prev).normalized()
        points.append(p.copy())
    return [pt + Vector(TAIL_ROOT) for pt in points]


def polylines_object(name, strands):
    verts, edges, strand_id = [], [], []
    for k, pts in enumerate(strands):
        first = len(verts)
        for pt in pts:
            verts.append(tuple(to_blender(pt.x, pt.y, pt.z)))
            strand_id.append(float(k))
        edges.extend((first + i, first + i + 1) for i in range(len(pts) - 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, edges, [])
    attr = me.attributes.new("_strand", "FLOAT", "POINT")
    attr.data.foreach_set("value", strand_id)
    me.update()
    obj = bpy.data.objects.new(name, me)
    link(obj)
    return obj


def build_hair():
    for o in [o for o in bpy.data.objects if o.name.startswith(("horse_mane_guides", "horse_tail_spine"))]:
        bpy.data.objects.remove(o, do_unlink=True)
    bare, hooded, spine = mane_guides(False), mane_guides(True), tail_spine()
    polylines_object("horse_mane_guides", bare)
    polylines_object("horse_mane_guides_menko", hooded)
    polylines_object("horse_tail_spine", [spine])
    return {"mane_bare": (len(bare), sum(len(g) for g in bare)), "mane_menko": (len(hooded), sum(len(g) for g in hooded)), "tail_spine_points": len(spine)}


if ARGS.get("run_hair") == "1":
    print(build_hair())
