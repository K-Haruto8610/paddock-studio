"""Menko (racing hood) and bridle for the plush horse head (docs/STYLE_GUIDE.md). Builds head.py first, then adds:

  horse_menko      the hood: the head's surface grown by the cloth thickness and cut to the hood's outline, with eye
                   holes and a closed rim. Slots: MAT_menko (the silks' colour, set at runtime)
  horse_bridle_noseband / _headstall / _browband   leather straps lying on the head and hood. Slots: MAT_leather
  horse_bridle_joint_*   the flat leather patches where straps meet.

Ported from hoodDistance() and addBridle() in app/plushHorse.ts: same outline, same strap layout (noseband ring around the
muzzle, cheek pieces up to a crown ring behind the ears, browband in front of them), in the head's own frame.

    python blender/run_in_blender.py blender/horse.py [hood_cage=22] [bridle=1] [cell=0.07]
(menko.py is a library: horse.py builds the body, the head parts, then this.)
"""
if "head_shell" not in globals():
    ARGS["lib"] = "1"
    exec(open(ARGS["root"] + "/blender/head.py", encoding="utf-8").read())

HOOD_T = 0.035                        # cloth thickness, metres (HOOD_THICKNESS in the code)
HOOD_CAGE = int(ARGS.get("hood_cage", 22))
STRAP_W = 0.062 * HEAD_S              # strap width, metres
STRAP_H = 0.009 * HEAD_S              # strap thickness
STRAP_LIFT = 0.010                    # strap centre above the surface (the app adds its fibre length on top)


def smooth_max(a, b, k): return -smooth_union(-a, -b, k)


def hood_outline(px, py, pz):
    """Negative where the hood cloth is (hoodDistance's outline and eye hole), in world metres."""
    ux, uy = (px - HEAD_X) / HEAD_S, (py - HEAD_Y) / HEAD_S
    k = 0.035
    border = ((ux - 0.5) - (uy + 0.39) * 0.76) / math.hypot(1, 0.76)
    noseband = ((ux - 0.778) - (uy + 0.15) * 0.34) / math.hypot(1, 0.34)
    front = smooth_max(border, noseband, k)
    outline = smooth_max(smooth_max(front, -0.42 - uy, k), -0.5 - ux, k) * HEAD_S
    eye_r = 0.13 * 1.24 + 0.035
    hole = (math.hypot((ux - 0.47) / 1.05, uy - 0.26) - eye_r) * HEAD_S
    cut = smooth_max(outline, -hole, k * HEAD_S)
    return smooth_max(cut, face_distance(px, py, pz) - 0.07, k * HEAD_S)     # only where the head is, not the neck


def sdist(p):
    """Distance to the surface (the body's, when body.py has replaced surface_distance) with the hood on it."""
    cover = clamp(-hood_outline(*p) / 0.03, 0, 1)
    return surface_distance(*p) - HOOD_T * cover


def grad(f, p, e=0.002):
    return Vector((f((p[0] + e, p[1], p[2])) - f((p[0] - e, p[1], p[2])),
                   f((p[0], p[1] + e, p[2])) - f((p[0], p[1] - e, p[2])),
                   f((p[0], p[1], p[2] + e)) - f((p[0], p[1], p[2] - e))))


def unit(v):
    return v.normalized() if v.length > 0 else Vector((0, 1, 0))


# World frame used by the maths below: (forward, up, sideways = +z the horse's right). to_blender() converts at the end.
def surface_point(d):
    """Surface along direction d from the centre of the head."""
    d = Vector(d).normalized()
    cx, cy = CENTER
    lo, hi = 0.0, 2.2
    for _ in range(36):
        mid = (lo + hi) / 2
        if surface_distance(cx + d.x * mid, cy + d.y * mid, d.z * mid) < 0: lo = mid
        else: hi = mid
    r = (lo + hi) / 2
    return Vector((cx + d.x * r, cy + d.y * r, d.z * r))


def hood_mesh(pivot):
    n = HOOD_CAGE
    cx, cy = CENTER
    pts, dist = {}, {}

    def vert(a, b, c):
        key = (a, b, c)
        if key not in pts:
            p = surface_point((a, b, c))
            pts[key] = p
            dist[key] = hood_outline(*p)
        return key

    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        coord = [0, 0, 0]
                        coord[axis] = sign * (n // 2)
                        coord[(axis + 1) % 3], coord[(axis + 2) % 3] = i + di - n // 2, j + dj - n // 2
                        quad.append(vert(*coord))
                    faces.append(quad)
    kept = [q for q in faces if sum(1 for k in q if dist[k] < 0) >= 2]

    # Snap each outside corner of a kept quad to where the outline crosses the edge towards an inside neighbour.
    snapped = {}
    for q in kept:
        for idx, k in enumerate(q):
            if dist[k] < 0 or k in snapped: continue
            best, to = None, None
            for nb in (q[(idx - 1) % 4], q[(idx + 1) % 4]):
                if dist[nb] < 0:
                    t = dist[k] / (dist[k] - dist[nb])
                    if best is None or t < best: best, to = t, nb
            if best is not None:
                snapped[k] = surface_point(pts[k].lerp(pts[to], best) - Vector((cx, cy, 0)))
    pos = {k: snapped.get(k, pts[k]) for q in kept for k in q}

    bm = bmesh.new()
    outer, inner = {}, {}
    for k, p in pos.items():
        nrm = unit(grad(lambda v: surface_distance(*v), p))
        outer[k] = bm.verts.new(to_blender(*(p + nrm * HOOD_T)) - pivot)
        inner[k] = bm.verts.new(to_blender(*(p - nrm * 0.004)) - pivot)
    for q in kept:
        try: bm.faces.new([outer[k] for k in q])
        except ValueError: pass
    key_of = {v: k for k, v in outer.items()}
    for e in list(bm.edges):                              # rim: a wall from the cloth's edge down to the head
        if len(e.link_faces) == 1 and e.verts[0] in key_of and e.verts[1] in key_of:
            a, b = key_of[e.verts[0]], key_of[e.verts[1]]
            try: bm.faces.new((outer[a], outer[b], inner[b], inner[a]))
            except ValueError: pass
    for v in list(inner.values()):                        # inner verts the rim does not use are dropped
        if not v.link_faces: bm.verts.remove(v)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # make sure the outer cloth faces outward
    f0 = max(bm.faces, key=lambda f: f.calc_center_median().length)
    centre = to_blender(cx, cy, 0) - pivot
    if f0.normal.dot(f0.calc_center_median() - centre) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    me = bpy.data.meshes.new("horse_menko")
    bm.to_mesh(me)
    bm.free()
    return me


# --- straps -----------------------------------------------------------------------------------------------------
def head_to_world(x, y, z): return Vector((HEAD_X + x * HEAD_S, HEAD_Y + y * HEAD_S, z * HEAD_S))


def normal_at(p): return unit(grad(sdist, p))


def surface_along(origin, d, a, b):
    lo, hi = a, b
    for _ in range(26):
        mid = (lo + hi) / 2
        if sdist(origin + d * mid) < 0: lo = mid
        else: hi = mid
    return origin + d * ((lo + hi) / 2)


def lifted(p): return p + normal_at(p) * STRAP_LIFT


def ring(center, up, a, b, steps):
    side = Vector((0, 0, 1))
    out = []
    for i in range(steps + 1):
        phi = lerp(a, b, i / steps)
        d = up * math.cos(phi) + side * math.sin(phi)
        out.append(lifted(surface_along(center, d, 0, HEAD_S)))
    return out


def straight(a, b, steps):
    n = unit(normal_at(a) + normal_at(b))
    pts = [a.copy()]
    for i in range(1, steps):
        chord = a.lerp(b, i / steps)
        pts.append(lifted(surface_along(chord, n, -0.5 * HEAD_S, 0.3 * HEAD_S)))
    pts.append(b.copy())
    return pts


def fair(points, closed, pinned=()):
    for _ in range(6):
        copy = [p.copy() for p in points]
        for i in range(len(points)):
            if not closed and i in (0, len(points) - 1): continue
            if i in pinned: continue
            prev, nxt = copy[(i - 1) % len(points)], copy[(i + 1) % len(points)]
            points[i] = points[i].lerp((prev + nxt) / 2, 0.5)
        for i, p in enumerate(points):
            gap = sdist(p) - STRAP_LIFT
            if gap < 0: points[i] = p + normal_at(p) * -gap
    return points


def strap_mesh(name, points, closed, pivot):
    w, h, bev = STRAP_W / 2, STRAP_H / 2, 0.003
    section = [(-w + bev, h), (w - bev, h), (w, h - bev), (w, -h + bev), (w - bev, -h), (-w + bev, -h), (-w, -h + bev), (-w, h - bev)]
    bm = bmesh.new()
    rows = []
    cnt = len(points)
    for i in range(cnt):
        a = points[(i - 1) % cnt] if closed or i > 0 else points[i]
        b = points[(i + 1) % cnt] if closed or i < cnt - 1 else points[i]
        tangent = unit(b - a)
        c = points[i]
        outward = normal_at(c)
        outward = unit(outward - tangent * outward.dot(tangent))
        across = unit(tangent.cross(outward))
        rows.append([bm.verts.new(to_blender(*(c + across * x + outward * y)) - pivot) for x, y in section])
    m = len(section)
    for i in range(cnt if closed else cnt - 1):
        r0, r1 = rows[i], rows[(i + 1) % cnt]
        for e in range(m):
            bm.faces.new((r0[e], r0[(e + 1) % m], r1[(e + 1) % m], r1[e]))
    if not closed:
        for r in (rows[0], rows[-1]):
            bm.faces.new(r)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def tris(obj): return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def build_menko_and_bridle():
    pivot = to_blender(PIVOT[0], PIVOT[1], 0)
    head = bpy.data.objects["horse_head"]
    menko_mat = material("MAT_menko", "#327ca6", 0.95, sheen=0.5, sheen_roughness=0.6, grain=0.25)
    for o in bpy.data.objects:
        if o.name.startswith("horse_ear_"):
            o.data.materials.clear()
            o.data.materials.append(menko_mat)                     # the hood's ears are hood cloth

    hood = bpy.data.objects.new("horse_menko", hood_mesh(pivot))
    hood.location = pivot
    hood.data.materials.append(menko_mat)
    smooth(hood)
    link(hood, head)
    hood.matrix_parent_inverse = Matrix.Translation(-pivot)
    info = {"menko_tris": tris(hood)}

    if ARGS.get("bridle", "1") == "1":
        leather = material("MAT_leather", "#5b3826", 0.5, sheen=0.0, grain=0.1, coat=0.3)
        nose_up = unit(Vector((0.34, 1, 0)))
        nose = fair(ring(head_to_world(0.778, -0.15, 0), nose_up, 0, math.pi * 2, 32)[:-1], True)
        near = lambda pts, t: min(pts, key=lambda p: (p - t).length)
        nose_r = near(nose, head_to_world(0.74, -0.2, 0.6)).copy()
        nose_l = near(nose, head_to_world(0.74, -0.2, -0.6)).copy()
        crown = ring(head_to_world(-0.2, 0.05, 0), unit(Vector((-0.25, 1, 0))), -1.75, 1.75, 24)
        poll_l, poll_r = crown[0], crown[-1]
        cheek_l = straight(nose_l, poll_l, 10)
        cheek_r = straight(poll_r, nose_r, 10)
        headstall = cheek_l + crown[1:-1] + cheek_r
        corners = [len(cheek_l) - 1, len(cheek_l) + len(crown) - 2]
        fair(headstall, False, corners)
        corner = headstall[corners[1]]
        top = lifted(surface_along(head_to_world(0.08, 0.3, 0), Vector((0, 1, 0)), 0, HEAD_S))
        center = (corner + top) / 2
        center.z = 0
        up = unit(top - center)
        off = corner - center
        reach = math.atan2(off.z, off.dot(up))
        brow = ring(center, up, -reach, reach, 20)
        brow[0] = headstall[corners[0]].copy()
        brow[-1] = corner.copy()
        brow = fair(brow, False)
        for nm, pts, closed in (("noseband", nose, True), ("headstall", headstall, False), ("browband", brow, False)):
            o = bpy.data.objects.new("horse_bridle_" + nm, strap_mesh("horse_bridle_" + nm, pts, closed, pivot))
            o.location = pivot
            o.data.materials.append(leather)
            smooth(o)
            link(o, head)
            o.matrix_parent_inverse = Matrix.Translation(-pivot)
            info[nm + "_tris"] = tris(o)
        for i, p in enumerate((nose_l, nose_r, headstall[corners[0]], headstall[corners[1]])):
            nrm = normal_at(p)
            j = bpy.data.objects.new("horse_bridle_joint_%d" % i, ellipsoid("horse_bridle_joint_%d" % i, (STRAP_W * 0.54, STRAP_W * 0.54, STRAP_H * 0.65), 8, 6))
            j.rotation_mode = "QUATERNION"
            j.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector((nrm.x, -nrm.z, nrm.y)))   # to Blender axes
            j.location = to_blender(*(p + nrm * STRAP_H * 0.2))
            j.data.materials.append(leather)
            smooth(j)
            link(j, head)
            j.matrix_parent_inverse = Matrix.Translation(-pivot)
    return info


if ARGS.get("run_menko") == "1":                      # horse.py calls build_menko_and_bridle() itself
    print(build_menko_and_bridle())
