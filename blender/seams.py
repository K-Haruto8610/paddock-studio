"""Sewn seams of the plush horse beyond the centre line and the leg rings (docs/STYLE_GUIDE.md: 縫い目をディテールとして入れる).

The app sews the centre line (z = 0) and the four leg rings itself, from formulas (stitchSeam in app/plushHorse.ts). Every
other seam is drawn here, in Blender, as a guide curve on the body, and baked onto `horse_body` as two vertex attributes the
app's stitch shader reads:

  seamguide_<name>   a poly curve lying on the body (Blender frame). Move its points in Blender to re-sew; add another
                     curve named seamguide_* to add a seam. Not exported itself (blender/export.py exports horse_* only).
  _seam_at           on horse_body: the nearest point on any guide, in the APP's frame (x forward, y up, z the horse's right),
                     so the fragment shader measures the distance to the seam as |rest - _seam_at| across each triangle.
  _seam_along        on horse_body: arc length along that guide, in metres, for the stitch dashes.

blender/export.py bakes again before writing the .glb, so an edited guide always reaches the app.

    python blender/run_in_blender.py blender/seams.py            # rebuild the guides and bake
"""
import math
import numpy as np

_STANDALONE = ARGS.get("lib") != "1"                 # read before the libraries below set lib=1
if "body_np" not in globals():
    ARGS["lib"], ARGS["unified"] = "1", "1"
    exec(open(ARGS["root"] + "/blender/head.py", encoding="utf-8").read())
    exec(open(ARGS["root"] + "/blender/body.py", encoding="utf-8").read())

SEAM_PREFIX = "seamguide_"


def _on_body(x, y, z):
    return float(body_np(np.float64(x), np.float64(y), np.float64(z)))


def _ray(x0, y, dx, dz):
    """Where a level ray from (x0, y, 0) in direction (dx, 0, dz) leaves the body (app frame)."""
    lo, hi = 0.0, 1.5
    for _ in range(40):
        m = (lo + hi) / 2
        if _on_body(x0 + dx * m, y, dz * m) < 0: lo = m
        else: hi = m
    return (x0 + dx * lo, y, dz * lo)


def mouth_points(n=61, x0=1.62, y0=1.37, lift=0.13, sweep=62):
    """The mouth: a smile sewn across the lower front of the muzzle, under the nostrils, from cheek to cheek. Its middle
    crosses the centre seam at the front; the corners turn up a little toward the cheeks."""
    pts = []
    for i in range(n):
        t = -1 + 2 * i / (n - 1)
        th = math.radians(sweep) * t
        pts.append(_ray(x0, y0 + lift * abs(t) ** 2.2, math.cos(th), math.sin(th)))
    return pts


GUIDES = {"mouth": mouth_points}


def build_seam_guides():
    col = bpy.data.collections.get("seams") or bpy.data.collections.new("seams")
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    thread = material("MAT_thread", "#efe6d2", 0.9, sheen=0.3)
    made = []
    for name, make in GUIDES.items():
        full = SEAM_PREFIX + name
        old = bpy.data.objects.get(full)
        if old:
            curve = old.data
            bpy.data.objects.remove(old, do_unlink=True)
            if curve.users == 0: bpy.data.curves.remove(curve)
        cu = bpy.data.curves.new(full, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.006                       # only so the guide shows in Blender; the app sews its own thread
        pts = make()
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, (x, y, z) in zip(sp.points, pts):
            p.co = (*to_blender(x, y, z), 1)
        cu.materials.append(thread)
        obj = bpy.data.objects.new(full, cu)
        col.objects.link(obj)
        made.append(full)
    return made


def _guide_polylines():
    """Every seamguide_* curve's points, world space, converted to the app frame."""
    lines = []
    for obj in bpy.data.objects:
        if not obj.name.startswith(SEAM_PREFIX) or obj.type != "CURVE": continue
        for sp in obj.data.splines:
            pts = [obj.matrix_world @ Vector(p.co[:3]) for p in sp.points] if sp.type == "POLY" else \
                  [obj.matrix_world @ p.co for p in sp.bezier_points]
            if len(pts) > 1:
                lines.append(np.array([(p.x, p.z, -p.y) for p in pts]))
    return lines


def bake_seams(body_name="horse_body"):
    """Writes _seam_at and _seam_along onto the body from the guides. With no guides, every vertex points 100 m below the
    ground, so nothing is sewn."""
    body = bpy.data.objects[body_name]
    me = body.data
    V = np.array([body.matrix_world @ v.co for v in me.vertices])
    V = np.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1)              # Blender -> app frame
    best = np.full(len(V), np.inf)
    at = np.tile([0.0, -100.0, 0.0], (len(V), 1))
    along = np.zeros(len(V))
    for line in _guide_polylines():
        a, b = line[:-1], line[1:]
        seg = b - a
        length = np.linalg.norm(seg, axis=1)
        start = np.concatenate([[0], np.cumsum(length)[:-1]])
        # nearest point on every segment for every vertex: (verts, segs)
        t = np.clip(np.einsum("vsk,sk->vs", V[:, None, :] - a[None], seg) / np.maximum(length ** 2, 1e-12), 0, 1)
        p = a[None] + t[..., None] * seg[None]
        d = np.linalg.norm(V[:, None, :] - p, axis=2)
        k = d.argmin(axis=1)
        dk = d[np.arange(len(V)), k]
        closer = dk < best
        best[closer] = dk[closer]
        at[closer] = p[np.arange(len(V)), k][closer]
        along[closer] = (start[k] + t[np.arange(len(V)), k] * length[k])[closer]
    for name in ("_seam_at", "_seam_along"):
        if name in me.attributes: me.attributes.remove(me.attributes[name])
    a_at = me.attributes.new("_seam_at", "FLOAT_VECTOR", "POINT")
    a_at.data.foreach_set("vector", at.astype(np.float32).ravel())
    a_along = me.attributes.new("_seam_along", "FLOAT", "POINT")
    a_along.data.foreach_set("value", along.astype(np.float32))
    me.update()
    near = int((best < 0.05).sum())
    return f"seams baked onto {body_name}: {len(_guide_polylines())} guide(s), {near} vertices within 5 cm"


if _STANDALONE:
    print("guides:", build_seam_guides())
    print(bake_seams())
