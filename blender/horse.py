"""Builds the plush horse's body and head set in the open Blender (docs/STYLE_GUIDE.md).

    python blender/run_in_blender.py blender/horse.py [cell=0.07] [hood_cage=22] [bridle=1]

  horse_body   torso + neck + head + legs as one seamless shell                        (blender/body.py)
  horse_head   empty at the neck pivot (0.55, 1.6); ears, eyes, nostrils, menko and bridle hang on it  (head.py, menko.py)
  horse_saddle, horse_saddle_print_L/R   the saddle cloth and number panels                  (saddle.py)
  horse_mane_guides[_menko], horse_tail_spine   guide hairs the app grows the mane and tail around   (hair.py)
  seamguide_*  seams beyond the centre line and leg rings (the mouth), baked onto horse_body as _seam_at/_seam_along  (seams.py)
"""
ARGS["lib"], ARGS["unified"] = "1", "1"
for part in ("head.py", "body.py", "menko.py", "saddle.py", "hair.py", "seams.py"):
    exec(open(ARGS["root"] + "/blender/" + part, encoding="utf-8").read())

build_body()
build_seam_guides()
print(bake_seams())                       # the mouth seam onto the body
build()                                   # head group + ears, eyes, nostrils
info = build_menko_and_bridle()           # hood + bridle
print(build_saddle())                     # saddle cloth and its two number panels
print(build_hair())                       # mane and tail guides
tri = lambda o: sum(len(p.vertices) - 2 for p in o.data.polygons) if o.type == "MESH" else 0
counts = {o.name: tri(o) for o in bpy.data.objects if o.name.startswith("horse_")}
body_tris = counts["horse_body"]
print("body", body_tris, "| head parts", sum(v for k, v in counts.items() if k != "horse_body"), "| total", sum(counts.values()))
