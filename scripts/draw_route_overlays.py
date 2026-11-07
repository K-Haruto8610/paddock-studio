"""Draw each GⅠ route (outputs/route-check/routes.json, from export_route_paths.mjs)
on the JRA plan image: route path (red), start mark from the plan label (magenta
ring), route start (green dot), chute control points (yellow). Output:
outputs/route-check/<venue>-<surface><distance>.png (git-ignored).
"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/route-check'
PLANS = {'05': ('tokyo', 'tokyo/plan.gif'), '06': ('nakayama', 'nakayama/pic_course_heimenzu.gif'),
         '07': ('chukyo', 'chukyo/pic_course_heimenzu.gif'), '08': ('kyoto', 'kyoto/pic_course_heimenzu.gif'),
         '09': ('hanshin', 'hanshin/pic_course_heimenzu.gif')}
S = 3


def main():
    routes = json.loads((OUT / 'routes.json').read_text(encoding='utf-8'))
    for r in routes:
        slug, f = PLANS[r['venueCode']]
        im = Image.open(ROOT / 'references/jra' / f).convert('RGB')
        im = im.resize((im.width * S, im.height * S), Image.NEAREST)
        d = ImageDraw.Draw(im)
        pts = [(x * S, y * S) for x, y in r['path']]
        d.line(pts, fill=(220, 0, 0), width=3)
        for x, y in r['chute']:
            d.ellipse([x * S - 4, y * S - 4, x * S + 4, y * S + 4], outline=(255, 200, 0), width=2)
        mx, my = r['mark'][0] * S, r['mark'][1] * S
        d.ellipse([mx - 9, my - 9, mx + 9, my + 9], outline=(255, 0, 255), width=3)
        sx, sy = pts[0]
        d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(0, 200, 0))
        d.text((8, 8), f"{r['name']} {r['surface']}{r['distance']}m [{r['source']}] {r['plan']}", fill=(0, 0, 0))
        name = f"{slug}-{'dirt' if r['surface'] == 'ダート' else 'turf'}{r['distance']}{'-outer' if r['kind'] == 'outer' else ''}.png"
        im.save(OUT / name)
        print(name, r['source'], r['markError'])


if __name__ == '__main__':
    main()
