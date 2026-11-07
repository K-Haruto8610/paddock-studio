"""Digitize JRA official course plans and elevation sections into app/courseData.ts.

Inputs are the images saved under references/jra/<venue>/ (downloaded from
https://www.jra.go.jp/facilities/race/<venue>/course/index.html).

Outputs
- app/courseData.ts                generated rail pixels + elevation profiles
- public/courses/<venue>-<kind>.png one-lap crop of the official section graph,
                                    normalized so travel runs left -> right and
                                    the winning post is the right edge.
- outputs/course-digitize/*.png     debug overlays for visual review

The plans are schematic drawings, not survey coordinates. Each rail is later
calibrated to the official lap length in trackGeometry.ts.
Tokyo keeps its hand-digitized rails (see references/jra/tokyo/README.md) and is
only processed here for the seek-bar section crops.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'references/jra'
DEBUG = ROOT / 'outputs/course-digitize'
PUBLIC = ROOT / 'public/courses'

# goalX: x pixel of the winning post (▲ゴール) on the home straight.
# outer: screen-angle window (deg, atan2(dy,dx) with y down) where the outer course
#        leaves the inner course. Outside the window both courses share the rail.
# kinds: only the courses JRA flat GⅠ races are run on at that venue.
#   中山: 皐月賞・有馬記念・ホープフルS (内回り), スプリンターズS (外回り)
#   京都: 秋華賞 (内回り), 天皇賞(春)・菊花賞・エリザベス女王杯・マイルCS (外回り)
#   阪神: 大阪杯・宝塚記念 (内回り), 桜花賞・阪神JF・朝日杯FS (外回り)
#   中京: 高松宮記念 (芝), チャンピオンズC (ダート)
PLANS = {
    'nakayama': {'code': '06', 'left': False, 'goalX': 237, 'outer': (-178, -8), 'kinds': ['turf', 'outer']},
    'kyoto':    {'code': '08', 'left': False, 'goalX': 287, 'outer': (-75, 75), 'kinds': ['turf', 'outer']},
    'hanshin':  {'code': '09', 'left': False, 'goalX': 168, 'outer': (-80, 80), 'kinds': ['turf', 'outer']},
    'chukyo':   {'code': '07', 'left': True,  'goalX': 381, 'outer': None, 'kinds': ['turf', 'dirt']},
}

# Section graphs. top/bottom: metres at the top/bottom of the plot frame.
# goal: which frame edge is the winning post (remaining distance 0).
PROFILES = {
    ('tokyo', 'turf'):     {'file': 'tokyo/turf-profile.gif', 'lap': 2083.1, 'top': 4, 'bottom': -6, 'goal': 'right', 'frameY': (13.5, 127)},
    ('tokyo', 'dirt'):     {'file': 'tokyo/dirt-profile.gif', 'lap': 1899, 'top': 4, 'bottom': -6, 'goal': 'right', 'frameY': (15, 129)},
    ('nakayama', 'turf'):  {'file': 'nakayama/pic_course_turf.gif', 'lap': 1667.1, 'top': 4, 'bottom': -4, 'goal': 'left'},
    ('nakayama', 'outer'): {'file': 'nakayama/pic_course_turf2.gif', 'lap': 1839.7, 'top': 4, 'bottom': -4, 'goal': 'left'},
    ('kyoto', 'turf'):     {'file': 'kyoto/pic_course_shiba_in.gif', 'lap': 1782.8, 'top': 6, 'bottom': -2, 'goal': 'left'},
    ('kyoto', 'outer'):    {'file': 'kyoto/pic_course_shiba_out.gif', 'lap': 1894.3, 'top': 6, 'bottom': -2, 'goal': 'left'},
    ('hanshin', 'turf'):   {'file': 'hanshin/pic_course_turf.gif', 'lap': 1689, 'top': 2, 'bottom': -4, 'goal': 'left'},
    ('hanshin', 'outer'):  {'file': 'hanshin/pic_course_turf2.gif', 'lap': 2089, 'top': 2, 'bottom': -4, 'goal': 'left'},
    ('chukyo', 'turf'):    {'file': 'chukyo/pic_course_shiba.gif', 'lap': 1705.9, 'top': 2, 'bottom': -3, 'goal': 'right'},
    ('chukyo', 'dirt'):    {'file': 'chukyo/pic_course_durt.gif', 'lap': 1530, 'top': 2, 'bottom': -3, 'goal': 'right'},
}


def rgb(path):
    return np.array(Image.open(path).convert('RGB')).astype(int)


# ---------------------------------------------------------------- profiles
def profile_masks(a):
    mn, mx = a.min(axis=2), a.max(axis=2)
    light = mn >= 200
    neutral = (mx - mn) < 14
    fill = ~light & ~neutral
    return light, fill


def frame_of(a, cfg):
    light, fill = profile_masks(a)
    ys, xs = np.nonzero(fill)
    # The fill reaches the bottom of the frame over the whole lap.
    x0, x1 = xs.min(), xs.max()
    bottom = ys.max() + 1
    if 'frameY' in cfg:
        top, bottom = cfg['frameY']
    else:
        sky = light & ~(a.min(axis=2) >= 250)  # tinted background, not white
        sy, sx = np.nonzero(sky[:, x0:x1 + 1])
        top = sy.min() - 1
    return x0, x1 + 1, top, bottom


def digitize_profile(venue, kind, cfg):
    a = rgb(REF / cfg['file'])
    _, fill = profile_masks(a)
    x0, x1, top, bottom = frame_of(a, cfg)
    per_m = (bottom - top) / (cfg['top'] - cfg['bottom'])
    # Coloured horizontal gridlines are 1-2px tall; the fill body is not.
    solid = fill.copy()
    solid[:-3] &= fill[1:-2] & fill[2:-1] & fill[3:]
    tops = []
    for x in range(x0, x1):
        col = np.nonzero(solid[int(top):int(bottom) + 1, x])[0]
        tops.append(int(top) + col[0] if len(col) else None)
    # Vertical gridlines are 1px wide: a column median removes their spikes.
    valid = [t for t in tops if t is not None]
    tops = [t if t is not None else valid[0] for t in tops]
    tops = ndimage.median_filter(np.array(tops, dtype=float), size=5, mode='nearest')
    pts = []
    for x, y in zip(range(x0, x1), tops):
        h = cfg['top'] - (y - top) / per_m
        frac = (x + .5 - x0) / (x1 - x0)
        remaining = frac * cfg['lap'] if cfg['goal'] == 'left' else (1 - frac) * cfg['lap']
        pts.append((remaining, h))
    pts.sort()
    # Frame border columns are drawn lines, not terrain.
    pts = pts[2:-2]
    pts = [(0.0, pts[0][1])] + pts + [(cfg['lap'], pts[-1][1])]

    def rdp(seq, tol=.04):
        (xa, ya), (xb, yb) = seq[0], seq[-1]
        worst, idx = 0, 0
        for i in range(1, len(seq) - 1):
            x, y = seq[i]
            err = abs(ya + (yb - ya) * (x - xa) / max(xb - xa, 1e-9) - y)
            if err > worst:
                worst, idx = err, i
        if worst <= tol:
            return [seq[0], seq[-1]]
        return rdp(seq[:idx + 1], tol)[:-1] + rdp(seq[idx:], tol)
    # JRA courses consist of level stretches joined by straight, even slopes, but
    # the graph's 1px steps (~0.07m) read as stairs, which the 3D exaggeration
    # turns into visible ledges. Smooth the column noise, keep only the slope
    # breaks, then snap near-level segments to exactly level.
    xs = np.array([x for x, _ in pts]); hs = np.array([h for _, h in pts])
    hs = ndimage.gaussian_filter1d(hs, 3, mode='nearest')
    simplified = [list(q) for q in rdp(list(zip(xs, hs)), tol=.12)]
    LEVEL = .15  # metres: a segment rising less than this is drawn level
    for _ in range(3):
        for i in range(len(simplified) - 1):
            a_, b_ = simplified[i], simplified[i + 1]
            if 0 < abs(b_[1] - a_[1]) < LEVEL:
                m = (a_[1] + b_[1]) / 2
                a_[1] = b_[1] = m
    # Merge collinear neighbours left after snapping.
    merged = [simplified[0]]
    for i in range(1, len(simplified) - 1):
        (xa, ya), (xb, yb), (xc, yc) = merged[-1], simplified[i], simplified[i + 1]
        if abs(ya + (yc - ya) * (xb - xa) / max(xc - xa, 1e-9) - yb) > 1e-6:
            merged.append(simplified[i])
    merged.append(simplified[-1])
    # The lap is a loop: the winning post height must match at both ends.
    merged[-1][1] = merged[0][1]
    simplified = [tuple(q) for q in merged]
    # Seek-bar crop: one lap, travel left->right, goal at the right edge.
    crop = Image.open(REF / cfg['file']).convert('RGB').crop((x0, int(round(top)), x1, int(round(bottom))))
    if cfg['goal'] == 'left':
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    crop.save(PUBLIC / f'{venue}-{kind}.png', optimize=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    dbg = Image.open(REF / cfg['file']).convert('RGB').resize((a.shape[1] * 2, a.shape[0] * 2), Image.NEAREST)
    dd = ImageDraw.Draw(dbg)
    to_x = (lambda r: x0 + r / cfg['lap'] * (x1 - x0)) if cfg['goal'] == 'left' else (lambda r: x1 - r / cfg['lap'] * (x1 - x0))
    dd.line([(to_x(r) * 2, (top + (cfg['top'] - h) * per_m) * 2) for r, h in simplified], fill=(255, 0, 0), width=2)
    dd.rectangle([x0 * 2, top * 2, x1 * 2, bottom * 2], outline=(0, 0, 255))
    dbg.save(DEBUG / f'{venue}-{kind}-profile.png')
    hs = [h for _, h in simplified]
    print(f'profile {venue}/{kind}: frame x={x0}..{x1} y={top}..{bottom} points={len(simplified)} range={min(hs):.2f}..{max(hs):.2f}m')
    return [[round(r, 1), round(h, 2)] for r, h in simplified]


# ---------------------------------------------------------------- plans
def plan_masks(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    dark = a.max(axis=2) < 110
    # Replace text/arrows with the nearest non-dark colour before classification.
    _, (iy, ix) = ndimage.distance_transform_edt(dark, return_indices=True)
    a = a[iy, ix]
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    green = (g > 170) & (r < 150) & (b < 150)
    orange = (r > 200) & (g > 140) & (g < 190) & (b < 120)
    return a, green, orange


def ray_samples(center, theta, rmax, step=.25):
    rs = np.arange(0, rmax, step)
    return rs, center[0] + rs * math.cos(theta), center[1] + rs * math.sin(theta)


def runs(mask_line, rs, max_gap):
    out, start, last = [], None, None
    for i, v in enumerate(mask_line):
        if v:
            if start is None:
                start = i
            elif rs[i] - rs[last] > max_gap:
                out.append((start, last)); start = i
            last = i
    if start is not None:
        out.append((start, last))
    return out


def smooth_closed(points, sigma_px, count=180):
    """Resample a closed line by arc length from points[0] and low-pass it.

    Radial scanning leaves pixel steps and small kinks (text, ticks, where the
    outer course leaves the inner one). In 3D the lanes are offset up to ~35m
    along the rail normal, which magnifies any wobble of the tangent, so the
    rail is Gaussian-smoothed along its length before export.
    """
    P = np.asarray(points, dtype=float)
    seg = np.hypot(*(np.roll(P, -1, axis=0) - P).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    dense = 1440
    t = np.linspace(0, total, dense, endpoint=False)
    closed = np.vstack([P, P[:1]])
    xs = np.interp(t, cum, closed[:, 0])
    ys = np.interp(t, cum, closed[:, 1])
    sig = sigma_px / (total / dense)
    xs = ndimage.gaussian_filter1d(xs, sig, mode='wrap')
    ys = ndimage.gaussian_filter1d(ys, sig, mode='wrap')
    step = dense // count
    return list(zip(xs[::step], ys[::step]))


def circular_median(v, k):
    n = len(v); half = k // 2
    return np.array([np.median([v[(i + j) % n] for j in range(-half, half + 1)]) for i in range(n)])


def digitize_plan(venue, cfg):
    path = REF / venue / 'pic_course_heimenzu.gif'
    a0 = rgb(path)
    a, green, orange = plan_masks(a0)
    H, W = green.shape
    # Non-green regions connected to the border are "outside"; enclosed gaps
    # between green bands separate the inner and outer turf courses.
    lab, _ = ndimage.label(~green)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    outside = np.isin(lab, list(border))
    # Keep the dirt oval (tick lines may split it into pieces) but drop legend
    # swatches; the centre is the middle of the oval's bounding box.
    olab, n = ndimage.label(orange)
    sizes = ndimage.sum(orange, olab, range(1, n + 1))
    orange = np.isin(olab, [i + 1 for i, v in enumerate(sizes) if v > 300])
    oy, ox = np.nonzero(orange)
    center = ((ox.min() + ox.max()) / 2, (oy.min() + oy.max()) / 2)
    rmax = math.hypot(W, H)
    N = 720
    thetas = np.linspace(-math.pi, math.pi, N, endpoint=False)
    dirt_in, turf_in, outer = [], [], []
    for th in thetas:
        rs, xs, ys = ray_samples(center, th, rmax)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        rs, xs, ys = rs[ok], xs[ok].astype(int), ys[ok].astype(int)
        o = runs(orange[ys, xs], rs, 3)
        main = max(o, key=lambda s: rs[s[1]] - rs[s[0]])
        di, do = rs[main[0]], rs[main[1]]
        gr = [s for s in runs(green[ys, xs], rs, 1.0) if rs[s[0]] >= do - 1]
        ti = rs[gr[0][0]] if gr else do
        oo = ti
        deg = math.degrees(th)
        if cfg['outer'] and cfg['outer'][0] <= deg <= cfg['outer'][1]:
            for prev, cur in zip(gr, gr[1:]):
                gap = slice(prev[1] + 1, cur[0])
                if gap.stop > gap.start and not outside[ys[gap], xs[gap]].any():
                    oo = rs[cur[0]]
        dirt_in.append(di); turf_in.append(ti); outer.append(oo)
    result = {}
    goal_theta = None
    for name, radii in [('turf', turf_in), ('outer', outer), ('dirt', dirt_in)]:
        if name not in cfg['kinds']:
            continue
        # Labels drawn on the outer band need a wider window than the clean inner rails.
        rr = circular_median(np.array(radii), 9 if name == 'turf' else 25)
        # Rail lines are smooth curves; a wide circular mean removes label/tick noise.
        k = 15
        rr = np.convolve(np.concatenate([rr[-(k // 2):], rr, rr[:k // 2]]), np.ones(k) / k, mode='valid')
        if name == 'turf':
            turf_rr = rr
        elif name == 'outer':
            # Where the courses share the rail the outer course must be the very same
            # line, otherwise the two ribbons overlap with a visible ledge. Model the
            # outer course as the inner rail plus a smooth outward bulge.
            bulge = ndimage.gaussian_filter1d(np.maximum(rr - turf_rr, 0), 6, mode='wrap')
            bulge[bulge < .3] = 0
            rr = turf_rr + ndimage.gaussian_filter1d(bulge, 8, mode='wrap')
        pts = [(center[0] + r * math.cos(t), center[1] + r * math.sin(t)) for r, t in zip(rr, thetas)]
        if goal_theta is None:
            # The winning post: the lowest (home-straight) rail point at x=goalX.
            bottom = [i for i, (x, y) in enumerate(pts) if y > center[1]]
            gi = min(bottom, key=lambda i: abs(pts[i][0] - cfg['goalX']))
            goal_theta = thetas[gi]
        # Order from the goal in the race direction. Screen y points down, so a
        # left-handed course runs with decreasing screen angle.
        gi = int(np.argmin(np.abs(np.angle(np.exp(1j * (thetas - goal_theta))))))
        step = -1 if cfg['left'] else 1
        ordered = [pts[(gi + step * k) % N] for k in range(N)]
        sampled = smooth_closed(ordered, 4)
        if name == 'dirt':
            # Distance labels sit on the dirt band and dent its inner edge. The band
            # is drawn with a constant width, so offset the clean turf-inner line
            # (= the dirt's outer edge) inwards by the band's median width instead.
            turf = np.array(result['turf'])
            width = float(np.median([np.min(np.hypot(*(turf - p).T)) for p in np.array(sampled)]))
            offset = []
            for i, p in enumerate(turf):
                tangent = turf[(i + 1) % len(turf)] - turf[i - 1]
                normal = np.array([-tangent[1], tangent[0]]) / np.hypot(*tangent)
                a, b = p + normal * width, p - normal * width
                offset.append(a if np.hypot(*(a - center)) < np.hypot(*(b - center)) else b)
            sampled = smooth_closed([tuple(q) for q in offset], 3)
            print(f'plan {venue}: dirt band width {width:.1f}px')
        result[name] = [[round(float(x), 2), round(float(y), 2)] for x, y in sampled]
    # Debug overlay
    DEBUG.mkdir(parents=True, exist_ok=True)
    im = Image.open(path).convert('RGB').resize((W * 2, H * 2), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for name, col in [('turf', (220, 0, 0)), ('outer', (160, 0, 200)), ('dirt', (0, 0, 220))]:
        if name in result:
            p = [(x * 2, y * 2) for x, y in result[name]]
            d.line(p + [p[0]], fill=col, width=2)
            d.ellipse([p[0][0] - 5, p[0][1] - 5, p[0][0] + 5, p[0][1] + 5], outline=col, width=2)
            d.ellipse([p[8][0] - 3, p[8][1] - 3, p[8][0] + 3, p[8][1] + 3], fill=col)
    im.save(DEBUG / f'{venue}-plan.png')
    print(f'plan {venue}: center=({center[0]:.1f},{center[1]:.1f}) size={W}x{H} rails={list(result)}')
    return {'size': [W, H], 'center': [round(center[0], 1), round(center[1], 1)], 'rails': result}


def main():
    data = {}
    for venue, cfg in PLANS.items():
        data[cfg['code']] = {'slug': venue, **digitize_plan(venue, cfg), 'profiles': {}}
    data.setdefault('05', {'slug': 'tokyo', 'profiles': {}})
    codes = {v['slug']: k for k, v in data.items()}
    for (venue, kind), cfg in PROFILES.items():
        data[codes[venue]]['profiles'][kind] = digitize_profile(venue, kind, cfg)
    body = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    (ROOT / 'app/courseData.ts').write_text(
        '// Generated by scripts/digitize_jra_courses.py from JRA official course plans and\n'
        '// elevation sections (references/jra). Schematic digitization, not survey data.\n'
        'export type Pixel=[number,number];\n'
        'export type VenueDrawing={slug:string;size?:[number,number];center?:[number,number];'
        'rails?:{turf:Pixel[];outer?:Pixel[];dirt?:Pixel[]};profiles:Record<string,[number,number][]>};\n'
        f'export const COURSE_DATA:Record<string,VenueDrawing>={body};\n', encoding='utf-8')
    print('wrote app/courseData.ts')


if __name__ == '__main__':
    main()
