"""F1 (追加指示10, 2026-11-19): 枠の不利の条件付き化. Hypothesis: an outside gate does not cost every horse the same at the first corner;
horses that are quick away (they cut in) and horses that will be dropped back anyway (they tuck in) lose little, the horses in the middle lose most.

  python scripts/experiment_f1_frame_gate.py measure        # first, measure where a bias really exists (training window only)
  python scripts/experiment_f1_frame_gate.py select|report|decide   # then the feature experiment (scripts/exp_common.py)

Measurement (training rows before 2025-06-01 only; the selection and report windows are never looked at):
  res1 = r1 - (a + b * past_r1): where a horse finished at the first corner relative to where its own history says (positive = further back).
  style = the horse's usual first-corner position (early_wish): front <= .15, forward <= .40, mid <= .70, back > .70.
  gate = (number - 1) / (field - 1): inner third / middle third / outer third.
  A bias counts as real only if the outer-minus-inner difference has the same sign in two time halves (before / from 2023-09-01), |t| >= 3 over
  both, and it survives Benjamini-Hochberg (10%) over all cells tested. Nothing is turned into a feature because it is a story.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as xc  # noqa: E402
import pace_model as pm  # noqa: E402

SPLIT = '20230901'
STYLES = ['front', 'forward', 'mid', 'back']
BANDS = [(0, 1400, 'sprint'), (1401, 1800, 'mile'), (1801, 2200, 'middle'), (2201, 4000, 'long')]


def style_of(w):
    return pd.cut(w, [-1, .15, .40, .70, 2], labels=STYLES).astype(object)


def diff_test(x, col='res1', lo=1 / 3, hi=2 / 3):
    """Outer third minus inner third of `col` and its Welch t."""
    a, b = x.loc[x.gate_out >= hi, col].dropna(), x.loc[x.gate_out <= lo, col].dropna()
    if len(a) < 30 or len(b) < 30:
        return None
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return {'diff': float(a.mean() - b.mean()), 'se': float(se), 't': float((a.mean() - b.mean()) / se), 'nOuter': int(len(a)), 'nInner': int(len(b))}


def frame_for_measure():
    d = xc.v19_frame()
    t = d[(d.date < '20250601') & d.complete & d.r1.notna() & d.past_r1.notna() & (d.field >= 8)].copy()
    t['gate_out'] = (t.UMABAN - 1) / (t.field - 1)
    b, a = np.polyfit(t.past_r1, t.r1, 1)
    t['res1'] = t.r1 - (a + b * t.past_r1)
    t['style'] = style_of(t.early_wish)
    t['band'] = pd.cut(t.distance, [0, 1400, 1800, 2200, 4000], labels=[x[2] for x in BANDS]).astype(object)
    t['half'] = np.where(t.date < SPLIT, 'A', 'B')
    t['course'] = t.venue.astype(str) + '/' + t.surface.astype(str) + '/' + t.distance.astype(int).astype(str)
    t['group'] = t.venue.astype(str) + '/' + t.surface.astype(str) + '/' + t.band
    return t, (a, b)


def bh(pvals, q=0.10):
    """Benjamini-Hochberg: boolean mask of the discoveries."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ok = np.zeros(len(p), bool)
    k = np.nonzero(p[order] <= q * (np.arange(len(p)) + 1) / len(p))[0]
    if len(k):
        ok[order[:k.max() + 1]] = True
    return ok


def cell_table(t, keys, label):
    """Outer-minus-inner res1 per cell of `keys` x style, in both halves."""
    rows = []
    for key, x in t.groupby(keys + ['style']):
        r = diff_test(x)
        if r is None:
            continue
        ra, rb = diff_test(x[x.half == 'A']), diff_test(x[x.half == 'B'])
        row = {'cell': key if isinstance(key, tuple) else (key,), **r, 'diffA': ra['diff'] if ra else None, 'diffB': rb['diff'] if rb else None,
               'replicates': bool(ra and rb and np.sign(ra['diff']) == np.sign(rb['diff']) == np.sign(r['diff']))}
        row['p'] = float(2 * stats.norm.sf(abs(r['t'])))
        rows.append(row)
    ok = bh([r['p'] for r in rows]) if rows else []
    for r, o in zip(rows, ok):
        r['bh'] = bool(o)
        r['real'] = bool(o and r['replicates'] and abs(r['t']) >= 3)
    return {'level': label, 'cells': len(rows), 'real': sum(r['real'] for r in rows), 'rows': rows}


def cmd_measure(args):
    t, (a, b) = frame_for_measure()
    res = {'rows': int(len(t)), 'races': int(t.RACE_CODE.nunique()), 'calibration': {'a': a, 'b': b}, 'meanRes1': float(t.res1.mean())}
    # 1. overall: gate x style
    tab = {}
    for s in STYLES:
        x = t[t.style == s]
        tab[s] = {'n': int(len(x)), 'all': diff_test(x), 'A': diff_test(x[x.half == 'A']), 'B': diff_test(x[x.half == 'B']),
                  'lane': diff_test(x, 'l1'),
                  'byGate': {g: float(x[(x.gate_out > lo) & (x.gate_out <= hi)].res1.mean()) for g, lo, hi in
                             (('inner', -1, 1 / 3), ('middle', 1 / 3, 2 / 3), ('outer', 2 / 3, 2))}}
    res['overall'] = tab
    mid = tab['mid']['all']['diff']
    ends = np.mean([tab['front']['all']['diff'], tab['back']['all']['diff']])
    res['humpTest'] = {'mid': mid, 'meanOfFrontAndBack': float(ends), 'note': 'hypothesis: mid > front, back (outer-minus-inner res1)'}
    # slope-based test of the interaction (gate_out x style) on the whole training window, race-clustered
    res['interaction'] = interaction(t)
    # 2. by distance band x surface, by course group and by course
    res['byBand'] = cell_table(t.assign(band_s=t.surface.astype(str) + '/' + t.band), ['band_s'], 'surface/band')
    res['byGroup'] = cell_table(t, ['group'], 'venue/surface/band')
    res['byCourse'] = cell_table(t, ['course'], 'venue/surface/distance')
    # 3. first-round idea A: Chukyo dirt 1800, going x style x course
    ch = t[(t.venue == 7) & (t.surface == 1) & (t.distance == 1800)]
    res['chukyoDirt1800'] = {'rows': int(len(ch)), 'races': int(ch.RACE_CODE.nunique()),
                             'byStyle': {s: diff_test(ch[ch.style == s]) for s in STYLES}}
    gg = []
    for (sf, band, going, style), x in t.groupby(['surface', 'band', 'going', 'style']):
        gg.append({'cell': (int(sf), band, int(going), style), 'n': int(len(x)), 'meanRes1': float(x.res1.mean()),
                   'se': float(x.res1.std() / np.sqrt(len(x)))})
    res['goingStyleBand'] = [r for r in gg if r['n'] >= 100]
    out = xc.ROOT / 'outputs' / 'f1-gate'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'measure.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    print_summary(res)


def interaction(t):
    """OLS of res1 on gate_out, style dummies and gate_out x style, standard errors clustered by race; the interaction block is tested
    with a Wald statistic against the model without it."""
    x = t.dropna(subset=['style', 'res1']).copy()
    S = pd.get_dummies(x.style, dtype=float)[STYLES]
    g = x.gate_out.to_numpy()[:, None]
    base = np.column_stack([S.to_numpy(), g])
    inter = S.to_numpy()[:, 1:] * g
    X = np.column_stack([base, inter])
    y = x.res1.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    codes = pd.factorize(x.RACE_CODE)[0]
    XtXi = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    s = X * resid[:, None]
    by = pd.DataFrame(s).groupby(codes).sum().to_numpy()
    meat = by.T @ by
    V = XtXi @ meat @ XtXi
    k = base.shape[1]
    idx = list(range(k, X.shape[1]))
    wald = float(beta[idx] @ np.linalg.inv(V[np.ix_(idx, idx)]) @ beta[idx])
    slopes = {'front': float(beta[4])}
    for i, sname in enumerate(STYLES[1:]):
        slopes[sname] = float(beta[4] + beta[k + i])
    se_slope = {'front': float(np.sqrt(V[4, 4]))}
    for i, sname in enumerate(STYLES[1:]):
        se_slope[sname] = float(np.sqrt(V[4, 4] + V[k + i, k + i] + 2 * V[4, k + i]))
    return {'slopePerFullGateRange': slopes, 'se': se_slope, 'waldInteraction': wald, 'dof': len(idx), 'p': float(stats.chi2.sf(wald, len(idx))),
            'races': int(len(set(codes)))}


def print_summary(res):
    print(f'training rows {res["rows"]}, races {res["races"]}; mean res1 {res["meanRes1"]:+.4f}')
    print('outer-third minus inner-third of res1 (positive = the outer gate finishes further back), by style:')
    for s, v in res['overall'].items():
        a, b = v['A'], v['B']
        print(f'  {s:>8} n={v["n"]:>6}  all {v["all"]["diff"]:+.4f} (t {v["all"]["t"]:+.1f})  A {a["diff"]:+.4f}  B {b["diff"]:+.4f}  lane diff {v["lane"]["diff"]:+.3f}  '
              f'by gate {v["byGate"]}')
    print('hump test', res['humpTest'])
    print('interaction (OLS, race-clustered):', res['interaction'])
    for k in ('byBand', 'byGroup', 'byCourse'):
        r = res[k]
        print(f'{k}: {r["cells"]} cells, real {r["real"]}')
        for row in sorted(r['rows'], key=lambda z: -abs(z['t']))[:12]:
            print(f'   {row["cell"]}: {row["diff"]:+.3f} (t {row["t"]:+.1f}) A {row["diffA"]} B {row["diffB"]} rep {row["replicates"]} bh {row["bh"]} real {row["real"]}')
    print('chukyo dirt 1800:', res['chukyoDirt1800'])


# ------------------------------------------------------------------------------------------------ feature variants
# What the measurement supports (README F1): the outside-gate cost is concentrated in forward / mid runners (front and back runners show
# none; interaction p = 0.0009), it is small (about 0.03 normalised rank over the whole gate range) and only one course cell replicates
# (Chukyo turf, middle distance). So the features are general forms, not per-course rules:
#   hump    gate_out, gate_out x a bump around the usual forward / mid position, gate_out x early position
#   rivals  how many horses that usually run as far forward (or further) start inside / outside of this one
#   course  static course descriptors: home-straight length (JRA published figures, rounded to 1 m, entered by hand, see STRAIGHT_M),
#           the number of corners the course passes and its first recorded corner (from the label structure before 2025-06-01)
STRAIGHT_M = {  # (venue code, surface 0 turf / 1 dirt) -> home straight in metres; turf outer-course override in OUTER_STRAIGHT_M
    (1, 0): 266, (2, 0): 262, (3, 0): 292, (4, 0): 359, (5, 0): 526, (6, 0): 310, (7, 0): 413, (8, 0): 328, (9, 0): 357, (10, 0): 293,
    (1, 1): 264, (2, 1): 260, (3, 1): 296, (4, 1): 354, (5, 1): 502, (6, 1): 308, (7, 1): 411, (8, 1): 329, (9, 1): 353, (10, 1): 291}
OUTER_STRAIGHT_M = {4: 659, 8: 404, 9: 474}  # Niigata / Kyoto / Hanshin outer turf courses
OUTER_TRACKS = (12, 14, 16, 18, 20, 22)       # JV-Data TRACK_CODE of turf outer courses (right-hand 18 / 20 / 22, left-hand 12 / 14 / 16)


def build_hump(d):
    d['gate_out'] = (d.UMABAN - 1) / (d.field - 1).where(d.field > 1)
    w = d.early_wish
    d['gate_hump'] = d.gate_out * np.exp(-0.5 * ((w - 0.35) / 0.22) ** 2)
    d['gate_x_early'] = d.gate_out * w


def _rivals(x):
    n, num, w = len(x), x.UMABAN.to_numpy(float), x.early_wish.to_numpy(float)
    wf = np.where(np.isnan(w), 0.5, w)
    inside, outside = num[None, :] < num[:, None], num[None, :] > num[:, None]
    forward = wf[None, :] <= 0.40
    ahead = wf[None, :] <= wf[:, None]
    return pd.DataFrame({'n_front_in': (inside & forward).sum(1), 'n_front_out': (outside & forward).sum(1),
                         'n_ahead_in': (inside & ahead).sum(1), 'n_ahead_out': (outside & ahead).sum(1)}, index=x.index)


def build_rivals(d):
    r = d.groupby('RACE_CODE', group_keys=False).apply(_rivals)
    for c in r:
        d[c] = r[c].reindex(d.index)


def build_course(d):
    sets = pm.corner_sets(d[d.date < '20250601'])
    key = list(zip(d.venue, d.track, d.distance))
    cache = {}
    for k in set(key):
        cs = sets(int(k[0]), int(k[1]), int(k[2]))
        cache[k] = (len(cs), cs[0] if cs else np.nan)
    d['n_corners_c'] = [cache[k][0] for k in key]
    d['first_corner'] = [cache[k][1] for k in key]
    d['straight_m'] = [OUTER_STRAIGHT_M[v] if (sf == 0 and t in OUTER_TRACKS and v in OUTER_STRAIGHT_M) else STRAIGHT_M.get((int(v), int(sf)), np.nan)
                       for v, sf, t in zip(d.venue, d.surface, d.track)]


def build_all(d):
    build_hump(d)
    build_rivals(d)
    build_course(d)


HUMP = ['gate_out', 'gate_hump', 'gate_x_early']
RIVALS = ['n_front_in', 'n_front_out', 'n_ahead_in', 'n_ahead_out']
COURSE = ['n_corners_c', 'first_corner', 'straight_m']
VARIANTS = {'base': xc.Variant(),
            'hump': xc.Variant(HUMP, build_all, note='gate x forward/mid interaction'),
            'rivals': xc.Variant(RIVALS, build_all, note='front-style rivals inside / outside'),
            'course': xc.Variant(COURSE, build_all, note='static course descriptors'),
            'hump+rivals': xc.Variant(HUMP + RIVALS, build_all),
            'all': xc.Variant(HUMP + RIVALS + COURSE, build_all)}
FOCUS = (1, 2)  # the corners F1 is meant to move; the guards are last3, gaps and lanes (exp_common)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'measure':
        xc.setup()
        cmd_measure(None)
    else:
        xc.main(VARIANTS, FOCUS, 'f1-gate')
