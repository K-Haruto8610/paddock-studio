"""Experiment (追加指示5, 2026-11-18): the same code retrained on the expanded DB, scored on the SAME 273 races.

The owner re-imported mykeibadb from JV-Link setup data (2023 onward, 2026-11-18 02:00-04:30). Every model, feature and
tuning step is unchanged; only the training data grew. Because 2025 also gained races (complete named races 641 ->
885), the published pipeline would now validate on more than 273 races, so this script pins Sep-Dec 2025 to the race
codes v1.6 validated on (taken from the pre-import frame) and reruns the full pipeline on each frame:

  before        pre-import frame (pickle of pace_model's frame, 2026-11-17 23:54, identical to v1.6's 27468/19006 rows);
                must reproduce models/pace/report.json v1.6 (sanity check)
  after         expanded frame, full pipeline (contention / move / interval tuning on Jun-Aug 2025 re-run, as the code does)
  after-v16P    expanded frame, v1.6's chosen parameters and interval margins (no re-tuning): the training-data effect alone
  after-seed7   'after' with every LightGBM seed 7 instead of 42 (size of run-to-run noise on the expanded frame)

Also reports the distribution shift that comes with the data: the course norm of the early 3F (PACE_NORM_END, before
2025-06) per course, and the pace label of the 273 validation races.

Run:  python scripts/experiment_data_expansion.py --before outputs/data-expansion/frame_before_20261118_0126.pkl \
        --after outputs/data-expansion/frame_after.pkl --out outputs/data-expansion/result.json [--only after,...]

Result 2026-11-18 -> ADOPTED as v1.7 (no code change; README「データ拡充（追加指示5）」):
  Frames: before 27468 history / 19006 training rows (1426 complete named races), after 48729 / 46401 (3442).
  The 273 validation races are the same in both frames, with identical labels; only past-run features and the training
  data changed (mean prior starts 5.1 -> 7.1). 'before' reproduces v1.6's report on all 13 metrics to 4 decimals.
  Validation MAE before -> after (seed-7 difference): r1 0.2360->0.2304 (0.0025), r2 0.2415->0.2345 (0.0011),
  r3 0.2523->0.2477 (0.0014), r4 0.2621->0.2574 (0.0010); g1 3.833->3.408, g2 3.826->3.395, g3 2.676->2.586,
  g4 2.389->2.367 (0.04/0.06/0.002/0.15); l1 0.432->0.436, l2 0.416->0.389, l3 0.534->0.516, l4 0.587->0.594;
  last3 0.884->0.814 s (0.003). after-v16P (no re-tuning) is the same within 0.001 (r4 0.2568, last3 0.814), so the gain
  is the data. Re-tuning picks front 0.45 / hold 0.06 (v1.6: 0.25 / 0.03).
  Shift: course norms on all 94 courses from 2547 -> 2759 races, mean |shift| 0.06 s (max 0.95); validation pace mean
  +0.041 -> +0.027 s, H/M/S 32/27/41% -> 32/28/40%.

Interim 2026-11-18 (追加指示8, 2022 imported; 2021/2020 still to come) -> NOT adopted yet, decision deferred to the
3-year run. --pin keeps the v1.6 frame's 273 races while --before is v1.7's frame:
  python scripts/experiment_data_expansion.py --pin outputs/data-expansion/frame_before_20261118_0126.pkl \
    --before outputs/data-expansion/frame_after.pkl --after outputs/data-expansion-v18/frame_2022.pkl \
    --old-report models/pace/report.json --out outputs/data-expansion-v18/result.json --only after,after-seed7
  2022: 3331 races / 914 named / 888 complete (2025: 3348/915/885). Frame 48729/46401 -> 60299/58319 rows, complete
  named races 3442 -> 4330; the 273 races' labels unchanged. Re-tuning picks hold 0.03 (v1.7 0.06), front stays 0.45.
  v1.7 -> +2022, seed 42 (seed 7): r1 0.2304->0.2290 (0.2329->0.2275), r2 0.2345->0.2328 (0.2356->0.2318),
  r3 0.2477->0.2470 (0.2463->0.2458), r4 0.2574->0.2541 (0.2564->0.2543); g1 3.408->3.492 (3.450->3.415),
  g2 3.395->3.370, g3 2.586->2.493, g4 2.367->2.209 (2.215->2.204); l1 0.4361->0.4370, l4 0.5936->0.5941 (seed 7
  0.5704); last3 0.814->0.823 (0.817->0.819). With v1.7's parameters fixed: r1-r4 0.2300/0.2341/0.2481/0.2549.
  -> at seed 42 only r4 clears 0.002 (bar: 3 corners); seed 7 and the two-seed mean clear it at r1/r2/r4. Borderline.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
VAL = ('20250901', '20251231')
ORIG_EVAL = pm.evaluation_races
ORIG = {k: getattr(pm, k) for k in ('lgbm', 'pace_lgbm', 'move_lgbm')}
PINNED = None  # set of RACE_CODE the validation window is restricted to


def evaluation_races(models, d, start, end, codes=None):
    # The pin restricts the window's races, not the frame: pace_model's course corner sets (E0) need the history before
    # the window. With the legacy label rule this is the same as filtering the frame, as before.
    if PINNED is not None and (start, end) == VAL:
        codes = PINNED if codes is None else set(codes) & PINNED
    return ORIG_EVAL(models, d, start, end, codes=codes)


pm.evaluation_races = evaluation_races


def set_seed(seed):
    for k, f in ORIG.items():
        setattr(pm, k, (lambda f: lambda *a: f(*a).set_params(random_state=seed))(f))


def validation_codes(d):
    """The races evaluation_races scores (>= 2 corners with every label), without fitting anything."""
    out = []
    for code, x in d[d.date.between(*VAL)].groupby('RACE_CODE'):
        if sum(x[f'r{c}'].notna().all() and x[f'g{c}'].notna().all() and x[f'l{c}'].notna().all() for c in pm.CORNERS) >= 2:
            out.append(code)
    return out


def frame(path):
    if path and Path(path).exists():
        return pd.read_pickle(path)
    d, _ = pm.load_frame(date.today().isoformat().replace('-', ''))
    d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))
    if path:
        d.to_pickle(path)
    return d


def pipeline(d, params=None, margins=None, margin_tuning=None, seed=42):
    set_seed(seed)
    tuning = None
    if params is None:
        contention, tuning = pm.tune_contention(d)
        params = pm.tune_moves(d, contention)
        margins, margin_tuning = pm.calibrate_intervals(d, params)
    metrics = pm.validate(d, params, margins, margin_tuning)
    set_seed(42)
    return {'params': params, 'contentionTuning': tuning, 'margins': margins, 'metrics': metrics}


def shift(before, after, codes):
    """Course norm and validation pace before/after the import."""
    def norms(d):
        tr = d[d.date < pm.PACE_NORM_END].drop_duplicates('RACE_CODE')
        g = tr.groupby(['venue', 'track', 'distance']).early3
        return pd.DataFrame({'norm': g.median(), 'n': g.count()})
    nb, na = norms(before), norms(after)
    both = nb.join(na, lsuffix='_b', rsuffix='_a', how='inner')
    pb = before[before.RACE_CODE.isin(codes)].drop_duplicates('RACE_CODE').set_index('RACE_CODE').pace
    pa = after[after.RACE_CODE.isin(codes)].drop_duplicates('RACE_CODE').set_index('RACE_CODE').pace
    lab = lambda s: {k: float(np.mean([pm.pace_label(v) == k for v in s.dropna()])) for k in 'HMS'}
    return {'courses': {'before': len(nb), 'after': len(na), 'common': len(both)},
            'normRaces': {'before': int(nb.n.sum()), 'after': int(na.n.sum())},
            'normShiftMeanAbs': float((both.norm_a - both.norm_b).abs().mean()),
            'normShiftMean': float((both.norm_a - both.norm_b).mean()),
            'normShiftMax': float((both.norm_a - both.norm_b).abs().max()),
            'validationPaceMean': {'before': float(pb.mean()), 'after': float(pa.mean())},
            'validationPaceShiftMeanAbs': float((pa - pb).abs().mean()),
            'validationLabelShare': {'before': lab(pb), 'after': lab(pa)}}


def label_diff(before, after, codes):
    """Are the 273 races' labels the same in both frames (the re-import could have rewritten rows)?"""
    labels = [f'{k}{c}' for c in pm.CORNERS for k in 'rgl'] + ['last3']
    past = ['past_' + t for t in pm.TARGETS] + ['starts']
    b = before[before.RACE_CODE.isin(codes)].set_index(['RACE_CODE', 'UMABAN']).sort_index()
    a = after[after.RACE_CODE.isin(codes)].set_index(['RACE_CODE', 'UMABAN']).reindex(b.index)

    def changed(cols):
        x, y = a[cols].astype(float), b[cols].astype(float)
        return ~(np.isclose(x, y) | (x.isna() & y.isna()))
    return {'rows': len(b), 'rowsMissingAfter': int(a.UMABAN.isna().sum()) if 'UMABAN' in a else int(a.date.isna().sum()),
            'labelCellsChanged': int(changed(labels).values.sum()),
            'rowsWithChangedPastFeatures': int(changed(past).any(axis=1).sum()),
            'meanStartsBefore': float(b.starts.mean()), 'meanStartsAfter': float(a.starts.mean())}


def main():
    global PINNED
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', help='pickle cache of the expanded frame (built from the DB when missing)')
    ap.add_argument('--old-report', default=str(pm.ROOT / 'outputs' / 'data-expansion' / 'pace_report_v1.6_before.json'))
    ap.add_argument('--out', default=str(pm.ROOT / 'outputs' / 'data-expansion' / 'result.json'))
    ap.add_argument('--only', default='before,after,after-v16P,after-seed7')
    ap.add_argument('--pin', help='frame whose validation races are pinned (default: --before; v1.6 frame for the 273)')
    args = ap.parse_args()
    old = json.loads(Path(args.old_report).read_text(encoding='utf-8'))
    before, after = frame(args.before), frame(args.after)
    codes = validation_codes(frame(args.pin) if args.pin else before)
    PINNED = set(codes)
    res = json.loads(Path(args.out).read_text(encoding='utf-8')) if Path(args.out).exists() else {}
    res['validationRaces'] = {'pinned': len(codes), 'allAfter': len(validation_codes(after)),
                              'pinnedPresentAfter': int(len(set(validation_codes(after)) & PINNED))}
    res['frames'] = {k: {'historyRows': int(len(f)), 'rows': int(f.complete.sum()),
                         'completeRaces': int(f[f.complete].RACE_CODE.nunique()), 'races': int(f.RACE_CODE.nunique())}
                     for k, f in (('before', before), ('after', after))}
    res['labels'] = label_diff(before, after, codes)
    res['shift'] = shift(before, after, codes)
    print(json.dumps({k: res[k] for k in ('validationRaces', 'frames', 'labels', 'shift')}, ensure_ascii=False, indent=1), flush=True)
    v16 = (old['contention'], old['intervals']['margins'], {t: v['tuning'] for t, v in old['intervals']['targets'].items()})
    runs = {'before': lambda: pipeline(before), 'after': lambda: pipeline(after), 'after-v16P': lambda: pipeline(after, *v16),
            'after-seed7': lambda: pipeline(after, seed=7)}
    for name in args.only.split(','):
        print(f'\n===== {name} =====', flush=True)
        res[name] = runs[name]()
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print('\n%-8s %8s' % ('metric', 'v1.6') + ''.join(f'{k:>12}' for k in runs if k in res))
    for t in old['metrics']:
        print('%-8s %8.4f' % (t, old['metrics'][t]['mae']) + ''.join(f"{res[k]['metrics'][t]['mae']:12.4f}" for k in runs if k in res))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
