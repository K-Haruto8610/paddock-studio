"""Experiment (追加指示4 フェーズ1, 2026-11-18): sequence features of the past runs for the per-horse models.

The current past_* columns are the plain mean of the last 5 runs, which flattens "the latest state and the
trend". Candidates (all from runs strictly before the race; same per-horse run sequence as past_*, i.e. the
named-race frame, so a value exists exactly when past_* would):
  last1   last1_{r1,r2,r3,r4,g4,l4,last3}: the latest run itself (lag 1)
  last2   last2_r4, last2_last3: two runs back (optional variant)
  trend   trend_r1, trend_r4, trend_last3: mean of the last 2 runs minus mean of runs 3-5 (needs both blocks)
  wmean   wmean_*: 5,4,3,2,1-weighted mean of the last 5 runs for every past_<target>. A SUBSTITUTION:
          the past_* columns are swapped for wmean_* in FEATURES (the rank blend keeps the plain past mean)
  std     std_r4, std_last3: spread of the last 5 runs (2+ runs)
  all     last1 + last2 + trend + std on top of the base (wmean is a substitution, tried separately)

Discipline (same as experiment_rel_features.py / experiment_jockey_class_margin.py): selection on Jun-Aug 2025
with models fitted before June; Sep-Dec 2025 report-only, baseline recomputed on the same frame, seed noise from
a second seed (7) on both windows. Adoption rule: continuous blended rank MAE better by >= 0.002 at 3+ corners on
validation, last3 / lane not worse by more than 0.01.

Result 2026-11-18 (tuning 230 / validation 273 races, current DB) -> NOT ADOPTED; FEATURES unchanged. This was the
last model-side experiment for the rank models: adding features to them is closed (README).
  Tuning, mean continuous rank MAE r1-r4: base 0.2330, seed 7 0.2331 (noise per corner 0.0004-0.0009).
  +last1 0.2330, +last2 0.2331, +trend 0.2333, +std 0.2331, wmean(swap) 0.2329, +all 0.2329: no group moves the
  mean by more than 0.0003; +all selected by 0.0001.
  Validation (+all vs base), continuous: r1 0.2182/0.2175, r2 0.2171/0.2180, r3 0.2236/0.2237, r4 0.2276/0.2284
  -> gains -0.0007/+0.0008/+0.0002/+0.0008 (validation seed noise up to 0.0007): fails ">= 0.002 at 3 corners",
  and the 2nd-corner lane gets worse by 0.0126 (> 0.01). last3 +0.0001.
  Reading: the latest run, the trend and the spread carry nothing the past-5 mean (and early_rel etc.) does not
  already give the trees; the weighted mean is the same as the plain mean within noise.

Run:  python scripts/experiment_sequence_features.py [--cache frame.pkl] [--out result.json]
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
pm.legacy_procedure('experiment_sequence_features.py')  # pre-E0: reproduces its record on the v1.8 procedure (E0 review finding 2)
import experiment_rel_features as rel  # noqa: E402

BASE = list(pm.FEATURES)
LAST1 = [f'last1_{t}' for t in ('r1', 'r2', 'r3', 'r4', 'g4', 'l4', 'last3')]
LAST2 = ['last2_r4', 'last2_last3']
TREND = ['trend_r1', 'trend_r4', 'trend_last3']
STD = ['std_r4', 'std_last3']
GROUPS = {'last1': LAST1, 'last2': LAST2, 'trend': TREND, 'std': STD}
WEIGHTS = (5, 4, 3, 2, 1)  # lag 1 (latest) .. lag 5


def add_sequence(d):
    """Sequence features per horse from earlier rows of the frame only (shift >= 1)."""
    d = d.sort_values(['date', 'RACE_CODE', 'UMABAN']).reset_index(drop=True)
    g = d.groupby('KETTO_TOROKU_BANGO')
    for t in pm.TARGETS:
        lags = [g[t].shift(k) for k in range(1, 6)]
        num = sum(w * s.fillna(0) for w, s in zip(WEIGHTS, lags))
        den = sum(w * s.notna() for w, s in zip(WEIGHTS, lags))
        d[f'wmean_{t}'] = (num / den).where(den > 0)
        if t in ('r1', 'r2', 'r3', 'r4', 'g4', 'l4', 'last3'):
            d[f'last1_{t}'] = lags[0]
        if t in ('r4', 'last3'):
            d[f'last2_{t}'] = lags[1]
            d[f'std_{t}'] = pd.concat(lags, axis=1).std(axis=1, skipna=True, ddof=1)
        if t in ('r1', 'r4', 'last3'):
            recent = pd.concat(lags[:2], axis=1).mean(axis=1, skipna=True)
            older = pd.concat(lags[2:], axis=1).mean(axis=1, skipna=True)
            d[f'trend_{t}'] = recent - older
    return d


def mean_cont(r):
    return float(np.mean(list(r['cont'].values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache')
    ap.add_argument('--out')
    args = ap.parse_args()
    today = date.today().isoformat().replace('-', '')
    if args.cache and Path(args.cache).exists():
        d = pd.read_pickle(args.cache)
    else:
        d, _ = pm.load_frame(today)
        d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))
        if args.cache:
            d.to_pickle(args.cache)
    d = add_sequence(d)
    wmean = [f'wmean_{t}' for t in pm.TARGETS]
    allf = [f for g in GROUPS.values() for f in g]
    print(d[allf + wmean[:4]].describe().T[['count', 'mean', 'std', 'min', 'max']].to_string(), flush=True)
    # Sanity: the latest run is not the race itself (would be a leak) -> lag-1 equals the previous row's label.
    chk = d.groupby('KETTO_TOROKU_BANGO').r4.shift()
    assert np.allclose(chk.fillna(-9), d.last1_r4.fillna(-9)), 'last1_r4 is not the previous run'

    swapped = [('wmean_' + f[5:]) if f in pm.PAST else f for f in BASE]
    configs = {'base': BASE, 'base-seed7': BASE, **{f'+{g}': BASE + f for g, f in GROUPS.items()},
               'wmean(swap)': swapped, '+all': BASE + allf}
    tune = rel.window(d, '20250601', '20250601', '20250831', configs, 'TUNING (selection)')
    base = mean_cont(tune['base'])
    noise = {t: abs(tune['base']['cont'][t] - tune['base-seed7']['cont'][t]) for t in rel.RANKS}
    print('\ntuning mean cont: ' + ' '.join(f'{k}={mean_cont(v):.4f}' for k, v in tune.items()), flush=True)
    print('seed noise per corner (tuning):', {t: round(v, 4) for t, v in noise.items()}, flush=True)
    cands = {k: v for k, v in configs.items() if not k.startswith('base')}
    best = min(cands, key=lambda k: mean_cont(tune[k]))
    print(f'selected on tuning: {best} (mean cont {mean_cont(tune[best]):.4f} vs base {base:.4f})', flush=True)

    val = rel.window(d, '20250901', '20250901', '20251231', {'base': BASE, 'base-seed7': BASE, best: cands[best]},
                     'VALIDATION (report only)')
    b, s = val['base'], val[best]
    gains = {t: b['cont'][t] - s['cont'][t] for t in rel.RANKS}
    vnoise = {t: abs(b['cont'][t] - val['base-seed7']['cont'][t]) for t in rel.RANKS}
    worse = {t: s['contention'][t] - b['contention'][t] for t in ['last3', 'l1', 'l2', 'l3', 'l4']}
    adopt = sum(v >= .002 for v in gains.values()) >= 3 and all(v <= .01 for v in worse.values())
    print('\nadoption check (validation, same data):', flush=True)
    print('  continuous rank gain per corner:', {t: round(v, 4) for t, v in gains.items()}, flush=True)
    print('  seed noise per corner (validation):', {t: round(v, 4) for t, v in vnoise.items()}, flush=True)
    print('  last3 / lane worsening:', {t: round(v, 4) for t, v in worse.items()}, flush=True)
    print('  ADOPT:', adopt, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({'groups': GROUPS, 'selected': best, 'selectedFeatures': cands[best], 'adopt': adopt,
                                              'gains': gains, 'seedNoise': {'tuning': noise, 'validation': vnoise},
                                              'tuning': tune, 'validation': val}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
