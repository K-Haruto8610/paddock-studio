"""Experiment (read-only, writes nothing in the repo unless an output path is given), phase 3 of 追加指示2:
a learned displacement layer instead of the contention layer's rank adjustment.

  target   disp{c} = actual r{c} - wish{c}, where wish{c} is the blended rank prediction
           (RANK_BLEND * model + (1 - RANK_BLEND) * past mean, as in pace_model.blend_ranks)
  inputs   the crowding features of scripts/experiment_rel_features.py (crowd/ahead per corner, lead_rivals,
           inside_rush, closer_traffic, pace_pressure_me) plus the wish itself, its rank within the race
           and the field size ('crowd' variant); 'wide' adds the base race/horse features.
  training wishes on the training rows are out-of-fold (4 date-ordered chunks of races, the per-horse rank
           models refitted on the other three), so the layer learns from wishes as noisy as at prediction.
  use      corrected wish = wish + shrink * predicted displacement, then
             corr-indep       integer ranks straight from the corrected wish (replaces the contention ranks)
             corr-contention  the corrected wish fed into the contention layer (gaps / lanes / last3 kept)

Selection (variant x shrink) on Jun-Aug 2025 with models fitted before June, by mean formed rank MAE of
corr-indep; Sep-Dec 2025 is report-only. Baselines are recomputed on the same data. 'base-seed7' (the
per-horse models refitted with another seed) sizes the run-to-run noise of formed ranks.

Result 2026-11-18 (current DB; tuning 230 / validation 273 races) -> NOT ADOPTED.
  Tuning, mean formed rank MAE r1-r4: base independent 0.2648, base contention 0.2651, best layer
  (wide x 0.5) 0.2648 = no gain; crowd x 0.5 0.2666, crowd x 1.0 0.2707, wide x 1.0 0.2684 (worse).
  The seed change alone moves a formed corner by up to 0.0027 (r2 0.2658 -> 0.2685).
  Validation (report only), formed rank, current contention / independent / corrected / corrected+contention:
    r1 0.2364/0.2360/0.2347/0.2363  r2 0.2415/0.2413/0.2391/0.2432
    r3 0.2521/0.2506/0.2506/0.2524  r4 0.2619/0.2596/0.2583/0.2635
  The corrected ranks gain 0.0013/0.0022/0.0000/0.0013 over independent, within the seed noise and with no
  gain at all on the selection window. Fed into the contention layer the correction is worse at r2/r4, and
  g1 +0.05 lengths. Continuous rank MAE barely moves (<= 0.0004).

Run:  python scripts/experiment_displacement.py [--cache frame.pkl] [--out result.json]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm  # noqa: E402
from experiment_rel_features import GROUPS, P15, add_candidates, mae, refit  # noqa: E402

CROWD = [f for g in GROUPS.values() for f in g]
VARIANTS = {'crowd': CROWD + ['wish', 'wish_rel', 'field'],
            'wide': CROWD + ['wish', 'wish_rel', 'past_r', 'early_rel', 'n_front'] + pm.BASE}
SHRINK = (0.5, 1.0)
RANKS = [f'r{c}' for c in pm.CORNERS]


def disp_lgbm():
    return LGBMRegressor(n_estimators=200, max_depth=4, num_leaves=15, learning_rate=.03, min_child_samples=60, subsample=.8,
                         subsample_freq=5, colsample_bytree=.8, reg_lambda=5, verbosity=-1, n_jobs=4, random_state=42)


def blend(model, past):
    return np.where(np.isnan(past), model, pm.RANK_BLEND * model + (1 - pm.RANK_BLEND) * past)


def corner_frame(x, c, wish, key):
    f = x[CROWD + pm.BASE + ['early_rel', 'n_front', key]].copy()
    f['wish'], f['past_r'] = wish, x[f'past_r{c}'].to_numpy(float)
    f['wish_rel'] = f.groupby(key).wish.rank(pct=True)
    return f


def oof_wishes(d, fit_end):
    """Out-of-fold blended rank wishes on the training rows (4 date-ordered chunks of races)."""
    fit = (d.date < fit_end) & d.complete
    races = d[fit].drop_duplicates('RACE_CODE').sort_values(['date', 'RACE_CODE']).RACE_CODE.to_numpy()
    out = {c: pd.Series(np.nan, index=d.index[fit]) for c in pm.CORNERS}
    for part in np.array_split(races, 4):
        held = fit & d.RACE_CODE.isin(part)
        for c in pm.CORNERS:
            t = f'r{c}'
            rows = fit & ~held & d[t].notna()
            m = pm.lgbm('l2').fit(d.loc[rows, pm.FEATURES], d.loc[rows, t])
            out[c][d.index[held]] = blend(m.predict(d.loc[held, pm.FEATURES]), d.loc[held, f'past_{t}'].to_numpy(float))
    return out


def fit_displacement(d, fit_end, wishes):
    fit = (d.date < fit_end) & d.complete
    models = {}
    for c in pm.CORNERS:
        x = d[fit]
        f = corner_frame(x, c, wishes[c].reindex(x.index).to_numpy(), 'RACE_CODE')
        ok = x[f'r{c}'].notna().to_numpy() & ~np.isnan(f.wish.to_numpy())
        y = x[f'r{c}'].to_numpy() - f.wish.to_numpy()
        for name, cols in VARIANTS.items():
            models[(name, c)] = disp_lgbm().fit(f.loc[ok, cols], y[ok])
    return models


def corrected(races, disp, name, s):
    out = []
    for x, corners, numbers, pred, spreads, early in races:
        p = dict(pred)
        for c in pm.CORNERS:
            f = corner_frame(x, c, np.asarray(pred[f'r{c}'], float), 'RACE_CODE')
            p[f'r{c}'] = np.asarray(pred[f'r{c}'], float) + s * disp[(name, c)].predict(f[VARIANTS[name]])
        out.append((x, corners, numbers, p, spreads, early))
    return out


def cont_mae(races):
    res = {}
    for t in RANKS:
        c = int(t[1])
        res[t] = mae([v for x, cs, _, p, *_ in races if c in cs for v in p[t]], [v for x, cs, *_ in races if c in cs for v in x[t]])
    return res


def summary(models, races):
    return {'cont': cont_mae(races), 'indep': pm.score(models, races, None), 'contention': pm.score(models, races, P15)}


def line(name, r):
    f = lambda k: ' '.join(f'{t}={r[k][t]:.4f}' for t in RANKS)
    return (f"{name:22} cont[{f('cont')}]  indep[{f('indep')}]  cont+[{f('contention')}]  "
            f"g4={r['contention']['g4']:.3f} l4={r['contention']['l4']:.4f} last3={r['contention']['last3']:.4f}")


def mean_ranks(r, k):
    return float(np.mean([r[k][t] for t in RANKS]))


def window(d, fit_end, start, end, label, settings=None):
    print(f'\n=== {label}: fit<{fit_end}, eval {start}..{end} ===', flush=True)
    models = pm.fit_models(d, fit_end)
    races = pm.evaluation_races(models, d, start, end)
    res = {'races': len(races), 'base': summary(models, races)}
    print(f"{len(races)} races", flush=True)
    print(line('base', res['base']), flush=True)
    if settings is None:  # tuning: seed noise reference
        pm_feats = pm.FEATURES
        seeded = refit(models, d, fit_end, pm_feats, seed=7)
        res['base-seed7'] = summary(seeded, pm.evaluation_races(seeded, d, start, end))
        print(line('base-seed7', res['base-seed7']), flush=True)
    disp = fit_displacement(d, fit_end, oof_wishes(d, fit_end))
    for name, s in settings or [(n, s) for n in VARIANTS for s in SHRINK]:
        res[f'{name}x{s}'] = summary(models, corrected(races, disp, name, s))
        print(line(f'{name}x{s}', res[f'{name}x{s}']), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', help='pickle of the prepared frame (created if missing)')
    ap.add_argument('--out')
    args = ap.parse_args()
    if args.cache and Path(args.cache).exists():
        d = pd.read_pickle(args.cache)
    else:
        d, _ = pm.load_frame(date.today().isoformat().replace('-', ''))
        d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))
        if args.cache:
            d.to_pickle(args.cache)
    d = add_candidates(d, 'RACE_CODE').copy()

    tune = window(d, '20250601', '20250601', '20250831', 'TUNING (selection)')
    keys = [k for k in tune if 'x' in k and not k.startswith('base')]
    best = min(keys, key=lambda k: mean_ranks(tune[k], 'indep'))
    name, s = best.split('x')[0], float(best.split('x')[1])
    print(f"\nselected on tuning: {best} (mean formed rank {mean_ranks(tune[best], 'indep'):.4f}; "
          f"base indep {mean_ranks(tune['base'], 'indep'):.4f}, base contention {mean_ranks(tune['base'], 'contention'):.4f})", flush=True)

    val = window(d, '20250901', '20250901', '20251231', 'VALIDATION (report only)', [(name, s)])
    b, r = val['base'], val[best]
    print('\nvalidation, formed rank MAE (current contention / independent / corrected independent / corrected + contention):')
    for t in RANKS:
        print(f"  {t}: {b['contention'][t]:.4f} / {b['indep'][t]:.4f} / {r['indep'][t]:.4f} / {r['contention'][t]:.4f}")
    worse = {t: r['contention'][t] - b['contention'][t] for t in ['last3', 'l1', 'l2', 'l3', 'l4', 'g1', 'g2', 'g3', 'g4']}
    print('  corrected + contention vs current, other targets:', {t: round(v, 4) for t, v in worse.items()}, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({'contention': P15, 'variants': VARIANTS, 'selected': best, 'tuning': tune,
                                              'validation': val}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
