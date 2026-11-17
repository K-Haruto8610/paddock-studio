"""Experiment (read-only, writes nothing in the repo unless an output path is given): move horse-to-horse
interaction from the contention rules into the per-horse models as race-relative features.

Background (追加指示2, 2026-11-17): the blended continuous rank predictions beat the past-5-run mean at
every corner, but forming integer ranks through the contention layer loses to it. Here each per-horse
model gets "how crowded is the position I want" features, computed only from today's field's past_* columns
(pre-race information; no label of the race itself is read).

Candidates (one group each):
  crowd     crowd{c}: other horses whose past_r{c} is within +-0.1 of mine (c = 1..4)
  ahead     ahead{c}: other horses whose past_r{c} is ahead of mine
  lead      lead_rivals: other horses with early_wish <= 0.2
  inside    inside_rush: horses drawn inside me whose early_wish is mine + 0.05 or faster
  closer    closer_traffic: if my past_r4 >= 0.5 (closer), horses whose past_r4 is ahead of mine, else 0
  pace      pace_pressure_me: n_front * (1 - early_rel)  (1 - early_rel: forwardness, so the product is the
            cost of going forward in a crowded front; n_front * early_rel would grow for back markers)

Discipline: selection on Jun-Aug 2025 (models fitted before June); Sep-Dec 2025 is report-only and only the
baseline and the selected set are scored there. The baseline is recomputed here on the same data (the DB
was reloaded since the published numbers; never compare with them directly).
Only the per-horse models (TARGETS) are refitted per configuration; pace, spread, move and closing models
do not use FEATURES and are shared. Contention uses the v1.5 parameters from models/pace/report.json.

Metrics per configuration:
  cont   continuous blended rank MAE (prepare_race output, before integer formation)
  indep  formed MAE without contention (independent reconstruct)
  cont+  formed MAE with contention (simulate)

Seed noise: 'base-seed7' refits the baseline with another seed (tuning only) to size run-to-run noise.
pastMeanFormed: the past-5-run mean turned into integer ranks within the race, the fair reference for the
formed outputs (pastMean is continuous and gains from shrinking toward the middle, which MAE rewards).

Result 2026-11-17 (current DB; tuning 230 / validation 273 races) -> NOT ADOPTED, phase 2 not run.
  Tuning, mean continuous rank MAE r1-r4: base 0.2330, seed 7 0.2332, best (+all) 0.2325. Every group moves
  a corner by at most 0.0010, the same size as the seed change (up to 0.0010). Five groups "improved"
  (ahead, lead, inside, closer, pace), but only by 0.0001-0.0004 on average; crowd made it worse.
  Validation (+all vs base), continuous: r1 0.2175/0.2175, r2 0.2173/0.2180, r3 0.2238/0.2237,
  r4 0.2281/0.2284 -> gains 0.0000/0.0006/-0.0001/0.0003: fails the ">= 0.002 at 3 corners" rule.
  (The formed-rank and lane/last3 rules would pass, but by under 0.003: within noise.)
  Main finding: formation does not lose to the past mean like for like. Validation formed MAE
  (independent / contention / past mean as integer ranks): r1 0.2360/0.2364/0.2363, r2 0.2413/0.2415/0.2437,
  r3 0.2506/0.2521/0.2520, r4 0.2596/0.2619/0.2619. The gap to the continuous past mean (0.2245-0.2363) is
  the cost of integer ranks under MAE (about +0.012 to +0.026 for the past mean itself), not the contention.
  Contention vs independent: +0.0004/+0.0002/+0.0015/+0.0023 (3-4 corners slightly worse: the move layer).

Run:  python scripts/experiment_rel_features.py [--cache frame.pkl] [--out result.json]
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

P15 = json.loads((pm.ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
BASE_FEATURES = list(pm.FEATURES)
GROUPS = {
    'crowd': [f'crowd{c}' for c in pm.CORNERS],
    'ahead': [f'ahead{c}' for c in pm.CORNERS],
    'lead': ['lead_rivals'],
    'inside': ['inside_rush'],
    'closer': ['closer_traffic'],
    'pace': ['pace_pressure_me'],
}
RANKS = [f'r{c}' for c in pm.CORNERS]


def add_candidates(x, key):
    """Race-relative crowding features from past_* only (same inputs exist for upcoming cards)."""
    cols = {f: np.full(len(x), np.nan) for g in GROUPS.values() for f in g}
    past = {c: x[f'past_r{c}'].to_numpy(float) for c in pm.CORNERS}
    wish, gate = x.early_wish.to_numpy(float), x.UMABAN.to_numpy(float)
    for idx in x.groupby(key, sort=False).indices.values():
        n = len(idx)
        other = ~np.eye(n, dtype=bool)
        for c in pm.CORNERS:
            p = past[c][idx]
            have = ~np.isnan(p)
            near = (np.abs(p[:, None] - p[None, :]) <= .1) & other
            front = p[None, :] < p[:, None]  # NaN compares False: horses without history are not counted
            cols[f'crowd{c}'][idx] = np.where(have, near.sum(1), np.nan)
            cols[f'ahead{c}'][idx] = np.where(have, front.sum(1), np.nan)
            if c == 4:
                cols['closer_traffic'][idx] = np.where(have, np.where(p >= .5, front.sum(1), 0), np.nan)
        w, g = wish[idx], gate[idx]
        cols['lead_rivals'][idx] = ((w[None, :] <= .2) & other).sum(1)
        rush = (g[None, :] < g[:, None]) & (w[None, :] <= w[:, None] + .05)
        cols['inside_rush'][idx] = np.where(np.isnan(w), np.nan, rush.sum(1))
    for k, v in cols.items():
        x[k] = v
    x['pace_pressure_me'] = x.n_front * (1 - x.early_rel)
    return x


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def refit(shared, d, end, feats, seed=42):
    fit, models = (d.date < end) & d.complete, dict(shared)
    for t in pm.TARGETS:
        rows = fit & d[t].notna()
        model = pm.lgbm('l1' if t[0] == 'g' else 'l2').set_params(random_state=seed)
        models[t] = model.fit(d.loc[rows, feats], d.loc[rows, t])
    return models


def evaluate(shared, d, fit_end, start, end, feats, seed=42):
    pm.FEATURES = feats  # prepare_race reads the module global
    try:
        models = refit(shared, d, fit_end, feats, seed)
        races = pm.evaluation_races(models, d, start, end)
        cont = {}
        for t in RANKS:
            c = int(t[1])
            p = [v for x, cs, _, pred, *_ in races if c in cs for v in pred[t]]
            a = [v for x, cs, *_ in races if c in cs for v in x[t]]
            cont[t] = mae(p, a)
        out = {'races': len(races), 'cont': cont, 'indep': pm.score(models, races, None), 'contention': pm.score(models, races, P15)}
    finally:
        pm.FEATURES = BASE_FEATURES
    tr = (d.date < fit_end) & d.complete
    past, past_formed = {}, {}
    for t in RANKS:
        c, med = int(t[1]), d.loc[tr, t].median()
        p = [x[f'past_{t}'].fillna(med).to_numpy() for x, cs, *_ in races if c in cs]
        a = [x[t].to_numpy() for x, cs, *_ in races if c in cs]
        past[t] = mae(np.concatenate(p), np.concatenate(a))
        # The past mean turned into integer ranks like the formation (ties by horse number = row order).
        formed = [np.argsort(np.argsort(v, kind='stable'), kind='stable') / (len(v) - 1) for v in p]
        past_formed[t] = mae(np.concatenate(formed), np.concatenate(a))
    out['pastMean'], out['pastMeanFormed'] = past, past_formed
    return out


def line(name, r):
    f = lambda k: ' '.join(f'{t}={r[k][t]:.4f}' for t in RANKS)
    return (f"{name:16} cont[{f('cont')}]  indep[{f('indep')}]  cont+[{f('contention')}]  "
            f"l4={r['contention']['l4']:.4f} last3={r['contention']['last3']:.4f}")


def window(d, fit_end, start, end, configs, label):
    shared = pm.fit_models(d, fit_end)
    print(f'\n=== {label}: fit<{fit_end}, eval {start}..{end} ===', flush=True)
    res = {}
    for name, feats in configs.items():
        seed = 7 if name == 'base-seed7' else 42
        res[name] = evaluate(shared, d, fit_end, start, end, feats, seed)
        print(line(name, res[name]), flush=True)
    ref = next(iter(res.values()))
    for k in ('pastMean', 'pastMeanFormed'):
        print(f"{k:16} " + ' '.join(f'{t}={ref[k][t]:.4f}' for t in RANKS), flush=True)
    return res


def mean_cont(r):
    return float(np.mean(list(r['cont'].values())))


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
    d = add_candidates(d, 'RACE_CODE')
    allf = [f for g in GROUPS.values() for f in g]
    print(d[allf].describe().T[['count', 'mean', 'std', 'max']].to_string(), flush=True)

    configs = {'base': BASE_FEATURES, 'base-seed7': BASE_FEATURES, **{f'+{g}': BASE_FEATURES + f for g, f in GROUPS.items()}, '+all': BASE_FEATURES + allf}
    tune = window(d, '20250601', '20250601', '20250831', configs, 'TUNING (selection)')
    base = mean_cont(tune['base'])
    good = [g for g in GROUPS if mean_cont(tune[f'+{g}']) < base]
    print('\ngroups improving mean continuous rank MAE on tuning:', good or 'none', flush=True)
    cands = {k: v for k, v in configs.items() if not k.startswith('base')}
    if len(good) > 1:
        cands['+good'] = BASE_FEATURES + [f for g in good for f in GROUPS[g]]
        tune.update(window(d, '20250601', '20250601', '20250831', {'+good': cands['+good']}, 'TUNING (combined good groups)'))
    best = min(cands, key=lambda k: mean_cont(tune[k]))
    print(f'selected on tuning: {best} (mean cont {mean_cont(tune[best]):.4f} vs base {base:.4f})', flush=True)

    val = window(d, '20250901', '20250901', '20251231', {'base': BASE_FEATURES, best: cands[best]}, 'VALIDATION (report only)')
    b, s = val['base'], val[best]
    print('\nadoption check (validation, same data):', flush=True)
    gains = {t: b['cont'][t] - s['cont'][t] for t in RANKS}
    print('  continuous rank gain >= 0.002 at 3+ corners:', sum(v >= .002 for v in gains.values()) >= 3,
          {t: round(v, 4) for t, v in gains.items()}, flush=True)
    formed_new = {t: min(s['indep'][t], s['contention'][t]) for t in RANKS}
    formed_cur = {t: b['contention'][t] for t in RANKS}
    print('  formed rank below current v1.5 (base+contention) at every corner:', all(formed_new[t] < formed_cur[t] for t in RANKS),
          {t: (round(formed_new[t], 4), round(formed_cur[t], 4)) for t in RANKS}, flush=True)
    worse = {t: s['contention'][t] - b['contention'][t] for t in ['last3', 'l1', 'l2', 'l3', 'l4']}
    print('  last3 / lane worsening within 0.01:', all(v <= .01 for v in worse.values()), {t: round(v, 4) for t, v in worse.items()}, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({'contention': P15, 'groups': GROUPS, 'improving': good, 'selected': best,
                                              'selectedFeatures': cands[best], 'tuning': tune, 'validation': val},
                                             ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
