"""Experiment (追加指示4 フェーズ2 (b), 2026-11-18): features for the race-level field-length (spread) models.

Why (b) and not (a) "gap to the horse ahead, accumulated": scripts/diagnose_margins.py showed on Sep-Dec 2025 that
the per-horse gap model alone already beats the past-5 mean (g1 3.33 vs 3.40, g2 3.44 vs 3.66) and that the
reconstruct loses through the order and the field length: with the true field length g1 falls 3.74 -> 2.91, with
the true order 3.74 -> 2.54. The order is the rank models, closed after phase 1; the field length is the part
still open. Accumulating predicted intervals (a) would add the order error up along the field.

Candidates (race level, pre-race only: today's field's past_* columns and earlier races):
  disp    wish_sd (spread of the field's usual early positions), last3_sd (spread of the usual closing 3F),
          closer_share (share of horses whose past_r4 >= 0.7)
  fieldg  pg_q80: 80% quantile of the field's past_g{c} (the field length the horses' own gaps add up to)
  hist    spread_hist: mean over the field of each horse's past-5 races' observed spread at corner c
          (were the races these horses ran strung out or bunched)
  course  course_spread: median spread at corner c on the same venue / track / distance, from fit-window races only
  all     all of the above

Discipline: selection on Jun-Aug 2025 (models fitted before June), Sep-Dec 2025 report-only, baseline recomputed on
the same frame, a second seed (7) for the spread models to size noise. Only the spread models are refitted; every
other model is shared. Adoption (追加指示4): g1 and g2 (contention, as published) below the same-data past-5 mean,
and g3 / g4 not worse by more than 0.05 lengths.

Result 2026-11-18 (tuning 230 / validation 273 races, current DB) -> NOT ADOPTED; SPREAD_FEATURES unchanged.
  Tuning, contention gap MAE g1-g4: base 3.709/3.993/2.946/2.645, past mean 3.758/4.150/2.967/2.838. g1 and g2 are the
  same for every configuration and for seed 7: the corner-1/2 spread models of that window are constants (43 / 53
  training races before June, under 2 x min_child_samples). Only g3/g4 could select: +course (mean 3.307 vs 3.323).
  Validation (+course vs base): g1 3.654/3.830, g2 3.797/3.817, g3 2.678/2.677, g4 2.448/2.390 (seed noise
  0.054/0.030/0.008/0.014); past mean 3.401/3.656/2.687/2.516. g1 improves beyond noise but stays above the past mean,
  g2 does not move, g4 is worse by 0.058 (> 0.05) -> fails both rules.
  Training races for the spread models: 43/53/110/110 (fit < 2025-06), 163/185/341/341 (< 2025-09), 278/323/616/616
  (< 2026, published). The field length is learned from a few hundred races at most and is barely better than a
  constant (diagnose_margins.py); this is a data-size limit, recorded in README, not chased further.

Run:  python scripts/experiment_spread_features.py [--cache frame.pkl] [--out result.json]
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
BASE = list(pm.SPREAD_FEATURES)
GROUPS = {'disp': ['wish_sd', 'last3_sd', 'closer_share'], 'fieldg': ['pg_q80'], 'hist': ['spread_hist'], 'course': ['course_spread']}
GAPS = [f'g{c}' for c in pm.CORNERS]
ORIG_RACE_LEVEL = pm.race_level
COURSE = {}  # corner -> {(venue, track, distance): median spread}, set per window from fit-window races


def race_level(x, key, c):
    out = ORIG_RACE_LEVEL(x, key, c)
    g = x.groupby(key, sort=False)
    out['wish_sd'] = g.early_wish.std()
    out['last3_sd'] = g.past_last3.std()
    out['closer_share'] = g.past_r4.agg(lambda s: float((s >= .7).mean()))
    out['pg_q80'] = g[f'past_g{c}'].quantile(.8)
    if f'past_spread{c}' in x:
        out['spread_hist'] = g[f'past_spread{c}'].mean()
    med = COURSE.get(c, {})
    out['course_spread'] = [med.get((v, t, ds), np.nan) for v, t, ds in zip(out.venue, out.track, out.distance)]
    return out


def add_spread_history(d):
    """past_spread{c}: mean observed spread at corner c over the horse's previous 5 races (shift 1)."""
    d = d.sort_values(['date', 'RACE_CODE', 'UMABAN']).reset_index(drop=True)
    for c in pm.CORNERS:
        g = d.groupby('RACE_CODE')[f'g{c}']
        spread = g.transform(lambda s: s.quantile(pm.SPREAD_Q)).where(g.transform('count') == g.transform('size'))
        d[f'spread{c}_obs'] = spread
        d[f'past_spread{c}'] = d.groupby('KETTO_TOROKU_BANGO')[f'spread{c}_obs'].transform(
            lambda s: s.shift().rolling(5, min_periods=1).mean())
    return d.copy()


def set_course(d, end):
    tr = d[(d.date < end) & d.complete]
    for c in pm.CORNERS:
        r = ORIG_RACE_LEVEL(tr, 'RACE_CODE', c).dropna(subset=['spread'])
        COURSE[c] = r.groupby(['venue', 'track', 'distance']).spread.median().to_dict()


def oof_train(d, end):
    """Training rows with out-of-fold pace predictions, exactly as fit_models builds them."""
    fit = (d.date < end) & d.complete
    train = d[fit].copy()
    races = ORIG_RACE_LEVEL(train, 'RACE_CODE', 1).dropna(subset=['pace']).sort_values('date')
    oof = pd.Series(np.nan, index=races.index)
    for part in np.array_split(races.index, 4):
        rest = races.index.difference(part)
        oof[part] = pm.pace_lgbm().fit(races.loc[rest, pm.PACE_FEATURES], races.loc[rest, 'pace']).predict(races.loc[part, pm.PACE_FEATURES])
    train['pace_pred'] = train.RACE_CODE.map(oof)
    return train


def evaluate(shared, train, d, start, end, feats, seed):
    models = dict(shared)
    for c in pm.CORNERS:
        r = race_level(train, 'RACE_CODE', c).dropna(subset=['spread'])
        models[f'spread{c}'] = pm.lgbm('l1').set_params(random_state=seed).fit(r[feats], r['spread'])
    pm.SPREAD_FEATURES = feats
    try:
        races = pm.evaluation_races(models, d, start, end)
    finally:
        pm.SPREAD_FEATURES = BASE
    cont, indep = pm.score(models, races, P15), pm.score(models, races, None)
    sp = {c: [] for c in pm.CORNERS}
    for x, corners, numbers, pred, spreads, early in races:
        for c in corners:
            sp[c].append(abs(spreads[c] - float(np.quantile(x[f'g{c}'], pm.SPREAD_Q))))
    tr = (d.date < start) & d.complete
    past = {}
    for t in GAPS:
        c = int(t[1])
        p = [np.abs(x[f'past_{t}'].fillna(d.loc[tr, t].median()) - x[t]) for x, cs, *_ in races if c in cs]
        past[t] = float(np.mean(np.concatenate(p)))
    return {'races': len(races), 'contention': cont, 'independent': indep, 'spreadMae': {f'g{c}': float(np.mean(v)) for c, v in sp.items()},
            'pastMean': past}


def line(name, r):
    f = lambda k: ' '.join(f'{t}={r[k][t]:.3f}' for t in GAPS)
    return f"{name:14} cont+[{f('contention')}]  indep[{f('independent')}]  spread[{f('spreadMae')}]  r4={r['contention']['r4']:.4f} last3={r['contention']['last3']:.4f}"


def window(d, end, start, stop, configs, label):
    set_course(d, end)
    pm.race_level = race_level  # fit_models / prepare_race call the module function
    try:
        shared = pm.fit_models(d, end)
        train = oof_train(d, end)
        print(f'\n=== {label}: fit<{end}, eval {start}..{stop} ===', flush=True)
        res = {}
        for name, feats in configs.items():
            res[name] = evaluate(shared, train, d, start, stop, feats, 7 if name == 'base-seed7' else 42)
            print(line(name, res[name]), flush=True)
    finally:
        pm.race_level = ORIG_RACE_LEVEL
    print('pastMean       ' + ' '.join(f'{t}={next(iter(res.values()))["pastMean"][t]:.3f}' for t in GAPS), flush=True)
    return res


def mean_gap(r):
    return float(np.mean([r['contention'][t] for t in GAPS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache')
    ap.add_argument('--out')
    args = ap.parse_args()
    if args.cache and Path(args.cache).exists():
        d = pd.read_pickle(args.cache)
    else:
        d, _ = pm.load_frame(date.today().isoformat().replace('-', ''))
        d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))
        if args.cache:
            d.to_pickle(args.cache)
    d = add_spread_history(d)
    allf = [f for g in GROUPS.values() for f in g]
    configs = {'base': BASE, 'base-seed7': BASE, **{f'+{g}': BASE + f for g, f in GROUPS.items()}, '+all': BASE + allf}
    tune = window(d, '20250601', '20250601', '20250831', configs, 'TUNING (selection)')
    cands = {k: v for k, v in configs.items() if not k.startswith('base')}
    best = min(cands, key=lambda k: mean_gap(tune[k]))
    print('\ntuning mean contention gap: ' + ' '.join(f'{k}={mean_gap(v):.4f}' for k, v in tune.items()), flush=True)
    print(f'selected on tuning: {best}', flush=True)
    val = window(d, '20250901', '20250901', '20251231', {'base': BASE, 'base-seed7': BASE, best: cands[best]}, 'VALIDATION (report only)')
    b, s = val['base'], val[best]
    below = {t: s['contention'][t] < s['pastMean'][t] for t in ('g1', 'g2')}
    worse = {t: s['contention'][t] - b['contention'][t] for t in ('g3', 'g4')}
    adopt = all(below.values()) and all(v <= .05 for v in worse.values())
    print('\nadoption check (validation, same data):', flush=True)
    print('  g1/g2 below past mean:', {t: (round(s['contention'][t], 3), round(s['pastMean'][t], 3)) for t in ('g1', 'g2')}, flush=True)
    print('  change vs base:', {t: round(s['contention'][t] - b['contention'][t], 3) for t in GAPS}, flush=True)
    print('  seed noise (validation):', {t: round(abs(b['contention'][t] - val['base-seed7']['contention'][t]), 3) for t in GAPS}, flush=True)
    print('  ADOPT:', adopt, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({'groups': GROUPS, 'selected': best, 'adopt': adopt, 'tuning': tune, 'validation': val},
                                             ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
