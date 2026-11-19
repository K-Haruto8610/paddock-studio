"""G1 guard re-check: the gap guard without the numerical blow-up of pace_model.reconstruct.

pace_model.reconstruct divides the gap model's shape by `anchor` (the shape at the 80% horse) whenever anchor > 1e-6. When the per-horse gap predictions
are nearly flat that is a division by a tiny number: in the 2025-10-26 Sep-Dec race 2025102608030909 (8 runners, predicted field length 3.68 lengths) the
largest predicted gap came out at 22,813 lengths, which alone moves the corner-3 gap MAE of the whole window by up to 6 lengths and has nothing to do with
the feature under test. The raw guard is reported in the README as it is; this script recomputes the gap (and lane) differences on the races where neither
system has a predicted gap above BLOWUP lengths (each system with its own re-tuned contention parameters, the seeds as in exp_common).

  python scripts/experiment_g1_gapcheck.py            # variants r1 and r1_rl, seeds 42 and 7, writes outputs/g1-rating/gapcheck.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_standard as es  # noqa: E402
import exp_common as xc  # noqa: E402
import experiment_data_expansion as ex  # noqa: E402
import experiment_g1_rating as g1  # noqa: E402
import pace_model as pm  # noqa: E402

BLOWUP = 60.0  # lengths: the longest real field length in the validation windows is about 20


def run(name, seed, d):
    rep = json.loads((xc.ROOT / 'outputs' / 'g1-rating' / f'report_{xc.slug(name)}_seed{seed}_retune.json').read_text(encoding='utf-8'))
    out = {}
    for system, variant in (('baseline', 'base'), ('candidate', name)):
        P = rep[system]['params']
        ex.set_seed(seed)
        xc.prepare(d, g1.VARIANTS[variant])
        models = pm.fit_models(d, xc.REP[0])
        races = pm.evaluation_races(models, d, xc.REP[1], xc.REP[2])
        rows = {}
        for x, corners, numbers, pred, spreads, early in races:
            con = pm.simulate(numbers, early, pred, spreads, corners, P)
            rows[x.RACE_CODE.iloc[0]] = {'maxgap': max(max(con[c]['gap']) for c in corners),
                                         'gap': {c: (np.abs(np.asarray(con[c]['gap'], float) - x[f'g{c}'].to_numpy()), list(numbers)) for c in corners if pm.labelled(x, c)},
                                         'lane': {c: (np.abs(np.asarray(con[c].get('localLane', con[c]['lane']), float) - x[f'l{c}'].to_numpy()), list(numbers))
                                                  for c in corners if pm.labelled(x, c)}}
        out[system] = rows
    xc.set_features()
    g1.uninstall()
    bad = sorted(r for r in out['baseline'] if out['baseline'][r]['maxgap'] > BLOWUP or out['candidate'][r]['maxgap'] > BLOWUP)
    tab = es.RaceTable()
    for system, rows in out.items():
        for race, v in rows.items():
            if race in bad:
                continue
            for c, (e, keys) in v['gap'].items():
                tab.add(f'g{c}', system, race, e, keys)
            for c, (e, keys) in v['lane'].items():
                tab.add(f'l{c}', system, race, e, keys)
    res = {'excludedRaces': bad, 'maxPredictedGap': {r: {s: out[s][r]['maxgap'] for s in out} for r in bad}, 'diffs': {}}
    for c in pm.CORNERS:
        for m in (f'g{c}', f'l{c}'):
            if (m, 'baseline') in tab.rows:
                res['diffs'][m] = tab.diff(m, 'candidate', 'baseline')
    return res


def main():
    xc.setup()
    d = xc.v19_frame()
    res = {}
    for name in ('r1', 'r1_rl'):
        for seed in (42, 7):
            res[f'{name}/{seed}'] = run(name, seed, d)
            r = res[f'{name}/{seed}']
            print(name, seed, 'excluded', r['excludedRaces'], {m: round(v['diff'], 4) for m, v in r['diffs'].items()}, flush=True)
    (xc.ROOT / 'outputs' / 'g1-rating' / 'gapcheck.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
