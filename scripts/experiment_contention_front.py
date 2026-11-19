"""Experiment (read-only, writes nothing in the repo): does the contention parameter `front`
(0.25 -> 0.45 in the v1.5 retune) explain the g2 regression (3.69 -> 3.92 lengths, while the
independent prediction improved 3.97 -> 3.90)?

Same data for both settings; everything except `front` is the v1.5 value from models/pace/report.json.
  1. Tuning window (fit < 2025-06, score Jun-Aug): per-metric error ratio vs independent for every
     PARAM_GRID point, i.e. the breakdown of tune_contention's objective.
  2. Validation window (fit < 2025-09, score Sep-Dec): MAE per target for front=0.45 and front=0.25.

Result 2026-11-17 (current DB, 273 validation races): g2 3.817 (0.45) vs 3.826 (0.25), independent
3.749; every other target within 0.003. Tuning objective 0.9822 vs 0.9820. front is not the cause;
v1.5 keeps 0.45 (see knownIssues in models/pace/report.json).

Run:  python scripts/experiment_contention_front.py [out.json]
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm  # noqa: E402
pm.legacy_procedure('experiment_contention_front.py')  # pre-E0: reproduces its record on the v1.8 procedure (E0 review finding 2)

V15 = json.loads((pm.ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
SETTINGS = {'front45': dict(V15), 'front25': {**V15, 'front': .25}}


def main():
    out = {'v15': V15}
    d, _ = pm.load_frame(date.today().isoformat().replace('-', ''))
    d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))

    models = pm.fit_models(d, '20250601')
    races = pm.evaluation_races(models, d, '20250601', '20250831')
    base = pm.score(models, races, None)
    grid = []
    for P in pm.PARAM_GRID:
        s = pm.score(models, races, P)
        ratios = {k: s[k] / base[k] for k in s}
        grid.append({'P': P, 'obj': float(np.mean(list(ratios.values()))), 'ratios': ratios})
        print('grid', P, f"obj={grid[-1]['obj']:.4f}", ' '.join(f'{k}={v:.3f}' for k, v in ratios.items()), flush=True)
    grid.sort(key=lambda r: r['obj'])
    out['tuning'] = {'races': len(races), 'independent': base, 'grid': grid}

    models = pm.fit_models(d, '20250901')
    races = pm.evaluation_races(models, d, '20250901', '20251231')
    ind = pm.score(models, races, None)
    val = {'races': len(races), 'independent': ind}
    for name, P in SETTINGS.items():
        val[name] = {'mae': pm.score(models, races, P), 'moves': pm.move_metrics(races, P),
                     'rankError': pm.rank_error(models, races, P)}
    out['validation'] = val
    print(f"\nvalidation {len(races)} races\n{'t':6} {'indep':>7} {'f=.45':>7} {'f=.25':>7}")
    for t in ind:
        print(f"{t:6} {ind[t]:7.3f} {val['front45']['mae'][t]:7.3f} {val['front25']['mae'][t]:7.3f}")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
