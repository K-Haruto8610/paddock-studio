"""v2.0 on the v3 comparison windows: per-horse published predictions saved for scripts/v3/evaluate.py.

  python -I scripts/v3/baseline_v20.py --window sel|rep --seed 42|7     -> outputs/v3/v20_<window>_seed<seed>.pkl

The frame and code are the ones main publishes with (pace_model.py v2.0 on outputs/data-expansion-v20/frame_2020.pkl,
rating_pre attached), the procedure the reviewed one (EVAL_CORNERS 'course', PACE_OOF 'forward', RANK_BLEND 0.75), the
contention / move parameters the published ones (models/pace/report.json, tuned on Jun-Aug 2025 with seed 42), as the
experiments of 追加指示10/11 fixed them. Windows (experiment-discipline): sel = models fitted before 2025-06-01, scored on
Jun-Aug 2025; rep = fitted before 2025-09-01, scored on Sep-Dec 2025 pinned to v1.6's 273 races.

Saved per horse: the published formation's rank and gap at each scored corner, the published closing 3F (traffic-aware,
`contention`), and the finish estimate exactly as race_output builds it (last-corner gap + closing difference x 5
lengths per second), in lengths and in seconds (/5).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import experiment_data_expansion as ex  # noqa: E402  (pins evaluation_races, set_seed)
import experiment_data_expansion_v20 as v20  # noqa: E402
import experiment_ewm_history as e1  # noqa: E402
import pace_model as pm  # noqa: E402

OUT = pm.ROOT / 'outputs' / 'v3'
WINDOWS = {'sel': ('20250601', '20250601', '20250831'), 'rep': ('20250901', '20250901', '20251231')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', choices=list(WINDOWS), required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    pm.EVAL_CORNERS, pm.PACE_OOF, pm.RANK_BLEND = 'course', 'forward', 0.75
    ex.PINNED = set(ex.validation_codes(pd.read_pickle(e1.PIN)))
    P = json.loads((pm.ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
    d = v20.prepared(v20.FRAMES['after'])
    fit_end, start, end = WINDOWS[args.window]
    ex.set_seed(args.seed)
    models = pm.fit_models(d, fit_end)
    races = pm.evaluation_races(models, d, start, end)
    rows = []
    for x, corners, numbers, pred, spreads, early in races:
        con = pm.simulate(numbers, early, pred, spreads, corners, P)
        last3 = pm.closing(models, x, con, corners, pred['pace'])[0]
        gaps = con[corners[-1]]['gap']
        best = float(min(last3))
        fin = np.array([gaps[i] + (float(last3[i]) - best) * pm.LENGTHS_PER_SECOND for i in range(len(numbers))])
        fin = fin - fin.min()
        for i, num in enumerate(numbers):
            r = {'RACE_CODE': x.RACE_CODE.iloc[0], 'UMABAN': int(num), 'corners': tuple(corners), 'last3': float(last3[i]),
                 'finLengths': float(fin[i]), 'finSeconds': float(fin[i] / pm.LENGTHS_PER_SECOND), 'pace': pred['pace']}
            for c in corners:
                r[f'rank{c}'] = float(con[c]['rank'][i])
                r[f'gap{c}'] = float(con[c]['gap'][i])
            rows.append(r)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_pickle(OUT / f'v20_{args.window}_seed{args.seed}.pkl')
    print(f'{args.window} seed {args.seed}: {out.RACE_CODE.nunique()} races, {len(out)} horses')


if __name__ == '__main__':
    main()
