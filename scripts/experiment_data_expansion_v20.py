"""Data expansion 2020/2021 (2026-11-19): v2.0 settings retrained on the grown frame, scored on the SAME validation races.

The DB now holds 2020-2026 in full (field-complete races 4,330 -> 6,115). The v2.0 code is unchanged (rating_pre included); only the training data
grew. `before` is the frame v1.9/v2.0 were adopted on (outputs/data-expansion-v18/frame_2022.pkl, 60,299 rows), `after` the new one
(outputs/data-expansion-v20/frame_2020.pkl, 84,827 rows). Both are run through the published pipeline (tune contention / moves / intervals on
Jun-Aug 2025, validate on Sep-Dec 2025) and the validation races are pinned to the ones `before` scores (279).

  python scripts/experiment_data_expansion_v20.py --frame before --seed 7      # writes outputs/data-expansion-v20/result_<frame>_seed<seed>.json
  python scripts/experiment_data_expansion_v20.py --frame after --seed 7

The seed-42 runs are the published pipeline itself (v2.0 report for `before`; `pace_model.py --frame outputs/data-expansion-v20/frame_2020.pkl
--raw-rows 319342` for `after`). Interval margins are re-calibrated in each run, as the published code does.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_data_expansion as ex  # noqa: E402  (pins evaluation_races to a race set, set_seed)
import pace_model as pm  # noqa: E402
import rating_index as ri  # noqa: E402

FRAMES = {'before': 'outputs/data-expansion-v18/frame_2022.pkl', 'after': 'outputs/data-expansion-v20/frame_2020.pkl'}


def prepared(path):
    d = pd.read_pickle(pm.ROOT / path)
    d = d[[k for k in d if not (k.startswith('past') or k in ('starts', 'rating_pre'))]]
    d = pm.add_traffic(pm.add_history_features(d))
    return ri.attach(d, ri.ratings(ri.load_runs(pd.Timestamp.today().strftime('%Y%m%d'))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', choices=list(FRAMES), required=True)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    ex.PINNED = set(ex.validation_codes(prepared(FRAMES['before'])))
    d = prepared(FRAMES[args.frame])
    print(args.frame, 'rows', len(d), 'pinned races', len(ex.PINNED), flush=True)
    out = ex.pipeline(d, seed=args.seed)
    out['standard'] = out['metrics'].pop('_standard')
    out['intervals'] = out['metrics'].pop('_intervals')
    for k in [k for k in out['metrics'] if k.startswith('_')]:
        out['metrics'].pop(k)
    p = pm.ROOT / 'outputs' / 'data-expansion-v20' / f'result_{args.frame}_seed{args.seed}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print('wrote', p)


if __name__ == '__main__':
    main()
