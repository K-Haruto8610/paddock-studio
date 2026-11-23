"""v3 S4: the race pace exactly as v2.0 scores it, so v3's predicted early 3F can be paired with v2.0's.

  python -I scripts/v3/pace_actual.py      -> outputs/v3/pace_actual.pkl, outputs/v3/pace_norms.json

v2.0's pace is the leaders' early 3F (race_shosai.ZENHAN_3F) minus the course's usual early 3F (pace_model.course_norm,
races before PACE_NORM_END = 2025-06-01, venue/track/distance else surface/distance). v3 predicts the leader's time at
600 m from the start (sim lead600); minus the same norm it is on v2.0's scale.

pace_actual.pkl : RACE_CODE, date, early3 (s), norm (s), pace (s), grade (v2.0 frame, every complete race)
pace_norms.json : {'exact': {"venue,track,distance": s}, 'loose': {"surface,distance": s}, 'sigma': {window: s},
                   'trainPace': {window: [...]}} -- sigma = v2.0's forward out-of-sample pace error sd before the window's
                   fit date (pace_model.pace_sigma), the scale of the H/M/S probabilities in the E0 RPS
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import experiment_data_expansion_v20 as v20  # noqa: E402
import pace_model as pm  # noqa: E402

OUT = pm.ROOT / 'outputs' / 'v3'
FIT_END = {'sel': '20250601', 'rep': '20250901'}


def main():
    pm.EVAL_CORNERS, pm.PACE_OOF, pm.RANK_BLEND = 'course', 'forward', 0.75
    d = v20.prepared(v20.FRAMES['after'])
    races = d.drop_duplicates('RACE_CODE')[['RACE_CODE', 'date', 'venue', 'track', 'distance', 'surface', 'early3', 'pace', 'grade', 'complete']]
    races = races.assign(norm=races.early3 - races.pace)
    races.to_pickle(OUT / 'pace_actual.pkl')
    tr = d[d.date < pm.PACE_NORM_END].drop_duplicates('RACE_CODE')
    exact = tr.groupby(['venue', 'track', 'distance']).early3.median()
    loose = tr.groupby(['surface', 'distance']).early3.median()
    out = {'normEnd': pm.PACE_NORM_END, 'exact': {f'{v},{t},{ds}': float(x) for (v, t, ds), x in exact.items()},
           'loose': {f'{s},{ds}': float(x) for (s, ds), x in loose.items()}, 'sigma': {}, 'trainPace': {}}
    for w, end in FIT_END.items():
        sigma, train = pm.pace_sigma(d, end)
        out['sigma'][w], out['trainPace'][w] = sigma, [float(v) for v in train]
        print(w, 'sigma', round(sigma, 3), 'train races', len(train), flush=True)
    (OUT / 'pace_norms.json').write_text(json.dumps(out), encoding='utf-8')
    print('races', len(races), 'with pace', int(races.pace.notna().sum()))


if __name__ == '__main__':
    main()
