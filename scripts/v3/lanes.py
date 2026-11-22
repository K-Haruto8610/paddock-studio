"""v3 S3 labels: each horse's lane inside its abreast group at the physical corners (read-only DB).

  python -I scripts/v3/lanes.py      -> outputs/v3/lanes.pkl (RACE_CODE, UMABAN, l1..l4)

The label is the one E0 scores (pace_model.parse_passing): the position inside an abreast group of the passing string
counted from the inside, 0 = innermost or running alone. Same rule as data.py's g1..g4: a corner reported twice keeps
its latest pass; a string that does not list exactly the race's starters is left out. Only races in races.pkl.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_forecasts import settings  # noqa: E402
import pace_model as pm  # noqa: E402

OUT = pm.ROOT / 'outputs' / 'v3'


def main():
    races = pd.read_pickle(OUT / 'races.pkl')
    runs = pd.read_pickle(OUT / 'runs.pkl')[['RACE_CODE', 'UMABAN']]
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    cols = ','.join(f'CORNER{k},KAKU_TSUKA_JUNI{k}' for k in (1, 2, 3, 4))
    with c.cursor() as q:
        q.execute(f"""SELECT RACE_CODE,{cols} FROM race_shosai WHERE KAISAI_NEN BETWEEN '2020' AND '2026'
          AND KEIBAJO_CODE BETWEEN '01' AND '10' AND TRACK_CODE BETWEEN '10' AND '29'""")
        strings = pd.DataFrame(q.fetchall()).drop_duplicates('RACE_CODE').set_index('RACE_CODE')
    c.close()
    strings = strings[strings.index.isin(races.index)]
    nums = runs.groupby('RACE_CODE').UMABAN.agg(lambda s: set(s.astype(int)))
    out = {k: {} for k in (1, 2, 3, 4)}
    for code, r in strings.iterrows():
        for k in (1, 2, 3, 4):
            corner = str(r[f'CORNER{k}'] or '0').strip()
            if corner not in ('1', '2', '3', '4'):
                continue
            try:
                parsed = pm.parse_passing(r[f'KAKU_TSUKA_JUNI{k}'])
            except ValueError:
                continue
            if not parsed or set(parsed) != nums.get(code, set()):
                continue
            for n, v in parsed.items():
                out[int(corner)][(code, n)] = v[1]
    lab = runs.copy()
    for k in (1, 2, 3, 4):
        lab[f'l{k}'] = [out[k].get((a, int(b)), np.nan) for a, b in zip(lab.RACE_CODE, lab.UMABAN)]
    lab.to_pickle(OUT / 'lanes.pkl')
    print('lanes', {k: float(lab[f'l{k}'].notna().mean()) for k in (1, 2, 3, 4)},
          'zero share', {k: float((lab[f'l{k}'] == 0).sum() / lab[f'l{k}'].notna().sum()) for k in (1, 2, 3, 4)})


if __name__ == '__main__':
    main()
