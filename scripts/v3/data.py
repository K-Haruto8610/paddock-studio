"""v3 S1 data: every JRA flat race since 2020 (named or not) with the leader's laps and each horse's times (read-only DB).

  python -I scripts/v3/data.py [--today YYYY-MM-DD]     -> outputs/v3/runs.pkl, outputs/v3/races.pkl

races : RACE_CODE, date, venue, track, kind, distance, surface, going, cls (class level, pace_model.class_level), field,
        named (race name and every horse name clean, the v2.0 training rule), complete (every starter present),
        lead (cumulative time of the leader at every 200 m mark counted back from the post, from LAP_TIME1..; NaN when
        the laps are missing).
runs  : one row per starter: horse id, UMABAN, gate (WAKUBAN), age, sex, carried weight (kg), blinker, normal (finished
        normally), soha (finish time s), last3, t600 (= soha - last3, the horse's own time at the 600 m-to-go mark),
        finish_gap (s behind the winner), fin (finishing place), r1..r4 (normalised places at the physical corners,
        the last pass, as pace_model), g1..g4 (lengths from the passing strings, pace_model.parse_passing).
Odds, popularity and prize columns are never read.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from build_forecasts import settings, number, clean  # noqa: E402
import pace_model as pm  # noqa: E402
import courses as cs  # noqa: E402

OUT = pm.ROOT / 'outputs' / 'v3'
NORMAL_IJO = ('', '0')


def load(today_s):
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    laps = ','.join(f'r.LAP_TIME{i}' for i in range(1, 26))
    with c.cursor() as q:
        q.execute(f"""SELECT r.RACE_CODE,r.KAISAI_NEN,r.KAISAI_GAPPI,r.KEIBAJO_CODE,r.KYOSOMEI_HONDAI,r.GRADE_CODE,
          r.KYOSO_JOKEN_CODE_SAIJAKUNEN,r.KYORI,r.TRACK_CODE,r.SHUSSO_TOSU,r.SHIBA_BABAJOTAI_CODE,r.DIRT_BABAJOTAI_CODE,{laps},
          r.CORNER1,r.KAKU_TSUKA_JUNI1,r.CORNER2,r.KAKU_TSUKA_JUNI2,r.CORNER3,r.KAKU_TSUKA_JUNI3,r.CORNER4,r.KAKU_TSUKA_JUNI4,
          u.UMABAN,u.WAKUBAN,u.KETTO_TOROKU_BANGO,u.BAMEI,u.SEIBETSU_CODE,u.BAREI,u.FUTAN_JURYO,u.BLINKER_SHIYO_KUBUN,
          u.IJO_KUBUN_CODE,u.KAKUTEI_CHAKUJUN,u.SOHA_TIME,u.TIME_SA,u.KOHAN_3F,
          u.CORNER1_JUNI,u.CORNER2_JUNI,u.CORNER3_JUNI,u.CORNER4_JUNI
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026'
          AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          ORDER BY r.KAISAI_NEN,r.KAISAI_GAPPI,r.RACE_CODE,u.UMABAN""")
        rows = pd.DataFrame(q.fetchall())
    c.close()
    rows['date'] = rows.KAISAI_NEN + rows.KAISAI_GAPPI
    return rows[rows.date < today_s].drop_duplicates(['RACE_CODE', 'UMABAN']).reset_index(drop=True)


def leader_marks(r):
    """Cumulative leader time at the 200 m marks counted back from the post: index k = time at distance - 200 k ...
    returned in race order (from the first mark after the start to the post). The first lap is distance mod 200
    (or 200)."""
    dist = int(r.KYORI)
    n = int(np.ceil(dist / 200))
    lap = np.array([number(r[f'LAP_TIME{i}']) for i in range(1, n + 1)], float) / 10
    if len(lap) < n or np.isnan(lap).any() or (lap <= 0).any() or lap.sum() < dist / 20:
        return None
    return np.cumsum(lap)


def build(rows):
    for col in ['KYORI', 'TRACK_CODE', 'SHUSSO_TOSU', 'UMABAN', 'WAKUBAN', 'BAREI', 'FUTAN_JURYO', 'KOHAN_3F', 'KAKUTEI_CHAKUJUN']:
        rows[col] = rows[col].map(number)
    rows = rows[rows.KYORI.between(1000, 3600) & rows.SHUSSO_TOSU.between(2, 18) & rows.UMABAN.between(1, 18)].copy()
    first = rows.drop_duplicates('RACE_CODE').set_index('RACE_CODE')
    names_ok = rows.groupby('RACE_CODE').BAMEI.agg(lambda s: bool(s.map(clean).all()))
    races = pd.DataFrame({
        'date': first.date, 'venue': first.KEIBAJO_CODE.astype(int), 'track': first.TRACK_CODE.astype(int),
        'distance': first.KYORI, 'field': first.SHUSSO_TOSU, 'surface': (first.TRACK_CODE >= 23).astype(int),
        'name_ok': first.KYOSOMEI_HONDAI.map(clean) & names_ok,
        'cls': [pm.class_level(g, k) for g, k in zip(first.GRADE_CODE, first.KYOSO_JOKEN_CODE_SAIJAKUNEN)],
    })
    going = np.where(races.surface == 1, first.DIRT_BABAJOTAI_CODE.map(number), first.SHIBA_BABAJOTAI_CODE.map(number))
    races['going'] = pd.Series(going, index=races.index).where(lambda g: g.between(1, 4))
    races['kind'] = [cs.kind_of(v, t) for v, t in zip(races.venue, races.track)]
    races['lead'] = [leader_marks(r) for _, r in first.iterrows()]
    counts = rows.groupby('RACE_CODE').UMABAN.nunique()
    races['complete'] = counts.reindex(races.index) == races.field
    races['named'] = races.name_ok
    races = races.drop(columns='name_ok')

    runs = pd.DataFrame({
        'RACE_CODE': rows.RACE_CODE, 'date': rows.date, 'horse': rows.KETTO_TOROKU_BANGO, 'UMABAN': rows.UMABAN.astype(int),
        'waku': rows.WAKUBAN, 'age': rows.BAREI, 'sex': rows.SEIBETSU_CODE.map(number), 'weight': rows.FUTAN_JURYO / 10,
        'blinker': rows.BLINKER_SHIYO_KUBUN.map({'0': 0, '1': 1}).fillna(0),
        'normal': rows.IJO_KUBUN_CODE.fillna('0').str.strip().isin(NORMAL_IJO),
        'fin': rows.KAKUTEI_CHAKUJUN.where(rows.KAKUTEI_CHAKUJUN.between(1, 18)),
    })
    runs['soha'] = rows.SOHA_TIME.map(pm.soha_seconds)
    runs['last3'] = (rows.KOHAN_3F / 10).where(rows.KOHAN_3F.between(280, 500))
    runs['t600'] = runs.soha - runs.last3
    sa = rows.TIME_SA.map(lambda v: number(str(v).strip()) if str(v).strip() not in ('', '9999') else np.nan)
    runs['finish_gap'] = (sa.clip(lower=0) / 10).where(sa.abs() < 9000)
    field = rows.RACE_CODE.map(races.field)
    for c in (1, 2, 3, 4):
        rank = rows[f'CORNER{c}_JUNI'].map(number)
        runs[f'r{c}'] = ((rank - 1) / (field - 1).clip(lower=1)).where(rank.between(1, field))
        runs[f'g{c}'] = np.nan
    # gaps from the passing strings (pace_model.add_labels: a repeated corner keeps the latest pass)
    g = {c: np.full(len(runs), np.nan) for c in (1, 2, 3, 4)}
    pos = pd.Series(np.arange(len(runs)), index=runs.index)
    for code, idx in rows.groupby('RACE_CODE').groups.items():
        r = rows.loc[idx[0]]
        nums = rows.loc[idx, 'UMABAN'].astype(int).to_numpy()
        for k in (1, 2, 3, 4):
            corner = str(r[f'CORNER{k}'] or '0').strip()
            if corner not in ('1', '2', '3', '4'):
                continue
            try:
                parsed = pm.parse_passing(r[f'KAKU_TSUKA_JUNI{k}'])
            except ValueError:
                continue
            if not parsed or set(parsed) != set(nums):
                continue
            g[int(corner)][pos[idx].to_numpy()] = [parsed[n][0] for n in nums]
    for c in (1, 2, 3, 4):
        runs[f'g{c}'] = g[c]
    return races, runs.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--today')
    args = ap.parse_args()
    today_s = (args.today or date.today().isoformat()).replace('-', '')
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load(today_s)
    races, runs = build(rows)
    import rating_index as ri
    rt = ri.ratings(ri.load_runs(today_s))
    runs = runs.merge(rt[['RACE_CODE', 'UMABAN', 'rating_pre']], on=['RACE_CODE', 'UMABAN'], how='left')
    races.to_pickle(OUT / 'races.pkl')
    runs.to_pickle(OUT / 'runs.pkl')
    print(f'races {len(races)} (named {int(races.named.sum())}, complete named {int((races.named & races.complete).sum())}, '
          f'with laps {int(races.lead.notna().sum())}); runs {len(runs)}; t600 {runs.t600.notna().mean():.4f}; '
          f'rating {runs.rating_pre.notna().mean():.4f}')


if __name__ == '__main__':
    main()
