"""追加指示10 フェーズ0 (2026-11-19): DB の事前確認4件と、F2・F3 の土台になる文脈指数の健全性。モデルは変えない。

  python scripts/experiment_f0_context.py db        # 1 overseas records / 2 weather / 3 jockey affiliation / 4 distance change
  python scripts/experiment_f0_context.py indices   # opponent rating + day clock: health checks
  python scripts/experiment_f0_context.py cheap     # rest interval (overall effect) and distance change on the v1.8 frame
  python scripts/experiment_f0_context.py frame     # what changed in the DB since the v1.8 frame

Results go to outputs/f0-context/*.json. The indices themselves are in scripts/context_index.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_index as ci  # noqa: E402
import eval_standard as es  # noqa: E402
import pace_model as pm  # noqa: E402
from build_forecasts import settings  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
OUT = ci.CACHE
FRAME = pm.ROOT / 'outputs' / 'data-expansion-v18' / 'frame_2022.pkl'
REP = ('20250901', '20251231')
SEL = ('20250601', '20250831')


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def sql(q):
    import pymysql
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as cur:
        cur.execute(q)
        rows = cur.fetchall()
    c.close()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- db
def cmd_db(_):
    res = {}
    # 1. overseas records
    ov = sql("""SELECT r.KEIBAJO_CODE, r.RACE_CODE, concat(r.KAISAI_NEN, r.KAISAI_GAPPI) AS date, u.KETTO_TOROKU_BANGO AS horse,
                u.SOHA_TIME, u.KOHAN_3F, u.CORNER4_JUNI, u.KAKUTEI_CHAKUJUN
                FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
                WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026' AND r.KEIBAJO_CODE REGEXP '^[A-Z]'""")
    ov['year'] = ov.date.str[:4]
    ov['time_ok'] = ov.SOHA_TIME.map(pm.soha_seconds).notna()
    ov['l3_ok'] = pd.to_numeric(ov.KOHAN_3F, errors='coerce').between(280, 500)
    ov['c4_ok'] = pd.to_numeric(ov.CORNER4_JUNI, errors='coerce').between(1, 30)
    runs = ci.load_runs()
    jra = runs[['date', 'KETTO_TOROKU_BANGO', 'RACE_CODE', 'UMABAN']].rename(columns={'KETTO_TOROKU_BANGO': 'horse'})
    ov_last = ov.sort_values('date').drop_duplicates(['horse', 'date'])
    ovd = ov_last[['date', 'horse']].assign(ov_date=lambda x: pd.to_datetime(x.date, format='%Y%m%d')).drop(columns='date').sort_values('ov_date')
    j = jra.assign(dt=pd.to_datetime(jra.date, format='%Y%m%d')).sort_values('dt')
    m = pd.merge_asof(j, ovd, left_on='dt', right_on='ov_date', by='horse', direction='backward', allow_exact_matches=False)
    m['since'] = (m.dt - m.ov_date).dt.days
    named = pd.read_pickle(FRAME)[['RACE_CODE', 'UMABAN']]
    mm = m.merge(named.assign(UMABAN=named.UMABAN.astype(int)), on=['RACE_CODE', 'UMABAN'])
    res['overseas'] = {'races': int(ov.RACE_CODE.nunique()), 'rows': int(len(ov)), 'horses': int(ov.horse.nunique()),
                       'racesByYear': {k: int(v) for k, v in ov.groupby('year').RACE_CODE.nunique().items()},
                       'rowsWithTime': int(ov.time_ok.sum()), 'rowsWithLast3': int(ov.l3_ok.sum()), 'rowsWithCorner4': int(ov.c4_ok.sum()),
                       'juraRunsWithin270dAfterOverseas_all': int((m.since <= 270).sum()),
                       'juraRunsWithin270dAfterOverseas_frame': int((mm.since <= 270).sum()),
                       'juraRunsWithin120dAfterOverseas_frame': int((mm.since <= 120).sum()), 'frameRows': int(len(mm)),
                       'venues': {k: int(v) for k, v in ov.groupby('KEIBAJO_CODE').RACE_CODE.nunique().sort_values(ascending=False).head(8).items()}}
    # 2. weather
    w = runs.drop_duplicates('RACE_CODE')
    res['weather'] = {'codeShareByYear': {y: {str(int(k)): round(float(v), 4) for k, v in g.weather.value_counts(normalize=True).items()}
                                         for y, g in w.groupby(w.date.str[:4])},
                      'missingByYear': {y: float(g.weather.isna().mean() + (g.weather == 0).mean()) for y, g in w.groupby(w.date.str[:4])}}
    # 3. jockey affiliation
    km = sql("SELECT KISHUMEI_RYAKUSHO AS jockey, TOZAI_SHOZOKU_CODE AS tozai, KISHU_CODE FROM kishu_master")
    km['jockey'] = km.jockey.map(pm.jockey_name)
    km = km.dropna(subset=['jockey']).drop_duplicates('jockey')
    codes = sql("SELECT CODE, MEISHO1, MEISHO2 FROM tozai_shozoku_code") if False else None
    r25 = runs[runs.date.str[:4] == '2025']
    names = r25.KISHUMEI_RYAKUSHO.map(pm.jockey_name)
    t = names.map(km.set_index('jockey').tozai.to_dict())
    res['jockeyAffiliation'] = {'masterRows': int(len(km)), 'masterTozaiShare': {str(k): int(v) for k, v in km.tozai.value_counts().items()},
                                'rides2025': int(len(r25)), 'ridesWithAffiliation': float(t.notna().mean()),
                                'affiliationShareOfRides': {str(k): round(float(v), 4) for k, v in t.value_counts(normalize=True).items()},
                                'note': 'kishu_master is a current snapshot (not as-of the ride date); moves between 美浦/栗東 are rare but a retroactive column.',
                                'horseStableInRunTable': float(runs.TOZAI_SHOZOKU_CODE.astype(str).str.strip().isin(['1', '2']).mean())}
    # 4. distance change among the current features
    res['distanceChangeInFeatures'] = {'FEATURES': pm.FEATURES, 'hasPreviousDistance': any('dist' in f and f != 'distance' for f in pm.FEATURES),
                                       'MOVE_FEATURES_extra': [f for f in pm.MOVE_FEATURES if f not in pm.FEATURES]}
    save('db_checks.json', res)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


# ------------------------------------------------------------------------------------------------------ indices
def tau_by_race(df, col, target='fin_y', min_n=5):
    out = []
    for _, x in df.groupby('RACE_CODE'):
        if len(x) >= min_n:
            v = es.kendall(x[col].to_numpy(), x[target].to_numpy())
            if not np.isnan(v):
                out.append(v)
    return float(np.mean(out)), len(out)


def cmd_indices(_):
    runs = ci.load_runs()
    runs = runs.sort_values(['date', 'RACE_CODE', 'UMABAN']).reset_index(drop=True)
    print(f'runs {len(runs)}, races {runs.RACE_CODE.nunique()}, {runs.date.min()}..{runs.date.max()}', flush=True)
    res = {}
    # ---- opponent rating
    rt = ci.ratings(runs)
    pl = ci.plain_form(runs)
    rt = rt.merge(pl, on=['RACE_CODE', 'UMABAN'], how='left')
    rt.to_pickle(OUT / 'ratings.pkl')
    m = runs[['RACE_CODE', 'UMABAN', 'date', 'KAKUTEI_CHAKUJUN', 'SHUSSO_TOSU', 'cls', 'KETTO_TOROKU_BANGO']].merge(rt, on=['RACE_CODE', 'UMABAN'])
    m['fin_y'] = -m.KAKUTEI_CHAKUJUN
    rows = {}
    for name, (a, b) in {'selection 2025-06..08': SEL, 'report 2025-09..12': REP, 'all 2022-2025': ('20220101', '20251231')}.items():
        x = m[m.date.between(a, b)]
        row = {'races': int(x.RACE_CODE.nunique()), 'runs': int(len(x)), 'ratedShare': float((x.rating_conf > 0).mean())}
        known = x[x.rating_conf > 0]
        for col in ('rating_pre', 'plain_pre'):
            row[f'tau_{col}'], row['tauRaces'] = tau_by_race(x, col)
            # also among races where >= 5 horses have a history, scored on those horses only
            row[f'tauKnown_{col}'], row['tauKnownRaces'] = tau_by_race(known, col)
        row['ratingSd'] = float(x.rating_pre.std())
        row['oppMeanSd'] = float(x.opp_mean.std())
        rows[name] = row
    res['rating'] = rows
    byyear = {y: {'meanRating': float(g.rating_pre.mean()), 'sd': float(g.rating_pre.std()), 'ratedShare': float((g.rating_conf > 0).mean())}
              for y, g in m.groupby(m.date.str[:4])}
    res['ratingByYear'] = byyear
    # opponent term: does the field strength predict how a horse's own result differs from what its rating says?
    x = m[m.date.between('20220101', '20251231') & (m.rating_conf > 0)]
    res['opponentAdjustment'] = {'corr_oppMean_vs_perfRaw': float(np.corrcoef(x.opp_mean, x.perf_raw)[0, 1]),
                                 'note': 'perf_raw is the finish position y; its mean within a race is 0, so a correlation with opp_mean would be a bias'}
    # leakage check: ratings as of a day must not move when the future is dropped
    cut = '20250301'
    early = ci.ratings(runs[runs.date < cut])
    both = early.merge(rt, on=['RACE_CODE', 'UMABAN'], suffixes=('_cut', ''))
    res['leakCheck'] = {'rows': int(len(both)), 'maxAbsDiff': float((both.rating_pre_cut - both.rating_pre).abs().max()),
                        'oppMaxAbsDiff': float((both.opp_mean_cut - both.opp_mean).abs().max())}
    # ---- day clock
    t = ci.race_table(runs)
    c = ci.day_clock(t)
    c.to_pickle(OUT / 'clock.pkl')
    jra_all = c[c.date.between('20220101', '20251231')]
    ck = {}
    for surf, name in ((0, 'turf'), (1, 'dirt')):
        g = jra_all[(jra_all.surface == surf) & jra_all.dev_t.notna() & (jra_all.n_clock_full >= 3)]
        gd = pd.get_dummies(g.going.fillna(0).astype(int), prefix='g', dtype=float)
        ck[name] = {'races': int(len(g)), 'devT_sd': float(g.dev_t.std()), 'corr_devT_clockFull': float(np.corrcoef(g.dev_t, g.clock_full)[0, 1]),
                    'R2_clockFull': ci.r2(g.dev_t, g[['clock_full']])[0], 'R2_goingDummies': ci.r2(g.dev_t, gd.to_numpy())[0],
                    'R2_both': ci.r2(g.dev_t, np.column_stack([g.clock_full, gd.to_numpy()]))[0],
                    'R2_clockPrior': ci.r2(g.dev_t, g[['clock_prior']])[0] if g.clock_prior.notna().sum() > 100 else None,
                    'meanClockByGoing': {str(int(k)): float(v) for k, v in g.groupby('going').clock_full.mean().items()}}
        g3 = jra_all[(jra_all.surface == surf) & jra_all.dev_l3.notna() & (jra_all.n_clock3_full >= 3)]
        ck[name]['closing'] = {'races': int(len(g3)), 'devL3_sd': float(g3.dev_l3.std()), 'corr_devL3_clock3Full': float(np.corrcoef(g3.dev_l3, g3.clock3_full)[0, 1]),
                               'R2_clock3Full': ci.r2(g3.dev_l3, g3[['clock3_full']])[0],
                               'meanClock3ByGoing': {str(int(k)): float(v) for k, v in g3.groupby('going').clock3_full.mean().items()}}
        # split-half reliability: even- vs odd-numbered other races of the day
    rel = []
    for key, grp in jra_all.groupby(['date', 'venue', 'surface']):
        if grp.dev_t.notna().sum() >= 8:
            v = grp.dev_t.dropna().to_numpy()
            rel.append((v[0::2].mean(), v[1::2].mean()))
    rel = np.array(rel)
    ck['splitHalfDayReliability'] = {'days': int(len(rel)), 'corr': float(np.corrcoef(rel[:, 0], rel[:, 1])[0, 1])}
    rel3 = []
    for key, grp in jra_all.groupby(['date', 'venue', 'surface']):
        if grp.dev_l3.notna().sum() >= 8:
            v = grp.dev_l3.dropna().to_numpy()
            rel3.append((v[0::2].mean(), v[1::2].mean()))
    rel3 = np.array(rel3)
    ck['splitHalfDayReliabilityClosing'] = {'days': int(len(rel3)), 'corr': float(np.corrcoef(rel3[:, 0], rel3[:, 1])[0, 1])}
    # leakage check: a race's own time must not matter
    c2 = t.copy()
    pick = c[(c.date == '20250614') & c.t_win.notna() & (c.n_clock_prior >= 2)].RACE_CODE.iloc[0]  # a race with earlier races that day
    i = c2.index[c2.RACE_CODE == pick][0]
    c2.loc[i, 't_win'] += 5.0
    c2.loc[i, 'l3_med'] += 2.0
    a, b = ci.day_clock(c2), c.set_index('RACE_CODE')
    own = a.set_index('RACE_CODE').loc[t.RACE_CODE[i]]
    ck['ownTimeCheck'] = {'clock_full_changed': float(abs(own.clock_full - b.loc[t.RACE_CODE[i]].clock_full)),
                          'clock_prior_changed': float(abs(own.clock_prior - b.loc[t.RACE_CODE[i]].clock_prior)),
                          'note': 'race altered by +5 s / +2 s; its own clock must not move (differences must be 0 or NaN)'}
    ck['coverage'] = {'racesWithClockFull': float((jra_all.n_clock_full >= 1).mean()), 'racesWithClockPrior': float((jra_all.n_clock_prior >= 1).mean()),
                      'meanOthers': float(jra_all.n_clock_full.mean()), 'meanEarlier': float(jra_all.n_clock_prior.mean())}
    res['dayClock'] = ck
    save('indices.json', res)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


# --------------------------------------------------------------------------------------------------- cheap checks
def cmd_cheap(_):
    import experiment_ewm_history as e1
    runs = ci.load_runs().sort_values(['date', 'RACE_CODE', 'UMABAN'])
    g = runs.groupby('KETTO_TOROKU_BANGO')
    runs['prev_date'] = g.date.shift()
    runs['prev_distance'] = g.KYORI.shift()
    runs['prev_surface'] = g.surface.shift()
    runs['rest'] = (pd.to_datetime(runs.date, format='%Y%m%d') - pd.to_datetime(runs.prev_date, format='%Y%m%d')).dt.days
    d = e1.frame(0.2)
    d = d.merge(runs[['RACE_CODE', 'UMABAN', 'rest', 'prev_distance', 'prev_surface']].assign(UMABAN=lambda x: x.UMABAN.astype(float)),
                on=['RACE_CODE', 'UMABAN'], how='left')
    d = d[d.complete & (d.date < SEL[0])]
    d['dchg'] = d.distance - d.prev_distance
    res = {'rows': int(len(d))}
    targets = ['r1', 'r4', 'last3']
    for t in targets:
        d[f'res_{t}'] = d[t] - d[f'past_{t}']
    rest_b = pd.cut(d.rest, [-1, 13, 27, 55, 111, 10000], labels=['<=13', '14-27', '28-55', '56-111', '112+']).astype(object).fillna('none')
    dist_b = pd.cut(d.dchg, [-10000, -401, -1, 0, 400, 10000], labels=['<=-400', '-399..-1', 'same', '+1..+400', '>=+400']).astype(object).fillna('none')
    for nm, b in (('rest', rest_b), ('distanceChange', dist_b)):
        tab = {}
        for lv, x in d.groupby(b):
            row = {'n': int(len(x))}
            for t in targets:
                v = x[f'res_{t}'].dropna()
                row[t] = {'mean': float(v.mean()), 'se': float(v.std() / np.sqrt(len(v)))} if len(v) > 30 else None
            tab[str(lv)] = row
        res[nm] = tab
    save('cheap_checks.json', res)
    print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------------------------------------------- frame
def cmd_frame(_):
    d = pd.read_pickle(FRAME)
    runs = ci.load_runs()
    c = runs.drop_duplicates('RACE_CODE')
    now = sql("""SELECT KAISAI_NEN AS y, count(*) AS races FROM race_shosai WHERE KEIBAJO_CODE BETWEEN '01' AND '10' AND TRACK_CODE BETWEEN '10' AND '29'
                 AND KAISAI_NEN BETWEEN '2020' AND '2026' GROUP BY 1 ORDER BY 1""")
    res = {'frameRows': int(len(d)), 'frameFirstDate': d.date.min(), 'frameCompleteRaces': int(d[d.complete].RACE_CODE.nunique()),
           'frameYearsRaces': {y: int(g.RACE_CODE.nunique()) for y, g in d.groupby(d.date.str[:4])},
           'dbFlatRacesByYear': {r.y: int(r.races) for r in now.itertuples()},
           'dbRunsTable': {y: int(g.RACE_CODE.nunique()) for y, g in c.groupby(c.date.str[:4])}}
    save('frame_check.json', res)
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['db', 'indices', 'cheap', 'frame'])
    args = ap.parse_args()
    {'db': cmd_db, 'indices': cmd_indices, 'cheap': cmd_cheap, 'frame': cmd_frame}[args.cmd](args)


if __name__ == '__main__':
    main()
