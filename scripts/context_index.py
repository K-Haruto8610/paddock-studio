"""Shared context indices for the F1-F3 experiments (追加指示10, phase 0, 2026-11-19). Read-only DB, no model change.

Two indices, both built from ALL JRA flat races since 2020 (named or not; the training frame only keeps named races):

* opponent rating (`ratings`): every horse carries a time-series rating that is updated ONLY with results of earlier days
  (a whole race day is applied after the day, so same-day races never see each other) and weighted exponentially with the
  alpha E1 adopted (0.2). A run's performance is `perf = mean rating of the OTHER horses before the race + y`, where y in
  [-1, 1] is the finish position of the run (winner +1). The rating before a race is the shrunk exponentially weighted
  mean of the horse's earlier `perf`, so a past result is read against the field it was earned in. (The scale of y is
  arbitrary: every quantity is linear in it, so it does not matter for trees.) Exposed per run: `rating_pre`,
  `rating_conf` (weight behind it), `opp_mean` (the other horses' mean rating before the race), `perf`, `rating_pct`
  (percentile of the horse's rating in today's field).

* day clock (`day_clock`): how fast the track was on a day, from the OTHER races of the same day, venue and surface.
  Each race's time deviation from its par (winner time in % of par; field-median closing 3F in seconds) is averaged over
  the other races of that day/venue/surface (shrunk toward 0 by count). The par of a (venue, track, distance, class, age
  group) comes from races before NORM_END only, like the pace norms (no later period leaks in).
  `clock_full*` uses every other race of the day: valid for a PAST run (the target race is on a later date).
  `clock_prior*` uses only races with a smaller race number on that day: what was known before the race itself started.
  The race itself never enters its own index.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm  # noqa: E402
from build_forecasts import settings, number  # noqa: E402

ROOT = pm.ROOT
CACHE = ROOT / 'outputs' / 'f0-context'
NORM_END = pm.PACE_NORM_END          # '20250601': par times from the earliest training window only
RATING_ALPHA = 0.2                    # E1's history alpha
RATING_PRIOR = 1.0                    # weight of the prior (rating 0) in the shrunk mean
CLOCK_SHRINK = 1.5                    # pseudo-count shrinking a day's clock toward 0
NORMAL_IJO = ['', '0', '3', '4', '5', '6', '7']  # same rule as pace_model.load_frame


def load_runs(refresh=False):
    """One row per run of every JRA flat race 2020+ (venue 01-10, TRACK_CODE 10-29). Cached (pickle) because the query is slow."""
    f = CACHE / 'runs.pkl'
    if f.exists() and not refresh:
        return pd.read_pickle(f)
    import pymysql
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute("""SELECT r.RACE_CODE,r.KAISAI_NEN,r.KAISAI_GAPPI,r.KEIBAJO_CODE,r.RACE_BANGO,r.KYOSO_SHUBETSU_CODE,r.GRADE_CODE,
          r.KYOSO_JOKEN_CODE_SAIJAKUNEN,r.KYORI,r.TRACK_CODE,r.SHUSSO_TOSU,r.TENKO_CODE,r.SHIBA_BABAJOTAI_CODE,r.DIRT_BABAJOTAI_CODE,
          r.ZENHAN_3F,u.UMABAN,u.WAKUBAN,u.KETTO_TOROKU_BANGO,u.BAREI,u.SEIBETSU_CODE,u.KISHUMEI_RYAKUSHO,u.TOZAI_SHOZOKU_CODE,
          u.IJO_KUBUN_CODE,u.KAKUTEI_CHAKUJUN,u.SOHA_TIME,u.KOHAN_3F,u.CORNER1_JUNI,u.CORNER2_JUNI,u.CORNER3_JUNI,u.CORNER4_JUNI
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026'
          AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          ORDER BY r.KAISAI_NEN,r.KAISAI_GAPPI,r.RACE_CODE,u.UMABAN""")
        d = pd.DataFrame(q.fetchall())
    c.close()
    d['date'] = d.KAISAI_NEN + d.KAISAI_GAPPI
    for col in ['KYORI', 'TRACK_CODE', 'SHUSSO_TOSU', 'UMABAN', 'WAKUBAN', 'RACE_BANGO', 'BAREI', 'KAKUTEI_CHAKUJUN', 'KOHAN_3F',
                'ZENHAN_3F', 'KEIBAJO_CODE'] + [f'CORNER{i}_JUNI' for i in pm.CORNERS]:
        d[col] = pd.to_numeric(d[col].astype(str).str.strip(), errors='coerce')
    d['normal'] = d.IJO_KUBUN_CODE.fillna('0').str.strip().isin(NORMAL_IJO)
    d['surface'] = (d.TRACK_CODE >= 23).astype(int)
    d['soha'] = d.SOHA_TIME.map(pm.soha_seconds)
    d['last3'] = (d.KOHAN_3F / 10).where(d.KOHAN_3F.between(280, 500))
    d['early3'] = (d.ZENHAN_3F / 10).where(d.ZENHAN_3F.between(300, 450))
    going = np.where(d.surface == 1, pd.to_numeric(d.DIRT_BABAJOTAI_CODE, errors='coerce'), pd.to_numeric(d.SHIBA_BABAJOTAI_CODE, errors='coerce'))
    d['going'] = pd.Series(going, index=d.index).where(lambda g: g.between(1, 4))
    d['weather'] = pd.to_numeric(d.TENKO_CODE, errors='coerce')
    d['cls'] = [pm.class_level(g, k) for g, k in zip(d.GRADE_CODE, d.KYOSO_JOKEN_CODE_SAIJAKUNEN)]
    sh = d.KYOSO_SHUBETSU_CODE.astype(str).str.strip()
    d['ageg'] = np.where(sh == '11', 0, np.where(sh == '12', 1, 2))
    d = d.drop_duplicates(['RACE_CODE', 'UMABAN']).reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_pickle(f)
    return d


# ------------------------------------------------------------------------------------------------ opponent rating
from rating_index import ratings  # noqa: E402,F401  (the rating itself lives in rating_index.py so the model can use it without this module)


def plain_form(runs, alpha=RATING_ALPHA, prior=RATING_PRIOR):
    """The same exponentially weighted, shrunk mean of the finish position y WITHOUT the opponent term (baseline for the rating)."""
    ok = runs[runs.normal]
    out, num, den = [], {}, {}
    for date, day in ok.groupby('date', sort=True):
        updates = []
        for _, x in day.groupby('RACE_CODE', sort=False):
            n = len(x)
            if n < 3:
                continue
            ids = x.KETTO_TOROKU_BANGO.to_numpy()
            r = np.array([num.get(h, 0.0) / (den.get(h, 0.0) + prior) for h in ids])
            fin = x.KAKUTEI_CHAKUJUN.to_numpy(float)
            fin_ok = (fin >= 1) & (fin <= 18)
            m = int(fin_ok.sum())
            out.append(pd.DataFrame({'RACE_CODE': x.RACE_CODE.to_numpy(), 'UMABAN': x.UMABAN.to_numpy(), 'plain_pre': r}))
            if m >= 3:
                fr = pd.Series(np.where(fin_ok, fin, np.nan)).rank().to_numpy()
                y = ((m + 1) / 2 - fr) / ((m - 1) / 2)
                updates.extend((h, v) for h, v, f in zip(ids, y, fin_ok) if f)
        for h, y in updates:
            num[h] = (1 - alpha) * num.get(h, 0.0) + y
            den[h] = (1 - alpha) * den.get(h, 0.0) + 1.0
    return pd.concat(out, ignore_index=True)


# -------------------------------------------------------------------------------------------------- day clock
KEYS = [['venue', 'track', 'distance', 'cls', 'ageg'], ['venue', 'track', 'distance', 'cls'], ['surface', 'distance', 'cls'],
        ['surface', 'distance']]


def race_table(runs):
    """One row per race: winner time, field-median closing 3F, class, age group, going, weather, race number."""
    r = runs[runs.normal].copy()
    win = r[r.KAKUTEI_CHAKUJUN == 1].groupby('RACE_CODE').soha.min()
    med3 = r.groupby('RACE_CODE').last3.median()
    first = runs.groupby('RACE_CODE').first()
    t = pd.DataFrame({'date': first.date, 'venue': first.KEIBAJO_CODE.astype(int), 'track': first.TRACK_CODE.astype(int),
                      'surface': first.surface, 'distance': first.KYORI, 'cls': first.cls, 'ageg': first.ageg, 'going': first.going,
                      'weather': first.weather, 'race_no': first.RACE_BANGO, 'n': first.SHUSSO_TOSU})
    t['t_win'], t['l3_med'] = win, med3
    t['cls'] = t.cls.fillna(-1)  # unknown class is its own group
    return t.reset_index()


def par_times(t, norm_end=NORM_END):
    """par winner time and par median closing 3F per race, from races before norm_end with the first key level that has >= 4 races."""
    base = t[t.date < norm_end]
    par = {}
    for col in ('t_win', 'l3_med'):
        v = pd.Series(np.nan, index=t.index)
        for keys in KEYS[::-1]:  # finest key last so it overwrites where it has enough races
            g = base.groupby(keys)[col].agg(['median', 'count'])
            g = g[g['count'] >= 4]['median']
            m = pd.Series(t[keys].merge(g.rename('par'), left_on=keys, right_index=True, how='left').par.to_numpy(), index=t.index)
            v = m.where(m.notna(), v)
        par[col] = v
    return par


def day_clock(t, norm_end=NORM_END, shrink=CLOCK_SHRINK):
    """Add dev_t (winner time, % slower than par), dev_l3 (median closing 3F, s slower than par) and the day clocks
    clock_full / clock_prior (time) and clock3_full / clock3_prior (closing) of the race's day/venue/surface, each computed from
    OTHER races only; n_clock_full / n_clock_prior = how many races stand behind them."""
    t = t.copy()
    par = par_times(t, norm_end)
    t['dev_t'] = 100 * (t.t_win - par['t_win']) / par['t_win']
    t['dev_l3'] = t.l3_med - par['l3_med']
    t = t.sort_values(['date', 'venue', 'surface', 'race_no']).reset_index(drop=True)
    grp = t.groupby(['date', 'venue', 'surface'], sort=False)
    for col, name in (('dev_t', 'clock'), ('dev_l3', 'clock3')):
        v = t[col]
        valid = v.notna().astype(float)
        s_all = v.fillna(0).groupby([t.date, t.venue, t.surface]).transform('sum')
        n_all = valid.groupby([t.date, t.venue, t.surface]).transform('sum')
        n_oth = n_all - valid
        t[f'{name}_full'] = ((s_all - v.fillna(0)) / n_oth.where(n_oth > 0)) * (n_oth / (n_oth + shrink))
        t[f'n_{name}_full'] = n_oth
        s_prev = v.fillna(0).groupby([t.date, t.venue, t.surface]).cumsum() - v.fillna(0)
        n_prev = valid.groupby([t.date, t.venue, t.surface]).cumsum() - valid
        t[f'{name}_prior'] = (s_prev / n_prev.where(n_prev > 0)) * (n_prev / (n_prev + shrink))
        t[f'n_{name}_prior'] = n_prev
    return t


def r2(y, X):
    """R^2 of an OLS fit (with intercept) of y on the columns of X, on rows where everything is present."""
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y, X = y[ok], X[ok]
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()), int(ok.sum())
