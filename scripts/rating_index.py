"""Opponent-adjusted horse rating (追加指示10 phase 0, adopted into the model by 追加指示11 / G1, v2.0). Independent of pace_model (no import of it), so
pace_model, the experiments (context_index) and the pre-race card path all use the same code.

Every horse carries a time-series rating that is updated ONLY with results of earlier days (a whole race day is applied after the day, so same-day races never
see each other) and weighted exponentially with the alpha E1 adopted (0.2). A run's performance is `perf = mean rating of the OTHER horses before the race + y`,
y in [-1, 1] the finish place of the run (winner +1). The rating before a race is the shrunk exponentially weighted mean of the horse's earlier `perf`:
num / (den + PRIOR) with num <- (1 - alpha) * num + perf, den <- (1 - alpha) * den + 1 (a horse never seen has rating 0).
The field of a race is every horse that was in it before the off (not scratched / excluded before the race), whether or not it finished, so no column ever uses
who failed to finish; only horses with a finish place are scored and updated. Built from ALL JRA flat races since 2020 (named or not).

Exposed per run: rating_pre, rating_conf (the weight behind it), opp_mean (the other horses' mean rating before the race), perf, perf_raw, rating_pct
(percentile in the field). The model uses rating_pre only (scripts/experiment_g1_rating.py).
"""
import numpy as np
import pandas as pd

from build_forecasts import settings, number  # noqa: F401  (settings: the read-only DB connection)

RATING_ALPHA = 0.2   # E1's history alpha
RATING_PRIOR = 1.0   # weight of the prior (rating 0) in the shrunk mean
NORMAL_IJO = ['', '0', '3', '4', '5', '6', '7']  # same rule as pace_model.load_frame: 1 / 2 (scratched / excluded before the race) are out


def load_runs(today_s=None):
    """One row per run of every JRA flat race (venue 01-10, TRACK_CODE 10-29) since 2020 with a date before `today_s` (YYYYMMDD; None = all).
    Columns: RACE_CODE, date, UMABAN, KETTO_TOROKU_BANGO, normal, KAKUTEI_CHAKUJUN. Read-only."""
    import pymysql
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute("""SELECT r.RACE_CODE,r.KAISAI_NEN,r.KAISAI_GAPPI,u.UMABAN,u.KETTO_TOROKU_BANGO,u.IJO_KUBUN_CODE,u.KAKUTEI_CHAKUJUN
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026'
          AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          ORDER BY r.KAISAI_NEN,r.KAISAI_GAPPI,r.RACE_CODE,u.UMABAN""")
        d = pd.DataFrame(q.fetchall())
    c.close()
    d['date'] = d.KAISAI_NEN + d.KAISAI_GAPPI
    if today_s:
        d = d[d.date < today_s]
    for col in ('UMABAN', 'KAKUTEI_CHAKUJUN'):
        d[col] = pd.to_numeric(d[col].astype(str).str.strip(), errors='coerce')
    d['normal'] = d.IJO_KUBUN_CODE.fillna('0').str.strip().isin(NORMAL_IJO)
    return d.drop_duplicates(['RACE_CODE', 'UMABAN']).reset_index(drop=True)


def ratings(runs, alpha=RATING_ALPHA, prior=RATING_PRIOR, return_state=False):
    """Per-run rating columns (see the module docstring): a frame with RACE_CODE, UMABAN and the columns. `runs` sorted by date.
    return_state=True also returns {'num': ..., 'den': ...}: every horse's state after the last day, to rate a horse before a coming race."""
    ok = runs[runs.normal & runs.date.notna()]
    out = []
    num, den = {}, {}
    for _, day in ok.groupby('date', sort=True):
        updates = []
        for _, x in day.groupby('RACE_CODE', sort=False):
            n = len(x)
            if n < 3:
                continue
            ids = x.KETTO_TOROKU_BANGO.to_numpy()
            nu = np.array([num.get(h, 0.0) for h in ids])
            de = np.array([den.get(h, 0.0) for h in ids])
            r = nu / (de + prior)
            opp = (r.sum() - r) / (n - 1)
            fin = x.KAKUTEI_CHAKUJUN.to_numpy(float)
            fin_ok = (fin >= 1) & (fin <= 18)
            m = int(fin_ok.sum())
            y = np.full(n, np.nan)
            if m >= 3:
                fin_rank = pd.Series(np.where(fin_ok, fin, np.nan)).rank(method='average').to_numpy()  # ties (dead heats) share a place
                y = np.where(fin_ok, ((m + 1) / 2 - fin_rank) / ((m - 1) / 2), np.nan)
            perf = opp + y
            pct = pd.Series(r).rank(pct=True).to_numpy()
            out.append(pd.DataFrame({'RACE_CODE': x.RACE_CODE.to_numpy(), 'UMABAN': x.UMABAN.to_numpy(), 'rating_pre': r, 'rating_conf': de,
                                     'opp_mean': opp, 'perf': perf, 'perf_raw': y, 'rating_pct': pct}))
            updates.extend((h, p) for h, p, f in zip(ids, perf, fin_ok) if f and m >= 3)
        for h, p in updates:  # the day's results enter only after the whole day is scored
            num[h] = (1 - alpha) * num.get(h, 0.0) + p
            den[h] = (1 - alpha) * den.get(h, 0.0) + 1.0
    table = pd.concat(out, ignore_index=True)
    return (table, {'num': num, 'den': den, 'prior': prior}) if return_state else table


def rating_of(state, horse_id):
    """A horse's rating from a state returned by ratings(..., return_state=True) (a horse never seen: 0)."""
    return state['num'].get(horse_id, 0.0) / (state['den'].get(horse_id, 0.0) + state['prior'])


def attach(d, table):
    """Add rating_pre to a frame with RACE_CODE and UMABAN (the frame's rows keep their order; a run without a rating row stays NaN)."""
    key = d[['RACE_CODE', 'UMABAN']].astype({'UMABAN': int})
    t = table[['RACE_CODE', 'UMABAN', 'rating_pre']].astype({'UMABAN': int})
    d['rating_pre'] = key.merge(t, on=['RACE_CODE', 'UMABAN'], how='left').rating_pre.to_numpy()
    return d
