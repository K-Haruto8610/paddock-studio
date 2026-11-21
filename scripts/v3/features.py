"""v3 S1 inputs: per-run observations, the past-run tensor of each target run, and the target race's conditions.

Only runs on strictly earlier days enter a horse's history. A run's observation is what the record shows of that race
(docs/race-simulator-design.md 5.1): the horse's own mean / early / late speed, its time behind the first horse at the
600 m mark, its closing 3F against the race median, its finishing place and margin, its early and late corner places,
and the race's own early / late speed of the leader, with the conditions it was run under.
"""
import numpy as np
import pandas as pd

K = 10            # past runs per target
VENUES = 10
KINDS = ('turf', 'outer', 'dirt', 'straight')
V0 = 16.0         # m/s, centre of the speed features


def course_index(venue, kind):
    return (np.asarray(venue, int) - 1) * len(KINDS) + np.array([KINDS.index(k) for k in kind])


def run_observations(races, runs):
    """Adds per-run observation columns to `runs` (a copy) using its race's leader marks and field."""
    r = runs.merge(races[['date', 'venue', 'track', 'kind', 'distance', 'field', 'surface', 'going', 'cls', 'lead', 'named']]
                   .drop(columns='date'), left_on='RACE_CODE', right_index=True, how='left')
    d = r.distance
    ok = r.normal & r.soha.notna()
    r['o_speed'] = (d / r.soha).where(ok) - V0
    r['o_late'] = (600 / r.last3).where(ok) - V0
    r['o_early'] = ((d - 600) / r.t600).where(ok) - V0
    g = r[ok].groupby('RACE_CODE')
    r['o_gap600'] = (r.t600 - g.t600.transform('min').reindex(r.index)).clip(0, 6)
    r['o_l3rel'] = (r.last3 - g.last3.transform('median').reindex(r.index)).clip(-3, 3)
    r['o_place'] = ((r.fin - 1) / (r.field - 1).clip(lower=1))
    r['o_margin'] = r.finish_gap.clip(0, 5)
    r['o_r_early'] = r.r1.fillna(r.r2).fillna(r.r3)
    r['o_r_late'] = r.r4.fillna(r.r3)

    def lead_speeds(row):
        m = row.lead
        if not isinstance(m, np.ndarray):
            return (np.nan, np.nan)
        dist = row.distance
        n = len(m)  # marks at distance - 200 (n-1-k) ... the post
        t_post, t600 = m[-1], m[-4] if n >= 4 else np.nan
        return ((dist - 600) / t600 - V0, 600 / (t_post - t600) - V0)

    lr = races.apply(lead_speeds, axis=1, result_type='expand')
    lr.columns = ['lead_early', 'lead_late']
    r = r.join(lr, on='RACE_CODE')
    return r


OBS = ['o_speed', 'o_late', 'o_early', 'o_gap600', 'o_l3rel', 'o_place', 'o_margin', 'o_r_early', 'o_r_late',
       'lead_early', 'lead_late']


def condition_matrix(r):
    """Target-race conditions known before the off (one row per run)."""
    kind = r.kind.map({k: i for i, k in enumerate(KINDS)}).to_numpy()
    cols = [np.log(r.distance.to_numpy() / 1600), r.surface.to_numpy(float), (r.going.fillna(1).to_numpy() - 1) / 3,
            r.cls.fillna(2).to_numpy() / 7, r.field.to_numpy() / 18, (r.age.fillna(4).to_numpy() - 4) / 2,
            (r.weight.fillna(55).to_numpy() - 55) / 3, r.waku.fillna(4).to_numpy() / 8, (r.UMABAN.to_numpy() / r.field.to_numpy()),
            r.blinker.to_numpy(float), r.rating_pre.fillna(0).to_numpy(), r.rating_pre.isna().to_numpy(float)]
    sex = r.sex.fillna(1).astype(int).clip(1, 3).to_numpy()
    onehots = [np.eye(VENUES)[r.venue.to_numpy() - 1], np.eye(len(KINDS))[kind], np.eye(3)[sex - 1]]
    return np.column_stack(cols + onehots).astype(np.float32)


def history_tensor(r, pool, targets):
    """Past-run tensor [len(targets), K, F] and mask [len(targets), K] for the rows `targets` (index into r).
    History comes from the rows of r where `pool` is True, strictly earlier dates, most recent first.
    F = observations (filled 0) + their validity + the past run's conditions + its relation to the target."""
    obs = r[OBS].to_numpy(np.float32)
    valid = ~np.isnan(obs)
    obs = np.nan_to_num(obs)
    date = pd.to_datetime(r.date, format='%Y%m%d').to_numpy()
    cond = np.column_stack([np.log(r.distance.to_numpy() / 1600), r.surface.to_numpy(float), (r.going.fillna(1).to_numpy() - 1) / 3,
                            r.cls.fillna(2).to_numpy() / 7, r.field.to_numpy() / 18, (r.weight.fillna(55).to_numpy() - 55) / 3]).astype(np.float32)
    venue, kind = r.venue.to_numpy(), r.kind.to_numpy()
    F = obs.shape[1] * 2 + cond.shape[1] + 5
    X = np.zeros((len(targets), K, F), np.float32)
    M = np.zeros((len(targets), K), bool)
    pos = {i: p for p, i in enumerate(targets)}
    poolv = pool.to_numpy()
    for _, idx in r.groupby('horse', sort=False).indices.items():
        idx = idx[np.argsort(date[idx], kind='stable')]
        hist = idx[poolv[idx]]
        hd = date[hist]
        for i in idx:
            p = pos.get(i)
            if p is None:
                continue
            past = hist[hd < date[i]][::-1][:K]
            if not len(past):
                continue
            n = len(past)
            days = (date[i] - date[past]).astype('timedelta64[D]').astype(float)
            X[p, :n, :obs.shape[1]] = obs[past]
            X[p, :n, obs.shape[1]:2 * obs.shape[1]] = valid[past]
            X[p, :n, 2 * obs.shape[1]:2 * obs.shape[1] + cond.shape[1]] = cond[past]
            rel = np.column_stack([np.log1p(days) / 6, np.arange(n) / K, cond[past, 0] - np.log(r.distance.iat[i] / 1600),
                                   (venue[past] == venue[i]) & (kind[past] == kind[i]), cond[past, 1] == r.surface.iat[i]])
            X[p, :n, 2 * obs.shape[1] + cond.shape[1]:] = rel
            M[p, :n] = True
    return X, M
