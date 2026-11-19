"""Standard evaluation measures for the pace-forecast experiments (E0 audit, 2026-11-18).

Pure numpy / scipy helpers, no model code, so pace_model.validate() and every experiment script can share them.
What they standardise (README「評価の監査（E0）」):

- Ranks: the primary measures are the within-race Kendall tau of the published (integer) formation and its integer
  rank MAE. Continuous MAE is secondary and is taken on the model BEFORE the past-mean blend (`model_r<c>`), because the
  blend mixes an unchanged past mean into every candidate and the continuous MAE rewards shrinkage (an L1 objective
  lowers it by ~0.003 without changing the order).
- Lanes: the label is the position inside an abreast group from the inside (58-69% zeros), so MAE is reported against
  the constant-0 forecast (`ratioToZero` < 1 means better than "everyone on the inside / alone") together with the
  accuracy over the classes 0 / 1 / 2+.
- Pace: seconds MAE and correlation, plus the ranked probability score of the H/M/S forecast (Gaussian predictive
  distribution around the point forecast) against the climatology and the majority-class forecast.
- Differences between two systems: paired bootstrap over races (rows within a race stay together).
"""
import numpy as np
from scipy.stats import kendalltau, norm

LANE_CLASSES = (0, 1, 2)  # 2 = two or more horses inside this one in its abreast group


def kendall(pred, actual):
    """Within-race Kendall tau-b; NaN when either side is constant."""
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    ok = ~(np.isnan(pred) | np.isnan(actual))
    if ok.sum() < 3:
        return np.nan
    return float(kendalltau(pred[ok], actual[ok]).statistic)


def lane_class(v):
    """Lane slot -> class 0 / 1 / 2+ (continuous predictions are rounded first)."""
    return np.minimum(np.rint(np.clip(np.asarray(v, float), 0, None)), LANE_CLASSES[-1])


def lane_summary(pred, label):
    """MAE, the constant-0 MAE on the same rows, their ratio, and the class accuracy of pred and of constant 0."""
    pred, label = np.asarray(pred, float), np.asarray(label, float)
    mae, zero = float(np.mean(np.abs(pred - label))), float(np.mean(np.abs(label)))
    lc = lane_class(label)
    return {'mae': mae, 'zeroMae': zero, 'ratioToZero': mae / zero if zero > 0 else np.nan,
            'classAccuracy': float(np.mean(lane_class(pred) == lc)), 'zeroClassAccuracy': float(np.mean(lc == 0)),
            'classShare': {str(k): float(np.mean(lc == k)) for k in LANE_CLASSES}}


def pace_class(v, bands):
    """0 = H (fast, v <= bands[0]), 1 = M, 2 = S (slow, v >= bands[1]); same rule as pace_model.pace_label."""
    v = np.asarray(v, float)
    return np.where(v <= bands[0], 0, np.where(v >= bands[1], 2, 1))


def gaussian_class_probs(mu, sigma, bands):
    """P(H), P(M), P(S) for a normal predictive distribution N(mu, sigma) around each point forecast."""
    mu = np.asarray(mu, float)
    ph = norm.cdf((bands[0] - mu) / sigma)
    ps = 1 - norm.cdf((bands[1] - mu) / sigma)
    return np.column_stack([ph, 1 - ph - ps, ps])


def rps(probs, outcome):
    """Ranked probability score for ordered classes (lower is better; 0 = certain and right)."""
    probs = np.atleast_2d(np.asarray(probs, float))
    k = probs.shape[1]
    obs = np.zeros_like(probs)
    obs[np.arange(len(probs)), np.asarray(outcome, int)] = 1
    return float(np.mean(np.sum((np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(obs, axis=1)[:, :-1]) ** 2, axis=1) / (k - 1)))


def pace_summary(pred, actual, train_actual, sigma, bands):
    """Point and probabilistic scores of the race-pace forecast, with the constant / climatology / majority baselines
    (all baselines from the training races only)."""
    pred, actual, train_actual = (np.asarray(v, float) for v in (pred, actual, train_actual))
    ok = ~np.isnan(actual)
    pred, actual = pred[ok], actual[ok]
    train_actual = train_actual[~np.isnan(train_actual)]
    const = float(np.median(train_actual))
    cls, train_cls = pace_class(actual, bands), pace_class(train_actual, bands)
    share = np.array([np.mean(train_cls == k) for k in range(3)])
    majority = int(np.argmax(share))
    model_probs = gaussian_class_probs(pred, sigma, bands)
    return {'races': int(ok.sum()), 'mae': float(np.mean(np.abs(pred - actual))), 'constantMae': float(np.mean(np.abs(const - actual))),
            'corr': float(np.corrcoef(pred, actual)[0, 1]),
            'labelHit': float(np.mean(pace_class(pred, bands) == cls)), 'majorityClass': 'HMS'[majority],
            'majorityHit': float(np.mean(cls == majority)),
            'rps': rps(model_probs, cls), 'rpsClimatology': rps(np.tile(share, (len(cls), 1)), cls),
            'rpsMajority': rps(np.tile(np.eye(3)[majority], (len(cls), 1)), cls),
            'probHit': float(np.mean(np.argmax(model_probs, axis=1) == cls)), 'sigma': float(sigma),
            'trainShare': {k: float(v) for k, v in zip('HMS', share)}, 'bands': [float(b) for b in bands]}


class RaceTable:
    """Per-race sums of a per-horse error (or one per-race value such as tau) for several systems, so differences can
    be bootstrapped with the races paired."""

    def __init__(self):
        self.rows = {}  # (metric, system) -> {race: {row key: value}} (NaN kept, so systems stay aligned row by row)

    def add(self, metric, system, race, values, keys=None):
        """Per-horse errors of one race, added to what the race already has (e.g. several corners). `keys` name the rows
        (e.g. (corner, horse number)) so that diff() compares two systems on the same horses only; without keys the rows
        are numbered in the order they are added (the same order for every system). NaN errors are kept as missing."""
        v = np.asarray(values, float).ravel()
        r = self.rows.setdefault((metric, system), {}).setdefault(race, {})
        if keys is None:
            keys = [('#', len(r) + i) for i in range(len(v))]
        for k, e in zip(keys, v):
            r[k] = float(e)

    def add_value(self, metric, system, race, value):
        if value is not None and not np.isnan(value):
            self.rows.setdefault((metric, system), {})[race] = {'value': float(value)}

    def _sums(self, metric, system, races=None, pair=None):
        """Per-race (sum, count) over the non-missing rows; with `pair` (the other system's rows) only the rows both have."""
        r = self.rows.get((metric, system), {})
        out = {}
        for race in (r if races is None else races):
            rows = r.get(race, {})
            keys = [k for k, e in rows.items() if not np.isnan(e)]
            if pair is not None:
                other = pair.get(race, {})
                keys = [k for k in keys if k in other and not np.isnan(other[k])]
            if keys:
                out[race] = (sum(rows[k] for k in keys), len(keys))
        return out

    def mean(self, metric, system):
        s = self._sums(metric, system)
        return float(sum(v[0] for v in s.values()) / sum(v[1] for v in s.values())) if s else np.nan

    def count(self, metric, system):
        return int(sum(v[1] for v in self._sums(metric, system).values()))

    def races(self, metric, system):
        return len(self._sums(metric, system))

    def diff(self, metric, a, b, reps=2000, seed=812):
        """Mean of a minus mean of b over the rows both systems have (same race, same row key, both non-missing), with a
        95% paired bootstrap interval (races resampled). 'rows' = the paired rows."""
        ra, rb = self.rows.get((metric, a), {}), self.rows.get((metric, b), {})
        common = sorted(set(ra) & set(rb))
        pa, pb = self._sums(metric, a, common, rb), self._sums(metric, b, common, ra)
        races = sorted(set(pa) & set(pb))
        if not races:
            return {'diff': np.nan, 'lo': np.nan, 'hi': np.nan, 'races': 0, 'rows': 0}
        sa, na = np.array([pa[r][0] for r in races]), np.array([pa[r][1] for r in races], float)
        sb, nb = np.array([pb[r][0] for r in races]), np.array([pb[r][1] for r in races], float)
        point = sa.sum() / na.sum() - sb.sum() / nb.sum()
        idx = np.random.default_rng(seed).integers(len(races), size=(reps, len(races)))
        boot = sa[idx].sum(1) / na[idx].sum(1) - sb[idx].sum(1) / nb[idx].sum(1)
        lo, hi = np.quantile(boot, [.025, .975])
        return {'diff': float(point), 'lo': float(lo), 'hi': float(hi), 'races': len(races), 'rows': int(na.sum())}
