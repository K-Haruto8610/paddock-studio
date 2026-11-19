"""G1 (追加指示11, 2026-11-19): the opponent-adjusted rating (context_index.ratings) as a horse-ability feature.

Found as a by-product of F2 (README「追加指示10」, rating(info)): rating_pre / rating_pct / field_rating moved tau, integer MAE and the closing 3F the right way at
every corner in both seeds, but it broke two guards (lane MAE at corner 4, gap MAE at corner 2). So the same experiment, now formal, with variants that
keep the rating out of the gap and lane models:

  r3      rating_pre, rating_pct, field_rating in every model           (the F2 variant)
  r1      rating_pre only
  r3_rl   the three, in the rank and closing models only (gap and lane models keep the v1.9 features)
  r1_rl   rating_pre only, rank and closing models only

rating_pre = the horse's rating before the race (opponent-adjusted exponentially weighted results of ALL its earlier JRA flat races, unnamed ones too),
rating_pct = its percentile in today's field, field_rating = the mean rating of the other horses. Every one uses only days before the race
(scripts/experiment_f0_context.py leak checks). Procedure, windows, seeds and the adoption rule: scripts/exp_common.py. Focus: all four corners and the closing 3F.

  python scripts/experiment_g1_rating.py select|report|decide [--seed 42|7] [--variant NAME] [--retune] [--only a,b]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_index as ci  # noqa: E402
import exp_common as xc  # noqa: E402
import pace_model as pm  # noqa: E402

RATING3 = ['rating_pre', 'rating_pct', 'field_rating']


def build_rating(d):
    r = pd.read_pickle(ci.CACHE / 'ratings.pkl')[['RACE_CODE', 'UMABAN', 'rating_pre', 'rating_pct', 'opp_mean']]
    m = d[['RACE_CODE', 'UMABAN']].astype({'UMABAN': int}).merge(r.astype({'UMABAN': int}), on=['RACE_CODE', 'UMABAN'], how='left')
    d['rating_pre'], d['rating_pct'], d['field_rating'] = m.rating_pre.to_numpy(), m.rating_pct.to_numpy(), m.opp_mean.to_numpy()


class Sub:
    """A model that sees only some of the columns (the gap and lane models without the rating)."""

    def __init__(self, model, cols):
        self.model, self.cols = model, cols

    def fit(self, X, y):
        self.model.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.cols])


_state = {'on': False}
_orig = {}


def install():
    """Refit the gap (g) and lane (l) models without the rating columns, after pm.fit_models; every caller (select, report, the re-tuning pipeline) gets it."""
    if _state['on']:
        return
    _orig['fit'] = pm.fit_models

    def fit_models(d, end):
        models = _orig['fit'](d, end)
        fit = (d.date < end) & d.complete
        cols = [c for c in pm.FEATURES if c not in RATING3]
        for t in pm.TARGETS:
            if t[0] in ('g', 'l'):
                rows = fit & d[t].notna()
                models[t] = Sub(pm.lgbm('l1' if t[0] == 'g' else 'l2'), cols).fit(d.loc[rows, pm.FEATURES], d.loc[rows, t])
        return models

    pm.fit_models = fit_models
    _state['on'] = True


def uninstall():
    if _state['on']:
        pm.fit_models = _orig['fit']
        _state['on'] = False


def everywhere(d):
    build_rating(d)
    uninstall()


def rank_closing_only(d):
    build_rating(d)
    install()


VARIANTS = {'base': xc.Variant(build=lambda d: uninstall()),
            'r3': xc.Variant(RATING3, everywhere),
            'r1': xc.Variant(['rating_pre'], everywhere),
            'r3_rl': xc.Variant(RATING3, rank_closing_only),
            'r1_rl': xc.Variant(['rating_pre'], rank_closing_only)}
FOCUS = (1, 2, 3, 4)

if __name__ == '__main__':
    xc.main(VARIANTS, FOCUS, 'g1-rating', last3_focus=True)
