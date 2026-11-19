"""F2 (追加指示10, 2026-11-19): 当日の時計傾向で過去走を補正する. Hypothesis: the same closing 3F is a different thing on a fast day and on a slow day.

Each past run of a horse is read against the speed of ITS day: the day clock of context_index.day_clock (the other races of the same day,
venue and surface; the run's own race never enters; par times from races before 2025-06-01 only). Added on top of the v1.9 history summary
(exponentially weighted, alpha 0.2), nothing is replaced:

  past_last3_adj  exponentially weighted mean of (closing 3F - the closing clock of that day)
  past_clock3     ... of the closing clock itself (what kind of days the horse's numbers come from)
  past_clock      ... of the finish-time clock
  late_rel_adj    percentile of past_last3_adj in today's field (like late_rel)

The index of a past run uses the whole day (all other races), which is allowed because the target race is on a later date; the target
race's own day is never used (not known weeks ahead). `rating` is informational only (the opponent rating of phase 0 as a plain ability
feature; its integration is for 追加指示11) and is excluded from the choice.

  python scripts/experiment_f2_day_clock.py select|report|decide [--seed 42|7] [--variant NAME] [--retune]
Focus: closing 3F and the ranks at corners 3 and 4. Guards: gaps, lanes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_index as ci  # noqa: E402
import exp_common as xc  # noqa: E402
import pace_model as pm  # noqa: E402


def _ewm(d, col):
    return d.groupby('KETTO_TOROKU_BANGO')[col].transform(lambda s: s.shift().ewm(alpha=pm.HISTORY_ALPHA, ignore_na=True).mean())


def build_clock(d):
    c = pd.read_pickle(ci.CACHE / 'clock.pkl')[['RACE_CODE', 'clock_full', 'clock3_full']]
    m = d[['RACE_CODE']].merge(c, on='RACE_CODE', how='left')
    d['clock_run'], d['clock3_run'] = m.clock_full.to_numpy(), m.clock3_full.to_numpy()
    d['last3_adj'] = d.last3 - d.clock3_run
    d['past_last3_adj'] = _ewm(d, 'last3_adj')
    d['past_clock3'] = _ewm(d, 'clock3_run')
    d['past_clock'] = _ewm(d, 'clock_run')
    d['late_rel_adj'] = d.past_last3_adj.groupby(d.RACE_CODE).rank(pct=True)


def build_rating(d):
    build_clock(d)
    r = pd.read_pickle(ci.CACHE / 'ratings.pkl')[['RACE_CODE', 'UMABAN', 'rating_pre', 'rating_pct', 'opp_mean']]
    m = d[['RACE_CODE', 'UMABAN']].astype({'UMABAN': int}).merge(r.astype({'UMABAN': int}), on=['RACE_CODE', 'UMABAN'], how='left')
    d['rating_pre'], d['rating_pct'], d['field_rating'] = m.rating_pre.to_numpy(), m.rating_pct.to_numpy(), m.opp_mean.to_numpy()


VARIANTS = {'base': xc.Variant(),
            'l3adj': xc.Variant(['past_last3_adj'], build_clock, note='clock-adjusted closing history'),
            'l3adj+rel': xc.Variant(['past_last3_adj', 'late_rel_adj'], build_clock),
            'l3adj+ctx': xc.Variant(['past_last3_adj', 'past_clock3', 'past_clock'], build_clock),
            'l3adj+rel+ctx': xc.Variant(['past_last3_adj', 'late_rel_adj', 'past_clock3', 'past_clock'], build_clock),
            'rating(info)': xc.Variant(['rating_pre', 'rating_pct', 'field_rating'], build_rating, note='opponent rating as an ability feature; informational')}
FOCUS = (3, 4)

if __name__ == '__main__':
    xc.main(VARIANTS, FOCUS, 'f2-clock', last3_focus=True, exclude=('rating(info)',))
