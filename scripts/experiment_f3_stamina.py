"""F3 (追加指示10, 2026-11-19): スタミナ2型. Hypothesis: "a horse that can keep a fast pace for long" and "a horse that waits through a slow pace and
finishes strongly" are different horses, and one plain average of the past runs flattens the difference.

Every past run is sorted by the pace of ITS race (the leaders' early 3F against the course norm, `pace`; fast <= -0.35 s, slow >= +0.20 s: the
fixed PACE_BANDS; E2's gap600 was not adopted, so it is not used) and two families of history indicators are built, exponentially weighted with the
v1.9 alpha (0.2) over the runs of that family only:

  past_hold_fast   in fast races, how well a horse that ran in the front half (first-corner position <= 0.4) kept its place:
                   -(last observed corner rank - first observed corner rank)
  past_l3rel_fast  in fast races, its closing 3F against the median of its own race (negative = quicker than the field)
  past_l3rel_slow  in slow races, the same relative closing 3F
  n_fast / n_slow  how many runs stand behind them
combined with TODAY's predicted pace (`pace_pred`): out of fold on the training rows (the forward race-day chunks of pace_model.forward_pace),
the pace model's own forecast on the evaluation rows; never the actual pace of the target race. fit_fast = past_hold_fast x how much faster than the
norm today is predicted, fit_slow = past_l3rel_slow x how much slower.

Variants: stam (indicators only), pace_only (the predicted pace alone: control), stam+pace (indicators + pace_pred + the two products).
Focus: rank at corner 4 and the closing 3F. Guards: gaps, lanes, the pace forecast itself.

Side parts of the brief (README F3):
  survive   前に出た馬の逃げ残り率 by course x sole front-runner x pace (measured first; a real pattern would be handed to F3)
  pace      the category pace prior (grade, age, sex limit, distance, surface) on the pace-model side, scored with the pace measures

  python scripts/experiment_f3_stamina.py select|report|decide [--seed 42|7] [--variant NAME] [--retune]
  python scripts/experiment_f3_stamina.py survive | pace
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as xc  # noqa: E402
import pace_model as pm  # noqa: E402

FAST, SLOW = pm.PACE_BANDS


def _ewm(d, col):
    return d.groupby('KETTO_TOROKU_BANGO')[col].transform(lambda s: s.shift().ewm(alpha=pm.HISTORY_ALPHA, ignore_na=True).mean())


def build_stamina(d):
    first = d[[f'r{c}' for c in pm.CORNERS]].bfill(axis=1).iloc[:, 0]
    fast, slow = (d.pace <= FAST), (d.pace >= SLOW)
    d['last3_rel'] = d.last3 - d.groupby('RACE_CODE').last3.transform('median')
    held = (-d.mall).where((first <= 0.4) & (d.n_corners >= 2))
    d['_hold_fast'] = held.where(fast)
    d['_l3_fast'] = d.last3_rel.where(fast)
    d['_l3_slow'] = d.last3_rel.where(slow)
    d['_is_fast'], d['_is_slow'] = fast.astype(float).where(d.pace.notna()), slow.astype(float).where(d.pace.notna())
    d['past_hold_fast'] = _ewm(d, '_hold_fast')
    d['past_l3rel_fast'] = _ewm(d, '_l3_fast')
    d['past_l3rel_slow'] = _ewm(d, '_l3_slow')
    g = d.groupby('KETTO_TOROKU_BANGO')
    d['n_fast'] = g._is_fast.transform(lambda s: s.shift().cumsum())
    d['n_slow'] = g._is_slow.transform(lambda s: s.shift().cumsum())


def products(d):
    """Interactions with today's predicted pace (pace_pred; NaN stays NaN)."""
    d['fit_fast'] = d.past_hold_fast * np.clip(-d.pace_pred, 0, None)
    d['fit_slow'] = d.past_l3rel_slow * np.clip(d.pace_pred, 0, None)


# ----------------------------------------------------------------------------------------------- pace_pred plumbing
_state = {'on': False}
_orig = {}


def install():
    """Wrap pm.fit_models / pm.evaluation_races so every caller (select, report, the re-tuning pipeline) gets pace_pred before it is used."""
    if _state['on']:
        return
    _orig['fit'], _orig['eval'] = pm.fit_models, pm.evaluation_races

    def fit_models(d, end):
        fwd = pm.forward_pace(d, end)
        d['pace_pred'] = d.RACE_CODE.map(fwd['pred'])  # out of fold; races of the first chunk and later than `end` stay NaN
        products(d)
        return _orig['fit'](d, end)

    def evaluation_races(models, d, start, end, codes=None):
        rows = d[d.date.between(start, end)]
        races = pm.race_level(rows, 'RACE_CODE', 1)
        pred = pd.Series(models['pace'].predict(races[pm.PACE_FEATURES]), index=races.index)
        d.loc[rows.index, 'pace_pred'] = rows.RACE_CODE.map(pred).to_numpy()
        products(d)
        return _orig['eval'](models, d, start, end, codes=codes)

    pm.fit_models, pm.evaluation_races = fit_models, evaluation_races
    _state['on'] = True


def uninstall():
    if _state['on']:
        pm.fit_models, pm.evaluation_races = _orig['fit'], _orig['eval']
        _state['on'] = False


# ------------------------------------------------------------------------------------ category pace prior (pace-model side)
# The pace of a race depends on its category (grade, age, sex limit, distance, surface) more than on the single race: Oaks and Derby, say, have
# their own usual early pace. The prior of a race is the shrunk mean, over EARLIER races of its category, of its pace measured against the course's
# running median early 3F of the races before it (`pace_fwd`: forward only, independent of PACE_NORM_END), with a fallback to coarser categories.
PRIOR_K = 8.0
EXTRA_RACE_COLS = ['cat_pace_prior', 'cat_pace_n']
_race_level = {}


def install_race_level():
    if 'orig' in _race_level:
        return
    _race_level['orig'] = pm.race_level

    def race_level(x, key, c):
        out = _race_level['orig'](x, key, c)
        for col in EXTRA_RACE_COLS:
            if col in x:
                out[col] = x.groupby(key, sort=False)[col].first()
        return out

    pm.race_level = race_level


def uninstall_race_level():
    if 'orig' in _race_level:
        pm.race_level = _race_level.pop('orig')


def build_catprior(d):
    import bisect
    import context_index as ci
    runs = ci.load_runs()
    fem = runs.groupby('RACE_CODE').SEIBETSU_CODE.apply(lambda s: bool((s.astype(str).str.strip() == '2').all()))
    r = d[d.complete].groupby('RACE_CODE').agg(date=('date', 'first'), venue=('venue', 'first'), track=('track', 'first'), distance=('distance', 'first'),
                                                surface=('surface', 'first'), grade=('grade', 'first'), early3=('early3', 'first'), age=('age', 'max'))
    r = r.sort_values(['date']).reset_index()
    r['female'] = r.RACE_CODE.map(fem).fillna(False)
    r['band'] = pd.cut(r.distance, [0, 1400, 1800, 2200, 4000], labels=False)
    r['gg'] = r.grade.map({'A': 'G1', 'B': 'G2', 'C': 'G3'}).fillna('other')
    r['agec'] = np.where(r.age <= 2, '2', np.where(r.age <= 3, '3', 'o'))
    keys = [('gg', 'agec', 'female', 'band', 'surface'), ('gg', 'surface', 'band'), ('surface', 'band')]
    course_hist, sums = {}, [{} for _ in keys]
    prior, n_used = {}, {}
    for _, day in r.groupby('date', sort=True):
        staged = []
        for row in day.itertuples():
            h = course_hist.get((row.venue, row.track, row.distance), [])
            dev = row.early3 - h[len(h) // 2] if len(h) >= 5 and not np.isnan(row.early3) else np.nan
            est, cnt = 0.0, 0
            for lvl in (2, 1, 0):  # coarse to fine: each level shrinks toward the level above
                k = tuple(getattr(row, c) for c in keys[lvl])
                sm, ct = sums[lvl].get(k, (0.0, 0))
                est = (sm + PRIOR_K * est) / (ct + PRIOR_K)
                cnt = ct
            prior[row.RACE_CODE], n_used[row.RACE_CODE] = est, cnt
            staged.append((row, dev))
        for row, dev in staged:  # a day's races enter only after the whole day has been scored
            if not np.isnan(row.early3):
                bisect.insort(course_hist.setdefault((row.venue, row.track, row.distance), []), row.early3)
            if not np.isnan(dev):
                for lvl in range(3):
                    k = tuple(getattr(row, c) for c in keys[lvl])
                    sm, ct = sums[lvl].get(k, (0.0, 0))
                    sums[lvl][k] = (sm + dev, ct + 1)
    d['cat_pace_prior'] = d.RACE_CODE.map(prior)
    d['cat_pace_n'] = d.RACE_CODE.map(n_used)


# --------------------------------------------------------------------------------------- front-runner survival (measurement)
def cmd_survive(args):
    """逃げ残り率: how often the horse that leads at the first observed corner finishes in the first three, by course, sole front-runner or not
    and pace. Training rows (before 2025-06-01) only; a cell counts as real when both time halves agree on the sign against the same
    sole/pace cell over all courses, |z| >= 3 and it survives Benjamini-Hochberg (10%)."""
    from scipy import stats
    import context_index as ci
    bh = xc.bh
    d = xc.v19_frame()
    runs = ci.load_runs()[['RACE_CODE', 'UMABAN', 'KAKUTEI_CHAKUJUN']].astype({'UMABAN': int}).rename(columns={'UMABAN': '_u'})
    t = d[(d.date < '20250601') & d.complete].assign(_u=lambda x: x.UMABAN.astype(int)).merge(runs, on=['RACE_CODE', '_u'], how='left')
    t['first'] = t[[f'r{c}' for c in pm.CORNERS]].bfill(axis=1).iloc[:, 0]
    lead = t[(t['first'] == 0) & t.KAKUTEI_CHAKUJUN.between(1, 18)].drop_duplicates('RACE_CODE').copy()
    lead['top3'] = (lead.KAKUTEI_CHAKUJUN <= 3).astype(float)
    lead['win'] = (lead.KAKUTEI_CHAKUJUN == 1).astype(float)
    lead['sole'] = (lead.n_front <= 1).map({True: 'sole', False: 'contested'})
    lead['pace_c'] = pd.cut(lead.pace, [-9, pm.PACE_BANDS[0], pm.PACE_BANDS[1], 9], labels=['fast', 'mid', 'slow']).astype(object)
    lead['band'] = pd.cut(lead.distance, [0, 1400, 1800, 2200, 4000], labels=['sprint', 'mile', 'middle', 'long']).astype(object)
    lead['group'] = lead.venue.astype(str) + '/' + lead.surface.astype(str) + '/' + lead.band
    lead['half'] = np.where(lead.date < '20230901', 'A', 'B')
    res = {'leaders': int(len(lead)), 'top3Rate': float(lead.top3.mean()), 'winRate': float(lead.win.mean()),
           'byPaceAndSole': {f'{a}/{b}': {'n': int(len(x)), 'top3': float(x.top3.mean()), 'win': float(x.win.mean())}
                             for (a, b), x in lead.groupby(['pace_c', 'sole'])}}
    rows = []
    for key, x in lead.groupby(['group', 'sole', 'pace_c']):
        if len(x) < 25:
            continue
        pa, pb = x[x.half == 'A'], x[x.half == 'B']
        ref = lead[(lead.sole == key[1]) & (lead.pace_c == key[2])].top3.mean()
        z = (x.top3.mean() - ref) / np.sqrt(max(ref * (1 - ref), 1e-9) / len(x))
        sa = np.sign(pa.top3.mean() - ref) if len(pa) >= 8 else 0
        sb = np.sign(pb.top3.mean() - ref) if len(pb) >= 8 else 0
        rows.append({'cell': key, 'n': int(len(x)), 'top3': float(x.top3.mean()), 'ref': float(ref), 'z': float(z), 'p': float(2 * stats.norm.sf(abs(z))),
                     'replicates': bool(sa == sb == np.sign(z) and sa != 0)})
    ok = bh([r['p'] for r in rows]) if rows else []
    for r, o in zip(rows, ok):
        r['real'] = bool(o and r['replicates'] and abs(r['z']) >= 3)
    res['byCourse'] = {'cells': len(rows), 'real': sum(r['real'] for r in rows), 'top': sorted(rows, key=lambda r: -abs(r['z']))[:12]}
    fuchu = lead[(lead.venue == 5) & (lead.surface == 0)]
    res['tokyoTurf'] = {f'{a}/{b}': {'n': int(len(x)), 'top3': float(x.top3.mean()), 'ref': float(lead[(lead.sole == b) & (lead.pace_c == a)].top3.mean())}
                        for (a, b), x in fuchu.groupby(['pace_c', 'sole'])}
    out = xc.ROOT / 'outputs' / 'f3-stamina'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'survive.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def cmd_pace(args):
    """Pace forecast with and without the category prior, both seeds, both windows (the pace model's own measures: seconds MAE, correlation,
    RPS against the climatology and the majority-class forecast)."""
    import eval_standard as es
    from experiment_data_expansion import set_seed
    d = xc.v19_frame()
    out = {}
    for seed in (42, 7):
        for name, (fit_end, a, b) in {'selection': (xc.SEL[0], xc.SEL[1], xc.SEL[2]), 'report': (xc.REP[0], xc.REP[1], xc.REP[2])}.items():
            for system in ('base', 'catprior'):
                set_seed(seed)
                uninstall()
                if system == 'catprior':
                    build_catprior(d)
                    install_race_level()
                    xc.set_features([], EXTRA_RACE_COLS)
                else:
                    uninstall_race_level()
                    xc.set_features()
                models = pm.fit_models(d, fit_end)
                races = pm.evaluation_races(models, d, a, b)
                pp = np.array([pred['pace'] for x, corners, numbers, pred, *_ in races])
                pa = np.array([x.pace.iloc[0] for x, *_ in races])
                sigma, train_pace = pm.pace_sigma(d, fit_end)
                row = es.pace_summary(pp, pa, train_pace, sigma, pm.PACE_BANDS)
                out[f'{seed}/{name}/{system}'] = row
                print(f'seed {seed} {name} {system}: races {row["races"]} mae {row["mae"]:.4f} (const {row["constantMae"]:.4f}) corr {row["corr"]:.3f} '
                      f'hit {row["labelHit"]:.3f} (majority {row["majorityHit"]:.3f}) rps {row["rps"]:.4f} (clim {row["rpsClimatology"]:.4f}, '
                      f'majority {row["rpsMajority"]:.4f}) probHit {row["probHit"]:.3f}', flush=True)
    uninstall_race_level()
    xc.set_features()
    p = xc.ROOT / 'outputs' / 'f3-stamina'
    p.mkdir(parents=True, exist_ok=True)
    (p / 'pace_prior.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')


STAM = ['past_hold_fast', 'past_l3rel_fast', 'past_l3rel_slow', 'n_fast', 'n_slow']
PACE = ['pace_pred', 'fit_fast', 'fit_slow']


def build_plain(d):
    build_stamina(d)
    uninstall()


def build_pace(d):
    build_stamina(d)
    install()


VARIANTS = {'base': xc.Variant(build=lambda d: uninstall()),
            'stam': xc.Variant(STAM, build_plain, note='pace-split history indicators'),
            'pace_only': xc.Variant(['pace_pred'], build_pace, note='control: only the predicted pace of today'),
            'stam+pace': xc.Variant(STAM + PACE, build_pace, note='indicators x predicted pace')}
FOCUS = (4,)

if __name__ == '__main__':
    xc.main(VARIANTS, FOCUS, 'f3-stamina', extra_cmds={'survive': cmd_survive, 'pace': cmd_pace}, last3_focus=True)
