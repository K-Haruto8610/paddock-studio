"""展開予想モデル (pace-forecast model): read-only DB -> per-corner rank / gap / lane + last 3F.

Labels come from JV-Data:
- rank : umagoto_race_joho.CORNERn_JUNI (physical corner n)
- gap  : lengths behind the leader, rebuilt from race_shosai.KAKU_TSUKA_JUNIk using JRA notation
         ( )=abreast, under 1 length, listed inner first / ,=1-2 lengths / -=2-5 / ==5+ / no mark=under 1
- lane : position inside an abreast group counted from the inside (0 = innermost or running alone)
- last3: KOHAN_3F

Only strictly prior starts enter the features; odds, popularity and payouts are never read.
The output follows the import schema `paddock.pace-forecast/v1` (see README) so the app can load it,
and so can any other model that writes the same file.
"""
import sys, json, argparse, heapq
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
import pymysql
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_forecasts import settings, number, clean, load_upcoming_entries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'paddock.pace-forecast/v1'
VERSION = '1.3'
# Representative lengths for each JRA separator (midpoints of the published bands).
GAP_LENGTHS = {'': 0.6, ',': 1.5, '-': 3.5, '=': 6.5}
SEP_ORDER = ['', ',', '-', '=']
LENGTHS_PER_SECOND = 5.0  # JRA rule of thumb: 1 length is about 0.2 s
CORNERS = (1, 2, 3, 4)
TARGETS = [f'{kind}{c}' for c in CORNERS for kind in ('r', 'g', 'l')] + ['last3']


def parse_passing(text):
    """JRA corner passing string -> {umaban: (lengths behind leader, lane inside abreast group,
    abreast group size, unit directly ahead is under 1 length away)}."""
    s = (text or '').strip()
    out, gap, sep, first, i = {}, 0.0, '', True, 0
    while i < len(s):
        ch = s[i]
        if ch in ',-=':
            sep = max(sep, ch, key=SEP_ORDER.index)
            i += 1
            continue
        if ch in '* ':
            i += 1
            continue
        if ch == '(':
            j = s.index(')', i)
            members = [m.strip(' *') for m in s[i + 1:j].split(',')]
            i = j + 1
        elif ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            members = [s[i:j]]
            i = j
        else:
            raise ValueError(f'unexpected {ch!r} in {s!r}')
        close = not first and sep == ''
        if not first:
            gap += GAP_LENGTHS[sep]
        first, sep = False, ''
        members = [m for m in members if m.isdigit()]
        for lane, m in enumerate(members):
            out[int(m)] = (gap, lane, len(members), close)
    return out


def load_frame(today_s):
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute("""SELECT r.RACE_CODE,r.KAISAI_NEN,r.KAISAI_GAPPI,r.KEIBAJO_CODE,r.KYOSOMEI_HONDAI,
          r.KYORI,r.TRACK_CODE,r.SHUSSO_TOSU,r.ZENHAN_3F,r.GRADE_CODE,r.SHIBA_BABAJOTAI_CODE,r.DIRT_BABAJOTAI_CODE,
          r.CORNER1,r.KAKU_TSUKA_JUNI1,r.CORNER2,r.KAKU_TSUKA_JUNI2,r.CORNER3,r.KAKU_TSUKA_JUNI3,r.CORNER4,r.KAKU_TSUKA_JUNI4,
          u.UMABAN,u.WAKUBAN,u.KETTO_TOROKU_BANGO,u.BAMEI,u.BLINKER_SHIYO_KUBUN,u.BAREI,u.FUTAN_JURYO,
          u.CORNER1_JUNI,u.CORNER2_JUNI,u.CORNER3_JUNI,u.CORNER4_JUNI,u.KOHAN_3F,u.IJO_KUBUN_CODE
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026'
          AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          ORDER BY r.KAISAI_NEN,r.KAISAI_GAPPI,r.RACE_CODE,u.UMABAN""")
        rows = q.fetchall()
    c.close()
    d = pd.DataFrame(rows)
    d['date'] = d.KAISAI_NEN + d.KAISAI_GAPPI
    d = d[d.date < today_s]  # pre-race cards have no results: never history or training rows
    d = d[d.BAMEI.map(clean) & d.KYOSOMEI_HONDAI.map(clean)].copy()
    for col in ['KYORI', 'TRACK_CODE', 'SHUSSO_TOSU', 'UMABAN', 'WAKUBAN', 'BAREI', 'FUTAN_JURYO', 'KOHAN_3F'] + [f'CORNER{i}_JUNI' for i in CORNERS]:
        d[col] = d[col].map(number)
    d = d[d.KYORI.between(1000, 3600) & d.SHUSSO_TOSU.between(5, 18) & d.UMABAN.between(1, 18)]
    d = d[d.IJO_KUBUN_CODE.fillna('0').str.strip().isin(['', '0', '3', '4', '5', '6', '7'])]
    counts = d.groupby('RACE_CODE').UMABAN.transform('nunique')
    d = d[counts == d.SHUSSO_TOSU].drop_duplicates(['RACE_CODE', 'UMABAN']).reset_index(drop=True)
    d['field'] = d.SHUSSO_TOSU
    d['distance'] = d.KYORI
    d['surface'] = (d.TRACK_CODE >= 23).astype(int)
    d['venue'] = d.KEIBAJO_CODE.astype(int)
    d['track'] = d.TRACK_CODE.astype(int)
    d['gate'] = d.UMABAN / d.field
    d['blinker'] = d.BLINKER_SHIYO_KUBUN.map({'0': 0, '1': 1})
    d['weight'] = d.FUTAN_JURYO / 10
    d['age'] = d.BAREI
    d['last3'] = (d.KOHAN_3F / 10).where(d.KOHAN_3F.between(280, 500))
    # Race-level: early 3F of the leader (label for the pace model) and going (known before the off).
    z3 = d.ZENHAN_3F.map(number)
    d['early3'] = (z3 / 10).where(z3.between(300, 450))
    going = np.where(d.surface == 1, d.DIRT_BABAJOTAI_CODE.map(number), d.SHIBA_BABAJOTAI_CODE.map(number))
    d['going'] = pd.Series(going, index=d.index).where(lambda g: g.between(1, 4))
    d['grade'] = d.GRADE_CODE.fillna('').str.strip()  # only to describe GI pace, never a model input
    return d, len(rows)


def add_labels(d):
    """Per-corner rank (normalised), gap (lengths) and lane, from the race passing strings."""
    for c in CORNERS:
        rank = d[f'CORNER{c}_JUNI']
        d[f'r{c}'] = ((rank - 1) / (d.field - 1)).where(rank.between(1, d.field))
        d[f'g{c}'] = np.nan
        d[f'l{c}'] = np.nan
        d[f'out{c}'] = np.nan   # horses outside this one in its abreast group
        d[f'close{c}'] = np.nan  # the group ahead is under 1 length away
    bad = 0
    for _, idx in d.groupby('RACE_CODE').groups.items():
        r = d.loc[idx[0]]
        nums = d.loc[idx, 'UMABAN'].astype(int)
        for k in CORNERS:
            corner = str(r[f'CORNER{k}'] or '0').strip()
            if corner not in ('1', '2', '3', '4'):
                continue
            try:
                parsed = parse_passing(r[f'KAKU_TSUKA_JUNI{k}'])
            except ValueError:
                bad += 1
                continue
            if not parsed:
                continue
            # A repeated corner (long races pass it twice) keeps the latest pass, like CORNERn_JUNI.
            if set(parsed) != set(nums):
                bad += 1
                continue
            d.loc[idx, f'g{corner}'] = [parsed[n][0] for n in nums]
            d.loc[idx, f'l{corner}'] = [parsed[n][1] for n in nums]
            d.loc[idx, f'out{corner}'] = [parsed[n][2] - 1 - parsed[n][1] for n in nums]
            d.loc[idx, f'close{corner}'] = [float(parsed[n][3]) for n in nums]
    print(f'passing strings rejected (unparsable or field mismatch): {bad}', flush=True)
    return d


BASE = ['distance', 'surface', 'venue', 'track', 'gate', 'weight', 'age', 'field', 'blinker']
PAST = ['past_' + t for t in TARGETS]
RACE_REL = ['early_rel', 'late_rel', 'front_pressure', 'outer_pressure']
FEATURES = BASE + PAST + ['starts'] + RACE_REL
# Move models (道中の位置変化): the horse's past moves, the field's movers, the course's corner count.
MOVE_FEATURES = FEATURES + ['past_m2', 'past_m3', 'past_m4', 'past_mall', 'past_up', 'past_down', 'past_mall_rel',
                            'field_movers', 'n_corners', 'early_wish']
MOVE_CORNERS = (2, 3, 4)


def add_race_relative(x, race_key):
    """Field-composition features: where this horse's usual early position sits in today's field."""
    g = x.groupby(race_key)
    early = x[['past_r1', 'past_r2', 'past_r3']].mean(axis=1).fillna(x.past_r4)
    x['early_rel'] = early.groupby(x[race_key]).rank(pct=True)
    x['late_rel'] = x.past_last3.groupby(x[race_key]).rank(pct=True)
    x['front_pressure'] = x.past_r4.lt(.3).groupby(x[race_key]).transform('mean')
    x['outer_pressure'] = g.past_l4.transform('mean')
    # Mid-race movement: how this horse usually moves compared with today's field, and how many
    # habitual movers (gained a quarter of the field in a past race) are in the race.
    x['past_mall_rel'] = x.past_mall.groupby(x[race_key]).rank(pct=True)
    x['field_movers'] = x.past_up.lt(-.2).groupby(x[race_key]).transform('mean')
    x['early_wish'] = x.past_r1.fillna(x.past_r3).fillna(x.past_r4)
    # Pace pressure: how many horses want the lead and how fast the races they ran usually started.
    by_race = x.groupby(race_key)
    x['n_front'] = by_race.early_wish.transform(lambda s: (s <= .2).sum())
    x['n_leaders'] = by_race.past_led.transform(lambda s: (s >= .4).sum())
    x['front3_wish'] = by_race.early_wish.transform(lambda s: s.nsmallest(3).mean())
    x['front_pace_hist'] = by_race.past_pace.transform('mean')
    return x


MOVE_LABELS = ['m2', 'm3', 'm4', 'mall']


def add_move_labels(d):
    """m{c}: change of normalised rank since the previous observed corner; mall: first to last corner;
    cm{c}: cumulative change since the first observed corner (the move models' targets)."""
    ranks = d[[f'r{c}' for c in CORNERS]]
    last_seen = pd.Series(np.nan, index=d.index)
    for c in CORNERS:
        d[f'm{c}'] = d[f'r{c}'] - last_seen
        last_seen = d[f'r{c}'].where(d[f'r{c}'].notna(), last_seen)
    first = ranks.bfill(axis=1).iloc[:, 0]
    d['mall'] = ranks.ffill(axis=1).iloc[:, -1] - first
    for c in CORNERS:
        d[f'cm{c}'] = d[f'r{c}'] - first
    d['n_corners'] = ranks.notna().sum(axis=1)
    return d


def add_history_features(d):
    d = add_move_labels(d.sort_values(['date', 'RACE_CODE', 'UMABAN']).reset_index(drop=True))
    by_horse = d.groupby('KETTO_TOROKU_BANGO')
    for t in TARGETS + MOVE_LABELS:
        d['past_' + t] = by_horse[t].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    d['past_up'] = by_horse['mall'].transform(lambda s: s.shift().rolling(5, min_periods=1).min())
    d['past_down'] = by_horse['mall'].transform(lambda s: s.shift().rolling(5, min_periods=1).max())
    d['starts'] = by_horse.cumcount()
    d['pace'] = d.early3 - course_norm(d)
    first = d[[f'r{c}' for c in CORNERS]].bfill(axis=1).iloc[:, 0]
    d['led'] = (first <= .15).astype(float).where(first.notna())
    by_horse = d.groupby('KETTO_TOROKU_BANGO')
    d['past_pace'] = by_horse.pace.transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    d['past_led'] = by_horse.led.transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    return add_race_relative(d, 'RACE_CODE')


PACE_NORM_END = '20250601'  # course norms from the earliest training window only: no later period leaks in


def course_norm(d):
    """Usual early 3F of the course (venue/track/distance, else surface/distance), from races before
    PACE_NORM_END. pace = early 3F minus this: negative = the leaders went faster than usual."""
    tr = d[d.date < PACE_NORM_END].drop_duplicates('RACE_CODE')
    exact = tr.groupby(['venue', 'track', 'distance']).early3.median().to_dict()
    loose = tr.groupby(['surface', 'distance']).early3.median().to_dict()
    return pd.Series([exact.get((v, t, ds), loose.get((s, ds), np.nan))
                      for v, t, ds, s in zip(d.venue, d.track, d.distance, d.surface)], index=d.index, dtype=float)


SPREAD_Q = 0.8
# Race pace (前半のペース): field composition that drives the early speed, and the going.
PACE_FEATURES = ['distance', 'surface', 'venue', 'track', 'field', 'going', 'front_pressure', 'n_front', 'n_leaders',
                 'front3_wish', 'front_pace_hist']
SPREAD_FEATURES = ['distance', 'surface', 'venue', 'track', 'field', 'front_pressure', 'outer_pressure', 'mean_past_g', 'mean_early',
                   'going', 'n_front', 'n_leaders', 'front3_wish', 'front_pace_hist', 'pace_pred']
# Pace labels (seconds vs the course's usual early 3F); 'H' fast, 'S' slow. Bands are the training terciles.
PACE_BANDS = (-0.35, 0.2)


def lgbm(objective):
    return LGBMRegressor(objective=objective, n_estimators=220, max_depth=6, num_leaves=31, learning_rate=.04, min_child_samples=40,
                         subsample=.85, subsample_freq=5, colsample_bytree=.85, verbosity=-1, n_jobs=4, random_state=42)


def race_level(x, key, c):
    """One row per race: field composition and (when known) the observed spread at corner c."""
    g = x.groupby(key, sort=False)
    cols = ['distance', 'surface', 'venue', 'track', 'field', 'front_pressure', 'outer_pressure', 'going', 'n_front', 'n_leaders',
            'front3_wish', 'front_pace_hist'] + [k for k in ('pace_pred', 'pace') if k in x]
    out = g[cols].first()
    out['mean_past_g'] = g[f'past_g{c}'].mean()
    out['mean_early'] = g['past_r1'].mean()
    if f'g{c}' in x:
        # Robust length of the field: gap of the horse 80% of the way down the order (the very last
        # horse is often tailed off and makes the maximum mostly noise).
        out['spread'] = g[f'g{c}'].quantile(SPREAD_Q).where(g[f'g{c}'].count() == g.size())
    if 'date' in x:
        out['date'] = g['date'].first()
    return out


def reconstruct(rank_pred, gap_pred, spread, numbers):
    """Order from the rank model; shape of the gaps from the per-horse gap model (never decreasing down
    the order); total spread from the race-level model. Per-horse regression alone shrinks every gap
    toward the mean and squeezes the field together.
    The horse at SPREAD_Q of the order gets exactly the predicted spread."""
    n = len(numbers)
    order = sorted(range(n), key=lambda i: (rank_pred[i], numbers[i]))
    raw = np.maximum.accumulate([float(gap_pred[i]) for i in order])
    shape = raw - raw[0]
    anchor = shape[int(round(SPREAD_Q * (n - 1)))]
    even = np.linspace(0, 1 / SPREAD_Q, n)
    # Half model shape, half even spacing: the median fits plateau, which would stack horses.
    shape = .5 * shape / anchor + .5 * even if anchor > 1e-6 else even
    ranks, gaps = [0] * n, [0.0] * n
    for pos, i in enumerate(order):
        ranks[i], gaps[i] = pos + 1, float(shape[pos] * max(0.0, spread))
    return ranks, gaps


TRAFFIC = ['traffic_out', 'traffic_close']
CLOSING_BASE = FEATURES + ['going']  # going (馬場状態) lowers the closing error (validation 0.951 -> 0.925 s)
LAST3_FEATURES = CLOSING_BASE + TRAFFIC
# Contention-layer parameters; tuned on Jun-Aug 2025 (see tune_contention). These are the defaults.
DEFAULT_PARAMS = {'alpha': .5, 'delta': .08, 'width': 3, 'front': .35, 'hold': .06}
PARAM_GRID = [{'alpha': a, 'delta': dl, 'width': w, 'front': f, 'hold': h}
              for a in (.5, .7) for dl in (.04, .08) for w in (2, 3) for f in (.25, .45) for h in (.03, .06, .1)]


def add_traffic(d):
    """Traffic at the last corner, from the passing string. Only a training label input for the closing
    model: at prediction time these come from the simulated formation, never from the race itself."""
    d['traffic_out'] = d.out4.fillna(d.out3)
    d['traffic_close'] = d.close4.fillna(d.close3)
    return d


def fit_models(d, end):
    """Per-horse models (intentions), race spread models, and the traffic-aware closing model."""
    fit, models = d.date < end, {}
    for t in TARGETS:
        rows = fit & d[t].notna()
        # Gap shape: median (L1) resists tailed-off horses. Lanes, ranks, closing: mean.
        models[t] = lgbm('l1' if t[0] == 'g' else 'l2').fit(d.loc[rows, FEATURES], d.loc[rows, t])
    rows = fit & d.last3.notna()
    # Traffic can only slow a horse down: monotone (+) in horses outside and a wall ahead.
    closing_model = lgbm('l2').set_params(monotone_constraints=[0] * len(CLOSING_BASE) + [1] * len(TRAFFIC))
    models['last3_traffic'] = closing_model.fit(d.loc[rows, LAST3_FEATURES], d.loc[rows, 'last3'])
    # Pace model, and out-of-fold pace predictions for the spread models' training rows (so they learn
    # from predictions as noisy as the ones they get before a race, not from the actual pace).
    train = d[fit].copy()
    races = race_level(train, 'RACE_CODE', 1).dropna(subset=['pace']).sort_values('date')
    models['pace'] = pace_lgbm().fit(races[PACE_FEATURES], races['pace'])
    oof = pd.Series(np.nan, index=races.index)
    for part in np.array_split(races.index, 4):
        rest = races.index.difference(part)
        oof[part] = pace_lgbm().fit(races.loc[rest, PACE_FEATURES], races.loc[rest, 'pace']).predict(races.loc[part, PACE_FEATURES])
    train['pace_pred'] = train.RACE_CODE.map(oof)
    for c in CORNERS:
        races = race_level(train, 'RACE_CODE', c).dropna(subset=['spread'])
        models[f'spread{c}'] = lgbm('l1').fit(races[SPREAD_FEATURES], races['spread'])
    for c in MOVE_CORNERS:
        # Cumulative move since the first corner, learned directly (not chained corner to corner, which
        # compounds the error of the simulated previous position). Small, regularised trees: the signal
        # is weak and the target heavy-tailed.
        rows = fit & d[f'cm{c}'].notna() & d[[f'r{k}' for k in CORNERS if k < c]].notna().any(axis=1)
        models[f'move{c}'] = move_lgbm().fit(d.loc[rows, MOVE_FEATURES], d.loc[rows, f'cm{c}'])
    return models


def pace_lgbm():
    return LGBMRegressor(n_estimators=200, max_depth=3, num_leaves=7, learning_rate=.03, min_child_samples=30, subsample=.8,
                         subsample_freq=5, colsample_bytree=.8, reg_lambda=5, verbosity=-1, n_jobs=4, random_state=42)


def pace_label(v, bands=PACE_BANDS):
    return 'H' if v <= bands[0] else 'S' if v >= bands[1] else 'M'


def move_lgbm():
    return LGBMRegressor(n_estimators=200, max_depth=4, num_leaves=15, learning_rate=.03, min_child_samples=60, subsample=.8,
                         subsample_freq=5, colsample_bytree=.8, reg_lambda=5, verbosity=-1, n_jobs=4, random_state=42)


def independent(numbers, pred, spreads, corners):
    """Previous behaviour: every corner predicted on its own, no interaction between horses."""
    n, form = len(numbers), {}
    for c in corners:
        rank, gap = reconstruct(pred[f'r{c}'], pred[f'g{c}'], spreads[c], numbers)
        form[c] = {'rank': rank, 'gap': gap, 'lane': list(np.clip(pred[f'l{c}'], 0, 4)), 'held': [False] * n,
                   'out': [0] * n, 'close': [0.0] * n, 'want': rank}
    return form


def simulate(numbers, early, pred, spreads, corners, P):
    """Position contention, corner by corner.

    Each horse wants the position its own model predicts; after the first corner it can only move so far
    from where it already is (alpha blends the wish with the previous position). Horses claim spots in
    order of that pressure. Horses within delta of the row's leader run abreast, at most width wide.
    When the row is full, a front-running type (early speed in the top 'front' share of the field) goes
    wide to keep its spot; any other horse tucks in behind (控える) and loses the position it wanted.
    Traffic at each corner (horses outside, wall under one length ahead) is kept for the closing model.

    With the move models on (P['k'] > 0), a later corner's pressure is instead
    (1 - beta) * wish + beta * (first-corner position + k * predicted move since the first corner):
    the move is relative to the field (centred), so horses that usually go forward down the back
    straight pass the ones that drop back, rather than every horse being pulled to where it was.
    """
    n, form, prev, prev_lane, base = len(numbers), {}, None, None, None
    width = int(P['width'])
    move_k, move_beta = P.get('k', 0.0), P.get('beta', 0.0)  # not `k`: the loop below reuses that name
    for c in corners:
        want = np.asarray(pred[f'r{c}'], float)
        lane_want = np.asarray(pred[f'l{c}'], float)
        if prev is None:
            prio = want
        elif move_k > 0 and f'move{c}' in pred:
            move = np.asarray(pred[f'move{c}'], float)
            prio = (1 - move_beta) * want + move_beta * (base + move_k * (move - move.mean()))
        else:
            prio = (1 - P['alpha']) * want + P['alpha'] * prev
        side = lane_want if prev_lane is None else .5 * prev_lane + .5 * lane_want
        order = sorted(range(n), key=lambda i: (prio[i], numbers[i]))
        want_rank = [0] * n
        for pos, i in enumerate(order):
            want_rank[i] = pos + 1
        rows, held = [], [False] * n
        queue = [(prio[i], numbers[i], i) for i in order]
        heapq.heapify(queue)
        while queue:
            p, _, i = heapq.heappop(queue)
            if rows and p - rows[-1]['p0'] < P['delta']:
                row = rows[-1]['m']
                if len(row) < width or (early[i] <= P['front'] and len(row) < width + 1):
                    row.append(i)
                    continue
                if not held[i]:
                    # The spot is taken: tuck in behind (控える). The horse drops back by `hold` and
                    # horses still pushing within that distance go past it before it finds a place.
                    held[i] = True
                    heapq.heappush(queue, (p + P['hold'], numbers[i], i))
                    continue
            rows.append({'p0': p, 'm': [i]})
        rank, lane, row_of, pos = [0] * n, [0] * n, [0] * n, 0
        for ri, row in enumerate(rows):
            # Rail side first by preferred lane; a horse that went wide stays outermost.
            row['m'] = sorted(row['m'][:width], key=lambda i: (side[i], numbers[i])) + row['m'][width:]
            for li, i in enumerate(row['m']):
                pos += 1
                rank[i], lane[i], row_of[i] = pos, li, ri
        _, g = reconstruct(rank, pred[f'g{c}'], spreads[c], numbers)
        row_gap = []
        for ri, row in enumerate(rows):
            rg = min(g[i] for i in row['m'])
            row_gap.append(max(rg, row_gap[-1] + GAP_LENGTHS['']) if ri else 0.0)
        # The minimum step between rows stretches the field; scale back so the horse at SPREAD_Q of the
        # order sits at the predicted field length again.
        k = int(round(SPREAD_Q * (n - 1)))
        target, have = sorted(g)[k], sorted(row_gap[row_of[i]] for i in range(n))[k]
        if have > 1e-6:
            row_gap = [v * target / have for v in row_gap]
        form[c] = {'rank': rank, 'gap': [row_gap[row_of[i]] for i in range(n)], 'lane': lane, 'held': held,
                   'out': [len(rows[row_of[i]]['m']) - 1 - lane[i] for i in range(n)],
                   'close': [float(row_of[i] > 0 and row_gap[row_of[i]] - row_gap[row_of[i] - 1] < 1.0) for i in range(n)],
                   'want': want_rank}
        prev, prev_lane = (np.array(rank) - 1) / max(1, n - 1), np.array(lane, float)
        if base is None:
            base = prev
    return form


def prepare_race(models, x, key, corners):
    numbers = [int(v) for v in x.UMABAN]
    pred = {t: models[t].predict(x[FEATURES]) for t in TARGETS}
    xm = x.assign(n_corners=len(corners))  # the course's corner count, as in the training label
    for c in MOVE_CORNERS:
        if f'move{c}' in models:
            pred[f'move{c}'] = models[f'move{c}'].predict(xm[MOVE_FEATURES])
    pace = float(models['pace'].predict(race_level(x, key, 1)[PACE_FEATURES])[0])
    pred['pace'] = pace
    xp = x.assign(pace_pred=pace)
    spreads = {c: float(models[f'spread{c}'].predict(race_level(xp, key, c)[SPREAD_FEATURES])[0]) for c in CORNERS}
    return numbers, pred, spreads, x.early_rel.fillna(.5).to_numpy()


def closing(models, x, form, corners):
    """Closing 3F with the traffic the formation leaves at the last corner, and with a clear run."""
    last = form[corners[-1]]
    xt = x[CLOSING_BASE].copy()
    xt['traffic_out'], xt['traffic_close'] = np.array(last['out'], float), np.array(last['close'], float)
    free = xt.copy()
    free['traffic_out'], free['traffic_close'] = 0.0, 0.0
    return models['last3_traffic'].predict(xt[LAST3_FEATURES]), models['last3_traffic'].predict(free[LAST3_FEATURES])


def evaluation_races(models, d, start, end):
    """Cached per-race predictions for a period: only corners where every horse has all labels."""
    out = []
    for _, x in d[d.date.between(start, end)].groupby('RACE_CODE'):
        corners = [c for c in CORNERS if x[f'r{c}'].notna().all() and x[f'g{c}'].notna().all() and x[f'l{c}'].notna().all()]
        if len(corners) < 2:
            continue
        out.append((x, corners, *prepare_race(models, x, 'RACE_CODE', corners)))
    return out


BIG_MOVE = .25  # a quarter of the field between the first and the last corner


def move_metrics(races, P):
    """How well the formation reproduces mid-race moves (first to last predicted corner).
    corr: correlation of predicted and actual change of normalised rank; upHit / downHit: share of horses
    that actually gained / lost a quarter of the field and were predicted to move that way (>5%);
    falseBig: share of all horses predicted to move 15%+ that actually moved under 10%."""
    mp, ma = [], []
    for x, corners, numbers, pred, spreads, early in races:
        form, n = simulate(numbers, early, pred, spreads, corners, P), len(numbers)
        a, b = corners[0], corners[-1]
        mp.extend((np.array(form[b]['rank']) - np.array(form[a]['rank'])) / (n - 1))
        ma.extend(x[f'r{b}'].to_numpy() - x[f'r{a}'].to_numpy())
    mp, ma = np.array(mp), np.array(ma)
    up, down = ma <= -BIG_MOVE, ma >= BIG_MOVE
    return {'corr': float(np.corrcoef(mp, ma)[0, 1]), 'upHit': float(np.mean(mp[up] < -.05)), 'downHit': float(np.mean(mp[down] > .05)),
            'falseBig': float(np.mean((np.abs(mp) > .15) & (np.abs(ma) < .1))), 'predictedMove': float(np.mean(np.abs(mp))),
            'actualMove': float(np.mean(np.abs(ma))), 'upRows': int(up.sum()), 'downRows': int(down.sum())}


MOVE_GRID = [{'beta': b, 'k': k} for b in (.7, 1.0) for k in (1, 1.5, 2, 3)]
MOVE_RANK_TOLERANCE = 1.002  # the move layer may cost at most 0.2% of the mean rank error


def rank_error(models, races, P):
    s = score(models, races, P)
    return float(np.mean([v for k, v in s.items() if k[0] == 'r']))


def tune_moves(d, P):
    """Strength of the move layer, chosen on Jun-Aug 2025 like the contention parameters: the largest
    correlation with actual moves among settings whose mean rank error stays within the tolerance of the
    formation without moves. Rank error alone always prefers no movement (it rewards shrinking)."""
    models = fit_models(d, '20250601')
    races = evaluation_races(models, d, '20250601', '20250831')
    limit = rank_error(models, races, P) * MOVE_RANK_TOLERANCE
    best, best_corr = {**P, 'beta': 0.0, 'k': 0.0}, move_metrics(races, P)['corr']
    for M in MOVE_GRID:
        Q = {**P, **M}
        corr = move_metrics(races, Q)['corr']
        if rank_error(models, races, Q) <= limit and corr > best_corr:
            best, best_corr = Q, corr
    print(f'move layer tuned: beta={best["beta"]} k={best["k"]} (move corr {best_corr:.3f})', flush=True)
    return best


def score(models, races, P):
    """Validation errors per target. P=None scores the independent (no-interaction) predictions."""
    err = {}
    for x, corners, numbers, pred, spreads, early in races:
        n = len(numbers)
        form = independent(numbers, pred, spreads, corners) if P is None else simulate(numbers, early, pred, spreads, corners, P)
        for c in corners:
            err.setdefault(f'r{c}', []).extend(np.abs((np.array(form[c]['rank']) - 1) / (n - 1) - x[f'r{c}'].to_numpy()))
            err.setdefault(f'g{c}', []).extend(np.abs(np.array(form[c]['gap']) - x[f'g{c}'].to_numpy()))
            err.setdefault(f'l{c}', []).extend(np.abs(np.array(form[c]['lane'], float) - x[f'l{c}'].to_numpy()))
        ok = x.last3.notna().to_numpy()
        last3 = pred['last3'] if P is None else closing(models, x, form, corners)[0]
        err.setdefault('last3', []).extend(np.abs(last3 - x.last3.to_numpy())[ok])
    return {k: float(np.mean(v)) for k, v in err.items()}


def tune_contention(d):
    """Pick contention parameters on Jun-Aug 2025 with models fitted on data before June (usable history
    starts in 2025); the Sep-Dec validation period is never seen while tuning."""
    models = fit_models(d, '20250601')
    races = evaluation_races(models, d, '20250601', '20250831')
    base = score(models, races, None)
    best, best_obj = DEFAULT_PARAMS, None
    for P in PARAM_GRID:
        s = score(models, races, P)
        obj = float(np.mean([s[k] / base[k] for k in s]))
        if best_obj is None or obj < best_obj:
            best, best_obj = P, obj
    print(f'contention tuned on {len(races)} races: {best} (error ratio vs independent {best_obj:.3f})', flush=True)
    return best


def validate(d, P):
    models = fit_models(d, '20250901')
    races = evaluation_races(models, d, '20250901', '20251231')
    with_layer, without = score(models, races, P), score(models, races, None)
    tr = d.date < '20250901'
    metrics = {}
    for t in with_layer:
        base = []
        for x, corners, *_ in races:
            if t == 'last3':
                ok = x.last3.notna()
                base.extend(np.abs(x.loc[ok, 'past_last3'].fillna(d.loc[tr, 'last3'].median()) - x.loc[ok, 'last3']))
            elif int(t[1:]) in corners:
                base.extend(np.abs(x['past_' + t].fillna(d.loc[tr, t].median()) - x[t]))
        metrics[t] = {'mae': with_layer[t], 'independentMae': without[t], 'baselineMae': float(np.mean(base)),
                      'validationRows': len(base)}
        print(f"{t}: contention={with_layer[t]:.3f} independent={without[t]:.3f} past-mean={metrics[t]['baselineMae']:.3f}", flush=True)
    # How much traffic costs, as the closing model learned it (validation races, simulated formations).
    delays, held = [], []
    for x, corners, numbers, pred, spreads, early in races:
        form = simulate(numbers, early, pred, spreads, corners, P)
        t3, free = closing(models, x, form, corners)
        delays.extend(t3 - free)
        held.extend(form[corners[0]]['held'])
    metrics['_summary'] = {'races': len(races), 'meanKickDelay': float(np.mean(np.maximum(delays, 0))),
                           'delayedShare': float(np.mean(np.array(delays) > .05)), 'heldShareFirstCorner': float(np.mean(held))}
    print(metrics['_summary'], flush=True)
    # Mid-race moves with and without the move layer, and the past-5-run mean move as a reference.
    still = {**P, 'k': 0.0}
    past, actual = [], []
    for x, corners, *_ in races:
        a, b = corners[0], corners[-1]
        past.extend((x[f'past_r{b}'] - x[f'past_r{a}']).fillna(0))
        actual.extend(x[f'r{b}'] - x[f'r{a}'])
    metrics['_moves'] = {'model': move_metrics(races, P), 'withoutMoves': move_metrics(races, still),
                         'pastMeanCorr': float(np.corrcoef(past, actual)[0, 1]),
                         'rankError': rank_error(models, races, P), 'rankErrorWithoutMoves': rank_error(models, races, still),
                         'beta': P.get('beta', 0.0), 'k': P.get('k', 0.0)}
    print(metrics['_moves'], flush=True)
    # Race pace: predicted vs actual early 3F against the course norm.
    pp = np.array([pred['pace'] for x, corners, numbers, pred, *_ in races])
    pa = np.array([x.pace.iloc[0] for x, *_ in races])
    ok = ~np.isnan(pa)
    const = float(np.median(d.loc[tr].drop_duplicates('RACE_CODE').pace.dropna()))
    hit = np.mean([pace_label(a) == pace_label(b) for a, b in zip(pp[ok], pa[ok])])
    metrics['_pace'] = {'corr': float(np.corrcoef(pp[ok], pa[ok])[0, 1]), 'mae': float(np.mean(np.abs(pp[ok] - pa[ok]))),
                        'constantMae': float(np.mean(np.abs(const - pa[ok]))), 'labelHit': float(hit),
                        'labelShare': {k: float(np.mean([pace_label(v) == k for v in pa[ok]])) for k in 'HMS'}, 'races': int(ok.sum()),
                        'bands': list(PACE_BANDS)}
    print(metrics['_pace'], flush=True)
    return metrics


def corner_sets(d):
    """Corners a course actually passes, from history of the same venue/track/distance."""
    seen = d.groupby('RACE_CODE').agg(venue=('venue', 'first'), track=('track', 'first'), distance=('distance', 'first'),
                                      **{f'has{c}': (f'r{c}', lambda s: bool(s.notna().all())) for c in CORNERS})
    seen['set'] = seen[[f'has{c}' for c in CORNERS]].apply(lambda r: tuple(c for c in CORNERS if r[f'has{c}']), axis=1)
    seen = seen[seen.set.map(len) >= 2]
    exact = seen.groupby(['venue', 'track', 'distance']).set.agg(lambda s: s.mode().iloc[0]).to_dict()
    loose = seen.groupby(['venue', 'distance']).set.agg(lambda s: s.mode().iloc[0]).to_dict()
    return lambda v, t, dist: list(exact.get((v, t, dist)) or loose.get((v, dist)) or (3, 4))


def race_output(models, x, key, corners, P, gi_bands=PACE_BANDS):
    """One race: contention formation per corner, traffic-aware closing, finish estimate."""
    numbers, pred, spreads, early = prepare_race(models, x, key, corners)
    form = simulate(numbers, early, pred, spreads, corners, P)
    last3, free = closing(models, x, form, corners)
    n = len(numbers)
    horses = {}
    for i, num in enumerate(numbers):
        positions = []
        for c in corners:
            f = form[c]
            positions.append({'corner': c, 'rank': f['rank'][i], 'gap': round(f['gap'][i], 2), 'lane': float(f['lane'][i]),
                              'wantRank': f['want'][i], 'held': bool(f['held'][i]),
                              'boxed': bool(f['out'][i] > 0 and f['close'][i] > 0)})
        horses[str(num)] = {'positions': positions, 'last3': round(float(last3[i]), 1), 'last3Free': round(float(free[i]), 1),
                            'kickDelay': round(max(0.0, float(last3[i] - free[i])), 2)}
    # Finish = last-corner gap plus the closing difference over the final 3F (lengths/second rule);
    # the closing time already carries the delay from being shut in.
    gaps = form[corners[-1]]['gap']
    best = float(min(last3))
    fin = [gaps[i] + (float(last3[i]) - best) * LENGTHS_PER_SECOND for i in range(n)]
    lead = min(fin)
    for pos, i in enumerate(sorted(range(n), key=lambda i: (fin[i], numbers[i]))):
        horses[str(numbers[i])]['finish'] = {'rank': pos + 1, 'gap': round(fin[i] - lead, 2)}
    # Race pace: predicted early 3F vs the course's usual, and the horses that usually want the lead.
    wish = x.early_wish.to_numpy()
    led = x.past_led.to_numpy()
    front = [numbers[i] for i in np.argsort(np.nan_to_num(wish, nan=1.0)) if (wish[i] <= .2) or (led[i] >= .4)]
    # The label ranks the prediction among GI races (terciles of the model's own predictions for past GI
    # races): GIs start faster than the all-class course norm, and predictions shrink toward the mean.
    # 'early' stays in seconds against the course norm.
    pace = {'early': round(pred['pace'], 2), 'label': pace_label(pred['pace'], gi_bands),
            'bands': [round(b, 2) for b in gi_bands], 'frontRunners': front}
    return {'corners': corners, 'horses': horses, 'pace': pace}


GOING = {'良': 1, '稍重': 2, '重': 3, '不良': 4}  # JV-Data 馬場状態コード


def upcoming_rows(snap, d):
    race = snap['race']
    active = [h for h in snap['horses'] if h['status'] != 'scratched']
    hist = d[d.KETTO_TOROKU_BANGO.isin([h['horseId'] for h in active]) & (d.date < race['date'])]
    rows = []
    for h in active:
        past = hist[hist.KETTO_TOROKU_BANGO == h['horseId']].tail(5)
        row = {'distance': race['distance'], 'surface': int(race['surface'] == 'ダート'), 'venue': int(race['venueCode']),
               'track': int(race['trackCode']), 'gate': h['number'] / len(active), 'weight': h['weight'], 'age': h['age'],
               'field': len(active), 'blinker': np.nan if h['blinker'] is None else int(h['blinker']),
               'starts': int((hist.KETTO_TOROKU_BANGO == h['horseId']).sum()), 'race': snap['raceId'], 'UMABAN': h['number']}
        for t in TARGETS + MOVE_LABELS:
            row['past_' + t] = float(past[t].mean()) if past[t].notna().any() else np.nan
        row['going'] = GOING.get((snap.get('conditions') or {}).get('going'), np.nan)  # absent on fetched cards
        for t in ('pace', 'led'):
            row['past_' + t] = float(past[t].mean()) if past[t].notna().any() else np.nan
        row['past_up'] = float(past.mall.min()) if past.mall.notna().any() else np.nan
        row['past_down'] = float(past.mall.max()) if past.mall.notna().any() else np.nan
        rows.append(row)
    return add_race_relative(pd.DataFrame(rows), 'race')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--today', help='YYYY-MM-DD; rows on/after it are not history (default: today)')
    ap.add_argument('--forecasts', default=str(ROOT / 'public' / 'data' / 'forecasts.json'))
    ap.add_argument('--entries-dir', default=str(ROOT / 'data' / 'entries'))
    ap.add_argument('--out', default=str(ROOT / 'public' / 'data' / 'pace_forecasts.json'))
    args = ap.parse_args()
    today_s = (args.today or date.today().isoformat()).replace('-', '')
    d, raw = load_frame(today_s)
    d = add_traffic(add_history_features(add_labels(d)))
    print(d[TARGETS].notna().sum().to_string(), flush=True)
    params = tune_moves(d, tune_contention(d))
    metrics = validate(d, params)
    models = fit_models(d, '20260101')
    sets = corner_sets(d)
    gi_rows = d[(d.date < '20260101') & (d.grade == 'A')]
    gi_pace = float(gi_rows.drop_duplicates('RACE_CODE').pace.median())
    gi_pred = models['pace'].predict(race_level(gi_rows, 'RACE_CODE', 1)[PACE_FEATURES])
    gi_bands = tuple(float(v) for v in np.quantile(gi_pred, [1 / 3, 2 / 3]))
    print(f'GI races: actual pace {gi_pace:+.2f} s vs course norm; predicted terciles {gi_bands} ({len(gi_pred)} races)', flush=True)
    published = json.loads(Path(args.forecasts).read_text(encoding='utf-8'))['races']
    upcoming = {s['raceId']: s for s in load_upcoming_entries(args.entries_dir, today_s)}
    out_races = {}
    for race in published:
        if race.get('historical') is False:
            if race['id'] not in upcoming:
                continue
            x = upcoming_rows(upcoming[race['id']], d)
        else:
            x = d[d.RACE_CODE == race['id']]
        if x.empty:
            print('WARNING no feature rows for', race['id'], race['name'])
            continue
        nums = [int(v) for v in x.UMABAN]
        if sorted(nums) != sorted(h['number'] for h in race['horses']):
            print('WARNING field mismatch, skipped', race['id'], race['name'])
            continue
        corners = sets(int(x.venue.iloc[0]), int(x.track.iloc[0]), int(x.distance.iloc[0]))
        key = 'RACE_CODE' if 'RACE_CODE' in x else 'race'
        out_races[race['id']] = race_output(models, x, key, corners, params, gi_bands)
        print(race['name'], 'corners', corners)
    (ROOT / 'models' / 'pace').mkdir(parents=True, exist_ok=True)
    for t, m in models.items():
        m.booster_.save_model(str(ROOT / 'models' / 'pace' / f'{t}.txt'))
    model = {'name': 'Paddock展開予想モデル', 'version': VERSION, 'trainingEnd': '2025-12-31', 'validation': '2025-09〜12',
             'rows': int(len(d)), 'excludedRows': int(raw - len(d)), 'features': FEATURES, 'closingFeatures': LAST3_FEATURES,
             'moveFeatures': MOVE_FEATURES, 'contention': params, 'summary': metrics.pop('_summary'),
             'moves': metrics.pop('_moves'), 'pace': {**metrics.pop('_pace'), 'giPace': gi_pace, 'giBands': list(gi_bands)}, 'paceFeatures': PACE_FEATURES, 'metrics': metrics,
             'notes': ['各角の順位・先頭からの馬身差・併走群内の内外位置と上がり3Fを個別に学習（LightGBM）。',
                       '馬身差は「順位モデルの並び × 馬ごとの差モデルの形 × レース単位の隊列の長さモデル」で再構成。',
                       'ポジション争い：各馬の希望位置（順位モデル）と前の角の位置から押し出しの強さを決め、同じ列に並べる頭数を制限。列が埋まると先行型は外を回り、それ以外は後ろで控える（位置を失う）。',
                       '前半のペース：出走馬の先行型の頭数・過去に逃げた割合・その馬たちが走ったレースの前半の速さ・馬場状態から、前半3Fがコースの標準より何秒速いかを予想（H/M/S）。隊列の長さモデルに使う（予想値で学習）。上がりモデルには馬場状態を入れる。',
                       '道中の位置変化：最初の角からの累積移動量を角ごとに学習（過去走の移動・最大の上げ幅・出走馬中の動く馬の割合・角の数）。2角以降は「最初の角の位置＋予想移動量」と希望位置を混ぜて並べ直す。強さは2025年6〜8月で、順位誤差の悪化を0.2%以内に抑える範囲で移動の相関が最大になる値を選ぶ。',
                       '仕掛け遅れ：上がり3Fモデルは最終角の渋滞（外に並ぶ頭数・1馬身未満で前が壁）を特徴量に学習し、予想時はポジション争いで決まった隊列の渋滞を入れる。前が開いた場合との差を仕掛け遅れ（秒）として出す。',
                       '馬身差と内外はJRAの通過順表記から復元。記号の幅（,=1〜2馬身, -=2〜5, ==5以上）は中央値で近似。',
                       '内外は併走群の内からの位置（0=最内または単独）。ラチからの距離ではない。',
                       'ゴールは最終角の馬身差に上がり3Fの差（1秒=5馬身）を加えた推定。着順モデルではない。',
                       '過去5走のみを使用。オッズ・人気・払戻・対象レースの結果は使わない。']}
    out = {'schema': SCHEMA, 'generatedAt': datetime.now().isoformat(timespec='seconds'), 'model': model,
           'units': {'gap': '馬身（先頭から）', 'lane': '併走群の内からの位置（0=最内/単独）', 'last3': '秒'}, 'races': out_races}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (ROOT / 'models' / 'pace' / 'report.json').write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {len(out_races)} races -> {args.out}')


if __name__ == '__main__':
    main()
