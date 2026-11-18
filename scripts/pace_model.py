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
import sys, json, argparse, heapq, re
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
# Revision of the v1 format (README): 1.1 adds the optional rankRange / last3Range fields; v1 readers ignore them.
SCHEMA_REVISION = '1.1'
VERSION = '1.9'  # 1.9: past_<target> = EWM of all earlier runs (alpha 0.2), RANK_BLEND 0.75, E0 validation procedure (E1)
# 1.8: no code change; refitted and re-tuned after 2022 was imported (追加指示8; 1.7: 2023+, 追加指示5)
# 1.6: forecast ranges (rankRange / last3Range); same point model, refitted and re-tuned on the current DB# Representative lengths for each JRA separator (midpoints of the published bands).
GAP_LENGTHS = {'': 0.6, ',': 1.5, '-': 3.5, '=': 6.5}
SEP_ORDER = ['', ',', '-', '=']
LENGTHS_PER_SECOND = 5.0  # JRA rule of thumb: 1 length is about 0.2 s
# 5.7 (least squares of the official margins on TIME_SA) was tested on 2026-11-18 and NOT adopted: finish-gap MAE
# -0.07 lengths but finish-order correlation -0.002 to -0.010 (scripts/experiment_spread_after_expansion.py).
CORNERS = (1, 2, 3, 4)
TARGETS = [f'{kind}{c}' for c in CORNERS for kind in ('r', 'g', 'l')] + ['last3']
# Evaluation / validation procedure (E0 audit, 2026-11-18; README「評価の監査（E0）」). The defaults are the audited
# procedure. The legacy values reproduce the v1.8 report and the earlier experiment records exactly.
EVAL_CORNERS = 'course'  # corners a validation race is formed on: 'course' = the course's usual corner set from races
#                          before the window (what race_output does before a race); 'labels' (legacy) = the corners whose
#                          labels happen to be complete in the race itself. Scoring always needs complete labels.
PACE_OOF = 'forward'     # pace input of the spread models' training rows: 'forward' = predicted by a pace model fitted
#                          on earlier races only; 'kfold' (legacy) = fitted on the other three date chunks, later races too
PACE_OOF_CHUNKS = 8      # date chunks of the forward scheme; the first chunk has no earlier races and stays out of the
#                          spread models' training rows


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
          u.CORNER1_JUNI,u.CORNER2_JUNI,u.CORNER3_JUNI,u.CORNER4_JUNI,u.KOHAN_3F,u.IJO_KUBUN_CODE,u.SOHA_TIME,u.TIME_SA
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026'
          AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          ORDER BY r.KAISAI_NEN,r.KAISAI_GAPPI,r.RACE_CODE,u.UMABAN""")
        rows = q.fetchall()
    c.close()
    d = pd.DataFrame(rows)
    d['date'] = d.KAISAI_NEN + d.KAISAI_GAPPI
    d = d[d.date < today_s]  # pre-race cards have no results: never history or training rows
    # Named races only. Adding the unnamed ordinary races (about 3.7x the rows) was tested on 2026-11-13 and
    # made the named-race validation worse (finish-order corr 0.205 -> 0.192): lower classes run differently.
    d = d[d.BAMEI.map(clean) & d.KYOSOMEI_HONDAI.map(clean)].copy()
    for col in ['KYORI', 'TRACK_CODE', 'SHUSSO_TOSU', 'UMABAN', 'WAKUBAN', 'BAREI', 'FUTAN_JURYO', 'KOHAN_3F'] + [f'CORNER{i}_JUNI' for i in CORNERS]:
        d[col] = d[col].map(number)
    d = d[d.KYORI.between(1000, 3600) & d.SHUSSO_TOSU.between(5, 18) & d.UMABAN.between(1, 18)]
    d = d[d.IJO_KUBUN_CODE.fillna('0').str.strip().isin(['', '0', '3', '4', '5', '6', '7'])]
    counts = d.groupby('RACE_CODE').UMABAN.transform('nunique')
    # Same rule as build_forecasts.py: partially imported races stay in the frame
    # as history for past-* features, but are flagged and never trained on.
    d['complete'] = counts == d.SHUSSO_TOSU
    d = d.drop_duplicates(['RACE_CODE', 'UMABAN']).reset_index(drop=True)
    print(f"field-complete races: {int(d.groupby('RACE_CODE')['complete'].first().sum())} / {d.RACE_CODE.nunique()}", flush=True)
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
    # Finish time and margin (E2): labels / checks only, never features of the race itself.
    d['soha'] = d.SOHA_TIME.map(soha_seconds)
    sa = d.TIME_SA.map(lambda v: number(str(v).strip()) if str(v).strip() not in ('', '9999') else np.nan)
    d['finish_gap'] = (sa.clip(lower=0) / 10).where(sa.abs() < 9000)  # seconds behind the winner (winner 0)
    # Race-level: early 3F of the leader (label for the pace model) and going (known before the off).
    z3 = d.ZENHAN_3F.map(number)
    d['early3'] = (z3 / 10).where(z3.between(300, 450))
    going = np.where(d.surface == 1, d.DIRT_BABAJOTAI_CODE.map(number), d.SHIBA_BABAJOTAI_CODE.map(number))
    d['going'] = pd.Series(going, index=d.index).where(lambda g: g.between(1, 4))
    d['grade'] = d.GRADE_CODE.fillna('').str.strip()  # only to describe GI pace, never a model input
    return d, len(rows)


def soha_seconds(v):
    """JV-Data SOHA_TIME (MSSs, e.g. 1345 = 1:34.5) -> seconds; 0000 / 9999 / malformed -> NaN."""
    s = str(v or '').strip()
    if len(s) != 4 or not s.isdigit() or s in ('0000', '9999'):
        return np.nan
    return int(s[0]) * 60 + int(s[1:3]) + int(s[3]) / 10


def add_time_labels(d):
    """t600: the horse's own time at the 600 m-to-go mark = finish time - its own last 3F (the per-horse last 3F is
    timed from the horse's own pass of the mark). gap600: seconds behind the first horse to pass the mark, only for
    full-field races where every horse has t600. The mark is not a corner (Nakayama: about the 3rd corner, Tokyo: late
    in the 4th): used to calibrate the corner labels and as a second truth near corners 3-4, never as a corner label (E2)."""
    d['t600'] = d.soha - d.last3
    by_race = d.groupby('RACE_CODE').t600
    ok = by_race.transform(lambda s: s.notna().all()).astype(bool) & d.complete
    d['gap600'] = (d.t600 - by_race.transform('min')).where(ok)
    return d


def add_past_gap600(d):
    """past_gap600: the history summary (HISTORY_ALPHA) of the horse's earlier gap600 (E2 candidate feature). d sorted by date."""
    by_horse = d.groupby('KETTO_TOROKU_BANGO').gap600
    if HISTORY_ALPHA is None:
        d['past_gap600'] = by_horse.transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    else:
        d['past_gap600'] = by_horse.transform(lambda s: s.shift().ewm(alpha=HISTORY_ALPHA, ignore_na=True).mean())
    return d


# --- Jockey / class change / last-start margin (追加指示3, 2026-11-18) -------------------------------------
# Not used: body-weight change (announced just before the off, so a published forecast cannot contain it).
# Columns checked against the local DB (2026-11-18):
#   jockey : umagoto_race_joho.KISHUMEI_RYAKUSHO, whitespace removed. KISHU_CODE is NOT usable as the id: it is
#            '00000' on about a third of 2025 rows (22380 of 61221) although the name is filled, and the same
#            jockey appears under both a code and 00000. Names are unique enough (573 names / 269 non-zero codes).
#   class  : race_shosai.GRADE_CODE (A=GI, B=GII, C=GIII, D=non-graded stakes, E=other special race, L=listed,
#            blank=ordinary) + KYOSO_JOKEN_CODE_SAIJAKUNEN (the youngest-age condition; every flat race of 2020+
#            has it filled): 701 maiden debut, 703 maiden, 005 1-win (500万下), 010 2-win, 016 3-win, 999 open.
#   margin : umagoto_race_joho.TIME_SA, signed, in 0.1 s, to the winner (the winner holds a NEGATIVE value: the
#            margin to the runner-up). Checked on 2025 rows: winners always <= 0, every other horse >= 0 and equal
#            to SOHA_TIME minus the winner's time in 95% of rows (the rest are minute rollovers). '9999' = none.
#            So no length-code table is needed: last_margin = max(TIME_SA, 0) / 10 seconds, winner = 0.
# History is built from ALL races (named or not) since 2020: the training frame only keeps named races, and
# class_last / last_margin / jockey stats from named races alone would skip most of a horse's career.
CLASS_BY_CONDITION = {'701': 0, '703': 0, '005': 1, '010': 2, '016': 3, '999': 4}  # new/maiden 0, 1-win 1, 2-win 2, 3-win 3, open 4
CLASS_BY_GRADE = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'L': 4}  # GI 7, GII 6, GIII 5, non-graded stakes / listed 4
JMC_FEATURES = ['jockey_same', 'jockey_rides', 'jockey_mean_r1', 'jockey_mean_fin',
                'class_today', 'class_last', 'class_diff', 'last_margin']
JMC_VARIANTS = ['class_mean5', 'margin_mean3']
JMC_GROUPS = {'jockey': JMC_FEATURES[:4], 'class': JMC_FEATURES[4:7], 'margin': ['last_margin']}


def class_level(grade, condition):
    g, k = (grade or '').strip(), (condition or '').strip()
    return float(CLASS_BY_GRADE[g]) if g in CLASS_BY_GRADE else float(CLASS_BY_CONDITION[k]) if k in CLASS_BY_CONDITION else np.nan


def jockey_name(s):
    s = re.sub(r'[\s　]+', '', str(s or ''))
    return s if s and clean(s) else None


def load_history(today_s):
    """One row per run of every race (named or not) since 2020 before today_s, with jockey / class / margin /
    first-corner position / finish. Same venue, track and abnormal-result filters as load_frame."""
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute("""SELECT r.RACE_CODE,r.KAISAI_NEN,r.KAISAI_GAPPI,r.GRADE_CODE,r.KYOSO_JOKEN_CODE_SAIJAKUNEN,r.SHUSSO_TOSU,
          u.UMABAN,u.KETTO_TOROKU_BANGO,u.KISHUMEI_RYAKUSHO,u.CORNER1_JUNI,u.KAKUTEI_CHAKUJUN,u.TIME_SA,u.IJO_KUBUN_CODE
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026'
          AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          ORDER BY r.KAISAI_NEN,r.KAISAI_GAPPI,r.RACE_CODE,u.UMABAN""")
        h = pd.DataFrame(q.fetchall())
    c.close()
    h['date'] = h.KAISAI_NEN + h.KAISAI_GAPPI
    h = h[h.date < today_s]
    h = h[h.IJO_KUBUN_CODE.fillna('0').str.strip().isin(['', '0', '3', '4', '5', '6', '7'])]
    h = h.drop_duplicates(['RACE_CODE', 'UMABAN']).reset_index(drop=True)
    h['UMABAN'] = h.UMABAN.map(number)  # numeric like the training frame, so the merge keys match
    field = h.SHUSSO_TOSU.map(number)
    r1, fin = h.CORNER1_JUNI.map(number), h.KAKUTEI_CHAKUJUN.map(number)
    h['jockey'] = h.KISHUMEI_RYAKUSHO.map(jockey_name)
    h['cls'] = [class_level(g, k) for g, k in zip(h.GRADE_CODE, h.KYOSO_JOKEN_CODE_SAIJAKUNEN)]
    h['r1'] = ((r1 - 1) / (field - 1)).where(r1.between(1, field) & (field > 1))
    h['fin'] = ((fin - 1) / (field - 1)).where(fin.between(1, field) & (field > 1))
    sa = h.TIME_SA.map(lambda v: number(str(v).strip()) if str(v).strip() not in ('', '9999') else np.nan)
    h['margin'] = (sa.clip(lower=0) / 10).where(sa.abs() < 9000)  # seconds behind the winner; winner (negative) -> 0
    return h[['RACE_CODE', 'UMABAN', 'KETTO_TOROKU_BANGO', 'date', 'jockey', 'cls', 'r1', 'fin', 'margin']]


def jockey_stats(rides, jockey, date):
    """(rides, mean first-corner position, mean normalised finish) over the jockey's rides strictly before `date`.
    Unseen jockey -> all NaN (LightGBM handles the missing values)."""
    p = rides[(rides.jockey == jockey) & (rides.date < date)] if jockey else rides.iloc[:0]
    if p.empty:
        return np.nan, np.nan, np.nan
    return float(len(p)), float(p.r1.mean()) if p.r1.notna().any() else np.nan, float(p.fin.mean()) if p.fin.notna().any() else np.nan


def add_jockey_class_margin(d, h):
    """Attach JMC_FEATURES (+ JMC_VARIANTS) to the training frame from the all-race history h. Every value uses
    only runs / rides on earlier dates (jockey stats exclude the same day), so nothing of the race itself leaks."""
    h = h.sort_values(['date', 'RACE_CODE', 'UMABAN']).reset_index(drop=True)
    g = h.groupby('KETTO_TOROKU_BANGO')
    h['jockey_prev'] = g.jockey.shift()
    h['class_last'] = g.cls.shift()
    h['last_margin'] = g.margin.shift()
    h['class_mean5'] = g.cls.transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    h['margin_mean3'] = g.margin.transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    h['jockey_same'] = (h.jockey == h.jockey_prev).astype(float).where(h.jockey.notna() & h.jockey_prev.notna())
    h['class_today'] = h.cls
    h['class_diff'] = h.cls - h.class_last
    # Jockey history by date: cumulative sums over earlier dates only (the same day's rides are subtracted).
    day = h.dropna(subset=['jockey']).assign(n1=lambda x: x.r1.notna().astype(float), s1=lambda x: x.r1.fillna(0),
                                             nf=lambda x: x.fin.notna().astype(float), sf=lambda x: x.fin.fillna(0), n=1.0)
    day = day.groupby(['jockey', 'date'])[['n', 'n1', 's1', 'nf', 'sf']].sum().sort_index()
    prior = day.groupby(level='jockey').cumsum() - day
    prior['jockey_rides'] = prior.n.where(prior.n > 0)
    prior['jockey_mean_r1'] = (prior.s1 / prior.n1).where(prior.n1 > 0)
    prior['jockey_mean_fin'] = (prior.sf / prior.nf).where(prior.nf > 0)
    h = h.join(prior[['jockey_rides', 'jockey_mean_r1', 'jockey_mean_fin']], on=['jockey', 'date'])
    keep = ['RACE_CODE', 'UMABAN'] + JMC_FEATURES + JMC_VARIANTS
    return d.merge(h[keep], on=['RACE_CODE', 'UMABAN'], how='left')


def upcoming_jmc(h, horse_id, jockey, date, class_today):
    """Same features for a declared entry (strictly earlier runs / rides only). The pipeline only imports GI
    cards, so the caller passes class_today = 7."""
    runs = h[(h.KETTO_TOROKU_BANGO == horse_id) & (h.date < date)].sort_values(['date', 'RACE_CODE'])
    jockey = jockey_name(jockey)
    rides, r1, fin = jockey_stats(h, jockey, date)
    last = runs.iloc[-1] if len(runs) else None
    row = {'jockey_same': np.nan if last is None or jockey is None or last.jockey is None else float(last.jockey == jockey),
           'jockey_rides': rides, 'jockey_mean_r1': r1, 'jockey_mean_fin': fin, 'class_today': float(class_today),
           'class_last': np.nan if last is None else last.cls, 'last_margin': np.nan if last is None else last.margin,
           'class_mean5': runs.cls.tail(5).mean() if len(runs) else np.nan,
           'margin_mean3': runs.margin.tail(3).mean() if len(runs) else np.nan}
    row['class_diff'] = row['class_today'] - row['class_last']
    return row


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


# Summary of a horse's earlier runs for the per-target history columns past_<target> (E1, 2026-11-18): None = mean of
# the last 5 runs (v1.8 and earlier); a float = exponentially weighted mean of ALL earlier runs with that alpha (weights
# 1, 1 - alpha, (1 - alpha)^2, ... from the latest run back; runs without the label are skipped). Moves, pace and led stay
# 5-run means. The 5-run means are always kept as past5_<target>: the report's past-mean baseline.
# 0.2 chosen on Jun-Aug 2025 among {None, 0.2, 0.3, 0.5} together with RANK_BLEND (scripts/experiment_ewm_history.py, E1).
HISTORY_ALPHA = 0.2


def summarize_history(d):
    """past_<target> / past5_<target> for TARGETS; d sorted by date, one row per run (the race itself excluded by shift)."""
    by_horse = d.groupby('KETTO_TOROKU_BANGO')
    for t in TARGETS:
        d['past5_' + t] = by_horse[t].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        if HISTORY_ALPHA is None:
            d['past_' + t] = d['past5_' + t]
        else:
            d['past_' + t] = by_horse[t].transform(lambda s: s.shift().ewm(alpha=HISTORY_ALPHA, ignore_na=True).mean())
    return d


def runs_summary(runs, t):
    """The same summary for a pre-race card: `runs` are the horse's earlier runs in date order."""
    v = runs[t]
    if not v.notna().any():
        return np.nan
    if HISTORY_ALPHA is None:
        return float(v.tail(5).mean()) if v.tail(5).notna().any() else np.nan
    return float(v.ewm(alpha=HISTORY_ALPHA, ignore_na=True).mean().iloc[-1])


def add_history_features(d):
    d = add_move_labels(d.sort_values(['date', 'RACE_CODE', 'UMABAN']).reset_index(drop=True))
    by_horse = d.groupby('KETTO_TOROKU_BANGO')
    for t in MOVE_LABELS:
        d['past_' + t] = by_horse[t].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    d = summarize_history(d)
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
# Energy spent before the straight (脚が溜まらない). Tested on Sep-Dec 2025 (see README): being near the front
# in a fast early pace costs most; being further forward early than usual (二の足 / かかり) a little. Moving up
# mid-race (早仕掛け) and slow-pace keenness showed no cost in the data and are left out.
ENERGY = ['e_exposure', 'e_ahead']
CLOSING_BASE = FEATURES + ['going']  # going (馬場状態) lowers the closing error (validation 0.951 -> 0.925 s)
LAST3_FEATURES = CLOSING_BASE + TRAFFIC + ENERGY
# Contention-layer parameters; tuned on Jun-Aug 2025 (see tune_contention).
# Keep the requested four-wide limit fixed during tuning; front runners may run five wide.
DEFAULT_PARAMS = {'alpha': .5, 'delta': .08, 'width': 4, 'front': .35, 'hold': .06}
PARAM_GRID = [{'alpha': a, 'delta': dl, 'width': w, 'front': f, 'hold': h}
              for a in (.5, .7) for dl in (.04, .08) for w in (4,) for f in (.25, .45) for h in (.03, .06, .1)]
# Written to report.json as knownIssues. Checked 2026-11-17 (scripts/experiment_contention_front.py).
KNOWN_ISSUES = [
    '2角の馬身差（g2）はポジション争いありの方が独立予測より悪い（v1.4 3.69 → v1.5 3.92馬身、独立は3.97 → 3.90）。'
    'v1.5の再チューニングで変わったfront（0.25→0.45）が原因ではない：同じデータで他を固定してfrontだけ0.25に戻しても、'
    '検証（9〜12月273レース）のg2は3.817→3.826とむしろわずかに悪化し、他の指標の差も0.003以内。'
    '6〜8月のチューニングでもfrontの差は目的関数で0.0002しかなく、ほぼ効いていない。'
    '目的関数（全指標の誤差比の平均）は2角の内外（0.83）と上がり（0.95）の改善が大きく、馬身差（g1〜g4は1.00〜1.02）の悪化を許す形で選ばれる。'
    'なお同じv1.5の設定でも現在のDBで再実行すると独立予測のg2は3.75になり、データ更新だけで0.15馬身動く。',
    'v1.6（2026-11-18）は現在のDB（行数17041→19006）で再チューニングし、frontは0.25が選ばれた（6〜8月の目的関数で0.45との差0.0002）。'
    'v1.5比の検証誤差：g2 3.92→3.83、l2 0.429→0.416、上がり0.894→0.884は改善、g1 3.82→3.83、l3 0.523→0.534、r1/r2 +0.002は悪化（DB更新の影響を含む）。',
    'v1.7（2026-11-18）はコード無変更で、取り込み直したDB（2023年以降の全レース。学習対象の全頭そろう名前付きレース1426→3442）で再学習・再チューニングした（frontは0.45）。'
    '同じ検証273レースで、順位 r1〜r4 0.236/0.242/0.252/0.262→0.230/0.235/0.248/0.257、上がり0.884→0.814秒、馬身差 g1/g2 3.83/3.83→3.41/3.39。'
    '内外の1角・4角は0.004/0.007悪化（許容0.01以内）。2020〜2022年はまだ一部しかない（scripts/experiment_data_expansion.py）。',
    '1・2角の馬身差（g1/g2）は拡充前は過去5走平均に負けていた（3.83/3.83 対 3.40/3.66）。拡充後は同一データの過去平均3.30/3.46に対し3.41/3.39で、'
    'g1はなお0.10馬身負け、g2は勝つが差はシード差（0.03/0.05）並み。どちらも記号からの馬身差ラベルの誤差（1頭あたり約0.8馬身）より小さい。'
    '隊列長モデルは1・2角の学習レースが278/323→1214/1421になり、定数予想をわずかに上回るようになった（scripts/experiment_spread_after_expansion.py）。',
    'v1.8（2026-11-18）はコード無変更で、2022年を取り込んだDB（学習対象の全頭そろう名前付きレース3442→4330）で再学習・再チューニングした（frontは0.45、holdは0.06→0.03）。'
    '同じ検証273レースで、順位 r1〜r4 0.2304/0.2345/0.2477/0.2574→0.2290/0.2328/0.2470/0.2541（シード7では1・2・4角で0.002超の改善、シード42では4角のみ）、'
    '馬身差 g3/g4 2.59/2.37→2.49/2.21、g1 3.41→3.49（シード7では3.45→3.42）、上がり0.814→0.823秒（許容0.01以内）。'
    '4角のポジション争いあり（g4 2.21）は独立予測（2.19）よりなお悪いが、差は0.17→0.01馬身に縮んだ。2020・2021年はまだ一部しかない。',
    '評価の監査（E0、2026-11-18）：公開の内外は全角で「全頭0」に MAE・クラス正解率とも負ける（定数0比 1.20/1.02/1.10/1.09）。'
    '採点は localLane（行内の番号）、画面の lane は進路スロットで、表示している lane は採点されていない。'
    'ポジション争いは順位を良くしておらず（選定期間では独立より悪い）、上がりの改善はポジション争いではなく上がりモデル自体による。'
    '天皇賞（春）2026 の角の組は全データの最頻値（1〜4角）で、当日より前のレースだけなら1・2角になる（自分の結果が角の組を決めている）。',
    'v1.9（2026-11-18）：各角の順位・馬身差・内外と上がりの過去走要約を、直近5走の平均から全過去走の指数加重平均（α=0.2）に替え、順位の混合を0.5→0.75にした'
    '（6〜8月で選定、scripts/experiment_ewm_history.py）。検証は E0 の手順（コースの角の組、前方のみのペース交差予測）。'
    '同じ273レース・再チューニングありで、公開隊列の τ は1〜4角 +0.015/+0.018/+0.023/+0.011（シード7 +0.010/+0.006/+0.011/+0.006）、'
    '整数順位MAE −0.007/−0.007/−0.007/−0.003、上がり −0.009秒、馬身差・内外は許容内。',
]


def energy_cost(first, pace, usual):
    """first: normalised rank at the first observed corner; pace: early 3F vs course norm (s, negative =
    fast); usual: the horse's usual early position. e_exposure: near the front in a fast pace;
    e_ahead: further forward than usual early (used early speed / keen)."""
    first, pace, usual = (np.asarray(v, float) for v in (first, pace, usual))
    return {'e_exposure': np.clip(-np.nan_to_num(pace), 0, None) * (1 - first),
            'e_ahead': np.clip(np.nan_to_num(usual - first, nan=0.0), 0, None)}


def add_traffic(d):
    """Traffic at the last corner and energy spent before it, from the race itself. Only training inputs
    for the closing model: at prediction time these come from the simulated formation and the predicted
    pace, never from the race itself."""
    d['traffic_out'] = d.out4.fillna(d.out3)
    d['traffic_close'] = d.close4.fillna(d.close3)
    first = d[[f'r{c}' for c in CORNERS]].bfill(axis=1).iloc[:, 0]
    for k, v in energy_cost(first, d.pace, d.early_wish).items():
        d[k] = v
    return d


def fit_models(d, end):
    """Per-horse models (intentions), race spread models, and the traffic-aware closing model."""
    fit, models = (d.date < end) & d.complete, {}
    for t in TARGETS:
        rows = fit & d[t].notna()
        # Gap shape: median (L1) resists tailed-off horses. Lanes, ranks, closing: mean.
        models[t] = lgbm('l1' if t[0] == 'g' else 'l2').fit(d.loc[rows, FEATURES], d.loc[rows, t])
    rows = fit & d.last3.notna()
    # Traffic and spent energy can only slow a horse down: monotone (+).
    closing_model = lgbm('l2').set_params(monotone_constraints=[0] * len(CLOSING_BASE) + [1] * (len(TRAFFIC) + len(ENERGY)))
    models['last3_traffic'] = closing_model.fit(d.loc[rows, LAST3_FEATURES], d.loc[rows, 'last3'])
    # Pace model, and out-of-fold pace predictions for the spread models' training rows (so they learn
    # from predictions as noisy as the ones they get before a race, not from the actual pace).
    train = d[fit].copy()
    races = race_level(train, 'RACE_CODE', 1).dropna(subset=['pace']).sort_values('date')
    models['pace'] = pace_lgbm().fit(races[PACE_FEATURES], races['pace'])
    oof, skip = pace_oof(races)
    train['pace_pred'] = train.RACE_CODE.map(oof)
    for c in CORNERS:
        races = race_level(train, 'RACE_CODE', c).dropna(subset=['spread'])
        races = races[~races.index.isin(skip)]
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


def pace_oof(races):
    """Out-of-sample pace predictions for the training races (index = RACE_CODE, sorted by date), as noisy as the ones a
    spread model gets before a race. Returns (predictions, races left out of the spread models' training rows).
    'forward': each date chunk is predicted by a pace model fitted on the earlier chunks only (the first chunk has no
    earlier races, so it gets no prediction and is left out). 'kfold' (legacy): the other three chunks, later races too."""
    oof = pd.Series(np.nan, index=races.index)
    if PACE_OOF == 'kfold':
        for part in np.array_split(races.index, 4):
            rest = races.index.difference(part)
            oof[part] = pace_lgbm().fit(races.loc[rest, PACE_FEATURES], races.loc[rest, 'pace']).predict(races.loc[part, PACE_FEATURES])
        return oof, pd.Index([])
    parts = np.array_split(races.index.to_numpy(), PACE_OOF_CHUNKS)
    for k in range(1, len(parts)):
        rest = np.concatenate(parts[:k])
        oof[parts[k]] = pace_lgbm().fit(races.loc[rest, PACE_FEATURES], races.loc[rest, 'pace']).predict(races.loc[parts[k], PACE_FEATURES])
    return oof, pd.Index(parts[0])


def pace_sigma(d, end):
    """Spread of the forward out-of-sample pace errors on the training races before `end` (seconds): the scale of the
    normal predictive distribution used for the H/M/S probabilities (RPS). Uses the forward scheme whatever PACE_OOF is."""
    races = race_level(d[(d.date < end) & d.complete], 'RACE_CODE', 1).dropna(subset=['pace']).sort_values('date')
    parts = np.array_split(races.index.to_numpy(), PACE_OOF_CHUNKS)
    err = []
    for k in range(1, len(parts)):
        rest = np.concatenate(parts[:k])
        pred = pace_lgbm().fit(races.loc[rest, PACE_FEATURES], races.loc[rest, 'pace']).predict(races.loc[parts[k], PACE_FEATURES])
        err.extend(pred - races.loc[parts[k], 'pace'].to_numpy())
    return float(np.std(err)), races['pace'].to_numpy()


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
    order of that pressure. Horses within delta of the row's leader run abreast.
    The width limit applies only before corner 4 and to horses not moving up from their previous rank.
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
        order = sorted(range(n), key=lambda i: (prio[i], numbers[i]))
        want_rank = [0] * n
        for pos, i in enumerate(order):
            want_rank[i] = pos + 1
        # Keep a track-relative path across rows/corners instead of resetting each row to lane 0.
        # A horse trying to improve its rank goes outside the horses it intends to pass.
        advancing = [False] * n
        side = np.maximum(0, lane_want).copy() if prev_lane is None else prev_lane.copy()
        if prev is not None:
            previous_rank = np.rint(prev * max(1, n - 1)).astype(int) + 1
            for i in range(n):
                advancing[i] = want_rank[i] < previous_rank[i]
                if advancing[i]:
                    crossed = [j for j in range(n) if want_rank[i] <= previous_rank[j] < previous_rank[i]]
                    side[i] = max(side[i], max(prev_lane[j] for j in crossed) + 1)
        rows, held = [], [False] * n
        queue = [(prio[i], numbers[i], i) for i in order]
        heapq.heapify(queue)
        while queue:
            p, _, i = heapq.heappop(queue)
            if rows and p - rows[-1]['p0'] < P['delta']:
                row = rows[-1]['m']
                if c >= 4 or advancing[i] or len(row) < width or (early[i] <= P['front'] and len(row) < width + 1):
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
            row['m'] = sorted(row['m'], key=lambda i: (side[i], numbers[i]))
            outer = -1.0
            for i in row['m']:
                pos += 1
                outer = max(float(side[i]), outer + 1)
                rank[i], lane[i], row_of[i] = pos, outer, ri
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
                   # Observed labels describe position within a row, not the track-relative path. At the first corner the
                   # row index is a poor proxy (validation l1 0.58 vs 0.43 independent), so the model's own lane is used.
                   'localLane': ([float(v) for v in np.clip(lane_want, 0, 4)] if c == corners[0] else
                                 [rows[row_of[i]]['m'].index(i) for i in range(n)]),
                   'out': [sum(lane[j] > lane[i] for j in rows[row_of[i]]['m']) for i in range(n)],
                   'close': [float(any(0 < row_gap[row_of[i]] - row_gap[row_of[j]] < 1.0
                                       and abs(lane[j] - lane[i]) < 1.0 for j in range(n))) for i in range(n)],
                   'want': want_rank}
        prev, prev_lane = (np.array(rank) - 1) / max(1, n - 1), np.array(lane, float)
        if base is None:
            base = prev
    return form


# Weight of the independent model in the rank prediction (the rest is the horse's past summary past_r<c>), applied
# before the formation is built. Gaps, lanes and last3 stay model-only. v1.9: 0.75, re-chosen on Jun-Aug 2025 with
# HISTORY_ALPHA by the tau of the published formation (scripts/experiment_ewm_history.py). Before: 0.5 by the
# continuous MAE (scripts/experiment_rank_blend.py, pre-import frame).
RANK_BLEND = 0.75


def blend_ranks(pred, x):
    for c in CORNERS:
        past = x[f'past_r{c}'].to_numpy(float)
        model = np.asarray(pred[f'r{c}'], float)
        pred[f'model_r{c}'] = model  # kept for the pre-blend continuous MAE (E0 standard)
        pred[f'r{c}'] = np.where(np.isnan(past), model, RANK_BLEND * model + (1 - RANK_BLEND) * past)
    return pred


def prepare_race(models, x, key, corners):
    numbers = [int(v) for v in x.UMABAN]
    pred = blend_ranks({t: models[t].predict(x[FEATURES]) for t in TARGETS}, x)
    xm = x.assign(n_corners=len(corners))  # the course's corner count, as in the training label
    for c in MOVE_CORNERS:
        if f'move{c}' in models:
            pred[f'move{c}'] = models[f'move{c}'].predict(xm[MOVE_FEATURES])
    pace = float(models['pace'].predict(race_level(x, key, 1)[PACE_FEATURES])[0])
    pred['pace'] = pace
    xp = x.assign(pace_pred=pace)
    spreads = {c: float(models[f'spread{c}'].predict(race_level(xp, key, c)[SPREAD_FEATURES])[0]) for c in CORNERS}
    return numbers, pred, spreads, x.early_rel.fillna(.5).to_numpy()


def closing_inputs(x, form, corners, pace):
    """Closing-model inputs with the formation's traffic at the last corner and its first-corner energy cost."""
    last, n = form[corners[-1]], len(form[corners[-1]]['rank'])
    xt = x[CLOSING_BASE].copy()
    xt['traffic_out'], xt['traffic_close'] = np.array(last['out'], float), np.array(last['close'], float)
    first = (np.array(form[corners[0]]['rank']) - 1) / max(1, n - 1)
    for k, v in energy_cost(first, np.full(n, pace), x.early_wish.to_numpy()).items():
        xt[k] = v
    return xt


def closing(models, x, form, corners, pace, parts=False):
    """Closing 3F with the traffic the formation leaves at the last corner and the energy its first-corner
    position cost in the predicted pace; also with a clear run (traffic removed, energy kept).
    parts=True also returns the closing with neither traffic nor energy cost (for paceCost)."""
    xt = closing_inputs(x, form, corners, pace)
    free = xt.copy()
    free['traffic_out'], free['traffic_close'] = 0.0, 0.0
    m = models['last3_traffic']
    out = (m.predict(xt[LAST3_FEATURES]), m.predict(free[LAST3_FEATURES]))
    if parts:
        fresh = free.copy()
        for k in ENERGY:
            fresh[k] = 0.0
        out += (m.predict(fresh[LAST3_FEATURES]),)
    return out


# --- Forecast intervals (追加指示4 フェーズ3, 2026-11-18) ---------------------------------------------------
# Quantile LightGBM (alpha 0.25 / 0.75) fitted beside the point models, for the per-corner rank and the closing
# 3F. The rank quantiles use FEATURES like the rank models; the closing quantiles use LAST3_FEATURES and get the
# formation's traffic and energy at prediction time, like the closing (LightGBM's quantile objective does not
# accept monotone constraints, so these trees are unconstrained).
# What the app shows is checked, not the raw quantiles: ranks become whole positions (1 + q * (n - 1), rounded),
# the closing is rounded to 0.1 s, and each range is widened to contain the published point.
# Calibration: raw quantile trees are overconfident out of time (closing 3F: 30-33% of actual values inside the
# 25-75% band). Each side is moved out (or in) by a margin chosen on Jun-Aug 2025 (models fitted before June) so the
# SHOWN range covers 50% there (split conformal on the time split); Sep-Dec 2025 then reports the coverage.
# A target's range is published only if its validation coverage is in COVERAGE_BAND.
INTERVAL_Q = (0.25, 0.75)
RANK_INTERVALS = [f'r{c}' for c in CORNERS]
INTERVAL_TARGETS = RANK_INTERVALS + ['last3']
COVERAGE_BAND = (0.40, 0.60)
MARGIN_GRID = {'r': np.round(np.arange(-.08, .0801, .005), 3), 'last3': np.round(np.arange(-.2, 1.2001, .02), 2)}
Q_LO, Q_HI = (f'q{int(a * 100)}' for a in INTERVAL_Q)


def fit_intervals(d, end):
    fit, models = (d.date < end) & d.complete, {}
    for a, q in zip(INTERVAL_Q, (Q_LO, Q_HI)):
        for t in RANK_INTERVALS:
            rows = fit & d[t].notna()
            models[f'{q}_{t}'] = lgbm('quantile').set_params(alpha=a).fit(d.loc[rows, FEATURES], d.loc[rows, t])
        rows = fit & d.last3.notna()
        models[f'{q}_last3'] = lgbm('quantile').set_params(alpha=a).fit(d.loc[rows, LAST3_FEATURES], d.loc[rows, 'last3'])
    return models


def raw_quantiles(models, x, form, corners, pace):
    """{target: (lo, hi)} raw quantile predictions (normalised rank / seconds), crossing removed."""
    out = {}
    for c in corners:
        t = f'r{c}'
        if f'{Q_LO}_{t}' in models:
            out[t] = models[f'{Q_LO}_{t}'].predict(x[FEATURES]), models[f'{Q_HI}_{t}'].predict(x[FEATURES])
    if f'{Q_LO}_last3' in models:
        xt = closing_inputs(x, form, corners, pace)
        out['last3'] = models[f'{Q_LO}_last3'].predict(xt[LAST3_FEATURES]), models[f'{Q_HI}_last3'].predict(xt[LAST3_FEATURES])
    return {t: (np.minimum(lo, hi), np.maximum(lo, hi)) for t, (lo, hi) in out.items()}


def shown_range(t, lo, hi, point, n, margin):
    """The range the app shows: margin applied to each side, whole positions / 0.1 s, widened to the point."""
    lo, hi = np.asarray(lo, float) - margin, np.asarray(hi, float) + margin
    mid = (lo + hi) / 2
    lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)  # a negative margin can shrink a side only to the middle
    point = np.asarray(point, float)
    if t == 'last3':
        return np.minimum(np.round(lo, 1), point), np.maximum(np.round(hi, 1), point)
    to_pos = lambda v: np.clip(np.rint(1 + v * (n - 1)), 1, n)
    return np.minimum(to_pos(lo), point), np.maximum(to_pos(hi), point)


def race_intervals(models, x, form, corners, pace, last3, margins):
    """Shown ranges per horse: {'r<c>': [[lo, hi], ...] whole positions, 'last3': [[lo, hi], ...] seconds}."""
    n, out = len(x), {}
    for t, (lo, hi) in raw_quantiles(models, x, form, corners, pace).items():
        point = np.round(last3, 1) if t == 'last3' else np.array(form[int(t[1])]['rank'], float)
        a, b = shown_range(t, lo, hi, point, n, margins.get(t, 0.0))
        out[t] = [[round(float(u), 1), round(float(v), 1)] if t == 'last3' else [int(u), int(v)] for u, v in zip(a, b)]
    return out


def interval_rows(models, races, P):
    """Per target: raw quantiles, published point, actual value and field size of every evaluation row."""
    rows = {t: {'lo': [], 'hi': [], 'point': [], 'actual': [], 'n': []} for t in INTERVAL_TARGETS}
    for x, corners, numbers, pred, spreads, early in races:
        n = len(numbers)
        form = simulate(numbers, early, pred, spreads, corners, P)
        last3 = closing(models, x, form, corners, pred['pace'])[0]
        for t, (lo, hi) in raw_quantiles(models, x, form, corners, pred['pace']).items():
            if t == 'last3':
                ok = x.last3.notna().to_numpy()
                point, actual = np.round(last3, 1), np.round(x.last3.to_numpy(), 1)
            else:
                if not labelled(x, int(t[1])):
                    continue  # formed (course corner) but not scorable
                ok = np.ones(n, bool)
                point, actual = np.array(form[int(t[1])]['rank'], float), np.rint(x[t].to_numpy() * (n - 1)) + 1
            r = rows[t]
            for k, v in (('lo', lo), ('hi', hi), ('point', point), ('actual', actual), ('n', np.full(n, n))):
                r[k].extend(np.asarray(v, float)[ok])
    return {t: {k: np.array(v) for k, v in r.items()} for t, r in rows.items() if r['lo']}


def coverage(t, r, margin):
    """Coverage and mean width of the shown range for one margin (rows of mixed field sizes)."""
    lo, hi = np.empty(len(r['lo'])), np.empty(len(r['lo']))
    for n in np.unique(r['n']):
        m = r['n'] == n
        lo[m], hi[m] = shown_range(t, r['lo'][m], r['hi'][m], r['point'][m], int(n), margin)
    eps = 1e-9
    return float(np.mean((lo - eps <= r['actual']) & (r['actual'] <= hi + eps))), float(np.mean(hi - lo))


def raw_coverage(t, r):
    actual = r['actual'] if t == 'last3' else (r['actual'] - 1) / (r['n'] - 1)
    return float(np.mean((r['lo'] <= actual) & (actual <= r['hi'])))


def calibrate_intervals(d, P):
    """Margins chosen on Jun-Aug 2025 with models fitted before June: the shown coverage closest to 50%."""
    models = fit_models(d, '20250601')
    models.update(fit_intervals(d, '20250601'))
    rows = interval_rows(models, evaluation_races(models, d, '20250601', '20250831'), P)
    margins, tuning = {}, {}
    for t, r in rows.items():
        grid = MARGIN_GRID['last3' if t == 'last3' else 'r']
        cov = [coverage(t, r, float(m))[0] for m in grid]
        best = int(np.argmin([abs(c - .5) for c in cov]))
        margins[t] = float(grid[best])
        tuning[t] = {'margin': margins[t], 'coverage': cov[best], 'coverageWithoutMargin': coverage(t, r, 0.0)[0],
                     'rawQuantileCoverage': raw_coverage(t, r), 'rows': len(r['lo'])}
    print('interval margins (Jun-Aug):', tuning, flush=True)
    return margins, tuning


def interval_report(models, races, P, margins, tuning):
    """Validation coverage of the shown ranges per target; 'shown' decides what is published."""
    out = {}
    for t, r in interval_rows(models, races, P).items():
        cov, width = coverage(t, r, margins[t])
        out[t] = {'coverage': cov, 'coverageWithoutMargin': coverage(t, r, 0.0)[0], 'rawQuantileCoverage': raw_coverage(t, r),
                  'meanWidth': width, 'margin': margins[t], 'tuning': tuning[t], 'rows': len(r['lo']),
                  'shown': bool(COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1])}
    return {'alpha': list(INTERVAL_Q), 'band': list(COVERAGE_BAND), 'margins': margins,
            'unit': {'r': '番手（整数、公開値を含むよう拡張）', 'last3': '秒（0.1秒、公開値を含むよう拡張）', 'margin': 'r=正規化順位 / last3=秒'},
            'targets': out}


def labelled(x, c):
    """Every horse of the race has the rank, gap and lane label at corner c (needed to score that corner)."""
    return bool(x[f'r{c}'].notna().all() and x[f'g{c}'].notna().all() and x[f'l{c}'].notna().all())


def evaluation_races(models, d, start, end, codes=None):
    """Cached per-race predictions for a period (only the race codes in `codes` when given).
    EVAL_CORNERS 'course': every full-field race is formed on its course's usual corner set, taken from the races before
    `start` like race_output does before a race; corners without complete labels are formed but not scored.
    'labels' (legacy): only the corners where every horse has all labels, and races with two or more of them."""
    window = d[d.date.between(start, end)]
    if codes is not None:
        window = window[window.RACE_CODE.isin(codes)]
    sets = corner_sets(d[d.date < start]) if EVAL_CORNERS == 'course' else None
    out = []
    for _, x in window.groupby('RACE_CODE'):
        if sets is None:
            corners = [c for c in CORNERS if labelled(x, c)]
        elif not x.complete.all():
            continue
        else:
            r0 = x.iloc[0]
            corners = sets(int(r0.venue), int(r0.track), int(r0.distance))
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
        seen = [c for c in corners if labelled(x, c)]
        if len(seen) < 2:
            continue
        form, n = simulate(numbers, early, pred, spreads, corners, P), len(numbers)
        a, b = seen[0], seen[-1]
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
            if not labelled(x, c):
                continue
            err.setdefault(f'r{c}', []).extend(np.abs((np.array(form[c]['rank']) - 1) / (n - 1) - x[f'r{c}'].to_numpy()))
            err.setdefault(f'g{c}', []).extend(np.abs(np.array(form[c]['gap']) - x[f'g{c}'].to_numpy()))
            local_lane = form[c].get('localLane', form[c]['lane'])
            err.setdefault(f'l{c}', []).extend(np.abs(np.array(local_lane, float) - x[f'l{c}'].to_numpy()))
        ok = x.last3.notna().to_numpy()
        last3 = pred['last3'] if P is None else closing(models, x, form, corners, pred['pace'])[0]
        err.setdefault('last3', []).extend(np.abs(last3 - x.last3.to_numpy())[ok])
    return {k: float(np.mean(v)) for k, v in err.items()}


def tune_contention(d):
    """Pick contention parameters on Jun-Aug 2025 with models fitted on data before June (complete history
    starts in 2023 since the 2026-11-18 re-import; before it, in 2025); the Sep-Dec validation period is never seen while tuning."""
    models = fit_models(d, '20250601')
    races = evaluation_races(models, d, '20250601', '20250831')
    base = score(models, races, None)
    best, best_obj, best_ratios = DEFAULT_PARAMS, None, None
    for P in PARAM_GRID:
        s = score(models, races, P)
        ratios = {k: s[k] / base[k] for k in s}
        obj = float(np.mean([v for k, v in ratios.items() if k not in OBJECTIVE_EXCLUDES]))
        if best_obj is None or obj < best_obj:
            best, best_obj, best_ratios = P, obj, ratios
    print(f'contention tuned on {len(races)} races: {best} (error ratio vs independent {best_obj:.3f})', flush=True)
    print('  per-metric error ratio: ' + ' '.join(f'{k}={v:.3f}' for k, v in best_ratios.items()), flush=True)
    # Kept in report.json: which metrics the chosen setting trades against each other.
    tuning = {'races': len(races), 'objective': best_obj, 'ratios': best_ratios, 'objectiveExcludes': list(OBJECTIVE_EXCLUDES)}
    return best, tuning


# l1 is left out of the contention objective and only listed (E0): at the first corner the scored lane is the model's
# own lane with and without the layer, so its ratio is 1.0 for every setting and adds nothing to the choice.
OBJECTIVE_EXCLUDES = ('l1',)


def validate(d, P, margins=None, tuning=None):
    models = fit_models(d, '20250901')
    models.update(fit_intervals(d, '20250901'))
    races = evaluation_races(models, d, '20250901', '20251231')
    with_layer, without = score(models, races, P), score(models, races, None)
    tr = d.date < '20250901'
    metrics = {}
    for t in with_layer:
        base = []
        for x, corners, *_ in races:
            col = past5(x, t)  # the past-5-run mean, whatever summary the model uses (HISTORY_ALPHA)
            if t == 'last3':
                ok = x.last3.notna()
                base.extend(np.abs(x.loc[ok, col].fillna(d.loc[tr, 'last3'].median()) - x.loc[ok, 'last3']))
            elif int(t[1:]) in corners and labelled(x, int(t[1:])):
                base.extend(np.abs(x[col].fillna(d.loc[tr, t].median()) - x[t]))
        metrics[t] = {'mae': with_layer[t], 'independentMae': without[t], 'baselineMae': float(np.mean(base)),
                      'validationRows': len(base)}
        print(f"{t}: contention={with_layer[t]:.3f} independent={without[t]:.3f} past-mean={metrics[t]['baselineMae']:.3f}", flush=True)
    # How much traffic costs, as the closing model learned it (validation races, simulated formations).
    delays, held = [], []
    for x, corners, numbers, pred, spreads, early in races:
        form = simulate(numbers, early, pred, spreads, corners, P)
        t3, free = closing(models, x, form, corners, pred['pace'])
        delays.extend(t3 - free)
        held.extend(form[corners[0]]['held'])
    metrics['_summary'] = {'races': len(races), 'meanKickDelay': float(np.mean(np.maximum(delays, 0))),
                           'delayedShare': float(np.mean(np.array(delays) > .05)), 'heldShareFirstCorner': float(np.mean(held))}
    print(metrics['_summary'], flush=True)
    # Mid-race moves with and without the move layer, and the past-5-run mean move as a reference.
    still = {**P, 'k': 0.0}
    past, actual = [], []
    for x, corners, *_ in races:
        seen = [c for c in corners if labelled(x, c)]
        if len(seen) < 2:
            continue
        a, b = seen[0], seen[-1]
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
    if margins is not None:
        metrics['_intervals'] = interval_report(models, races, P, margins, tuning)
    print(metrics['_intervals'], flush=True)
    metrics['_standard'] = standard_metrics(models, races, P, d, '20250901')
    print(json.dumps(metrics['_standard'], ensure_ascii=False), flush=True)
    return metrics


def past5(x, t):
    """Column of the past-5-run mean of target t (past5_<t>; frames built before E1 only have past_<t>)."""
    return 'past5_' + t if 'past5_' + t in x else 'past_' + t


def standard_metrics(models, races, P, d, end):
    """E0 standard measures (scripts/eval_standard.py) for one evaluation window; models fitted before `end`.
    ranks : tau and integer MAE of the published (contention) formation, of the independent formation and of the
            past mean turned into integer ranks; continuous MAE of the model BEFORE the blend, of the blend and of the
            past mean (secondary).
    lanes : the scored lane (localLane) and the independent lane against the constant-0 forecast, class accuracy over
            0 / 1 / 2+, and the displayed lane (track-relative path slot, `lane` in the output) on the same label: the
            app shows `lane`, the scores use `localLane`.
    gaps  : the per-horse gap model on its own (a reference only: not a formation, so not drawable) next to the
            published and the independent formation (independentMae is the RECONSTRUCTED independent formation).
    last3 : the same closing model on the independent and the contention formation, with and without the traffic
            input, and the FEATURES-only closing model (independentMae in metrics).
    pace  : seconds MAE / correlation, H/M/S hit and ranked probability score against the climatology and the majority
            class (bands = PACE_BANDS, the training terciles of the actual pace; the screen's GI label uses other bands)."""
    import eval_standard as es
    tr = (d.date < end) & d.complete
    tab, lanes = es.RaceTable(), {}
    for x, corners, numbers, pred, spreads, early in races:
        race, n = x.RACE_CODE.iloc[0], len(numbers)
        nums = np.array(numbers, float)
        con = simulate(numbers, early, pred, spreads, corners, P)
        ind = independent(numbers, pred, spreads, corners)
        for c in corners:
            if not labelled(x, c):
                continue
            y = x[f'r{c}'].to_numpy()
            past = x[past5(x, f'r{c}')].fillna(d.loc[tr, f'r{c}'].median()).to_numpy()
            past_rank = np.argsort(np.argsort(past + 1e-9 * nums, kind='stable'), kind='stable') / (n - 1)
            forms = {'contention': (np.array(con[c]['rank']) - 1) / (n - 1), 'independent': (np.array(ind[c]['rank']) - 1) / (n - 1),
                     'pastMean': past_rank}
            for k, v in forms.items():
                tab.add(f'r{c}.integer', k, race, np.abs(v - y))
                tab.add_value(f'r{c}.tau', k, race, es.kendall(v, y))
            for k, v in (('model', pred[f'model_r{c}']), ('blend', pred[f'r{c}']), ('pastMean', past)):
                tab.add(f'r{c}.continuous', k, race, np.abs(np.asarray(v, float) - y))
            g = x[f'g{c}'].to_numpy()
            for k, v in (('contention', con[c]['gap']), ('independent', ind[c]['gap']), ('gapModel', pred[f'g{c}'])):
                tab.add(f'g{c}', k, race, np.abs(np.asarray(v, float) - g))
            lab = x[f'l{c}'].to_numpy()
            lane = lanes.setdefault(f'l{c}', {'label': [], 'contention': [], 'independent': [], 'displayed': []})
            lane['label'].extend(lab)
            lane['contention'].extend(np.asarray(con[c].get('localLane', con[c]['lane']), float))
            lane['independent'].extend(np.clip(pred[f'l{c}'], 0, 4))
            lane['displayed'].extend(np.asarray(con[c]['lane'], float))
        ok = x.last3.notna().to_numpy()
        if ok.any():
            l3 = x.last3.to_numpy()[ok]
            t3_con, free_con = closing(models, x, con, corners, pred['pace'])
            t3_ind = closing(models, x, ind, corners, pred['pace'])[0]  # the independent formation carries no traffic
            for k, v in (('contention', t3_con), ('contentionNoTraffic', free_con), ('independentFormation', t3_ind),
                         ('featuresOnlyModel', pred['last3'])):
                tab.add('last3', k, race, np.abs(np.asarray(v)[ok] - l3))
    out = {'ranks': {}, 'gaps': {}, 'lanes': {}, 'last3': {}}
    for c in CORNERS:
        if (f'r{c}.tau', 'contention') not in tab.rows:
            continue
        out['ranks'][f'r{c}'] = {
            'races': tab.races(f'r{c}.tau', 'contention'),
            'tau': {k: tab.mean(f'r{c}.tau', k) for k in ('contention', 'independent', 'pastMean')},
            'integerMae': {k: tab.mean(f'r{c}.integer', k) for k in ('contention', 'independent', 'pastMean')},
            'continuousMae': {k: tab.mean(f'r{c}.continuous', k) for k in ('model', 'blend', 'pastMean')},
            'tauContentionMinusIndependent': tab.diff(f'r{c}.tau', 'contention', 'independent'),
            'integerContentionMinusIndependent': tab.diff(f'r{c}.integer', 'contention', 'independent')}
        out['gaps'][f'g{c}'] = {k: tab.mean(f'g{c}', k) for k in ('contention', 'independent', 'gapModel')}
        lane = lanes[f'l{c}']
        out['lanes'][f'l{c}'] = {k: es.lane_summary(lane[k], lane['label']) for k in ('contention', 'independent', 'displayed')}
        out['lanes'][f'l{c}']['displayedDiffersFromScored'] = float(np.mean(np.array(lane['displayed']) != np.array(lane['contention'])))
    out['last3'] = {k: tab.mean('last3', k) for k in ('contention', 'contentionNoTraffic', 'independentFormation', 'featuresOnlyModel')}
    out['last3']['trafficInput'] = tab.diff('last3', 'contention', 'contentionNoTraffic')
    out['last3']['formation'] = tab.diff('last3', 'contentionNoTraffic', 'independentFormation')
    out['last3']['closingModelVsFeaturesOnly'] = tab.diff('last3', 'independentFormation', 'featuresOnlyModel')
    sigma, train_pace = pace_sigma(d, end)
    pp = np.array([pred['pace'] for x, corners, numbers, pred, *_ in races])
    pa = np.array([x.pace.iloc[0] for x, *_ in races])
    out['pace'] = es.pace_summary(pp, pa, train_pace, sigma, PACE_BANDS)
    return out


def corner_sets(d):
    """Corners a course actually passes, from history of the same venue/track/distance."""
    seen = d.groupby('RACE_CODE').agg(venue=('venue', 'first'), track=('track', 'first'), distance=('distance', 'first'),
                                      **{f'has{c}': (f'r{c}', lambda s: bool(s.notna().all())) for c in CORNERS})
    seen['set'] = seen[[f'has{c}' for c in CORNERS]].apply(lambda r: tuple(c for c in CORNERS if r[f'has{c}']), axis=1)
    seen = seen[seen.set.map(len) >= 2]
    exact = seen.groupby(['venue', 'track', 'distance']).set.agg(lambda s: s.mode().iloc[0]).to_dict()
    loose = seen.groupby(['venue', 'distance']).set.agg(lambda s: s.mode().iloc[0]).to_dict()
    return lambda v, t, dist: list(exact.get((v, t, dist)) or loose.get((v, dist)) or (3, 4))


def race_output(models, x, key, corners, P, gi_bands=PACE_BANDS, shown=(), margins=None):
    """One race: contention formation per corner, traffic-aware closing, finish estimate.
    shown: interval targets ('r1'..'r4', 'last3') whose validation coverage passed; only those get ranges."""
    numbers, pred, spreads, early = prepare_race(models, x, key, corners)
    form = simulate(numbers, early, pred, spreads, corners, P)
    last3, free, fresh = closing(models, x, form, corners, pred['pace'], parts=True)
    iv = race_intervals(models, x, form, corners, pred['pace'], last3, margins or {}) if shown else {}
    n = len(numbers)
    horses = {}
    for i, num in enumerate(numbers):
        positions = []
        for c in corners:
            f = form[c]
            positions.append({'corner': c, 'rank': f['rank'][i], 'gap': round(f['gap'][i], 2), 'lane': float(f['lane'][i]),
                              'wantRank': f['want'][i], 'held': bool(f['held'][i]),
                              'boxed': bool(f['out'][i] > 0 and f['close'][i] > 0)})
            if f'r{c}' in shown and f'r{c}' in iv:
                positions[-1]['rankRange'] = iv[f'r{c}'][i]
        horses[str(num)] = {'positions': positions, 'last3': round(float(last3[i]), 1), 'last3Free': round(float(free[i]), 1),
                            'kickDelay': round(max(0.0, float(last3[i] - free[i])), 2),
                            'paceCost': round(max(0.0, float(free[i] - fresh[i])), 2)}
        if 'last3' in shown and 'last3' in iv:
            horses[str(num)]['last3Range'] = iv['last3'][i]
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
        runs = hist[hist.KETTO_TOROKU_BANGO == h['horseId']]  # d is sorted by date (add_history_features)
        past = runs.tail(5)
        row = {'distance': race['distance'], 'surface': int(race['surface'] == 'ダート'), 'venue': int(race['venueCode']),
               'track': int(race['trackCode']), 'gate': h['number'] / len(active), 'weight': h['weight'], 'age': h['age'],
               'field': len(active), 'blinker': np.nan if h['blinker'] is None else int(h['blinker']),
               'starts': int((hist.KETTO_TOROKU_BANGO == h['horseId']).sum()), 'race': snap['raceId'], 'UMABAN': h['number']}
        for t in TARGETS:
            row['past_' + t] = runs_summary(runs, t)  # same summary as training (HISTORY_ALPHA)
        for t in MOVE_LABELS:
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
    contention, tuning = tune_contention(d)
    params = tune_moves(d, contention)
    margins, margin_tuning = calibrate_intervals(d, params)
    metrics = validate(d, params, margins, margin_tuning)
    models = fit_models(d, '20260101')
    models.update(fit_intervals(d, '20260101'))
    intervals = metrics.pop('_intervals')
    shown = [t for t, v in intervals['targets'].items() if v['shown']]
    print('interval ranges published for:', shown or 'none', flush=True)
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
        out_races[race['id']] = race_output(models, x, key, corners, params, gi_bands, shown, margins)
        print(race['name'], 'corners', corners)
    (ROOT / 'models' / 'pace').mkdir(parents=True, exist_ok=True)
    for t, m in models.items():
        m.booster_.save_model(str(ROOT / 'models' / 'pace' / f'{t}.txt'))
    model = {'name': 'Paddock展開予想モデル', 'version': VERSION, 'trainingEnd': '2025-12-31', 'validation': '2025-09〜12',
             'rows': int(d.complete.sum()), 'historyRows': int(len(d)), 'excludedRows': int(raw - len(d)), 'features': FEATURES, 'closingFeatures': LAST3_FEATURES,
             'moveFeatures': MOVE_FEATURES, 'contention': params, 'contentionTuning': tuning, 'knownIssues': KNOWN_ISSUES,
             'summary': metrics.pop('_summary'), 'standard': metrics.pop('_standard'),
             'moves': metrics.pop('_moves'), 'pace': {**metrics.pop('_pace'), 'giPace': gi_pace, 'giBands': list(gi_bands)}, 'paceFeatures': PACE_FEATURES,
             'intervals': {**intervals, 'published': shown}, 'metrics': metrics,
             'notes': ['各角の順位・先頭からの馬身差・併走群内の内外位置と上がり3Fを個別に学習（LightGBM）。',
                       '馬身差は「順位モデルの並び × 馬ごとの差モデルの形 × レース単位の隊列の長さモデル」で再構成。',
                       'ポジション争い：各馬の希望位置（順位モデル）と前の角の位置から押し出しの強さを決め、通常は4頭、先行型は5頭まで。前の角より位置を上げる馬は相手の外へ進み、頭数制限を受けない。4角以降も頭数制限なし。控える馬は後方へ回り、各列の左右位置は前の角から引き継ぐ。',
                       '前半のペース：出走馬の先行型の頭数・過去に逃げた割合・その馬たちが走ったレースの前半の速さ・馬場状態から、前半3Fがコースの標準より何秒速いかを予想（H/M/S）。隊列の長さモデルに使う（予想値で学習）。上がりモデルには馬場状態を入れる。',
                       '道中の位置変化：最初の角からの累積移動量を角ごとに学習（過去走の移動・最大の上げ幅・出走馬中の動く馬の割合・角の数）。2角以降は「最初の角の位置＋予想移動量」と希望位置を混ぜて並べ直す。強さは2025年6〜8月で、順位誤差の悪化を0.2%以内に抑える範囲で移動の相関が最大になる値を選ぶ。',
                       '脚の消耗：前半のペースが速いほど・前にいるほど、また普段より前で運ぶ（二の足・かかり）ほど上がりが遅くなる量を上がりモデルで学習（遅くする方向のみ）。予想時は予想ペースと1角の予想位置から入れ、paceCost（秒）として出す。',
                       '仕掛け遅れ：上がり3Fモデルは最終角の渋滞（外に並ぶ頭数・1馬身未満で前が壁）を特徴量に学習し、予想時はポジション争いで決まった隊列の渋滞を入れる。前が開いた場合との差を仕掛け遅れ（秒）として出す。',
                       '馬身差と内外はJRAの通過順表記から復元。記号の幅（,=1〜2馬身, -=2〜5, ==5以上）は中央値で近似。',
                       '学習用の内外は併走群内の位置。出力の内外は前の角から引き継ぐ進路スロット（0=内側）で、実測のラチ距離ではない。',
                       'ゴールは最終角の馬身差に上がり3Fの差（1秒=5馬身）を加えた推定。着順モデルではない。',
                       '予想レンジ：各角の番手と上がり3Fに、分位点回帰（LightGBM quantile, α=0.25/0.75）による50%区間を付ける。幅は2025年6〜8月で表示上の被覆率が50%になるよう補正し、9〜12月の被覆率が40〜60%の項目だけ出す。',
                       '過去走は対象日より前だけ。各角の順位・馬身差・内外と上がりは全過去走の指数加重平均（α=0.2、新しい走ほど重い）、道中の移動・ペース・逃げ率は直近5走の平均。',
                       'オッズ・人気・払戻・対象レースの結果は使わない。']}
    out = {'schema': SCHEMA, 'revision': SCHEMA_REVISION, 'generatedAt': datetime.now().isoformat(timespec='seconds'), 'model': model,
           'units': {'gap': '馬身（先頭から）', 'lane': '内側からの進路スロット（0=内側、列ごとのリセットなし）', 'last3': '秒'}, 'races': out_races}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (ROOT / 'models' / 'pace' / 'report.json').write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {len(out_races)} races -> {args.out}')


if __name__ == '__main__':
    main()
