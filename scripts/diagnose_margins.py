"""Diagnosis (追加指示4 フェーズ2, read-only): how noisy are the gap labels (g1-g4), and is the model's loss to the
past mean on g1/g2 a label problem or a model problem?

The gap labels are rebuilt from the JRA passing notation by giving each separator the middle of its band
(no mark 0.6 / , 1.5 / - 3.5 / = 6.5 lengths; parentheses = abreast, 0). There is no time at the corners,
so the error of that rebuild is measured where the truth is known, at the finish:

A. Label noise.
   A1  For every complete race in the period, the finish order is written in the same notation: the official
       margin to the horse ahead (CHAKUSA_CODE1: ハナ/アタマ/クビ/同着 -> abreast, under 1 length -> no mark,
       1-2 -> ',', 2-5 -> '-', 5+ -> '=') and the lengths behind the winner are rebuilt with the band
       middles, exactly like add_labels(). The truth is TIME_SA (0.1 s, signed to the winner; winner -> 0)
       at 1 s = 5 lengths. Also reported: the official margins summed (finer than 0.1 s) against TIME_SA, so
       the time-to-length rule itself is checked.
   A2  Per separator: the distribution of the true step inside each band (bias and spread vs the middle).
   A3  Per corner: separator composition of the actual passing strings in the period, and the expected label
       error per horse from that composition (bias and variance of each separator ahead of the horse, from A2,
       accumulated; assumes the within-band distribution at a corner is like the finish, see caveat below).
B. Ceiling. Models fitted before 2025-09 (same discipline as validate()), Sep-Dec 2025 races. Gap MAE per corner
   for: contention (v1.5 parameters), independent reconstruct, the per-horse gap model alone, the past-5 mean,
   and oracles (true order / true field length / both) to see which part of the reconstruct loses. The
   independent-minus-past-mean difference is then split by the expected label noise of the race (terciles)
   and of the horse: if the loss disappears where labels are clean, it is the labels; if it stays, the model.

Caveat: corner gaps are smaller than finish gaps (the field is bunched), so the within-band spread at a corner
is probably smaller than at the finish for '-' and '='; A3 is an upper-side estimate of the corner noise.

Result 2026-11-18 (current DB; finish: 269 validation / 614 races of 2025; ceiling: 273 validation races):
  A1  Official margins summed vs TIME_SA x 5: MAE 1.25 lengths, bias +1.16. The least-squares rate is 5.7 lengths/s,
      so "1 s = 5 lengths" itself understates finish gaps by ~12%. Notation rebuild vs official margins: MAE 1.03
      lengths, bias +0.23, p90 2.35 (vs TIME_SA: 1.69 / +1.39, mostly the conversion). The error grows down the order
      (2nd 0.44 ... 10th+ 2.63 lengths vs TIME_SA) and with the share of '-'/'=' in the string (race terciles
      0.89 / 1.49 / 2.65, corr 0.57).
  A2  True step inside each band (official, 2025): () 0.23 (middle 0), none 0.61 (0.6), ',' 1.31 (1.5), '-' 2.73 (3.5),
      '=' 7.88 (6.5, sd 2.7). '-' is over- and '=' under-stated by its middle; both carry most of the noise.
  A3  Corner strings are much tighter than finishes: '-' 13.5/12.4/7.0/5.1% and '=' 0.5/0.7/0.6/0.9% of the steps at
      corners 1-4. Expected label error per horse 0.77/0.82/0.75/0.76 lengths.
  B   Validation gap MAE g1/g2/g3/g4 - contention 3.83/3.82/2.68/2.39, independent 3.74/3.75/2.61/2.37, per-horse gap
      model alone 3.33/3.44/2.38/2.16, past mean 3.40/3.66/2.69/2.52; oracles with the true order 2.54/2.75/1.98/1.76,
      true field length 2.91/2.80/2.06/1.80, both 1.46/1.31/1.04/0.84. Spread model vs constant: 2.94/2.97, 3.44/3.25,
      2.40/2.32, 2.08/2.00 (corr 0.24/-0.01/0.06/0.13): the field length is not predicted at all.
      Independent minus past mean by race noise tercile (low/mid/high): g1 +0.54/+0.43/+0.01, g2 -0.16/+0.07/+0.34.
  Verdict: the label noise (~0.8 lengths) is of the same size as the model-vs-past-mean gap (g1 0.34-0.43, g2
  0.09-0.16), but it does not explain g1: the loss is largest in the cleanest races. g2's loss is in the noisy races
  (label side). The model side that is left is the field length (order is closed after phase 1) -> experiment (b),
  scripts/experiment_spread_features.py (not adopted: the spread models have 43-616 training races).

Run:  python scripts/diagnose_margins.py [--cache frame.pkl] [--out result.json] [--start 20250901 --end 20251231]
"""
import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm  # noqa: E402
from build_forecasts import settings, number  # noqa: E402

P15 = json.loads((pm.ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
# JV-Data 着差コード -> lengths. 'T' 大差 (10+), 'Z' is used for 10 lengths in this DB (mean TIME_SA 4.0 s).
CHAKUSA = {'D': 0.0, 'H': 0.1, 'A': 0.2, 'K': 0.3, '12': 0.5, '34': 0.75, '1': 1.0, '114': 1.25, '112': 1.5, '134': 1.75,
           '2': 2.0, '214': 2.25, '212': 2.5, '234': 2.75, '3': 3.0, '312': 3.5, '4': 4.0, '5': 5.0, '6': 6.0, '612': 6.5,
           '7': 7.0, '8': 8.0, '9': 9.0, 'Z': 10.0, 'T': 12.0}
SEPS = ['()', '', ',', '-', '=']  # '()' = abreast inside a group (gap 0)
MIDDLE = {'()': 0.0, **pm.GAP_LENGTHS}


def band(step):
    """Separator JRA would write for a step (lengths) to the unit ahead."""
    return '()' if step <= .3 else '' if step < 1 else ',' if step < 2 else '-' if step < 5 else '='


def separators(text):
    """Separator before every horse of a passing string, in running order (first horse: None)."""
    s, out, sep, first, i = (text or '').strip(), [], '', True, 0
    while i < len(s):
        ch = s[i]
        if ch in ',-=':
            sep = max(sep, ch, key=pm.SEP_ORDER.index)
            i += 1
            continue
        if ch in '* ':
            i += 1
            continue
        if ch == '(':
            j = s.index(')', i)
            members = [m.strip(' *') for m in s[i + 1:j].split(',') if m.strip(' *').isdigit()]
            i = j + 1
        elif ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            members, i = [s[i:j]], j
        else:
            raise ValueError(text)
        for k, m in enumerate(members):
            out.append((int(m), None if first and k == 0 else sep if k == 0 else '()'))
        first, sep = False, ''
    return out


def load_finish(codes):
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    rows = []
    with c.cursor() as q:
        codes = list(codes)
        for k in range(0, len(codes), 500):
            part = codes[k:k + 500]
            q.execute('SELECT RACE_CODE,UMABAN,KAKUTEI_CHAKUJUN,TIME_SA,CHAKUSA_CODE1 FROM umagoto_race_joho WHERE RACE_CODE IN ('
                      + ','.join(['%s'] * len(part)) + ')', part)
            rows += q.fetchall()
    c.close()
    f = pd.DataFrame(rows)
    f['UMABAN'] = f.UMABAN.map(number)
    f['fin'] = f.KAKUTEI_CHAKUJUN.map(number)
    sa = f.TIME_SA.map(lambda v: number(str(v).strip()) if str(v).strip() not in ('', '9999') else np.nan)
    f['time_len'] = (sa.clip(lower=0) / 10 * pm.LENGTHS_PER_SECOND).where(sa.abs() < 9000)
    f['step_off'] = f.CHAKUSA_CODE1.fillna('').str.strip().map(CHAKUSA)
    return f


def stats(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'n': int(len(v)), 'mae': float(np.mean(np.abs(v))), 'bias': float(np.mean(v)), 'median': float(np.median(v)),
            'p90abs': float(np.quantile(np.abs(v), .9)), 'sd': float(np.std(v))}


def finish_noise(d, f, start, end):
    """A1/A2 on complete races of the period."""
    races = d[d.date.between(start, end) & d.complete].groupby('RACE_CODE').size()
    f = f[f.RACE_CODE.isin(races.index)]
    rec_err, off_err, rec_off, by_pos, steps, per_race = [], [], [], {}, {s: [] for s in SEPS}, []
    sec, offl = [], []
    used = 0
    for code, x in f.groupby('RACE_CODE'):
        x = x[x.fin.between(1, 18)].sort_values(['fin', 'UMABAN'])
        if len(x) != races[code] or x.time_len.isna().any() or x.step_off.iloc[1:].isna().any():
            continue
        used += 1
        step = x.step_off.to_numpy().copy()
        step[0] = 0.0
        official = np.cumsum(step)
        sep = [band(s) for s in step[1:]]
        recon = np.concatenate([[0.0], np.cumsum([MIDDLE[s] for s in sep])])
        truth = x.time_len.to_numpy()
        e = recon[1:] - truth[1:]
        rec_err.extend(e)
        off_err.extend(official[1:] - truth[1:])
        rec_off.extend(recon[1:] - official[1:])
        sec.extend(truth[1:] / pm.LENGTHS_PER_SECOND)
        offl.extend(official[1:])
        for pos, v in enumerate(e, start=2):
            by_pos.setdefault(min(pos, 10), []).append(v)
        for s, v in zip(sep, step[1:]):
            steps[s].append(v)
        share = {s: sep.count(s) / len(sep) for s in SEPS}
        per_race.append({'wide': share['-'] + share['='], 'mae': float(np.mean(np.abs(e)))})
    pr = pd.DataFrame(per_race)
    pr['tercile'] = pd.qcut(pr.wide.rank(method='first'), 3, labels=['low', 'mid', 'high'])
    sep_stats = {s: {**stats(np.array(v) - MIDDLE[s]), 'meanStep': float(np.mean(v)) if v else None, 'middle': MIDDLE[s]}
                 for s, v in steps.items() if v}
    sec, offl = np.array(sec), np.array(offl)
    scale = float(np.sum(sec * offl) / np.sum(sec * sec))  # official lengths per second (least squares through 0)
    return {'races': used, 'rebuiltVsTime': stats(rec_err), 'officialVsTime': stats(off_err), 'rebuiltVsOfficial': stats(rec_off),
            'officialLengthsPerSecond': scale,
            'rebuiltByFinishPos': {str(k) + ('+' if k == 10 else ''): stats(v) for k, v in sorted(by_pos.items())},
            'stepWithinBand': sep_stats,
            'raceErrorByWideShare': {str(k): {'races': int(len(g)), 'wideShare': float(g.wide.mean()), 'mae': float(g.mae.mean())}
                                     for k, g in pr.groupby('tercile', observed=True)},
            'corrWideShareRaceMae': float(np.corrcoef(pr.wide, pr.mae)[0, 1])}


def expected_abs(mu, var):
    if var <= 1e-12:
        return abs(mu)
    s = math.sqrt(var)
    return s * math.sqrt(2 / math.pi) * math.exp(-mu * mu / (2 * var)) + mu * math.erf(mu / (s * math.sqrt(2)))


def corner_noise(x, c, cal):
    """Expected |label error| per horse at corner c of one race: separators ahead accumulate bias and variance."""
    corner_of = {int(str(x.iloc[0][f'CORNER{k}'] or '0').strip() or 0): k for k in pm.CORNERS}
    k = corner_of.get(c)
    if k is None:
        return None, None
    seq = separators(x.iloc[0][f'KAKU_TSUKA_JUNI{k}'])
    mu = var = 0.0
    out, comp = {}, []
    for num, sep in seq:
        if sep is not None:
            mu, var = mu + cal[sep]['bias'], var + cal[sep]['sd'] ** 2
            comp.append(sep)
        out[num] = expected_abs(mu, var)
    return out, comp


def ceiling(d, cal, start, end):
    """B: gap MAE per corner for the model variants, the past mean and oracles, split by expected label noise."""
    models = pm.fit_models(d, start)
    races = pm.evaluation_races(models, d, start, end)
    tr = (d.date < start) & d.complete
    names = ['contention', 'independent', 'gapModel', 'pastMean', 'trueOrder', 'trueSpread', 'trueOrderSpread']
    rows, comp, spread_rows = [], {c: [] for c in pm.CORNERS}, []
    hist = {c: pm.race_level(d[tr], 'RACE_CODE', c).dropna(subset=['spread']) for c in pm.CORNERS}
    course_med = {c: h.groupby(['venue', 'track', 'distance']).spread.median().to_dict() for c, h in hist.items()}
    all_med = {c: float(h.spread.median()) for c, h in hist.items()}
    for x, corners, numbers, pred, spreads, early in races:
        form = pm.simulate(numbers, early, pred, spreads, corners, P15)
        for c in corners:
            noise, seps = corner_noise(x, c, cal)
            if noise is None or set(noise) != set(numbers):
                continue
            comp[c].extend(seps)
            actual = x[f'g{c}'].to_numpy()
            true_spread = float(np.quantile(actual, pm.SPREAD_Q))
            r0 = x.iloc[0]
            spread_rows.append({'corner': c, 'true': true_spread, 'model': spreads[c], 'courseMedian':
                                course_med[c].get((r0.venue, r0.track, r0.distance), all_med[c]), 'constant': all_med[c]})
            truth_rank = x[f'r{c}'].to_numpy()
            v = {'contention': form[c]['gap'],
                 'independent': pm.reconstruct(pred[f'r{c}'], pred[f'g{c}'], spreads[c], numbers)[1],
                 'gapModel': pred[f'g{c}'],
                 'pastMean': x[f'past_g{c}'].fillna(d.loc[tr, f'g{c}'].median()).to_numpy(),
                 'trueOrder': pm.reconstruct(truth_rank, pred[f'g{c}'], spreads[c], numbers)[1],
                 'trueSpread': pm.reconstruct(pred[f'r{c}'], pred[f'g{c}'], true_spread, numbers)[1],
                 'trueOrderSpread': pm.reconstruct(truth_rank, pred[f'g{c}'], true_spread, numbers)[1]}
            race_noise = float(np.mean([noise[n] for n in numbers]))
            for i, num in enumerate(numbers):
                rows.append({'race': x.RACE_CODE.iloc[0], 'corner': c, 'noise': noise[num], 'raceNoise': race_noise, 'actual': actual[i],
                             **{k: abs(float(v[k][i]) - actual[i]) for k in names}})
    r, sp = pd.DataFrame(rows), pd.DataFrame(spread_rows)
    out = {'races': len(races), 'corners': {}, 'spread': {}}
    for c, g in sp.groupby('corner'):
        out['spread'][f'g{c}'] = {'races': int(len(g)), 'meanTrue': float(g.true.mean()), 'corrModel': float(np.corrcoef(g.model, g.true)[0, 1]),
                                  **{f'mae_{k}': float(np.mean(np.abs(g[k] - g.true))) for k in ('model', 'courseMedian', 'constant')}}
    for c, g in r.groupby('corner'):
        rt = g.drop_duplicates('race').raceNoise.rank(method='first')
        terc = pd.qcut(rt, 3, labels=['low', 'mid', 'high'])
        g = g.merge(pd.DataFrame({'race': g.drop_duplicates('race').race, 'tercile': terc.values}), on='race')
        h = pd.qcut(g.noise.rank(method='first'), 3, labels=['low', 'mid', 'high'])
        cnt = pd.Series(comp[c]).value_counts(normalize=True)
        out['corners'][f'g{c}'] = {
            'rows': int(len(g)), 'mae': {k: float(g[k].mean()) for k in names},
            'expectedLabelNoise': float(g.noise.mean()),
            'separatorShare': {s: float(cnt.get(s, 0.0)) for s in SEPS},
            'byRaceNoise': {str(t): {'rows': int(len(q)), 'noise': float(q.noise.mean()),
                                     **{k: float(q[k].mean()) for k in ('independent', 'contention', 'pastMean')},
                                     'indepMinusPast': float(q.independent.mean() - q.pastMean.mean())}
                            for t, q in g.groupby('tercile', observed=True)},
            'byHorseNoise': {str(t): {'rows': int(len(q)), 'noise': float(q.noise.mean()),
                                      'indepMinusPast': float(q.independent.mean() - q.pastMean.mean()),
                                      'contMinusPast': float(q.contention.mean() - q.pastMean.mean())}
                             for t, q in g.groupby(h, observed=True)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache')
    ap.add_argument('--out')
    ap.add_argument('--start', default='20250901')
    ap.add_argument('--end', default='20251231')
    args = ap.parse_args()
    if args.cache and Path(args.cache).exists():
        d = pd.read_pickle(args.cache)
    else:
        d, _ = pm.load_frame(date.today().isoformat().replace('-', ''))
        d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))
        if args.cache:
            d.to_pickle(args.cache)
    d = d.copy()
    f = load_finish(d[d.date >= '20250101'].RACE_CODE.unique())
    res = {'period': [args.start, args.end]}
    res['finishValidation'] = finish_noise(d, f, args.start, args.end)
    res['finish2025'] = finish_noise(d, f, '20250101', '20251231')
    cal = res['finish2025']['stepWithinBand']
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=1), flush=True)
    res['ceiling'] = ceiling(d, cal, args.start, args.end)
    print(json.dumps(res['ceiling'], ensure_ascii=False, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
