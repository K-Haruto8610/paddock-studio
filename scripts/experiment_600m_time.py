"""E2 (2026-11-18): each horse's time at the 600 m-to-go mark, the separator widths of the passing notation calibrated on
it, past-run gap600 as a history feature, and a finish estimate from the identity SOHA = t600 + last 3F. Report only.

Data (pace_model.load_frame / add_time_labels): t600 = SOHA_TIME - the horse's own KOHAN_3F (the per-horse last 3F is
timed from the horse's own pass of the 600 m mark); gap600 = seconds behind the first horse to pass the mark, for
full-field races where every horse has it; finish_gap = TIME_SA (seconds behind the winner, winner 0).
The 600 m mark is NOT a corner (Nakayama: around the 3rd corner, Tokyo: late in the 4th), so gap600 never replaces a
corner label; it is used to calibrate the corner labels and as a second truth near corners 3-4.

  data      coverage of t600 / gap600 and within-race Spearman of the t600 order with the corner ranks (current DB).
  widths    per corner, seconds per separator by least squares of the time steps between consecutive units of the
            passing string (time at the mark relative to the first-listed horse) on the separator counts, fitted ONLY on
            races before each fit end (selection: 2025-06, report: 2025-09). Compared with GAP_LENGTHS (best scale fitted
            on the same races) by the label-vs-time disagreement (MAE, seconds) and correlation on the window.
            Rule fixed in advance: if the calibrated widths cut the disagreement by 30% or more at corners 3 and 4 in
            both windows (the review saw ~40%), the corner-3/4 gap labels are rebuilt with them (lengths = seconds x the
            rate that keeps the current label scale) and the models refitted; gaps are then reported against the old
            labels, the new labels and gap600.
  features  past gap600 (the published history summary) added to FEATURES, or in place of past_g3/past_g4.
  finish    finish time gaps from (current) last-corner gap / 5 + last-3F difference, (identity-a) last-corner gap /
            measured rate + last-3F difference, (identity-b) a per-horse gap600 model + last-3F difference; against
            TIME_SA: MAE in seconds, bias, within-race Spearman of the finish order.
Selection Jun-Aug 2025 (fit < 2025-06), report Sep-Dec 2025 pinned to v1.6's 273 races, seeds 42 and 7.

Run:  python scripts/experiment_600m_time.py frame          (builds outputs/e2-600m/frame.pkl from the DB, read-only)
      python scripts/experiment_600m_time.py data
      python scripts/experiment_600m_time.py widths
      python scripts/experiment_600m_time.py features --seed 42   (and --seed 7)
      python scripts/experiment_600m_time.py finish --seed 42     (and --seed 7)
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_data_expansion as ex  # noqa: E402
import pace_model as pm  # noqa: E402
import diagnose_margins as dm  # noqa: E402
import eval_standard as es  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pm.ROOT
DATA = ROOT / 'outputs' / 'e2-600m'  # frame, data and widths (no model, procedure-free)
FRAME = DATA / 'frame.pkl'
OUT = DATA  # model results (features / finish): e2-600m-reviewed with the reviewed procedure
PROCEDURES = {'reviewed': (('course', 'forward'), 'e2-600m-reviewed'), 'e0': (('course_e0', 'forward_ids'), 'e2-600m')}
V18 = ROOT / 'outputs' / 'data-expansion-v18' / 'frame_2022.pkl'
PIN = ROOT / 'outputs' / 'data-expansion' / 'frame_before_20261118_0126.pkl'
SEL, REP = ('20250601', '20250601', '20250831'), ('20250901', '20250901', '20251231')
SEPS = dm.SEPS  # '()', '', ',', '-', '='
CUT = 0.30


def load():
    d = pd.read_pickle(FRAME)
    return d


def cmd_frame(args):
    today = date.today().isoformat().replace('-', '')
    d, raw = pm.load_frame(today)
    d = pm.add_time_labels(pm.add_traffic(pm.add_history_features(pm.add_labels(d))))
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_pickle(FRAME)
    # Same rows and labels as the v1.8 frame? (the DB could have changed since)
    old = pd.read_pickle(V18)
    keys = ['RACE_CODE', 'UMABAN']
    cols = [f'{k}{c}' for c in pm.CORNERS for k in 'rgl'] + ['last3']
    a = d.set_index(keys).sort_index()
    b = old.set_index(keys).reindex(a.index)
    same = {c: bool(np.allclose(a[c].astype(float).fillna(-9), b[c].astype(float).fillna(-9))) for c in cols}
    print(f'frame rows {len(d)} (v1.8 {len(old)}), complete rows {int(d.complete.sum())} (v1.8 {int(old.complete.sum())}); '
          f'labels identical to v1.8: {all(same.values())}', flush=True)


def time_rows(d, end_before=None, window=None):
    """(race, corner, cumulative separator counts, time gap at the mark relative to the first-listed horse, date) for every
    horse of full-field races where all horses have t600 and the corner string parses with the right horses."""
    q = d[d.complete & d.gap600.notna()]
    if end_before:
        q = q[q.date < end_before]
    if window:
        q = q[q.date.between(*window)]
    rows = []
    for code, x in q.groupby('RACE_CODE'):
        r0 = x.iloc[0]
        t = dict(zip(x.UMABAN.astype(int), x.t600))
        for k in pm.CORNERS:
            corner = str(r0[f'CORNER{k}'] or '0').strip()
            if corner not in ('1', '2', '3', '4'):
                continue
            try:
                seq = dm.separators(r0[f'KAKU_TSUKA_JUNI{k}'])
            except ValueError:
                continue
            if {u for u, _ in seq} != set(t):
                continue
            counts = dict.fromkeys(SEPS, 0)
            t0 = t[seq[0][0]]
            for u, sep in seq:
                if sep is not None:
                    counts[sep] += 1
                rows.append({'race': code, 'corner': int(corner), 'date': r0.date, 'time': t[u] - t0,
                             **{f'n{s}': v for s, v in counts.items()}})
    return pd.DataFrame(rows)


COLS = [f'n{s}' for s in SEPS]
MIDDLES = np.array([dm.MIDDLE[s] for s in SEPS])


def calibration(train):
    """From training rows only: seconds per separator (least squares, no intercept, on the cumulative counts) and the
    seconds per length of the current band-middle labels (least squares through 0)."""
    A = train[COLS].to_numpy(float)
    w, *_ = np.linalg.lstsq(A, train.time.to_numpy(), rcond=None)
    lab = A @ MIDDLES
    return dict(zip(SEPS, map(float, w))), float(np.sum(lab * train.time) / np.sum(lab ** 2))


def compare(train, test):
    """Disagreement of the current band middles (best scale on train) and of the calibrated widths with the time gap."""
    w, rate = calibration(train)
    lab_te = test[COLS].to_numpy(float) @ MIDDLES
    cal = test[COLS].to_numpy(float) @ np.array([w[s] for s in SEPS])
    cur = lab_te * rate
    return {'rowsTrain': int(len(train)), 'rowsTest': int(len(test)), 'secondsPerLabelLength': rate,
            'widthsSeconds': w, 'widthsRelativeToComma': {s: w[s] / w[','] for s in SEPS},
            'maeCurrent': float(np.mean(np.abs(cur - test.time))), 'maeCalibrated': float(np.mean(np.abs(cal - test.time))),
            'corrCurrent': float(np.corrcoef(cur, test.time)[0, 1]), 'corrCalibrated': float(np.corrcoef(cal, test.time)[0, 1]),
            'cut': float(1 - np.mean(np.abs(cal - test.time)) / np.mean(np.abs(cur - test.time)))}


def cmd_data(args):
    d = load()
    q = d[d.complete & d.date.between('20220101', '20251231')]
    res = {'rows': int(len(q)), 't600Valid': float(q.t600.notna().mean()), 'soha': float(q.soha.notna().mean()),
           'racesAllTimed': int(q.groupby('RACE_CODE').gap600.apply(lambda s: s.notna().all()).sum()), 'races': int(q.RACE_CODE.nunique()),
           'finishGapValid': float(q.finish_gap.notna().mean())}
    for c in (3, 4):
        z = q.dropna(subset=['gap600', f'r{c}'])
        res[f'spearman_r{c}'] = float(z.groupby('RACE_CODE').apply(lambda g: g[f'r{c}'].corr(g.gap600, method='spearman')).mean())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


def cmd_widths(args):
    d = load()
    res = {}
    for name, (fit_end, start, end) in {'selection': SEL, 'report': REP}.items():
        tr, te = time_rows(d, end_before=fit_end), time_rows(d, window=(start, end))
        if name == 'report':
            te = te[te.race.isin(ex.PINNED)]
        res[name] = {f'c{c}': compare(tr[tr.corner == c], te[te.corner == c]) for c in pm.CORNERS}
    res['rebuild'] = bool(all(res[w][f'c{c}']['cut'] >= CUT for w in ('selection', 'report') for c in (3, 4)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'widths.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    for w in ('selection', 'report'):
        for c, v in res[w].items():
            print(f'{w} {c}: MAE {v["maeCurrent"]:.3f} -> {v["maeCalibrated"]:.3f} s (cut {v["cut"]:.0%}), corr {v["corrCurrent"]:.3f} -> '
                  f'{v["corrCalibrated"]:.3f}; widths/"," ' + ' '.join(f'{s or "none"}:{r:.2f}' for s, r in v['widthsRelativeToComma'].items()))
    print('rebuild corner-3/4 labels:', res['rebuild'])


def relabel(d, fit_end):
    """Corner-3/4 gap labels rebuilt with widths calibrated on races before fit_end, in lengths at the scale of the
    current labels (lengths per second = 1 / the current labels' seconds per length on the same races)."""
    tr = time_rows(d, end_before=fit_end)
    d = d.copy()
    for c in (3, 4):
        w, rate = calibration(tr[tr.corner == c])
        d[f'g{c}'] = rebuilt_gaps(d, c, {s: w[s] / rate for s in SEPS})
    return d


def label_rates(d, fit_end):
    """Lengths per second of the current corner labels, per corner, from races before fit_end."""
    tr = time_rows(d, end_before=fit_end)
    return {c: 1 / calibration(tr[tr.corner == c])[1] for c in pm.CORNERS}


def rebuilt_gaps(d, c, widths):
    out = pd.Series(np.nan, index=d.index)
    for code, x in d.groupby('RACE_CODE'):
        r0 = x.iloc[0]
        for k in pm.CORNERS:
            if str(r0[f'CORNER{k}'] or '0').strip() != str(c):
                continue
            try:
                seq = dm.separators(r0[f'KAKU_TSUKA_JUNI{k}'])
            except ValueError:
                continue
            nums = x.UMABAN.astype(int)
            if {u for u, _ in seq} != set(nums):
                continue
            g, gaps = 0.0, {}
            for u, sep in seq:
                if sep is not None:
                    g += widths[sep]
                gaps[u] = g
            out[x.index] = [gaps[u] for u in nums]
    return out


def gap_truths(d, rate):
    """Time truth near corners 3-4 in label lengths: gap600 x lengths per second of the current labels."""
    return d.gap600 * rate


def evaluate(d, fit_end, start, end, P, truths):
    """E0 standard rows for one system: ranks (tau / integer), gaps against every truth column in `truths`, lanes, last3."""
    models = pm.fit_models(d, fit_end)
    races = pm.evaluation_races(models, d, start, end)
    tab = es.RaceTable()
    for x, corners, numbers, pred, spreads, early in races:
        race, n = x.RACE_CODE.iloc[0], len(numbers)
        con = pm.simulate(numbers, early, pred, spreads, corners, P)
        for c in corners:
            if not pm.labelled(x, c):
                continue
            y = x[f'r{c}'].to_numpy()
            formed = (np.array(con[c]['rank']) - 1) / (n - 1)
            tab.add(f'r{c}.integer', 'x', race, np.abs(formed - y))
            tab.add_value(f'r{c}.tau', 'x', race, es.kendall(formed, y))
            g = np.asarray(con[c]['gap'], float)
            for name, col in truths.items():
                if c in (3, 4) or name == 'label':
                    tv = x[col.format(c=c)].to_numpy(float)
                    tab.add(f'g{c}.{name}', 'x', race, np.abs(g - tv))
            tab.add(f'l{c}', 'x', race, np.abs(np.asarray(con[c].get('localLane', con[c]['lane']), float) - x[f'l{c}'].to_numpy()))
        ok = x.last3.notna().to_numpy()
        if ok.any():
            t3 = pm.closing(models, x, con, corners, pred['pace'])[0]
            tab.add('last3', 'x', race, np.abs(t3[ok] - x.last3.to_numpy()[ok]))
    out = {m: tab.mean(m, 'x') for (m, _) in tab.rows}
    return out, tab


def cmd_features(args):
    """Variants on the same frame: base, +past_gap600, past_gap600 instead of past_g3/past_g4; and (if the widths rule
    passed) the rebuilt corner-3/4 labels. Gaps scored against the old labels, the rebuilt labels and gap600."""
    ex.set_seed(args.seed)
    d0 = load()
    rebuild = json.loads((DATA / 'widths.json').read_text(encoding='utf-8'))['rebuild']
    P = json.loads((ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
    res = {'seed': args.seed, 'rebuild': rebuild, 'historyAlpha': pm.HISTORY_ALPHA}
    base = list(pm.FEATURES)
    for wname, (fit_end, start, end) in {'selection': SEL, 'report': REP}.items():
        d = d0.copy()
        rates = label_rates(d, fit_end)
        for c in pm.CORNERS:
            d[f'time_g{c}'] = d.gap600 * rates[c]  # gap600 in the current labels' lengths (scored at corners 3-4 only)
            d[f'label_g{c}'] = d[f'g{c}']
        d = pm.add_past_gap600(d)
        truths = {'label': 'label_g{c}', 'time': 'time_g{c}'}
        variants = {'base': base, 'add': base + ['past_gap600'],
                    'replace': [f for f in base if f not in ('past_g3', 'past_g4')] + ['past_gap600']}
        out = {}
        try:
            for name, feats in variants.items():
                pm.FEATURES[:] = feats
                out[name] = evaluate(d, fit_end, start, end, P, truths)[0]
        finally:
            pm.FEATURES[:] = base
        if rebuild:
            dr = relabel(d, fit_end)
            keep = [c for c in dr if not (c.startswith('past') or c == 'starts')]
            dr = pm.add_past_gap600(pm.add_traffic(pm.add_history_features(dr[keep])))
            out['rebuiltLabels'] = evaluate(dr, fit_end, start, end, P, {'label': 'label_g{c}', 'rebuilt': 'g{c}', 'time': 'time_g{c}'})[0]
        res[wname] = {'rates': rates, 'variants': out}
        print(wname, json.dumps(out, indent=1, default=float), flush=True)
    (OUT / f'features_seed{args.seed}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


def cmd_finish(args):
    ex.set_seed(args.seed)
    d = pm.add_past_gap600(load())
    P = json.loads((ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
    res = {'seed': args.seed}
    for wname, (fit_end, start, end) in {'selection': SEL, 'report': REP}.items():
        models = pm.fit_models(d, fit_end)
        rows = (d.date < fit_end) & d.complete & d.gap600.notna()
        g600 = pm.lgbm('l1').fit(d.loc[rows, pm.FEATURES], d.loc[rows, 'gap600'])
        rate = label_rates(d, fit_end)
        races = pm.evaluation_races(models, d, start, end)
        tab = es.RaceTable()
        for x, corners, numbers, pred, spreads, early in races:
            truth = x.finish_gap.to_numpy(float)
            if np.isnan(truth).any():
                continue
            race = x.RACE_CODE.iloc[0]
            form = pm.simulate(numbers, early, pred, spreads, corners, P)
            last3 = pm.closing(models, x, form, corners, pred['pace'])[0]
            gl = np.asarray(form[corners[-1]]['gap'], float)
            k3 = last3 - last3.min()
            systems = {'current': gl / pm.LENGTHS_PER_SECOND + k3,
                       'identityA': gl / rate[corners[-1]] + k3,
                       'identityB': g600.predict(x[pm.FEATURES]) + last3}
            for k, v in systems.items():
                v = v - v.min()
                tab.add('fin', k, race, np.abs(v - truth))
                tab.add('fin.bias', k, race, v - truth)
                tab.add_value('fin.spearman', k, race, float(pd.Series(v).corr(pd.Series(truth), method='spearman')))
        res[wname] = {k: {'mae': tab.mean('fin', k), 'bias': tab.mean('fin.bias', k), 'spearman': tab.mean('fin.spearman', k),
                          'races': tab.races('fin', k)} for k in ('current', 'identityA', 'identityB')}
        res[wname]['diffs'] = {k: tab.diff('fin', k, 'current') for k in ('identityA', 'identityB')}
        print(wname, json.dumps(res[wname], indent=1, default=float), flush=True)
    (OUT / f'finish_seed{args.seed}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['frame', 'data', 'widths', 'features', 'finish'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--procedure', choices=list(PROCEDURES), default='reviewed')
    args = ap.parse_args()
    global OUT
    (pm.EVAL_CORNERS, pm.PACE_OOF), out = PROCEDURES[args.procedure]
    if args.cmd in ('features', 'finish'):
        OUT = ROOT / 'outputs' / out
        OUT.mkdir(parents=True, exist_ok=True)
    ex.PINNED = set(ex.validation_codes(pd.read_pickle(PIN)))
    {'frame': cmd_frame, 'data': cmd_data, 'widths': cmd_widths, 'features': cmd_features, 'finish': cmd_finish}[args.cmd](args)


if __name__ == '__main__':
    main()
