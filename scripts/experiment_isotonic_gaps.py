"""E3 (2026-11-18): gaps by isotonic regression onto the formation, with a field-length (spread) penalty.

Starting point: Codex's prototype (selection window, 2026-11-18). Keep the published formation's order and abreast rows;
give every row the monotone gap closest (least squares) to the mean of its horses' per-horse gap predictions (the L1 gap
model, `pred['g<c>']`); the first row stays at 0. Codex found the per-horse gap MAE lower at every corner
(1st corner 3.51 -> 3.27) but the field squeezed (80%-position gap 5.97 against 8.63 actual), because the per-horse
predictions are medians and shrink toward the middle.

Penalty: with y the isotonic gaps, the published gaps are s * y, where s minimises
    sum_i (s * y_i - raw_i)^2 + lam * n * (s * q80(y) - spread)^2
(spread = the race-level spread model's prediction, q80 = the gap 80% of the way down the field, np.quantile like the
spread label). lam = 0 is the prototype (s = 1 for an isotonic fit); lam -> inf stretches the field to the predicted
spread. The closed form keeps the order, the rows and the monotonicity: s = (sum y raw + lam n q spread) / (sum y^2 +
lam n q^2).

Discipline: frame and configuration as published now (pace_model's HISTORY_ALPHA / RANK_BLEND, contention parameters of
models/pace/report.json), E0 procedure, seeds 42 and 7.
  select  lam grid on Jun-Aug 2025 (models fitted before June). Choice: the lowest 2-seed per-horse gap MAE (rows of all
          corners) among the lam whose corner-averaged |spread bias| is not above the published formation's.
  report  the chosen lam once on Sep-Dec 2025 (v1.6's 273 races) against the published formation; race-paired bootstrap.
Adoption: the gap-MAE improvement's 95% interval excludes 0, and the spread bias is not worse than the published one.

Run:  python scripts/experiment_isotonic_gaps.py select --seed 42   (and --seed 7)
      python scripts/experiment_isotonic_gaps.py report --seed 42   (and --seed 7)
      python scripts/experiment_isotonic_gaps.py summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_data_expansion as ex  # noqa: E402
import pace_model as pm  # noqa: E402
import eval_standard as es  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pm.ROOT
OUT = ROOT / 'outputs' / 'e3-isotonic'
FRAME = ROOT / 'outputs' / 'data-expansion-v18' / 'frame_2022.pkl'
PIN = ROOT / 'outputs' / 'data-expansion' / 'frame_before_20261118_0126.pkl'
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf)
SEL, REP = ('20250601', '20250601', '20250831'), ('20250901', '20250901', '20251231')


def isotonic_gaps(form_gap, raw, spread, lam):
    """Rows of the formation (horses sharing a gap) in formation order -> monotone gaps nearest to the rows' mean
    per-horse prediction (first row 0), then stretched toward the predicted spread by the penalty lam."""
    before, raw = np.asarray(form_gap, float), np.asarray(raw, float)
    _, idx, count = np.unique(before, return_inverse=True, return_counts=True)
    mean = np.bincount(idx, weights=raw) / count
    y = np.zeros(len(mean))
    if len(mean) > 1:
        y[1:] = np.maximum(0, isotonic_regression(mean[1:], weights=count[1:]).x)
    x = y[idx]
    q, n = float(np.quantile(x, pm.SPREAD_Q)), len(x)
    if lam == 0 or q <= 1e-9 or spread <= 0:
        return x
    if np.isinf(lam):
        return x * spread / q
    s = (np.sum(x * raw) + lam * n * q * spread) / (np.sum(x * x) + lam * n * q * q)
    return s * x


def frame():
    d = pd.read_pickle(FRAME)
    d = pm.summarize_history(d)  # the published history summary (HISTORY_ALPHA)
    return pm.add_traffic(pm.add_race_relative(d, 'RACE_CODE'))


def params():
    return json.loads((ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']


def collect(d, fit_end, start, end, P, lams):
    models = pm.fit_models(d, fit_end)
    races = pm.evaluation_races(models, d, start, end)
    tab = es.RaceTable()
    for x, corners, numbers, pred, spreads, early in races:
        race = x.RACE_CODE.iloc[0]
        form = pm.simulate(numbers, early, pred, spreads, corners, P)
        for c in corners:
            if not pm.labelled(x, c):
                continue
            obs = x[f'g{c}'].to_numpy(float)
            true_q = float(np.quantile(obs, pm.SPREAD_Q))
            systems = {'published': np.asarray(form[c]['gap'], float), 'perHorse': np.asarray(pred[f'g{c}'], float)}
            for lam in lams:
                systems[f'lam{lam}'] = isotonic_gaps(form[c]['gap'], pred[f'g{c}'], spreads[c], lam)
            for k, v in systems.items():
                tab.add(f'g{c}', k, race, np.abs(v - obs))
                tab.add('g', k, race, np.abs(v - obs))  # all corners of a race together: the bootstrap resamples races
                tab.add_value(f'q{c}.bias', k, race, float(np.quantile(v, pm.SPREAD_Q)) - true_q)
                tab.add_value(f'q{c}.abs', k, race, abs(float(np.quantile(v, pm.SPREAD_Q)) - true_q))
    return tab, len(races)


def table(tab, systems):
    out = {}
    for k in systems:
        row = {'gapMae': tab.mean('g', k)}
        for c in pm.CORNERS:
            if (f'g{c}', k) in tab.rows:
                row[f'g{c}'] = tab.mean(f'g{c}', k)
                row[f'q{c}Bias'] = tab.mean(f'q{c}.bias', k)
                row[f'q{c}Mae'] = tab.mean(f'q{c}.abs', k)
        biases = [abs(row[f'q{c}Bias']) for c in pm.CORNERS if f'q{c}Bias' in row]
        row['meanAbsBias'] = float(np.mean(biases))
        out[k] = row
    return out


def cmd_select(args):
    ex.set_seed(args.seed)
    tab, n = collect(frame(), SEL[0], SEL[1], SEL[2], params(), LAMBDAS)
    systems = ['published', 'perHorse'] + [f'lam{lam}' for lam in LAMBDAS]
    res = {'seed': args.seed, 'races': n, 'table': table(tab, systems)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'select_seed{args.seed}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    for k, v in res['table'].items():
        print(f'{k:>10}: gap {v["gapMae"]:.3f} |bias| {v["meanAbsBias"]:.3f} ' + ' '.join(
            f'g{c} {v[f"g{c}"]:.3f}/{v[f"q{c}Bias"]:+.2f}' for c in pm.CORNERS if f'g{c}' in v), flush=True)


def choose():
    sel = [json.loads((OUT / f'select_seed{s}.json').read_text(encoding='utf-8'))['table'] for s in (42, 7)]
    mean = lambda k, m: float(np.mean([t[k][m] for t in sel]))
    limit = mean('published', 'meanAbsBias')
    ok = [lam for lam in LAMBDAS if mean(f'lam{lam}', 'meanAbsBias') <= limit]
    pick = min(ok, key=lambda lam: mean(f'lam{lam}', 'gapMae')) if ok else None
    return pick, {f'lam{lam}': {'gapMae': mean(f'lam{lam}', 'gapMae'), 'meanAbsBias': mean(f'lam{lam}', 'meanAbsBias')} for lam in LAMBDAS}, limit


def cmd_report(args):
    lam, _, _ = choose()
    if lam is None:
        print('no lambda keeps the spread bias at the published level; nothing to report')
        return
    ex.set_seed(args.seed)
    tab, n = collect(frame(), REP[0], REP[1], REP[2], params(), [lam])
    name = f'lam{lam}'
    res = {'seed': args.seed, 'lambda': lam, 'races': n, 'table': table(tab, ['published', 'perHorse', name]),
           'diff': {'all': tab.diff('g', name, 'published'),
                    **{f'g{c}': tab.diff(f'g{c}', name, 'published') for c in pm.CORNERS if (f'g{c}', name) in tab.rows}}}
    (OUT / f'report_seed{args.seed}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    print(json.dumps(res, indent=1, default=float), flush=True)


def cmd_summary(args):
    lam, sel, limit = choose()
    print(f'chosen lambda {lam} (published |bias| on the selection window {limit:.3f})')
    for k, v in sel.items():
        print(f'  {k:>10}: 2-seed gap MAE {v["gapMae"]:.4f}  mean |spread bias| {v["meanAbsBias"]:.3f}')
    reps = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('report_seed*.json'))}
    (OUT / 'result.json').write_text(json.dumps({'lambda': lam, 'selection': sel, 'reports': reps}, indent=1, default=float),
                                     encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['select', 'report', 'summary'])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    ex.PINNED = set(ex.validation_codes(pd.read_pickle(PIN)))
    {'select': cmd_select, 'report': cmd_report, 'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
