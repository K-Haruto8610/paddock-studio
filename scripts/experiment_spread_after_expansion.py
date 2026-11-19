"""Experiment (追加指示7, 2026-11-18): does the expanded DB lift the field-length (spread) limit? Diagnosis only.

Phase 2 (scripts/diagnose_margins.py, experiment_spread_features.py) blamed the g1/g2 loss to the past mean on too few
races for the race-level spread models (43/53 races before 2025-06 at corners 1/2, constants in the selection window).
After the import (追加指示5) the same questions are asked again, before vs after, with no model change:

  1. Training races of the spread models per corner and fit end (2025-06 / 2025-09 / 2026-01).
  2. Spread model vs a constant (training median, the L1 optimum), at the spread level (MAE, corr) and end to end
     (contention gap MAE with the model's spread vs the constant). Seed 7 refit for noise. Candidate only if clearly
     better than the constant in both windows.
  3. g1/g2 against the same-data past-5 mean (contention, independent, per-horse gap model alone), with the seed
     difference. Ranks are compared with the past mean turned into integer ranks (pastMeanFormed), as the published
     ranks are integers.
  4. Conversion only: the finish estimate (last-corner gap + closing-3F difference x lengths/s) with 5.0 (current)
     and 5.7 (least squares of the official margins on TIME_SA, diagnose_margins A1). Truth = the official margins
     summed (CHAKUSA_CODE1), which does not depend on the conversion; also the finish-order rank correlation.
  5. Differences inside the label noise (about 0.8 lengths per horse) or the seed difference are noise.

Discipline: selection window Jun-Aug 2025 (fit before June), validation Sep-Dec 2025 pinned to the 273 races of v1.6,
baselines recomputed on the same frame. One set of contention parameters for both frames: the retrained model's
(front 0.45, settled).

Run:  python scripts/experiment_spread_after_expansion.py --before outputs/data-expansion/frame_before_20261118_0126.pkl \
        --after outputs/data-expansion/frame_after.pkl --out outputs/data-expansion/spread_result.json

Result 2026-11-18 -> diagnosis only, no model change (README「データ拡充後の隊列長（追加指示7）」):
  1. Spread training races c1-c4: fit<2025-06 43/53/110/110 -> 979/1151/2060/2060; <2025-09 163/185/341/341 ->
     1099/1283/2291/2291; <2026 278/323/616/616 -> 1214/1421/2566/2566.
  2. Validation spread MAE model vs constant: before 2.94/3.44/2.40/2.08 vs 2.97/3.25/2.33/2.00, after 2.90/3.07/2.27/1.96
     vs 2.95/3.25/2.33/1.99 (corr 0.19/0.33/0.12/0.22). End to end (contention gaps, constant minus model): tuning
     0.05/0.12/0.10/0.03 (seed 7 0.09/0.10/0.10/0.05), validation 0.03/0.03/0.03/0.01 (seed 7 -0.02/0.01/0.03/0.02);
     before, validation -0.13/-0.15/-0.05/-0.10. The spread model now beats the constant; it is already the model in use.
  3. After, validation g1/g2: contention 3.41/3.39 (seed 7 3.43/3.44), independent 3.40/3.41, per-horse gap model 3.09/3.19,
     past mean 3.30/3.46. g1 still loses by 0.10-0.13, g2 wins by 0.02-0.07; both inside the ~0.8-length label noise.
     Ranks vs past mean as integer ranks: model ahead by 0.0021/0.0049/0.0000/0.0008 (seed 7 0.0003/0.0043/0.0017/0.0024).
  4. 5.7 lengths/s vs 5.0: finish-gap MAE 4.57 -> 4.50 (seed 7 4.44 -> 4.37; before 4.21 -> 4.18) but finish-order
     Spearman 0.228 -> 0.224 (seed 7 0.212 -> 0.211; tuning 0.232 -> 0.221) -> NOT adopted. Predicted finish gaps are
     ~3.5 lengths too small on average either way (bias -3.3 to -3.7).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_data_expansion as ex  # noqa: E402  (pins the validation window to v1.6's races)
import diagnose_margins as dm  # noqa: E402
import pace_model as pm  # noqa: E402
pm.legacy_procedure('experiment_spread_after_expansion.py')  # pre-E0: reproduces its record on the v1.8 procedure (E0 review finding 2)

sys.stdout.reconfigure(encoding='utf-8')
WINDOWS = {'tuning': ('20250601', '20250601', '20250831'), 'validation': ('20250901', '20250901', '20251231')}
FIT_ENDS = ['20250601', '20250901', '20260101']
RATES = (5.0, 5.7)


def spread_counts(d):
    out = {}
    for end in FIT_ENDS:
        train = d[(d.date < end) & d.complete]
        out[end] = [int(len(pm.race_level(train, 'RACE_CODE', c).dropna(subset=['spread']))) for c in pm.CORNERS]
    return out


def mae(v):
    return float(np.mean(np.abs(v))) if len(v) else float('nan')


def finish_truth(f, x):
    """Official lengths behind the winner (margins summed) and the finish position, by horse number; None if unusable."""
    g = f[(f.RACE_CODE == x.RACE_CODE.iloc[0]) & f.fin.between(1, 18)].sort_values(['fin', 'UMABAN'])
    if len(g) != len(x) or g.step_off.iloc[1:].isna().any():
        return None
    step = g.step_off.to_numpy().copy()
    step[0] = 0.0
    return dict(zip(g.UMABAN.astype(int), np.cumsum(step))), dict(zip(g.UMABAN.astype(int), g.fin))


def window(d, f, P, fit_end, start, end, seed):
    ex.set_seed(seed)
    models = pm.fit_models(d, fit_end)
    ex.set_seed(42)
    races = pm.evaluation_races(models, d, start, end)
    tr = (d.date < fit_end) & d.complete
    const = {c: float(pm.race_level(d[tr], 'RACE_CODE', c).dropna(subset=['spread']).spread.median()) for c in pm.CORNERS}
    err = {k: {c: [] for c in pm.CORNERS} for k in ('contention', 'contConst', 'independent', 'gapModel', 'pastMean',
                                                     'rank', 'rankPast', 'rankPastFormed')}
    sp = {c: [] for c in pm.CORNERS}
    fin = {r: [] for r in RATES}
    fin_rank = {r: [] for r in RATES}
    used_fin = 0
    for x, corners, numbers, pred, spreads, early in races:
        form = pm.simulate(numbers, early, pred, spreads, corners, P)
        form_c = pm.simulate(numbers, early, pred, {c: const[c] for c in pm.CORNERS}, corners, P)
        n = len(numbers)
        for c in corners:
            actual = x[f'g{c}'].to_numpy()
            sp[c].append((float(np.quantile(actual, pm.SPREAD_Q)), spreads[c], const[c]))
            err['contention'][c].extend(np.array(form[c]['gap']) - actual)
            err['contConst'][c].extend(np.array(form_c[c]['gap']) - actual)
            err['independent'][c].extend(np.array(pm.reconstruct(pred[f'r{c}'], pred[f'g{c}'], spreads[c], numbers)[1]) - actual)
            err['gapModel'][c].extend(np.asarray(pred[f'g{c}']) - actual)
            err['pastMean'][c].extend(x[f'past_g{c}'].fillna(d.loc[tr, f'g{c}'].median()).to_numpy() - actual)
            ra = x[f'r{c}'].to_numpy()
            err['rank'][c].extend((np.array(form[c]['rank']) - 1) / (n - 1) - ra)
            pr = x[f'past_r{c}'].fillna(d.loc[tr, f'r{c}'].median()).to_numpy()
            err['rankPast'][c].extend(pr - ra)
            err['rankPastFormed'][c].extend(np.argsort(np.argsort(pr, kind='stable'), kind='stable') / (n - 1) - ra)
        truth = finish_truth(f, x)
        if truth is None:
            continue
        used_fin += 1
        lengths, pos = truth
        last3 = pm.closing(models, x, form, corners, pred['pace'])[0]
        gaps, best = form[corners[-1]]['gap'], float(min(last3))
        for r in RATES:
            est = np.array([gaps[i] + (float(last3[i]) - best) * r for i in range(n)])
            est = est - est.min()
            fin[r].extend(est - np.array([lengths[num] for num in numbers]))
            order = np.empty(n)
            for k, i in enumerate(sorted(range(n), key=lambda i: (est[i], numbers[i]))):
                order[i] = k + 1
            fin_rank[r].append(float(pd.Series(order).corr(pd.Series([float(pos[num]) for num in numbers]), method='spearman')))
    out = {'races': len(races), 'constantSpread': const}
    for k, v in err.items():
        out[k] = {f'{"r" if k.startswith("rank") else "g"}{c}': mae(v[c]) for c in pm.CORNERS}
    out['spread'] = {}
    for c, rows in sp.items():
        a = np.array(rows)
        if len(a) > 2:
            out['spread'][f'c{c}'] = {'races': int(len(a)), 'meanTrue': float(a[:, 0].mean()), 'maeModel': mae(a[:, 1] - a[:, 0]),
                                      'maeConstant': mae(a[:, 2] - a[:, 0]), 'corrModel': float(np.corrcoef(a[:, 1], a[:, 0])[0, 1]),
                                      'sdModel': float(a[:, 1].std())}
    out['finish'] = {'races': used_fin, **{str(r): {'gapMae': mae(fin[r]), 'gapBias': float(np.mean(fin[r])),
                                                    'rankSpearman': float(np.nanmean(fin_rank[r]))} for r in RATES}}
    return out


def run(d, f, P):
    res = {'spreadTrainingRaces': spread_counts(d)}
    for name, (fit_end, start, end) in WINDOWS.items():
        for seed in (42, 7):
            key = f'{name}' + ('' if seed == 42 else '-seed7')
            print(f'--- {key}', flush=True)
            res[key] = window(d, f, P, fit_end, start, end, seed)
            print(json.dumps({k: res[key][k] for k in ('races', 'contention', 'contConst', 'pastMean', 'spread', 'finish')},
                             ensure_ascii=False), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--params-report', default=str(pm.ROOT / 'models' / 'pace' / 'report.json'),
                    help='report whose contention parameters are used for both frames (default: the retrained model)')
    ap.add_argument('--out', default=str(pm.ROOT / 'outputs' / 'data-expansion' / 'spread_result.json'))
    ap.add_argument('--only', default='before,after')
    args = ap.parse_args()
    P = json.loads(Path(args.params_report).read_text(encoding='utf-8'))['contention']
    frames = {'before': pd.read_pickle(args.before), 'after': pd.read_pickle(args.after)}
    ex.PINNED = set(ex.validation_codes(frames['before']))
    codes = set()
    for d in frames.values():
        codes |= set(d[d.date.between('20250601', '20251231')].RACE_CODE)
    f = dm.load_finish(sorted(codes))
    res = json.loads(Path(args.out).read_text(encoding='utf-8')) if Path(args.out).exists() else {}
    res['params'] = P
    for name in args.only.split(','):
        print(f'\n===== {name} =====', flush=True)
        res[name] = run(frames[name], f, P)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
