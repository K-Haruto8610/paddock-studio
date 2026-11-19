"""E0 audit (2026-11-18): reproduce the design-review claims inside the repo and size the two validation fixes.
No model and no published value changes: this script only reads frames and writes outputs/e0-audit/.

Frame: the v1.8 training frame (outputs/data-expansion-v18/frame_2022.pkl); report window pinned to v1.6's 273 races
(--pin, via experiment_data_expansion); selection window Jun-Aug 2025 with models fitted before June.

  measure   legacy procedure (EVAL_CORNERS='labels', PACE_OOF='kfold') and v1.8's contention parameters, both windows:
            1 lanes      scored lane (localLane) / independent / rounded independent / displayed lane vs the constant-0
                         forecast, class accuracy 0 / 1 / 2+, share of horses whose displayed lane differs from the scored
            2 last3      the same closing model on {independent, contention} formation x {traffic input on, off} (the
                         independent formation's traffic is derived with the contention layer's own rule), the energy
                         input off, and the FEATURES-only model: the 0.067 s split into model / energy / formation / traffic
            3 pace       seconds MAE, corr, H/M/S hit, RPS vs climatology and majority class
            4 ranks      tau and integer MAE of the formations, continuous MAE before / after the blend
            5 objective  contention objective with and without l1 over the whole grid: same choice?
            6 corners    published 2026 races: corner set from all data (race_output) vs from races before the race day
  pipeline  the full published procedure (tune_contention -> tune_moves -> calibrate_intervals -> validate) for one
            variant x seed: legacy (must reproduce v1.8), fixA (course corner sets), fixB (forward pace OOF), fixAB = E0
            as committed (a6c38ce); reviewed = after the Codex review of 2026-11-19 (straight courses not formed, pace
            OOF by race day with forward norms, every chunk race predicted). --nopin: the whole report window.
  summary   combine the files into outputs/e0-audit/result.json and print the tables used in the README.

Run:  python scripts/audit_e0.py measure
      python scripts/audit_e0.py pipeline --variant legacy --seed 42     (fixA, fixB, fixAB; seed 7 for legacy/fixAB)
      python scripts/audit_e0.py summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_data_expansion as ex  # noqa: E402  (pins the report window to v1.6's 273 races)
import pace_model as pm  # noqa: E402
import eval_standard as es  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pm.ROOT
OUT = ROOT / 'outputs' / 'e0-audit'
FRAME = ROOT / 'outputs' / 'data-expansion-v18' / 'frame_2022.pkl'
PIN = ROOT / 'outputs' / 'data-expansion' / 'frame_before_20261118_0126.pkl'
WINDOWS = {'selection': ('20250601', '20250601', '20250831'), 'report': ('20250901', '20250901', '20251231')}
VARIANTS = {'legacy': ('labels', 'kfold'), 'fixA': ('course_e0', 'kfold'), 'fixB': ('labels', 'forward_ids'),
            'fixAB': ('course_e0', 'forward_ids'), 'reviewed': ('course', 'forward'),
            'reviewedCorners': ('course', 'forward_ids'), 'reviewedOof': ('course_e0', 'forward')}
V18 = json.loads((ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))


def load(frame, pin):
    # The audit is of v1.8: its model settings (5-run means as stored in the v1.8 frame, rank blend 0.5), whatever the
    # current pace_model defaults are (v1.9 changed RANK_BLEND to 0.75).
    pm.HISTORY_ALPHA, pm.RANK_BLEND = None, 0.5
    d = pd.read_pickle(frame)
    ex.PINNED = set(ex.validation_codes(pd.read_pickle(pin))) if pin else None
    return d


def set_variant(name):
    pm.EVAL_CORNERS, pm.PACE_OOF = VARIANTS[name]


traffic_of = pm.formation_traffic  # moved to pace_model (standard_metrics uses it for the independent formation)


def closing_variant(models, x, form, corners, pace, traffic=True, energy=True):
    xt = pm.closing_inputs(x, form, corners, pace)
    if not traffic:
        xt['traffic_out'], xt['traffic_close'] = 0.0, 0.0
    if not energy:
        for k in pm.ENERGY:
            xt[k] = 0.0
    return models['last3_traffic'].predict(xt[pm.LAST3_FEATURES])


def measure_window(d, name, P):
    fit_end, start, end = WINDOWS[name]
    models = pm.fit_models(d, fit_end)
    races = pm.evaluation_races(models, d, start, end)
    out = {'races': len(races), 'standard': pm.standard_metrics(models, races, P, d, fit_end)}
    # 1 lanes: rounded independent too (the contention row index is an integer)
    lanes = {}
    for x, corners, numbers, pred, spreads, early in races:
        con = pm.simulate(numbers, early, pred, spreads, corners, P)
        for c in corners:
            if not pm.labelled(x, c):
                continue
            z = lanes.setdefault(c, {'label': [], 'round': []})
            z['label'].extend(x[f'l{c}'].to_numpy())
            z['round'].extend(np.rint(np.clip(pred[f'l{c}'], 0, 4)))
    out['lanesRoundedIndependent'] = {f'l{c}': es.lane_summary(z['round'], z['label']) for c, z in sorted(lanes.items())}
    # 2 last3 ablation with the same closing model
    tab, shift_rows, traffic_rows = es.RaceTable(), [], {}
    for x, corners, numbers, pred, spreads, early in races:
        ok = x.last3.notna().to_numpy()
        if not ok.any():
            continue
        race, l3 = x.RACE_CODE.iloc[0], x.last3.to_numpy()[ok]
        con = pm.simulate(numbers, early, pred, spreads, corners, P)
        ind = pm.independent(numbers, pred, spreads, corners)
        ind_t = {c: dict(ind[c]) for c in corners}
        last = corners[-1]
        ind_t[last]['out'], ind_t[last]['close'] = traffic_of(ind[last])
        variants = {
            'A_featuresOnlyModel': pred['last3'],
            'B_closing_noEnergy_noTraffic': closing_variant(models, x, ind, corners, pred['pace'], traffic=False, energy=False),
            'C_independent_energy_noTraffic': closing_variant(models, x, ind, corners, pred['pace'], traffic=False),
            'D_contention_energy_noTraffic': closing_variant(models, x, con, corners, pred['pace'], traffic=False),
            'E_contention_energy_traffic': closing_variant(models, x, con, corners, pred['pace']),
            'F_independent_energy_derivedTraffic': closing_variant(models, x, ind_t, corners, pred['pace']),
        }
        keys = list(np.asarray(numbers)[ok])
        for k, v in variants.items():
            tab.add('last3', k, race, np.abs(np.asarray(v)[ok] - l3), keys)
            tab.add('last3.bias', k, race, np.asarray(v)[ok] - l3, keys)  # signed: the energy / traffic inputs only slow a horse
        shift_rows.append((race, keys, np.asarray(variants['B_closing_noEnergy_noTraffic'])[ok],
                           np.asarray(variants['C_independent_energy_noTraffic'])[ok], l3))
        z = traffic_rows.setdefault('n', [0, 0, 0])
        z[0] += int(np.sum(np.asarray(ind_t[last]['close'])[ok] > 0))
        z[1] += int(np.sum(np.asarray(ind_t[last]['out'])[ok] > 0))
        z[2] += int(ok.sum())
    keys = sorted({k for (m, k) in tab.rows if m == 'last3'})
    out['last3Ablation'] = {'mae': {k: tab.mean('last3', k) for k in keys}, 'bias': {k: tab.mean('last3.bias', k) for k in keys},
                            'rows': tab.count('last3', keys[0]),
                            'steps': {s: tab.diff('last3', b, a) for s, (a, b) in {
                                'model (A->B)': ('A_featuresOnlyModel', 'B_closing_noEnergy_noTraffic'),
                                'energy input (B->C)': ('B_closing_noEnergy_noTraffic', 'C_independent_energy_noTraffic'),
                                'formation via energy (C->D)': ('C_independent_energy_noTraffic', 'D_contention_energy_noTraffic'),
                                'traffic input on contention (D->E)': ('D_contention_energy_noTraffic', 'E_contention_energy_traffic'),
                                'traffic input on independent (C->F)': ('C_independent_energy_noTraffic', 'F_independent_energy_derivedTraffic'),
                                'total (A->E)': ('A_featuresOnlyModel', 'E_contention_energy_traffic')}.items()}}
    # Review finding 6: how much of B->C is a level shift? B plus the mean of C - B for every horse, scored like C.
    shift = float(np.mean(np.concatenate([c - b for _, _, b, c, _ in shift_rows])))
    for race, keys, b, c, l3 in shift_rows:
        tab.add('last3', 'B_plus_meanShift', race, np.abs(b + shift - l3), keys)
    out['last3Ablation']['levelShift'] = {
        'meanCminusB': shift, 'maeB': tab.mean('last3', 'B_closing_noEnergy_noTraffic'), 'maeBshifted': tab.mean('last3', 'B_plus_meanShift'),
        'maeC': tab.mean('last3', 'C_independent_energy_noTraffic'),
        'shareOfBtoC': (tab.mean('last3', 'B_closing_noEnergy_noTraffic') - tab.mean('last3', 'B_plus_meanShift'))
                       / (tab.mean('last3', 'B_closing_noEnergy_noTraffic') - tab.mean('last3', 'C_independent_energy_noTraffic'))}
    # Review finding 5: the independent formation does leave traffic at the last corner (scored horses).
    n = traffic_rows.get('n', [0, 0, 0])
    out['last3Ablation']['independentTraffic'] = {'close': n[0], 'out': n[1], 'horses': n[2],
                                                  'treeSplitsOnTrafficClose': tree_splits(models['last3_traffic'], 'traffic_close')}
    return out, models, races


def tree_splits(model, feature):
    """How many splits of the fitted closing model use `feature` (0 = the input cannot change a prediction)."""
    names = model.booster_.feature_name()
    return int(model.booster_.feature_importance('split')[names.index(feature)])


def objective_check(models, races):
    """Every grid setting scored once; the argmin of the mean ratio with and without l1."""
    base = pm.score(models, races, None)
    rows = []
    for P in pm.PARAM_GRID:
        s = pm.score(models, races, P)
        ratios = {k: s[k] / base[k] for k in s}
        rows.append({'P': P, 'l1': ratios.get('l1'), 'with': float(np.mean(list(ratios.values()))),
                     'without': float(np.mean([v for k, v in ratios.items() if k != 'l1']))})
    a, b = min(rows, key=lambda r: r['with']), min(rows, key=lambda r: r['without'])
    return {'settings': len(rows), 'l1RatioMin': min(r['l1'] for r in rows), 'l1RatioMax': max(r['l1'] for r in rows),
            'choiceWithL1': a['P'], 'choiceWithoutL1': b['P'], 'sameChoice': a['P'] == b['P'],
            'rankingSame': [r['P'] for r in sorted(rows, key=lambda r: r['with'])] == [r['P'] for r in sorted(rows, key=lambda r: r['without'])]}


def corner_set_check(d):
    """Published 2026 races: corner set from all data (as race_output does) vs from the races before the race day."""
    pub = json.loads((ROOT / 'public' / 'data' / 'forecasts.json').read_text(encoding='utf-8'))['races']
    full = pm.corner_sets(d)
    out = []
    for r in pub:
        x = d[d.RACE_CODE == r['id']]
        if x.empty:
            continue
        r0 = x.iloc[0]
        key = (int(r0.venue), int(r0.track), int(r0.distance))
        before = pm.corner_sets(d[d.date < r0.date])(*key)
        out.append({'race': r['name'], 'date': r0.date, 'allData': full(*key), 'beforeRace': before, 'same': full(*key) == before})
    return out


def cmd_measure(args):
    d = load(args.frame, args.pin)
    set_variant('legacy')
    P = V18['contention']
    res = {'frame': str(args.frame), 'params': P}
    for name in WINDOWS:
        res[name], models, races = measure_window(d, name, P)
        if name == 'selection':
            res['objective'] = objective_check(models, races)
        print(name, json.dumps(res[name]['last3Ablation'], ensure_ascii=False), flush=True)
    res['cornerSets'] = corner_set_check(d)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'measure.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print('wrote', OUT / 'measure.json')


def cmd_pipeline(args):
    d = load(args.frame, args.pin)
    set_variant(args.variant)
    res = ex.pipeline(d, seed=args.seed)
    res.update({'variant': args.variant, 'seed': args.seed, 'evalCorners': pm.EVAL_CORNERS, 'paceOof': pm.PACE_OOF})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'pipeline_{args.variant}_seed{args.seed}{"_nopin" if args.pin is None else ""}.json'
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print('wrote', path)


def cmd_summary(args):
    runs = {p.stem.replace('pipeline_', ''): json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('pipeline_*.json'))}
    measure = json.loads((OUT / 'measure.json').read_text(encoding='utf-8')) if (OUT / 'measure.json').exists() else None
    targets = list(V18['metrics'])
    print('%-8s %8s' % ('metric', 'v1.8') + ''.join(f'{k:>18}' for k in runs))
    for t in targets:
        print('%-8s %8.4f' % (t, V18['metrics'][t]['mae']) + ''.join(f"{r['metrics'][t]['mae']:18.4f}" for r in runs.values()))
    for t in targets:
        print('%-8s rows  ' % t + ''.join(f"{r['metrics'][t]['validationRows']:18d}" for r in runs.values()))
    print('params:')
    for k, r in runs.items():
        print(' ', k, r['params'])
    print('standard (tau contention / integer contention / continuous model):')
    for k, r in runs.items():
        s = r['metrics']['_standard']['ranks']
        print(' ', k, ' '.join(f"{c}:{v['tau']['contention']:.4f}/{v['integerMae']['contention']:.4f}/{v['continuousMae']['model']:.4f}" for c, v in s.items()))
    legacy = runs.get('legacy_seed42')
    if legacy:
        diff = max(abs(legacy['metrics'][t]['mae'] - V18['metrics'][t]['mae']) for t in targets)
        print(f'legacy seed 42 vs published v1.8: max |diff| over the 13 metrics = {diff:.2e}')
    res = {'measure': measure, 'pipelines': runs}
    (OUT / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print('wrote', OUT / 'result.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['measure', 'pipeline', 'summary'])
    ap.add_argument('--frame', default=str(FRAME))
    ap.add_argument('--pin', default=str(PIN))
    ap.add_argument('--nopin', action='store_true', help='score the whole report window (not only v1.6\'s 273 races)')
    ap.add_argument('--variant', default='legacy', choices=list(VARIANTS))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.nopin:
        args.pin = None
    {'measure': cmd_measure, 'pipeline': cmd_pipeline, 'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
