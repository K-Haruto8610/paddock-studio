"""E1 (2026-11-18): past_<target> as an exponentially weighted mean of ALL prior runs instead of the plain mean of the
last 5 runs, with the rank blend re-selected. Formal experiment after the design review's quick check (README「E1」).

Why this is not the closed sequence-feature test (追加指示4 フェーズ1): that one weighted the runs INSIDE the 5-run
window, ran on the pre-import frame (mean prior starts 5.1) and was judged on the continuous MAE after the blend. Here
older runs are kept (v1.8 frame: 7.4 prior starts on average) and the judgement uses the E0 standard measures.

Change under test (pace_model.HISTORY_ALPHA): for every per-target history column (rank / gap / lane per corner and the
closing 3F) the summary of a horse's earlier runs is ewm(alpha, ignore_na).mean() over all of them; the race-relative
features and the energy inputs are rebuilt from it. Moves, pace and led stay 5-run means. The 5-run means are kept as
past5_* for the report's past-mean baseline. Training (add_history_features) and pre-race cards (upcoming_rows) use the
same summary.

Discipline (owner's rules + E0): v1.8 frame; E0 procedure (course corner sets, forward pace OOF); selection Jun-Aug 2025
with models fitted before June; report Sep-Dec 2025 pinned to v1.6's 273 races, report only; seeds 42 and 7.
  select   alpha in {None (5-run mean), 0.2, 0.3, 0.5} x RANK_BLEND in {0, 0.25, 0.5, 0.75, 1.0}, contention parameters
           fixed to the E0 pipeline's (fixAB, seed 42). Choice: the largest 2-seed mean of the corner-averaged tau of the
           published formation; within 0.002 of it, the lowest corner-averaged integer MAE.
  report   baseline (5-run mean, blend 0.5) and the chosen setting on the report window, with the contention parameters
           fixed and re-tuned (full pipeline), both seeds; race-paired bootstrap intervals.
Adoption: (1) on the selection window both seeds move tau, integer MAE and the pre-blend continuous MAE (corner averages)
the right way; (2) the report window moves the same way; (3) guards within tolerance on the report window: gaps
+0.05 lengths, lanes +0.01 (MAE; also shown as the ratio to constant 0), last3 +0.01 s, at every corner.

Run:  python scripts/experiment_ewm_history.py select --seed 42      (and --seed 7)
      python scripts/experiment_ewm_history.py report --seed 42 [--retune]   (and --seed 7)
      python scripts/experiment_ewm_history.py summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_data_expansion as ex  # noqa: E402
import pace_model as pm  # noqa: E402
import eval_standard as es  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pm.ROOT
OUT = ROOT / 'outputs' / 'e1-ewm'
FRAME = ROOT / 'outputs' / 'data-expansion-v18' / 'frame_2022.pkl'
PIN = ROOT / 'outputs' / 'data-expansion' / 'frame_before_20261118_0126.pkl'
E0_PARAMS = ROOT / 'outputs' / 'e0-audit' / 'pipeline_fixAB_seed42.json'
ALPHAS = (None, 0.2, 0.3, 0.5)
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEL, REP = ('20250601', '20250601', '20250831'), ('20250901', '20250901', '20251231')
TOL = {'g': 0.05, 'l': 0.01, 'last3': 0.01}


def frame(alpha):
    """The v1.8 frame with the history summary of `alpha` (None = 5-run mean, as stored)."""
    d = pd.read_pickle(FRAME)
    pm.HISTORY_ALPHA = alpha
    d = pm.summarize_history(d)
    d = pm.add_traffic(pm.add_race_relative(d, 'RACE_CODE'))
    return d


def contention_params():
    if E0_PARAMS.exists():
        return json.loads(E0_PARAMS.read_text(encoding='utf-8'))['params']
    return json.loads((ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']


def collect(tab, system, models, races, P, d, fit_end):
    """Per-race rows of the E0 standard measures for one system into a shared RaceTable (races paired by code)."""
    tr = (d.date < fit_end) & d.complete
    for x, corners, numbers, pred, spreads, early in races:
        race, n = x.RACE_CODE.iloc[0], len(numbers)
        con = pm.simulate(numbers, early, pred, spreads, corners, P)
        for c in corners:
            if not pm.labelled(x, c):
                continue
            y = x[f'r{c}'].to_numpy()
            formed = (np.array(con[c]['rank']) - 1) / (n - 1)
            tab.add(f'r{c}.integer', system, race, np.abs(formed - y))
            tab.add_value(f'r{c}.tau', system, race, es.kendall(formed, y))
            tab.add(f'r{c}.model', system, race, np.abs(np.asarray(pred[f'model_r{c}'], float) - y))
            tab.add(f'r{c}.blend', system, race, np.abs(np.asarray(pred[f'r{c}'], float) - y))
            tab.add(f'g{c}', system, race, np.abs(np.asarray(con[c]['gap'], float) - x[f'g{c}'].to_numpy()))
            lab = x[f'l{c}'].to_numpy()
            tab.add(f'l{c}', system, race, np.abs(np.asarray(con[c].get('localLane', con[c]['lane']), float) - lab))
            tab.add(f'l{c}.zero', system, race, np.abs(lab))
        ok = x.last3.notna().to_numpy()
        if ok.any():
            t3 = pm.closing(models, x, con, corners, pred['pace'])[0]
            tab.add('last3', system, race, np.abs(t3[ok] - x.last3.to_numpy()[ok]))


def summarize(tab, system):
    out = {}
    for c in pm.CORNERS:
        if (f'r{c}.tau', system) not in tab.rows:
            continue
        out[f'c{c}'] = {'tau': tab.mean(f'r{c}.tau', system), 'integer': tab.mean(f'r{c}.integer', system),
                        'model': tab.mean(f'r{c}.model', system), 'blend': tab.mean(f'r{c}.blend', system),
                        'gap': tab.mean(f'g{c}', system), 'lane': tab.mean(f'l{c}', system),
                        'laneRatioToZero': tab.mean(f'l{c}', system) / tab.mean(f'l{c}.zero', system),
                        'races': tab.races(f'r{c}.tau', system)}
    out['last3'] = tab.mean('last3', system)
    cs = [v for k, v in out.items() if k.startswith('c')]
    out['mean'] = {k: float(np.mean([v[k] for v in cs])) for k in ('tau', 'integer', 'model', 'blend')}
    return out


def cmd_select(args):
    ex.set_seed(args.seed)
    P = contention_params()
    res = {'seed': args.seed, 'params': P, 'runs': {}}
    for alpha in ALPHAS:
        d = frame(alpha)
        models = pm.fit_models(d, SEL[0])
        for blend in BLENDS:
            pm.RANK_BLEND = blend
            races = pm.evaluation_races(models, d, SEL[1], SEL[2])
            tab = es.RaceTable()
            collect(tab, 'x', models, races, P, d, SEL[0])
            key = f'{alpha}/{blend}'
            res['runs'][key] = summarize(tab, 'x')
            m = res['runs'][key]['mean']
            print(f'seed {args.seed} alpha {alpha} blend {blend}: tau {m["tau"]:.4f} integer {m["integer"]:.4f} '
                  f'model {m["model"]:.4f} races {len(races)}', flush=True)
    pm.RANK_BLEND = 0.5
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'select_seed{args.seed}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


def choose():
    runs = [json.loads((OUT / f'select_seed{s}.json').read_text(encoding='utf-8'))['runs'] for s in (42, 7)]
    keys = list(runs[0])
    tau = {k: np.mean([r[k]['mean']['tau'] for r in runs]) for k in keys}
    integer = {k: np.mean([r[k]['mean']['integer'] for r in runs]) for k in keys}
    best = max(tau.values())
    near = [k for k in keys if tau[k] >= best - 0.002]
    pick = min(near, key=lambda k: integer[k])
    alpha, blend = pick.split('/')
    return (None if alpha == 'None' else float(alpha)), float(blend), {'tau': tau, 'integer': integer, 'near': near}


def cmd_report(args):
    ex.set_seed(args.seed)
    alpha, blend, _ = choose() if args.alpha is None else (float(args.alpha), float(args.blend), None)
    res = {'seed': args.seed, 'alpha': alpha, 'blend': blend, 'retune': args.retune}
    tab = es.RaceTable()
    for system, (a, b) in {'baseline': (None, 0.5), 'candidate': (alpha, blend)}.items():
        d = frame(a)
        pm.RANK_BLEND = b
        if args.retune:
            run = ex.pipeline(d, seed=args.seed)
            P = run['params']
            res[system] = {'params': P, 'contentionTuning': run['contentionTuning'], 'metrics': run['metrics']}
        else:
            P = contention_params()
            res[system] = {'params': P}
        ex.set_seed(args.seed)
        models = pm.fit_models(d, REP[0])
        races = pm.evaluation_races(models, d, REP[1], REP[2])
        collect(tab, system, models, races, P, d, REP[0])
        res[system]['summary'] = summarize(tab, system)
    pm.RANK_BLEND = 0.5
    diffs = {}
    for c in pm.CORNERS:
        for m in ('tau', 'integer', 'model', 'blend') if (f'r{c}.tau', 'baseline') in tab.rows else ():
            diffs[f'r{c}.{m}'] = tab.diff(f'r{c}.{m}', 'candidate', 'baseline')
        for m in (f'g{c}', f'l{c}'):
            if (m, 'baseline') in tab.rows:
                diffs[m] = tab.diff(m, 'candidate', 'baseline')
    diffs['last3'] = tab.diff('last3', 'candidate', 'baseline')
    res['diffs'] = diffs
    OUT.mkdir(parents=True, exist_ok=True)
    name = f'report_seed{args.seed}{"_retune" if args.retune else ""}.json'
    (OUT / name).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    print(json.dumps({'alpha': alpha, 'blend': blend, 'baseline': res['baseline']['summary'], 'candidate': res['candidate']['summary'],
                      'diffs': diffs}, indent=1, default=float), flush=True)


def cmd_summary(args):
    alpha, blend, sel = choose()
    print(f'chosen alpha {alpha} blend {blend}; within 0.002 of the best tau: {sel["near"]}')
    for k in sel['tau']:
        print(f'  {k:>10}: 2-seed tau {sel["tau"][k]:.4f} integer {sel["integer"][k]:.4f}')
    rows = {}
    for p in sorted(OUT.glob('report_seed*.json')):
        rows[p.stem] = json.loads(p.read_text(encoding='utf-8'))
    for name, r in rows.items():
        print(name, 'baseline', json.dumps(r['baseline']['summary']['mean']), 'candidate', json.dumps(r['candidate']['summary']['mean']))
        for k, v in r['diffs'].items():
            print(f'   {k:>12}: {v["diff"]:+.4f} [{v["lo"]:+.4f}, {v["hi"]:+.4f}]')
    (OUT / 'result.json').write_text(json.dumps({'choice': {'alpha': alpha, 'blend': blend, **sel}, 'reports': rows}, indent=1,
                                                default=float), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['select', 'report', 'summary'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alpha')
    ap.add_argument('--blend')
    ap.add_argument('--retune', action='store_true')
    args = ap.parse_args()
    ex.PINNED = set(ex.validation_codes(pd.read_pickle(PIN)))
    pm.EVAL_CORNERS, pm.PACE_OOF = 'course', 'forward'
    {'select': cmd_select, 'report': cmd_report, 'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
