"""Shared harness of the F1-F3 experiments (追加指示10, 2026-11-19), built on experiment_ewm_history (E1) and the E0 standard measures.

An experiment script defines VARIANTS = {name: Variant(columns builder, extra feature columns)} next to the baseline 'base' (v1.9
as is, the stored v1.8 frame rebuilt with HISTORY_ALPHA 0.2, RANK_BLEND 0.75) and calls main(). The harness then does what the owner
fixed for every experiment:

  select   Jun-Aug 2025 (models fitted before June), contention parameters fixed to the reviewed-procedure pipeline (seed 42),
           every variant against 'base', seeds 42 and 7.
  report   Sep-Dec 2025 pinned to v1.6's 273 races (report only: never used to choose), 'base' against the chosen variant, with the
           contention layer fixed and with it re-tuned on Jun-Aug (the published procedure), seeds 42 and 7; race-paired bootstrap
           (2,000 draws) for every difference.
  decide   the three adoption conditions: (1) selection: both seeds move tau, integer MAE and the pre-blend continuous MAE of the
           focus corners the right way and the corner-averaged tau does not fall; (2) the report window moves the same way in both
           seeds; (3) guards on the report window (re-tuned): gaps +0.05 lengths, lanes +0.01 (MAE), last3 +0.01 s, at every corner.
           0.002 is about one standard error, so a difference is never accepted on its size alone.

Procedure: the reviewed one (EVAL_CORNERS 'course', PACE_OOF 'forward'). The stored v1.8 frame is used on purpose: the DB has grown
(2020/2021 imported, 2026-11-19) and a rebuilt frame would be another data-expansion experiment, not this one.
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_standard as es  # noqa: E402
import experiment_data_expansion as ex  # noqa: E402
import experiment_ewm_history as e1  # noqa: E402
import pace_model as pm  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pm.ROOT
BASE_FEATURES = list(pm.FEATURES)
MOVE_EXTRA = pm.MOVE_FEATURES[len(pm.FEATURES):]
SEL, REP = e1.SEL, e1.REP
TOL = e1.TOL
CACHE = ROOT / 'outputs' / 'f0-context' / 'frame_v19.pkl'
PARAMS = ROOT / 'outputs' / 'e0-audit' / 'pipeline_reviewed_seed42.json'


@dataclass
class Variant:
    features: list = field(default_factory=list)        # extra FEATURES columns (also reach the closing and move models)
    build: Callable = None                               # build(d) -> None: adds the columns to the frame in place
    pace_features: list = field(default_factory=list)   # extra PACE_FEATURES columns (race-level; must exist in race_level's output)
    note: str = ''


def v19_frame(refresh=False):
    """The stored v1.8 frame with the v1.9 history summary (alpha 0.2), the E0-reviewed corner/pace procedure."""
    if CACHE.exists() and not refresh:
        d = pd.read_pickle(CACHE)
    else:
        d = e1.frame(0.2)
        d.to_pickle(CACHE)
    pm.HISTORY_ALPHA = 0.2
    return d


def set_features(extra=(), pace_extra=()):
    pm.FEATURES = BASE_FEATURES + [f for f in extra if f not in BASE_FEATURES]
    pm.CLOSING_BASE = pm.FEATURES + ['going']
    pm.LAST3_FEATURES = pm.CLOSING_BASE + pm.TRAFFIC + pm.ENERGY
    pm.MOVE_FEATURES = pm.FEATURES + MOVE_EXTRA
    pm.PACE_FEATURES = list(PACE_BASE) + [f for f in pace_extra if f not in PACE_BASE]


PACE_BASE = list(pm.PACE_FEATURES)


def setup(params=PARAMS):
    pm.EVAL_CORNERS, pm.PACE_OOF = 'course', 'forward'
    pm.RANK_BLEND = 0.75
    e1.E0_PARAMS = params
    ex.PINNED = set(ex.validation_codes(pd.read_pickle(e1.PIN)))


def prepare(d, v):
    if v.build is not None:
        v.build(d)
    set_features(v.features, v.pace_features)


def cmd_select(args, variants):
    d = v19_frame()
    out = e1.OUT
    P = e1.contention_params()
    res = {'seed': args.seed, 'params': P, 'runs': {}}
    f = out / f'select_seed{args.seed}.json'
    only = set(args.only.split(',')) if args.only else None
    if only and f.exists():  # re-run some variants and keep the others
        res = json.loads(f.read_text(encoding='utf-8'))
    for name, v in variants.items():
        if only and name not in only:
            continue
        ex.set_seed(args.seed)
        prepare(d, v)
        models = pm.fit_models(d, SEL[0])
        races = pm.evaluation_races(models, d, SEL[1], SEL[2])
        tab = es.RaceTable()
        e1.collect(tab, 'x', models, races, P, d, SEL[0])
        res['runs'][name] = e1.summarize(tab, 'x')
        res['runs'][name]['pace'] = pace_row(races)
        m = res['runs'][name]['mean']
        print(f'seed {args.seed} {name}: tau {m["tau"]:.4f} integer {m["integer"]:.4f} model {m["model"]:.4f} '
              f'last3 {res["runs"][name]["last3"]:.4f} races {len(races)}', flush=True)
    set_features()
    out.mkdir(parents=True, exist_ok=True)
    (out / f'select_seed{args.seed}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


def pace_row(races):
    pp = np.array([pred['pace'] for x, corners, numbers, pred, *_ in races])
    pa = np.array([x.pace.iloc[0] for x, *_ in races])
    ok = ~np.isnan(pa)
    return {'races': int(ok.sum()), 'mae': float(np.mean(np.abs(pp[ok] - pa[ok]))), 'corr': float(np.corrcoef(pp[ok], pa[ok])[0, 1])}


def cmd_report(args, variants):
    d = v19_frame()
    cand = args.variant
    res = {'seed': args.seed, 'candidate': cand, 'retune': args.retune}
    tab = es.RaceTable()
    for system, name in (('baseline', 'base'), ('candidate', cand)):
        ex.set_seed(args.seed)
        prepare(d, variants[name])
        if args.retune:
            run = ex.pipeline(d, seed=args.seed)
            P = run['params']
            res[system] = {'params': P, 'contentionTuning': run['contentionTuning'], 'metrics': run['metrics']}
        else:
            P = e1.contention_params()
            res[system] = {'params': P}
        ex.set_seed(args.seed)
        models = pm.fit_models(d, REP[0])
        races = pm.evaluation_races(models, d, REP[1], REP[2])
        e1.collect(tab, system, models, races, P, d, REP[0])
        res[system]['summary'] = e1.summarize(tab, system)
        res[system]['pace'] = pace_row(races)
        res[system]['standard'] = pm.standard_metrics(models, races, P, d, REP[0])
    set_features()
    diffs = {}
    for c in pm.CORNERS:
        if (f'r{c}.tau', 'baseline') in tab.rows:
            for m in ('tau', 'integer', 'places', 'model', 'blend'):
                diffs[f'r{c}.{m}'] = tab.diff(f'r{c}.{m}', 'candidate', 'baseline')
        for m in (f'g{c}', f'l{c}'):
            if (m, 'baseline') in tab.rows:
                diffs[m] = tab.diff(m, 'candidate', 'baseline')
    diffs['last3'] = tab.diff('last3', 'candidate', 'baseline')
    res['diffs'] = diffs
    e1.OUT.mkdir(parents=True, exist_ok=True)
    name = f'report_{slug(cand)}_seed{args.seed}{"_retune" if args.retune else ""}.json'
    (e1.OUT / name).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    print(name, json.dumps({k: v for k, v in res['candidate']['summary'].items() if k in ('mean', 'last3')}, default=float), flush=True)


def corner_mean(s, key, corners):
    return float(np.mean([s[f'c{c}'][key] for c in corners if f'c{c}' in s]))


def decide(variants, focus, last3_focus=False, exclude=(), choose_by='tau'):
    """Print and save the selection table and the three adoption conditions for the variant chosen on the selection window."""
    sel = [json.loads((e1.OUT / f'select_seed{s}.json').read_text(encoding='utf-8'))['runs'] for s in (42, 7)]
    names = [n for n in variants if n != 'base' and n not in exclude]
    rows = {}
    for n in names:
        row = {}
        for si, s in enumerate((42, 7)):
            b, v = sel[si]['base'], sel[si][n]
            row[s] = {'tauFocus': corner_mean(v, 'tau', focus) - corner_mean(b, 'tau', focus),
                      'integerFocus': corner_mean(v, 'integer', focus) - corner_mean(b, 'integer', focus),
                      'modelFocus': corner_mean(v, 'model', focus) - corner_mean(b, 'model', focus),
                      'tauAll': v['mean']['tau'] - b['mean']['tau'], 'integerAll': v['mean']['integer'] - b['mean']['integer'],
                      'last3': v['last3'] - b['last3'],
                      'gap': {c: v[f'c{c}']['gap'] - b[f'c{c}']['gap'] for c in pm.CORNERS if f'c{c}' in v},
                      'lane': {c: v[f'c{c}']['lane'] - b[f'c{c}']['lane'] for c in pm.CORNERS if f'c{c}' in v}}
        rows[n] = row
        print(f'{n:>22}: ' + ' | '.join(f'seed {s}: tauF {r["tauFocus"]:+.4f} intF {r["integerFocus"]:+.4f} modelF {r["modelFocus"]:+.4f} '
                                         f'tauAll {r["tauAll"]:+.4f} last3 {r["last3"]:+.4f}' for s, r in row.items()), flush=True)
    key = {'tau': lambda n: np.mean([rows[n][s]['tauFocus'] for s in (42, 7)])}[choose_by]
    best = max(names, key=key)
    ok1 = all(rows[best][s]['tauFocus'] > 0 and rows[best][s]['integerFocus'] < 0 and rows[best][s]['modelFocus'] < 0
              and rows[best][s]['tauAll'] > -0.002 and (not last3_focus or rows[best][s]['last3'] < 0) for s in (42, 7))
    print(f'chosen on the selection window: {best} (2-seed focus tau gain {key(best):+.4f}); condition 1 (both seeds, focus tau / integer / '
          f'continuous, corner-averaged tau not falling): {ok1}', flush=True)
    return best, rows, ok1


def slug(name):
    return ''.join(ch if ch.isalnum() else '-' for ch in name).strip('-')


def decide_report(best, focus, retune=True, last3_focus=False):
    suffix = '_retune' if retune else ''
    reps = [json.loads((e1.OUT / f'report_{slug(best)}_seed{s}{suffix}.json').read_text(encoding='utf-8')) for s in (42, 7)]
    out = {'candidate': best, 'seeds': {}}
    ok2, ok3 = True, True
    for s, r in zip((42, 7), reps):
        b, c = r['baseline']['summary'], r['candidate']['summary']
        row = {'tauFocus': corner_mean(c, 'tau', focus) - corner_mean(b, 'tau', focus),
               'integerFocus': corner_mean(c, 'integer', focus) - corner_mean(b, 'integer', focus),
               'modelFocus': corner_mean(c, 'model', focus) - corner_mean(b, 'model', focus),
               'tauAll': c['mean']['tau'] - b['mean']['tau'], 'last3': c['last3'] - b['last3'],
               'gap': {k: c[f'c{k}']['gap'] - b[f'c{k}']['gap'] for k in pm.CORNERS if f'c{k}' in c},
               'lane': {k: c[f'c{k}']['lane'] - b[f'c{k}']['lane'] for k in pm.CORNERS if f'c{k}' in c},
               'laneRatioToZero': {k: (c[f'c{k}']['laneRatioToZero'], b[f'c{k}']['laneRatioToZero']) for k in pm.CORNERS if f'c{k}' in c},
               'diffs': {k: v for k, v in r['diffs'].items()}}
        row['same'] = row['tauFocus'] > 0 and row['integerFocus'] < 0 and row['modelFocus'] < 0 and (not last3_focus or row['last3'] < 0)
        row['guards'] = (max(row['gap'].values()) <= TOL['g'] and max(row['lane'].values()) <= TOL['l'] and row['last3'] <= TOL['last3'])
        ok2, ok3 = ok2 and row['same'], ok3 and row['guards']
        out['seeds'][s] = row
        print(f'seed {s} report: tauF {row["tauFocus"]:+.4f} intF {row["integerFocus"]:+.4f} modelF {row["modelFocus"]:+.4f} tauAll {row["tauAll"]:+.4f} '
              f'last3 {row["last3"]:+.4f} gaps {max(row["gap"].values()):+.3f} lanes {max(row["lane"].values()):+.4f} | same direction {row["same"]} guards {row["guards"]}',
              flush=True)
        for k in [f'r{c}.tau' for c in focus] + [f'r{c}.integer' for c in focus]:
            if k in r['diffs']:
                v = r['diffs'][k]
                print(f'      {k:>12}: {v["diff"]:+.4f} [{v["lo"]:+.4f}, {v["hi"]:+.4f}] ({v["races"]} races)', flush=True)
    out['condition2'], out['condition3'] = ok2, ok3
    return out


def main(variants, focus, out_name, extra_cmds=None, last3_focus=False, exclude=()):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['select', 'report', 'decide'] + list(extra_cmds or {}))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variant')
    ap.add_argument('--only', help='select: only these variants (comma separated), the others stay as recorded')
    ap.add_argument('--retune', action='store_true')
    args = ap.parse_args()
    e1.OUT = ROOT / 'outputs' / out_name
    setup()
    if extra_cmds and args.cmd in extra_cmds:
        return extra_cmds[args.cmd](args)
    if args.cmd == 'select':
        cmd_select(args, variants)
    elif args.cmd == 'report':
        cmd_report(args, variants)
    else:
        best, rows, ok1 = decide(variants, focus, last3_focus, exclude)
        res = {'chosen': best, 'selection': rows, 'condition1': ok1}
        for retune in (False, True):
            try:
                res['report_retune' if retune else 'report_fixed'] = decide_report(best, focus, retune, last3_focus)
            except FileNotFoundError:
                pass
        e1.OUT.mkdir(parents=True, exist_ok=True)
        (e1.OUT / 'result.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
