"""v3 vs v2.0 on the same races and horses (race-paired bootstrap), for one or more v3 runs.

  python -I scripts/v3/evaluate.py s1_named s1_all      -> outputs/v3/eval_<names>.json and a printed table

For every run name, window (sel, rep) and seed (42, 7) present in outputs/v3, the v3 predictions are paired with
v2.0's (baseline_v20.py) on the races both scored (rep: v1.6's 273 races) and the horses both have:

  last3      MAE of the closing 3F (s), normal finishers with a recorded 3F; v2.0 = its published traffic-aware closing
  finTau     within-race Kendall tau of the predicted finish against the official placing (normal finishers)
  finGap     MAE of the margin behind the winner (s); v2.0 = race_output's finish estimate / 5 lengths per second
  winner     share of races whose predicted winner won
  r<c>.tau   within-race Kendall tau of the corner order (v2.0 = its published formation) on the corners v2.0 scores
             (S1 may lose on these: no interaction yet)
Differences are v3 - v2.0 with a 95% race-paired bootstrap interval (2,000 draws).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import eval_standard as es  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / 'v3'
SEEDS = (42, 7)


def actuals():
    runs = pd.read_pickle(OUT / 'runs.pkl')
    return runs.set_index(['RACE_CODE', 'UMABAN'])


def as_base(name, window, seed):
    """A v3 run in the columns of a v2.0 baseline file (so a later stage can be compared with an earlier one)."""
    v20 = pd.read_pickle(OUT / f'v20_{window}_seed{seed}.pkl')
    corners = v20.drop_duplicates('RACE_CODE').set_index('RACE_CODE').corners
    v = pd.read_pickle(OUT / f'{name}_{window}_seed{seed}.pkl')
    v = v[v.RACE_CODE.isin(corners.index)].copy()
    v['corners'] = v.RACE_CODE.map(corners)
    v['finSeconds'] = v['T'] - v.groupby('RACE_CODE')['T'].transform('min')
    for c in (1, 2, 3, 4):
        v[f'rank{c}'] = v[f'ct{c}']
    return v


def table(name, window, seed, act, base_name='v20'):
    base = pd.read_pickle(OUT / f'v20_{window}_seed{seed}.pkl') if base_name == 'v20' else as_base(base_name, window, seed)
    v3 = pd.read_pickle(OUT / f'{name}_{window}_seed{seed}.pkl')
    common = sorted(set(base.RACE_CODE) & set(v3.RACE_CODE))
    tab = es.RaceTable()
    for code in common:
        b = base[base.RACE_CODE == code].set_index('UMABAN')
        v = v3[v3.RACE_CODE == code].set_index('UMABAN')
        nums = [n for n in b.index if n in v.index]
        a = act.loc[[(code, n) for n in nums]].reset_index(level=0, drop=True)
        normal = a.normal.to_numpy() & a.fin.notna().to_numpy()
        keys = list(nums)
        l3 = a.last3.to_numpy()
        ok3 = normal & ~np.isnan(l3)
        for sysname, p in (('v20', b.loc[nums, 'last3'].to_numpy()), ('v3', v.loc[nums, 'last3'].to_numpy())):
            tab.add('last3', sysname, code, np.where(ok3, np.abs(p - l3), np.nan), keys)
        fin_v3 = v.loc[nums, 'T'].to_numpy()
        fin = {'v20': b.loc[nums, 'finSeconds'].to_numpy(), 'v3': fin_v3 - np.nanmin(fin_v3)}
        gap = a.finish_gap.to_numpy()
        place = a.fin.to_numpy()
        for sysname, p in fin.items():
            tab.add('finGap', sysname, code, np.where(normal & ~np.isnan(gap), np.abs(p - gap), np.nan), keys)
            if normal.sum() >= 3:
                tab.add_value('finTau', sysname, code, es.kendall(p[normal], place[normal]))
                tab.add_value('winner', sysname, code, float(place[normal][np.argmin(p[normal])] == 1))
        corners = b.corners.iloc[0]
        for c in corners:
            y = a[f'r{c}'].to_numpy()
            if np.isnan(y).any() or f'ct{c}' not in v:
                continue
            ct = v.loc[nums, f'ct{c}'].to_numpy()
            if np.isnan(ct).any():
                continue
            tab.add_value(f'r{c}.tau', 'v20', code, es.kendall(b.loc[nums, f'rank{c}'].to_numpy(), y))
            tab.add_value(f'r{c}.tau', 'v3', code, es.kendall(ct, y))
    out = {'races': len(common), 'baseRaces': int(base.RACE_CODE.nunique()), 'v3Races': int(v3.RACE_CODE.nunique())}
    for m in ['last3', 'finTau', 'finGap', 'winner'] + [f'r{c}.tau' for c in (1, 2, 3, 4)]:
        if (m, 'v3') not in tab.rows:
            continue
        out[m] = {'v20': tab.mean(m, 'v20'), 'v3': tab.mean(m, 'v3'), 'diff': tab.diff(m, 'v3', 'v20')}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+')
    ap.add_argument('--base', default='v20', help="'v20' or an earlier v3 run name (e.g. s1_all_g); its columns show as 'v20'")
    args = ap.parse_args()
    names = args.names
    act = actuals()
    res = {}
    for name in names:
        for window in ('sel', 'rep'):
            for seed in SEEDS:
                if not (OUT / f'{name}_{window}_seed{seed}.pkl').exists():
                    continue
                if args.base != 'v20' and not (OUT / f'{args.base}_{window}_seed{seed}.pkl').exists():
                    continue
                r = table(name, window, seed, act, args.base)
                res[f'{name}/{window}/seed{seed}'] = r
                line = [f'{name:10} {window} seed {seed:2d} races {r["races"]:3d}']
                for m in ('last3', 'finTau', 'finGap', 'winner', 'r1.tau', 'r3.tau', 'r4.tau'):
                    if m in r:
                        d = r[m]['diff']
                        line.append(f'{m} {r[m]["v20"]:.3f}->{r[m]["v3"]:.3f} ({d["diff"]:+.3f} [{d["lo"]:+.3f},{d["hi"]:+.3f}])')
                print('  '.join(line))
    (OUT / f'eval_{"_".join(names)}{"" if args.base == "v20" else "_vs_" + args.base}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
