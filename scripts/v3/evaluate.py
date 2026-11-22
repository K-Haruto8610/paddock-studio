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
  g<c>       MAE of the lengths behind the corner's leader against the passing string (data.py g1..g4); v2.0 = its
             published formation's gap; v3 = its corner times x the learnt lengths per second (S1/S2 runs: x 5)
  l<c>       (S3 runs) MAE of the lane inside the abreast group (lanes.py, E0's label) against the constant-0 forecast:
             'v20' columns are the constant 0 here (design S3: lanes must beat the constant, v2.0 itself does not).
             v3's lane: horses whose corner times chain within ABREAST s form a group; the lane is the place of the
             horse's lateral position y in its group, from the inside
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
ABREAST = 0.1   # seconds: horses closer than this at a corner are one abreast group (the passing string's brackets)


def actuals():
    runs = pd.read_pickle(OUT / 'runs.pkl')
    lanes = pd.read_pickle(OUT / 'lanes.pkl')
    runs = runs.merge(lanes, on=['RACE_CODE', 'UMABAN'], how='left')
    return runs.set_index(['RACE_CODE', 'UMABAN'])


def v3_lengths(v, c):
    if f'gl{c}' in v:
        return v[f'gl{c}'].to_numpy()
    ct = v[f'ct{c}'].to_numpy()
    return (ct - np.nanmin(ct)) * 5.0


def v3_lanes(ct, y):
    """Lane inside the abreast group from corner times and lateral positions (0 = innermost or alone)."""
    order = np.argsort(ct)
    lane = np.zeros(len(ct))
    group = [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if ct[b] - ct[a] < ABREAST:
            group.append(b)
            continue
        lane[group] = np.argsort(np.argsort(y[group], kind='stable'), kind='stable')
        group = [b]
    lane[group] = np.argsort(np.argsort(y[group], kind='stable'), kind='stable')
    return lane


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
        v[f'gap{c}'] = v.groupby('RACE_CODE', group_keys=False).apply(lambda g: pd.Series(v3_lengths(g, c), index=g.index))
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
            g = a[f'g{c}'].to_numpy()
            if not np.isnan(g).any():
                tab.add(f'g{c}', 'v20', code, np.abs(b.loc[nums, f'gap{c}'].to_numpy() - g), keys)
                tab.add(f'g{c}', 'v3', code, np.abs(v3_lengths(v.loc[nums], c) - g), keys)
            lab = a[f'l{c}'].to_numpy()
            if f'y{c}' in v and not np.isnan(lab).any():
                tab.add(f'l{c}', 'v20', code, np.abs(lab), keys)
                lane = v.loc[nums, f'lane{c}'].to_numpy() if f'lane{c}' in v else v3_lanes(ct, v.loc[nums, f'y{c}'].to_numpy())
                tab.add(f'l{c}', 'v3', code, np.abs(lane - lab), keys)
    out = {'races': len(common), 'baseRaces': int(base.RACE_CODE.nunique()), 'v3Races': int(v3.RACE_CODE.nunique())}
    for m in ['last3', 'finTau', 'finGap', 'winner'] + [f'{k}{c}' for k in ('r', 'g', 'l') for c in (1, 2, 3, 4)]:
        m = m.replace('r1', 'r1.tau').replace('r2', 'r2.tau').replace('r3', 'r3.tau').replace('r4', 'r4.tau') if m[0] == 'r' else m
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
                for m in ('last3', 'finTau', 'finGap', 'winner', 'r1.tau', 'r3.tau', 'r4.tau', 'g3', 'g4', 'l3', 'l4'):
                    if m in r:
                        d = r[m]['diff']
                        line.append(f'{m} {r[m]["v20"]:.3f}->{r[m]["v3"]:.3f} ({d["diff"]:+.3f} [{d["lo"]:+.3f},{d["hi"]:+.3f}])')
                print('  '.join(line))
    (OUT / f'eval_{"_".join(names)}{"" if args.base == "v20" else "_vs_" + args.base}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
