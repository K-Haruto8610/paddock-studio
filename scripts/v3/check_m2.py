"""M2 check (design 4 / S2): does the forecast reproduce "a fast pace stops the front"?

  python -I scripts/v3/check_m2.py s2_all_g s1_all_g      -> outputs/v3/m2_<names>.json and a printed table

Per race: pace = the leader's time at 600 m from the start minus the usual time of the course (venue x course kind x
distance, median of the races before the window's fit date; negative = fast), and fb = mean closing 3F (against the
race median) of the front third minus that of the back third, the thirds taken at the last recorded corner (4th, else
3rd). The slope of fb on pace (seconds of closing difference per second of pace) is taken three ways on the same races:
from the actual records, from the v3 run (its own pace, positions and closing) and from v2.0 (its pace forecast, its
published 4th-corner formation and closing). 95% intervals by resampling races (2,000 draws).

Pass rule (fixed before the S2 runs were scored): same sign as the actual slope, at least half its size, and the two
95% intervals overlap.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / 'v3'
FIT = {'sel': '20250601', 'rep': '20250901'}


def time_at(lead, dist, x=600.0):
    if not isinstance(lead, np.ndarray):
        return np.nan
    n = len(lead)
    pos = np.array([dist - 200 * (n - 1 - i) for i in range(n)], float)
    return float(np.interp(x, np.r_[0, pos], np.r_[0, lead]))


def front_back(pos, l3):
    ok = ~(np.isnan(pos) | np.isnan(l3))
    pos, l3 = pos[ok], l3[ok]
    if len(pos) < 6:
        return np.nan
    share = (np.argsort(np.argsort(pos)) / (len(pos) - 1))
    rel = l3 - np.median(l3)
    f, b = rel[share <= 1 / 3], rel[share >= 2 / 3]
    return float(f.mean() - b.mean()) if len(f) and len(b) else np.nan


def slope(p, f, reps=2000, seed=812):
    ok = ~(np.isnan(p) | np.isnan(f))
    p, f = p[ok], f[ok]
    point = float(np.polyfit(p, f, 1)[0])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        i = rng.integers(len(p), size=len(p))
        boot.append(np.polyfit(p[i], f[i], 1)[0])
    lo, hi = np.quantile(boot, [.025, .975])
    return {'slope': point, 'lo': float(lo), 'hi': float(hi), 'races': int(ok.sum())}


def check(name, window, seed, races, runs):
    pred = pd.read_pickle(OUT / f'{name}_{window}_seed{seed}.pkl')
    base = pd.read_pickle(OUT / f'v20_{window}_seed{seed}.pkl')
    hist = races[(races.date < FIT[window]) & races.complete & (races.kind != 'straight')].copy()
    hist['t600lead'] = [time_at(l, d) for l, d in zip(hist.lead, hist.distance)]
    norm = hist.groupby(['venue', 'kind', 'distance']).t600lead.median()
    rows = []
    for code in sorted(set(pred.RACE_CODE) & set(base.RACE_CODE)):
        rc = races.loc[code]
        key = (rc.venue, rc.kind, rc.distance)
        if key not in norm.index:
            continue
        nrm = norm[key]
        v = pred[pred.RACE_CODE == code].set_index('UMABAN')
        b = base[base.RACE_CODE == code].set_index('UMABAN')
        a = runs[runs.RACE_CODE == code].set_index('UMABAN')
        nums = [n for n in v.index if n in b.index and n in a.index]
        a, v, b = a.loc[nums], v.loc[nums], b.loc[nums]
        ok = (a.normal & a.last3.notna()).to_numpy()
        apos = a.r4.fillna(a.r3).to_numpy()
        vpos = (v.ct4 if v.ct4.notna().all() else v.ct3).to_numpy()
        c_last = max(b.corners.iloc[0])
        bpos = b[f'rank{c_last}'].to_numpy()
        msk = lambda x: np.where(ok, x, np.nan)
        rows.append({'race': code,
                     'pace_actual': time_at(rc.lead, rc.distance) - nrm, 'fb_actual': front_back(msk(apos), msk(a.last3.to_numpy())),
                     'pace_v3': float(v.lead600.iloc[0]) - nrm, 'fb_v3': front_back(msk(vpos), msk(v.last3.to_numpy())),
                     'pace_v20': float(b.pace.iloc[0]), 'fb_v20': front_back(msk(bpos), msk(b.last3.to_numpy()))})
    t = pd.DataFrame(rows)
    out = {k: slope(t[f'pace_{k}'].to_numpy(), t[f'fb_{k}'].to_numpy()) for k in ('actual', 'v3', 'v20')}
    a, p = out['actual'], out['v3']
    out['pass'] = bool(np.sign(a['slope']) == np.sign(p['slope']) and abs(p['slope']) >= abs(a['slope']) / 2
                       and p['lo'] <= a['hi'] and a['lo'] <= p['hi'])
    out['paceCorr'] = float(t[['pace_actual', 'pace_v3']].corr().iloc[0, 1])
    out['paceCorrV20'] = float(t[['pace_actual', 'pace_v20']].corr().iloc[0, 1])
    return out


def main():
    names = sys.argv[1:]
    races = pd.read_pickle(OUT / 'races.pkl')
    runs = pd.read_pickle(OUT / 'runs.pkl')
    res = {}
    for name in names:
        for window in ('sel', 'rep'):
            for seed in (42, 7):
                if not (OUT / f'{name}_{window}_seed{seed}.pkl').exists():
                    continue
                r = check(name, window, seed, races, runs)
                res[f'{name}/{window}/seed{seed}'] = r
                f = lambda k: f'{r[k]["slope"]:+.3f} [{r[k]["lo"]:+.3f},{r[k]["hi"]:+.3f}]'
                print(f'{name:10} {window} seed {seed:2d} races {r["actual"]["races"]}  actual {f("actual")}  v3 {f("v3")}  '
                      f'v2.0 {f("v20")}  pass {r["pass"]}  pace corr v3 {r["paceCorr"]:.3f} v2.0 {r["paceCorrV20"]:.3f}')
    (OUT / f'm2_{"_".join(names)}.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
