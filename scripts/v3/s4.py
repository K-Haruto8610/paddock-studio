"""v3 S4: the published form of the race simulator (the split form fixed when S3 was closed, 2026-11-23).

  point forecast   S2_gwe, deterministic (no noise): corner order, corner gaps = its corner times x 5 lengths/s, closing
                   3F, finish order and margin, race pace (the leader's time 600 m from the start minus the course norm)
  sampled runs     S2_gwe's abilities fixed, run with S3's rules (lanes, blocking, corner distance) and FORM 0.15, WISHN 0
                   (trained on 4 runs per race: hold and gap_scale only, train.py --train-only hold,gap_scale; 32 runs
                   per race here). Used ONLY for the ranges (rankRange / last3Range) and the lane, which is a geometric
                   reference display, not a forecast

  python -I scripts/v3/s4.py runs --window sel|rep|pub --seed 42|7
        point + sampled runs of the window's races -> outputs/v3/s4_point_<w>_seed<s>.pkl, s4_samples_<w>_seed<s>.pkl
  python -I scripts/v3/s4.py calibrate --seed 42|7
        range margins on the selection window (models fitted before June 2025): the shown coverage closest to 50%, the
        v1.6 procedure (pace_model.calibrate_intervals) on the runs' quantiles -> outputs/v3/s4_margins_seed<s>.json
  python -I scripts/v3/s4.py evaluate
        E0's measures, v3 (S4) against v2.0 on the same races and horses, race-paired bootstrap, both windows and seeds,
        and the ranges' coverage -> outputs/v3/s4_eval.json
  python -I scripts/v3/s4.py forecast --window rep|pub --seed 42 [--out FILE]
        paddock.pace-forecast/v1 (revision 1.1) for the races of public/data/forecasts.json (past races from the run
        table, upcoming ones from data/entries cards) -> outputs/v3/pace_forecasts_v3_<w>.json unless --out is given.
        public/data is written only when --out names it (after S4 passes)

The ranges: per horse the 25% and 75% quantiles over the 32 runs of the normalised corner place / closing 3F, a margin
added to both sides, rounded to whole places / 0.1 s and widened to contain the point forecast (pace_model.shown_range).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import courses as cs  # noqa: E402
import evaluate as ev  # noqa: E402
import features as ft  # noqa: E402
import sim  # noqa: E402
import train as tr  # noqa: E402

OUT = tr.OUT
ROOT = tr.ROOT
POINT, SPLIT = 's2_all_gwe', 's3_all_gwe_split'
K_RUNS = 32
LPS = 5.0
Q = (0.25, 0.75)
GRID = {'r': np.round(np.arange(-.2, .4001, .005), 3), 'last3': np.round(np.arange(-.2, 2.0001, .02), 2)}
TARGET_COVERAGE = 0.5
COVERAGE_BAND = (0.40, 0.60)   # pace_model.COVERAGE_BAND
SEEDS = (42, 7)
VERSION = '3.0'


def mode(kind):
    """sim settings of the two halves (S2_gwe's: early contention span, wish merged into intent)."""
    sim.LOSS, sim.ABILITY_SET, sim.CONTEND_SPAN, sim.WISH, sim.TRAFFIC = 'centred', 'six', 'early', 'intent', 'full'
    sim.START, sim.WISHN, sim.READ = 0.0, 0.0, 'mean'
    if kind == 'point':
        sim.STAGE, sim.FORM, sim.SAMPLES = 's2', 0.0, 1
    else:
        sim.STAGE, sim.FORM, sim.SAMPLES = 's3', 0.15, 4


def make_data(races, r, codes):
    target = np.flatnonzero(r.RACE_CODE.isin(set(codes)).to_numpy())
    X, M = ft.history_tensor(r, pd.Series(True, index=r.index), target)
    C = ft.condition_matrix(r.iloc[target])
    rowpos = np.full(len(r), -1)
    rowpos[target] = np.arange(len(target))
    return tr.Data(races, r, list(codes), X, M, C, rowpos), X, C


def load_model(name, X, C):
    m = sim.Model(X.shape[2], C.shape[1]).to(tr.DEVICE)
    missing, unexpected = m.load_state_dict(torch.load(OUT / f'{name}.pt', map_location=tr.DEVICE), strict=False)
    assert set(missing) <= {'physics.hold', 'physics.gap_scale'} and not unexpected, (missing, unexpected)  # S2 files predate S3
    m.eval()
    return m


@torch.no_grad()
def point_runs(model, data, size=48):
    """One deterministic run per race: per horse T, t600, last3, corner times ct1..4, the leader's 600 m-from-start time."""
    mode('point')
    rows = []
    for ks in data.batches(size):
        b = data.batch(ks)
        t, ab = model(b, samples=1)
        T, t600, last3, corners, _ = sim.read_out(t, b)
        early = t[:, :, int(600 / sim.DX)].masked_fill(~b['present'], np.inf).min(1).values
        T, t600, last3, corners, early = (x.cpu().numpy() for x in (T, t600, last3, corners, early))
        for bi, k in enumerate(ks):
            for h in np.flatnonzero(data.present[k]):
                row = {'RACE_CODE': data.codes[k], 'UMABAN': int(data.umaban[k, h]), 'T': float(T[bi, h]), 't600': float(t600[bi, h]),
                       'last3': float(last3[bi, h]), 'lead600': float(early[bi])}
                for c in range(4):
                    row[f'ct{c + 1}'] = float(corners[bi, h, c]) if data.corner_idx[k, c] >= 0 else np.nan
                for a in sim.ABILITIES:
                    row[f'ab_{a}'] = float(ab[a][bi, h])
                rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def sampled_runs(model, data, K=K_RUNS, size=8, seed=12345):
    """K stochastic S3 runs per race: {code: {'umaban' [n], 'ct' [K, n, 4], 'last3' [K, n], 'lane' [K, n, 4]}}."""
    mode('split')
    gen = torch.Generator(device=tr.DEVICE).manual_seed(seed)
    out = {}
    for ks in data.batches(size):
        b = data.batch(ks)
        t, _ = model(b, samples=K, gen=gen)
        bk = sim.repeat_batch(b, K)
        _, _, last3, corners, _ = sim.read_out(t, bk)
        idx = bk['corner_idx'][:, None, :].expand(t.shape[0], t.shape[1], 4).clamp(min=0)
        y = torch.gather(model.y, 2, idx)
        n, H = len(ks), t.shape[1]
        ck = corners.view(K, n, H, 4).cpu().numpy()
        yk = y.view(K, n, H, 4).cpu().numpy()
        l3 = last3.view(K, n, H).cpu().numpy()
        for bi, k in enumerate(ks):
            hs = np.flatnonzero(data.present[k])
            lane = np.full((K, len(hs), 4), np.nan)
            ct = np.full((K, len(hs), 4), np.nan)
            for c in range(4):
                if data.corner_idx[k, c] < 0:
                    continue
                ct[:, :, c] = ck[:, bi, hs, c]
                for j in range(K):
                    lane[j, :, c] = ev.v3_lanes(ck[j, bi, hs, c], yk[j, bi, hs, c])
            out[data.codes[k]] = {'umaban': data.umaban[k, hs].astype(int), 'ct': ct, 'last3': l3[:, bi, hs], 'lane': lane}
    return out


def window_codes(races, window):
    fit_end, start, end = tr.WINDOWS[window]
    return races[races.complete & races.named & (races.kind != 'straight') & races.date.between(start, end)].index.tolist()


def cmd_runs(args):
    races, r = tr.load_tables()
    codes = window_codes(races, args.window)
    data, X, C = make_data(races, r, codes)
    point = point_runs(load_model(f'{POINT}_{args.window}_seed{args.seed}', X, C), data)
    point.to_pickle(OUT / f's4_point_{args.window}_seed{args.seed}.pkl')
    if args.point_only:
        print(f'{args.window} seed {args.seed}: {len(codes)} races, {len(point)} horses (point only)', flush=True)
        return
    samples = sampled_runs(load_model(f'{SPLIT}_{args.window}_seed{args.seed}', X, C), data)
    pd.to_pickle(samples, OUT / f's4_samples_{args.window}_seed{args.seed}.pkl')
    print(f'{args.window} seed {args.seed}: {len(codes)} races, {len(point)} horses, {K_RUNS} runs each', flush=True)


# --- ranges --------------------------------------------------------------------------------------------------
def int_rank(ct):
    """Whole places 1..n from times (ties by order)."""
    return np.argsort(np.argsort(ct, kind='stable'), kind='stable') + 1.0


def raw_ranges(s, point_ct, point_l3):
    """Per target: (lo, hi) raw quantiles of one race's runs (normalised place / s) and the point (whole place / 0.1 s)."""
    n = len(s['umaban'])
    out = {}
    for c in range(4):
        if np.isnan(s['ct'][0, :, c]).any():
            continue
        place = (np.argsort(np.argsort(s['ct'][:, :, c], 1, kind='stable'), 1, kind='stable')) / max(n - 1, 1)
        lo, hi = np.quantile(place, Q[0], 0), np.quantile(place, Q[1], 0)
        out[f'r{c + 1}'] = (lo, hi, int_rank(point_ct[:, c]))
    out['last3'] = (np.quantile(s['last3'], Q[0], 0), np.quantile(s['last3'], Q[1], 0), np.round(point_l3, 1))
    return out


def range_rows(window, seed, act, corners_of=None):
    """Per target, every scorable row of the window: raw lo / hi, point, actual, field size (v1.6's interval_rows)."""
    point = pd.read_pickle(OUT / f's4_point_{window}_seed{seed}.pkl')
    samples = pd.read_pickle(OUT / f's4_samples_{window}_seed{seed}.pkl')
    rows = {t: {k: [] for k in ('lo', 'hi', 'point', 'actual', 'n')} for t in ('r1', 'r2', 'r3', 'r4', 'last3')}
    for code, g in point.groupby('RACE_CODE'):
        s = samples[code]
        g = g.set_index('UMABAN').loc[s['umaban']]
        n = len(g)
        a = act.loc[[(code, u) for u in s['umaban']]]
        allowed = corners_of.get(code) if corners_of is not None else None
        if corners_of is not None and allowed is None:
            continue
        for t, (lo, hi, pt) in raw_ranges(s, g[[f'ct{c}' for c in (1, 2, 3, 4)]].to_numpy(), g['last3'].to_numpy()).items():
            if t == 'last3':
                y = np.round(a.last3.to_numpy(), 1)
                ok = ~np.isnan(y)
            else:
                c = int(t[1])
                if allowed is not None and c not in allowed:
                    continue
                lab = a[[f'r{c}', f'g{c}', f'l{c}']].to_numpy()
                if np.isnan(lab).any():
                    continue   # pace_model.labelled: every horse has the rank, gap and lane label
                y = np.rint(a[f'r{c}'].to_numpy() * (n - 1)) + 1
                ok = np.ones(n, bool)
            r = rows[t]
            for k, v in (('lo', lo), ('hi', hi), ('point', pt), ('actual', y), ('n', np.full(n, n))):
                r[k].extend(np.asarray(v, float)[ok])
    return {t: {k: np.array(v) for k, v in r.items()} for t, r in rows.items() if r['lo']}


def coverage(t, r, margin):
    import pace_model as pm
    return pm.coverage(t, r, margin)


def raw_coverage(t, r):
    actual = r['actual'] if t == 'last3' else (r['actual'] - 1) / (r['n'] - 1)
    return float(np.mean((r['lo'] <= actual) & (actual <= r['hi'])))


def v20_corners(window, seed):
    v20 = pd.read_pickle(OUT / f'v20_{window}_seed{seed}.pkl')
    return v20.drop_duplicates('RACE_CODE').set_index('RACE_CODE').corners.to_dict()


def cmd_calibrate(args):
    act = ev.actuals()
    rows = range_rows('sel', args.seed, act, v20_corners('sel', args.seed))
    margins, tuning = {}, {}
    for t, r in rows.items():
        grid = GRID['last3' if t == 'last3' else 'r']
        cov = [coverage(t, r, float(m))[0] for m in grid]
        best = int(np.argmin([abs(c - TARGET_COVERAGE) for c in cov]))
        margins[t] = float(grid[best])
        tuning[t] = {'margin': margins[t], 'coverage': cov[best], 'coverageWithoutMargin': coverage(t, r, 0.0)[0],
                     'rawQuantileCoverage': raw_coverage(t, r), 'meanWidth': coverage(t, r, margins[t])[1], 'rows': len(r['lo']),
                     'atGridEdge': best in (0, len(grid) - 1)}
        print(t, tuning[t], flush=True)
    (OUT / f's4_margins_seed{args.seed}.json').write_text(json.dumps({'margins': margins, 'tuning': tuning, 'quantiles': Q},
                                                                     indent=1), encoding='utf-8')


# --- E0 measures ---------------------------------------------------------------------------------------------
def load_pace():
    pa = pd.read_pickle(OUT / 'pace_actual.pkl').set_index('RACE_CODE')
    norms = json.loads((OUT / 'pace_norms.json').read_text(encoding='utf-8'))
    return pa, norms


def norm_of(norms, venue, track, distance, surface):
    v = norms['exact'].get(f'{int(venue)},{int(track)},{float(distance)}')   # keys as pace_actual.py wrote them
    v = v if v is not None else norms['loose'].get(f'{int(surface)},{float(distance)}')
    if v is None:
        raise KeyError(f'no course norm for {venue},{track},{distance}')
    return v


def e0_table(window, seed, act, races, pa, norms):
    """E0's measures on the races v2.0 scores, both systems row-paired (RaceTable keys = horse numbers)."""
    import eval_standard as es
    base = pd.read_pickle(OUT / f'v20_{window}_seed{seed}.pkl')
    v3 = pd.read_pickle(OUT / f's4_point_{window}_seed{seed}.pkl')
    common = sorted(set(base.RACE_CODE) & set(v3.RACE_CODE))
    tab = es.RaceTable()
    pace = {'v20': [], 'v3': [], 'actual': []}
    for code in common:
        b = base[base.RACE_CODE == code].set_index('UMABAN')
        v = v3[v3.RACE_CODE == code].set_index('UMABAN')
        nums = [n for n in b.index if n in v.index]
        n = len(nums)
        a = act.loc[[(code, u) for u in nums]].reset_index(level=0, drop=True)
        keys = list(nums)
        normal = a.normal.to_numpy() & a.fin.notna().to_numpy()
        l3 = a.last3.to_numpy()
        ok3 = normal & ~np.isnan(l3)
        for s, p in (('v20', b.loc[nums, 'last3'].to_numpy()), ('v3', v.loc[nums, 'last3'].to_numpy())):
            tab.add('last3', s, code, np.where(ok3, np.abs(p - l3), np.nan), keys)
        T = v.loc[nums, 'T'].to_numpy()
        fin = {'v20': b.loc[nums, 'finSeconds'].to_numpy(), 'v3': T - T.min()}
        gap, place = a.finish_gap.to_numpy(), a.fin.to_numpy()
        for s, p in fin.items():
            tab.add('finGap', s, code, np.where(normal & ~np.isnan(gap), np.abs(p - gap), np.nan), keys)
            if normal.sum() >= 3:
                tab.add_value('finTau', s, code, es.kendall(p[normal], place[normal]))
                tab.add_value('winner', s, code, float(place[normal][np.argmin(p[normal])] == 1))
        for c in b.corners.iloc[0]:
            lab = a[[f'r{c}', f'g{c}', f'l{c}']].to_numpy()
            ct = v.loc[nums, f'ct{c}'].to_numpy()
            if np.isnan(lab).any() or np.isnan(ct).any():
                continue
            y, g = a[f'r{c}'].to_numpy(), a[f'g{c}'].to_numpy()
            ranks = {'v20': (b.loc[nums, f'rank{c}'].to_numpy() - 1) / (n - 1), 'v3': (int_rank(ct) - 1) / (n - 1)}
            for s, p in ranks.items():
                tab.add_value(f'r{c}.tau', s, code, es.kendall(p, y))
                tab.add(f'r{c}.integer', s, code, np.abs(p - y), keys)
                tab.add(f'r{c}.places', s, code, np.abs(p - y) * (n - 1), keys)
            tab.add(f'g{c}', 'v20', code, np.abs(b.loc[nums, f'gap{c}'].to_numpy() - g), keys)
            tab.add(f'g{c}', 'v3', code, np.abs((ct - ct.min()) * LPS - g), keys)
        if code in pa.index and not np.isnan(pa.at[code, 'pace']):
            rc = races.loc[code]
            pv3 = float(v.lead600.iloc[0]) - norm_of(norms, rc.venue, rc.track, rc.distance, rc.surface)
            pv20, pact = float(b.pace.iloc[0]), float(pa.at[code, 'pace'])
            for s, p in (('v20', pv20), ('v3', pv3)):
                tab.add_value('pace.mae', s, code, abs(p - pact))
                tab.add_value('pace.hit', s, code, float(pace_class(p) == pace_class(pact)))
            pace['v20'].append(pv20)
            pace['v3'].append(pv3)
            pace['actual'].append(pact)
    out = {'races': len(common)}
    for m in sorted({k[0] for k in tab.rows}):
        out[m] = {'v20': tab.mean(m, 'v20'), 'v3': tab.mean(m, 'v3'), 'diff': tab.diff(m, 'v3', 'v20')}
    out['_pace'] = {s: {'corr': float(np.corrcoef(pace[s], pace['actual'])[0, 1]), 'mean': float(np.mean(pace[s]))} for s in ('v20', 'v3')}
    out['_pace']['actualMean'] = float(np.mean(pace['actual']))
    out['_pace']['races'] = len(pace['actual'])
    out['_paceSeries'] = pace
    return out


PACE_BANDS = (-0.35, 0.20)   # pace_model.PACE_BANDS (E0's fixed H/M/S bands)


def pace_class(v):
    return 0 if v <= PACE_BANDS[0] else 2 if v >= PACE_BANDS[1] else 1


# better = lower for errors, higher for tau / hit; the bar is every main measure better on both seeds (report window)
MAIN = ([(f'r{c}.tau', 'high') for c in (1, 2, 3, 4)] + [(f'r{c}.integer', 'low') for c in (1, 2, 3, 4)]
        + [(f'g{c}', 'low') for c in (1, 2, 3, 4)] + [('last3', 'low'), ('pace.mae', 'low'), ('finTau', 'high'), ('finGap', 'low')])
SECONDARY = [(f'r{c}.places', 'low') for c in (1, 2, 3, 4)] + [('pace.hit', 'high'), ('winner', 'high')]


def cmd_evaluate(args):
    import eval_standard as es
    act = ev.actuals()
    races = pd.read_pickle(OUT / 'races.pkl')
    pa, norms = load_pace()
    res = {'windows': {}, 'coverage': {}, 'rps': {}}
    for window in ('sel', 'rep'):
        for seed in SEEDS:
            key = f'{window}/seed{seed}'
            if not (OUT / f's4_point_{window}_seed{seed}.pkl').exists():
                print('skipped (no runs yet)', key, flush=True)
                continue
            t = e0_table(window, seed, act, races, pa, norms)
            series = t.pop('_paceSeries')
            res['windows'][key] = t
            # RPS of the H/M/S probabilities (normal predictive, E0): v2.0's sigma = its forward OOF error sd before the
            # fit date; v3's = the sd of its own pace errors on the selection window (out of sample for rep only)
            if window == 'rep':
                sel = e0_series('sel', seed, act, races, pa, norms)
                sig3 = float(np.std(np.array(sel['v3']) - np.array(sel['actual'])))
                train = np.array(norms['trainPace']['rep'])
                res['rps'][key] = {'v20': es.pace_summary(series['v20'], series['actual'], train, norms['sigma']['rep'], PACE_BANDS),
                                   'v3': es.pace_summary(series['v3'], series['actual'], train, sig3, PACE_BANDS)}
            mfile = OUT / f's4_margins_seed{seed}.json'
            if mfile.exists() and (OUT / f's4_samples_{window}_seed{seed}.pkl').exists():
                margins = json.loads(mfile.read_text(encoding='utf-8'))['margins']
                rows = range_rows(window, seed, act, v20_corners(window, seed))
                res['coverage'][key] = {tt: {'coverage': coverage(tt, r, margins[tt])[0], 'meanWidth': coverage(tt, r, margins[tt])[1],
                                             'coverageWithoutMargin': coverage(tt, r, 0.0)[0], 'rawQuantileCoverage': raw_coverage(tt, r),
                                             'margin': margins[tt], 'rows': len(r['lo'])} for tt, r in rows.items()}
    verdict = {'failed': [], 'passed': []}
    for m, better in MAIN:
        good = []
        for seed in SEEDS:
            d = res['windows'].get(f'rep/seed{seed}', {}).get(m, {}).get('diff', {}).get('diff', np.nan)
            good.append(bool(d < 0) if better == 'low' else bool(d > 0))
        (verdict['passed'] if all(good) else verdict['failed']).append(m)
    verdict['pass'] = not verdict['failed']
    res['verdict'] = verdict
    (OUT / 's4_eval.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    for key, t in res['windows'].items():
        print(f'== {key}  races {t["races"]}')
        for m, _ in MAIN + SECONDARY:
            if m in t:
                d = t[m]['diff']
                print(f'  {m:12} v2.0 {t[m]["v20"]:.3f}  v3 {t[m]["v3"]:.3f}  diff {d["diff"]:+.3f} [{d["lo"]:+.3f}, {d["hi"]:+.3f}]')
        if key in res['coverage']:
            print('  coverage', {tt: round(c['coverage'], 3) for tt, c in res['coverage'][key].items()})
    print('verdict', verdict)


def e0_series(window, seed, act, races, pa, norms):
    v3 = pd.read_pickle(OUT / f's4_point_{window}_seed{seed}.pkl').drop_duplicates('RACE_CODE').set_index('RACE_CODE')
    base = set(pd.read_pickle(OUT / f'v20_{window}_seed{seed}.pkl').RACE_CODE)
    out = {'v3': [], 'actual': []}
    for code, v in v3.iterrows():
        if code not in base or code not in pa.index or np.isnan(pa.at[code, 'pace']):
            continue
        rc = races.loc[code]
        out['v3'].append(float(v.lead600) - norm_of(norms, rc.venue, rc.track, rc.distance, rc.surface))
        out['actual'].append(float(pa.at[code, 'pace']))
    return out


# --- the published file ------------------------------------------------------------------------------------------
SEX = {'牡': 1, '牝': 2, 'セ': 3, '騸': 3}


def card_tables(races, runs, snap, grade):
    """The run table with an upcoming race's card appended (no observations): its race row and one row per runner."""
    import pace_model as pm
    import rating_index as ri
    race = snap['race']
    code = snap['raceId']
    active = [h for h in snap['horses'] if h['status'] != 'scratched']
    venue, track = int(race['venueCode']), int(race['trackCode'])
    going = pm.GOING.get((snap.get('conditions') or {}).get('going'), np.nan)
    races = pd.concat([races, pd.DataFrame([{
        'date': race['date'], 'venue': venue, 'track': track, 'distance': float(race['distance']), 'field': len(active),
        'surface': int(track >= 23), 'cls': pm.class_level(grade, ''), 'going': going, 'kind': cs.kind_of(venue, track),
        'lead': np.nan, 'complete': True, 'named': True}], index=[code])])
    state = ri.ratings(ri.load_runs(race['date']), return_state=True)[1]
    new = pd.DataFrame([{'RACE_CODE': code, 'date': race['date'], 'horse': h['horseId'], 'UMABAN': int(h['number']),
                         'waku': float(h['gate']), 'age': float(h['age']), 'sex': float(SEX.get(h['sex'], 1)),
                         'weight': float(h['weight']), 'blinker': float(bool(h['blinker'])), 'normal': True, 'fin': np.nan,
                         'soha': np.nan, 'last3': np.nan, 't600': np.nan, 'finish_gap': np.nan,
                         'rating_pre': ri.rating_of(state, h['horseId'])} for h in active])
    for c in (1, 2, 3, 4):
        new[f'r{c}'] = np.nan
        new[f'g{c}'] = np.nan
    return races, pd.concat([runs, new], ignore_index=True)


def gi_bands(model, races, r, pa, norms, before):
    """Terciles of v3's own pace predictions for past GI races (v2.0 labels a GI's pace the same way)."""
    gi = pa[(pa.grade == 'A') & (pa.date < before)].index
    codes = [c for c in gi if c in races.index and races.at[c, 'complete'] and races.at[c, 'kind'] != 'straight']
    data, _, _ = make_data(races, r, codes)
    p = point_runs(model, data).drop_duplicates('RACE_CODE')
    vals = [float(v.lead600) - norm_of(norms, races.at[v.RACE_CODE, 'venue'], races.at[v.RACE_CODE, 'track'],
                                       races.at[v.RACE_CODE, 'distance'], races.at[v.RACE_CODE, 'surface']) for _, v in p.iterrows()]
    return tuple(float(x) for x in np.quantile(vals, [1 / 3, 2 / 3])), len(vals)


def race_json(code, g, s, margins, races, norms, bands, shown_targets):
    """One race in the paddock.pace-forecast/v1 shape (revision 1.1)."""
    import pace_model as pm
    g = g.set_index('UMABAN').loc[s['umaban']]
    nums = [int(u) for u in s['umaban']]
    n = len(nums)
    ctm = g[[f'ct{c}' for c in (1, 2, 3, 4)]].to_numpy()
    corners = [c + 1 for c in range(4) if not np.isnan(ctm[:, c]).any()]
    ranges = raw_ranges(s, ctm, g['last3'].to_numpy())
    shown = {}
    for t, (lo, hi, pt) in ranges.items():
        a, b = pm.shown_range(t, lo, hi, pt, n, margins[t])
        shown[t] = (a, b)
    lane = np.nanmedian(s['lane'], 0)
    horses = {}
    for i, num in enumerate(nums):
        positions = []
        for c in corners:
            ct = ctm[:, c - 1]
            p = {'corner': c, 'rank': int(int_rank(ct)[i]), 'gap': round(float(min((ct[i] - ct.min()) * LPS, 60)), 2),
                 'lane': round(float(np.clip(lane[i, c - 1], 0, 12)), 2)}
            if f'r{c}' in shown_targets:
                lo, hi = shown[f'r{c}']
                p['rankRange'] = [int(lo[i]), int(hi[i])]
            positions.append(p)
        l3 = float(np.clip(round(float(g['last3'].iat[i]), 1), 28, 50))
        horses[str(num)] = {'positions': positions, 'last3': l3}
        if 'last3' in shown_targets:
            lo, hi = shown['last3']
            horses[str(num)]['last3Range'] = [float(np.clip(round(float(lo[i]), 1), 28, 50)), float(np.clip(round(float(hi[i]), 1), 28, 50))]
    T = g['T'].to_numpy()
    for pos, i in enumerate(sorted(range(n), key=lambda i: (T[i], nums[i]))):
        horses[str(nums[i])]['finish'] = {'rank': pos + 1, 'gap': round(float(min((T[i] - T.min()) * LPS, 80)), 2)}
    rc = races.loc[code]
    early = float(g['lead600'].iat[0]) - norm_of(norms, rc.venue, rc.track, rc.distance, rc.surface)
    first = ctm[:, corners[0] - 1]
    front = [nums[i] for i in np.argsort(first) if (int_rank(first)[i] - 1) / max(n - 1, 1) <= 0.2]
    pace = {'early': round(early, 2), 'label': pm.pace_label(early, bands), 'bands': [round(b, 2) for b in bands], 'frontRunners': front}
    return {'corners': corners, 'horses': horses, 'pace': pace}


def cmd_forecast(args):
    import pace_model as pm
    races, runs = pd.read_pickle(OUT / 'races.pkl'), pd.read_pickle(OUT / 'runs.pkl')
    published = json.loads((ROOT / 'public' / 'data' / 'forecasts.json').read_text(encoding='utf-8'))['races']
    today = (args.today or datetime.now().strftime('%Y-%m-%d')).replace('-', '')
    cards = {s['raceId']: s for s in pm.load_upcoming_entries(str(ROOT / 'data' / 'entries'), today)}
    wanted = []
    for race in published:
        code = race['id']
        if race.get('historical') is False:
            if code not in cards:
                continue
            races, runs = card_tables(races, runs, cards[code], race.get('grade') or 'A')
        elif code not in races.index:
            print('WARNING not in the run table', code, race['name'])
            continue
        if races.at[code, 'kind'] == 'straight':
            print('WARNING straight course, skipped', code, race['name'])
            continue
        wanted.append(code)
    r = ft.run_observations(races, runs).reset_index(drop=True)
    margins_file = json.loads((OUT / f's4_margins_seed{args.seed}.json').read_text(encoding='utf-8'))
    pa, norms = load_pace()
    data, X, C = make_data(races, r, wanted)
    pmodel = load_model(f'{POINT}_{args.window}_seed{args.seed}', X, C)
    point = point_runs(pmodel, data)
    samples = sampled_runs(load_model(f'{SPLIT}_{args.window}_seed{args.seed}', X, C), data)
    fit_end = tr.WINDOWS[args.window][0]
    bands, n_gi = gi_bands(pmodel, races, r, pa, norms, fit_end)
    print(f'GI pace terciles {bands} ({n_gi} races before {fit_end})', flush=True)
    ev_file = OUT / 's4_eval.json'
    evaluation = json.loads(ev_file.read_text(encoding='utf-8')) if ev_file.exists() else {}
    cov = evaluation.get('coverage', {}).get(f'rep/seed{args.seed}', {})
    # v1.6's rule: a target's ranges are published only if its report-window coverage is within COVERAGE_BAND
    targets = {t: {**c, 'shown': bool(COVERAGE_BAND[0] <= c['coverage'] <= COVERAGE_BAND[1])} for t, c in cov.items()}
    shown_targets = {t for t, c in targets.items() if c['shown']}
    if not cov:
        print('WARNING no report-window coverage (s4.py evaluate): no ranges published', flush=True)
    out_races = {}
    for code in wanted:
        out_races[code] = race_json(code, point[point.RACE_CODE == code], samples[code], margins_file['margins'], races, norms, bands,
                                    shown_targets)
    model = {'name': 'Paddock展開予想モデル', 'version': VERSION, 'trainingEnd': (pd.Timestamp(fit_end) - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
             'validation': '2025-09〜12', 'engine': 'v3 race simulator (S2_gwe point + S3 sampled runs)',
             'intervals': {'alpha': list(Q), 'margins': margins_file['margins'], 'tuning': margins_file['tuning'],
                           'band': list(COVERAGE_BAND), 'targets': targets, 'published': sorted(shown_targets)},
             'verdict': evaluation.get('verdict'),
             'notes': ['各馬の速度をスタートからゴールまで25m刻みで進める学習型シミュレーター（v3）。過去走の列から能力値を推定し、位置取りの競り合い・消耗・風よけを持つ。',
                       '点の予想（順位・馬身差・上がり・着順・ペース）は乱数なしの1回の走り（S2_gwe）から読み出す。馬身差は角の通過タイム差×5馬身/秒。',
                       '範囲（rankRange / last3Range）は、能力を固定し横位置・詰まり・コーナーの距離ロスの規則と、その日の出来の乱数（0.15 m/s）を入れた32回の走りの25〜75%分位に、2025年6〜8月で被覆率50%になる幅を足したもの。',
                       '内外（lane）は32回の走りの横位置から作った幾何による参考表示で、予想ではない（S3 の内外は全頭0の定数に勝てなかった）。',
                       '着順はゴールタイムの並び、ペースは先頭のスタートから600m地点の時刻とコースの標準（2025年6月より前の前半3Fの中央値）の差。',
                       'オッズ・人気・払戻・対象レースの結果は使わない。']}
    out = {'schema': 'paddock.pace-forecast/v1', 'revision': '1.1', 'generatedAt': datetime.now().isoformat(timespec='seconds'), 'model': model,
           'units': {'gap': '馬身（先頭から）', 'lane': '横位置の参考（併走群内の内からの順、幾何による参考表示）', 'last3': '秒'}, 'races': out_races}
    path = Path(args.out) if args.out else OUT / f'pace_forecasts_v3_{args.window}.json'
    path.write_text(json.dumps(out, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'wrote {len(out_races)} races -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['runs', 'calibrate', 'evaluate', 'forecast'])
    ap.add_argument('--window', choices=list(tr.WINDOWS), default='rep')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    ap.add_argument('--point-only', action='store_true', help='runs: the deterministic point forecast only (no sampled runs)')
    ap.add_argument('--today', default='', help='forecast: YYYY-MM-DD, cards of races on/after it are read (default: today)')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = ap.parse_args()
    tr.DEVICE = torch.device(args.device)
    {'runs': cmd_runs, 'calibrate': cmd_calibrate, 'evaluate': cmd_evaluate, 'forecast': cmd_forecast}[args.mode](args)


if __name__ == '__main__':
    main()
