"""v3 / v2.0 hybrid (docs/race-simulator-hybrid.md): v3's corner order, closing 3F and finish order (S2_gwe, deterministic);
v2.0's pace; corner gaps = v3's shape times a race-level spread per corner; the finish margin recomputed with the published
identity (pace_model.race_output, app/playback.ts):

    fin_i = gap_last_i + 5 (last3_i - min last3) - min_k [gap_last_k + 5 (last3_k - min last3)]      (lengths)

  python -I scripts/v3/hybrid.py v20parts --window sel|rep --seed 42|7
        v2.0 recomputed exactly as baseline_v20.py does (same frame, procedure, contention parameters and seed), keeping what
        the hybrid takes from it: per race the pace (s vs the course norm) and the spread models' predictions (C1), and the
        race-level rows its spread models were trained on (actual 80% spread, forward out-of-sample pace) for C2. Checks the
        ranks, gaps, closing, finish and pace against outputs/v3/v20_<w>_seed<s>.pkl -> outputs/v3/hybrid_v20_<w>_seed<s>.pkl
  python -I scripts/v3/hybrid.py valruns --window sel|rep --seed 42|7 [--device cpu]
        the window's S2_gwe deterministic run on its validation races (the 90 days before the fit date, used for early
        stopping only) -> outputs/v3/hybrid_valruns_<w>_seed<s>.pkl (C3's factors)
  python -I scripts/v3/hybrid.py candidates
        the spread candidates on the selection races (both seeds): E0's corner-gap MAE g<c>, the finish-margin MAE and the
        three consistency conditions per race -> outputs/v3/hybrid_candidates.json

Candidates (corner c, v3 corner times ct, leader = min ct, anchor = the horse at SPREAD_Q of v3's order):
  C0  gap = 5 (ct - lead)                                   (S4; finish also shown as S4 published it, 5 x finish-time gap)
  C1  gap = S_c (ct - lead) / (ct_anchor - lead), S_c = v2.0's spread<c> prediction
  C2  the same with S_c from a median linear regression on field, distance, surface, venue x surface and v2.0's pace
  C3  gap = k_c 5 (ct - lead), k_c = the MAE-optimal factor on the v3 model's validation races
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import evaluate as ev  # noqa: E402

OUT = ev.OUT
LPS = 5.0
SPREAD_Q = 0.8      # pace_model.SPREAD_Q
SEEDS = (42, 7)
TOL = 1e-9          # lengths: numerical tolerance of the consistency checks (no rounding)
CANDIDATES = ('C0', 'C1', 'C2', 'C3')


# --- v2.0 parts -----------------------------------------------------------------------------------------------
def cmd_v20parts(args):
    import baseline_v20 as bl
    import experiment_data_expansion as ex
    import experiment_data_expansion_v20 as v20
    import experiment_ewm_history as e1
    import pace_model as pm
    pm.EVAL_CORNERS, pm.PACE_OOF, pm.RANK_BLEND = 'course', 'forward', 0.75
    ex.PINNED = set(ex.validation_codes(pd.read_pickle(e1.PIN)))
    P = json.loads((pm.ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
    d = v20.prepared(v20.FRAMES['after'])
    fit_end, start, end = bl.WINDOWS[args.window]
    ex.set_seed(args.seed)
    models = pm.fit_models(d, fit_end)
    races = pm.evaluation_races(models, d, start, end)
    rows, per_race = [], []
    for x, corners, numbers, pred, spreads, early in races:
        con = pm.simulate(numbers, early, pred, spreads, corners, P)
        last3 = pm.closing(models, x, con, corners, pred['pace'])[0]
        gaps = con[corners[-1]]['gap']
        best = float(min(last3))
        fin = np.array([gaps[i] + (float(last3[i]) - best) * pm.LENGTHS_PER_SECOND for i in range(len(numbers))])
        fin = fin - fin.min()
        code = x.RACE_CODE.iloc[0]
        for i, num in enumerate(numbers):
            r = {'RACE_CODE': code, 'UMABAN': int(num), 'last3': float(last3[i]), 'finSeconds': float(fin[i] / pm.LENGTHS_PER_SECOND),
                 'pace': pred['pace']}
            for c in corners:
                r[f'rank{c}'], r[f'gap{c}'] = float(con[c]['rank'][i]), float(con[c]['gap'][i])
            rows.append(r)
        xp = x.assign(pace_pred=pred['pace'])
        feats = pm.race_level(xp, 'RACE_CODE', 1)[pm.SPREAD_FEATURES].iloc[0].to_dict()
        per_race.append({'RACE_CODE': code, 'pace': pred['pace'], 'pace_class': pm.pace_label(pred['pace']),
                         **{f'spread{c}': spreads[c] for c in pm.CORNERS}, **feats})
    # the same recomputation as baseline_v20.py: every saved value must match
    mine = pd.DataFrame(rows).set_index(['RACE_CODE', 'UMABAN']).sort_index()
    ref = pd.read_pickle(OUT / f'v20_{args.window}_seed{args.seed}.pkl').set_index(['RACE_CODE', 'UMABAN']).sort_index()
    assert mine.index.equals(ref.index), 'race / horse sets differ from v20 pkl'
    check = {k: float(np.nanmax(np.abs(mine[k].to_numpy() - ref[k].to_numpy()))) for k in mine.columns}
    print('max |recomputed - v20 pkl|', check, flush=True)
    # the spread models' training rows (fit_models: complete races before the fit date, forward pace, first chunk skipped)
    fwd = pm.forward_pace(d, fit_end)
    train = d[(d.date < fit_end) & d.complete].copy()
    train['pace_pred'] = train.RACE_CODE.map(fwd['pred'])
    tables = {}
    for c in pm.CORNERS:
        t = pm.race_level(train, 'RACE_CODE', c).dropna(subset=['spread'])
        tables[c] = t[~t.index.isin(fwd['skip'])][pm.SPREAD_FEATURES + ['spread', 'date']]
    pd.to_pickle({'races': pd.DataFrame(per_race).set_index('RACE_CODE'), 'train': tables, 'check': check, 'fitEnd': fit_end},
                 OUT / f'hybrid_v20_{args.window}_seed{args.seed}.pkl')
    print(f'{args.window} seed {args.seed}: {len(per_race)} races; spread training rows', {c: len(t) for c, t in tables.items()}, flush=True)


# --- v3 on the validation races (C3) -----------------------------------------------------------------------------
def cmd_valruns(args):
    import torch
    import s4
    import train as tr
    tr.DEVICE = torch.device(args.device)
    races, r = tr.load_tables()
    fit_end = tr.WINDOWS[args.window][0]
    val_start = (pd.Timestamp(fit_end) - pd.Timedelta(days=90)).strftime('%Y%m%d')
    ok = races.complete & races.lead.notna() & (races.kind != 'straight')
    codes = races[ok & races.named & races.date.between(val_start, fit_end, inclusive='left')].index.tolist()   # train.py's val_codes
    data, X, C = s4.make_data(races, r, codes)
    point = s4.point_runs(s4.load_model(f'{s4.POINT}_{args.window}_seed{args.seed}', X, C), data)
    point.to_pickle(OUT / f'hybrid_valruns_{args.window}_seed{args.seed}.pkl')
    print(f'{args.window} seed {args.seed}: {len(codes)} validation races ({val_start}..{fit_end}), {len(point)} horses', flush=True)


# --- candidates -------------------------------------------------------------------------------------------------
def labelled(a, c):
    return not np.isnan(a[[f'r{c}', f'g{c}', f'l{c}']].to_numpy()).any()


def anchor_of(ct):
    """Time behind the leader of the horse at SPREAD_Q of the order (v2.0's spread definition)."""
    rel = np.sort(ct - ct.min())
    return rel[int(round(SPREAD_Q * (len(ct) - 1)))]


def weighted_median(ratio, w):
    o = np.argsort(ratio)
    cw = np.cumsum(w[o])
    return float(ratio[o][np.searchsorted(cw, cw[-1] / 2)])


def fit_c3(valruns, act):
    """k_c minimising sum |k x - y| over the validation horses (x = 5 x v3's time gap, y = the passing string's lengths)."""
    k = {}
    for c in (1, 2, 3, 4):
        xs, ys = [], []
        for code, g in valruns.groupby('RACE_CODE'):
            ct = g[f'ct{c}'].to_numpy()
            if np.isnan(ct).any():
                continue
            a = act.loc[[(code, u) for u in g.UMABAN]]
            if not labelled(a, c):
                continue
            xs.append(LPS * (ct - ct.min()))
            ys.append(a[f'g{c}'].to_numpy())
        x, y = np.concatenate(xs), np.concatenate(ys)
        m = x > 1e-9
        k[c] = {'k': weighted_median(y[m] / x[m], x[m]), 'races': len(xs), 'rows': int(len(x))}
    # reference only (not a candidate: it uses v3's own finish-time identity, not the published one): the same factor on
    # the finish-time gap, seconds behind the winner, normal finishers with an official margin
    xs, ys = [], []
    for code, g in valruns.groupby('RACE_CODE'):
        a = act.loc[[(code, u) for u in g.UMABAN]]
        ok = a.normal.to_numpy() & a.finish_gap.notna().to_numpy()
        T = g['T'].to_numpy()
        xs.append((T - T.min())[ok])
        ys.append(a.finish_gap.to_numpy()[ok])
    x, y = np.concatenate(xs), np.concatenate(ys)
    m = x > 1e-9
    k['fin'] = {'k': weighted_median(y[m] / x[m], x[m]), 'races': len(xs), 'rows': int(len(x))}
    return k


def c2_design(t, venues):
    X = pd.DataFrame({'field': t['field'].astype(float), 'km': t['distance'].astype(float) / 1000, 'surface': t['surface'].astype(float),
                      'pace': t['pace_pred'].astype(float)}, index=t.index)
    vs = t['venue'].astype(int).astype(str) + '_' + t['surface'].astype(int).astype(str)
    for v in venues[1:]:
        X[f'v{v}'] = (vs == v).astype(float)
    return X


def fit_c2(train):
    from sklearn.linear_model import QuantileRegressor
    models = {}
    for c, t in train.items():
        t = t.dropna(subset=['field', 'distance', 'surface', 'venue', 'pace_pred', 'spread'])
        venues = sorted((t['venue'].astype(int).astype(str) + '_' + t['surface'].astype(int).astype(str)).unique())
        X = c2_design(t, venues)
        mu, sd = X[['field', 'km', 'pace']].mean(), X[['field', 'km', 'pace']].std()
        Xs = X.copy()
        Xs[['field', 'km', 'pace']] = (X[['field', 'km', 'pace']] - mu) / sd
        q = QuantileRegressor(quantile=0.5, alpha=1e-4, solver='highs').fit(Xs.to_numpy(), t['spread'].to_numpy())
        models[c] = {'model': q, 'venues': venues, 'mu': mu, 'sd': sd, 'rows': len(t),
                     'coef': dict(zip(Xs.columns, map(float, q.coef_))), 'intercept': float(q.intercept_)}
    return models


def predict_c2(models, c, feats):
    m = models[c]
    X = c2_design(pd.DataFrame([feats]), m['venues'])
    X[['field', 'km', 'pace']] = (X[['field', 'km', 'pace']] - m['mu']) / m['sd']
    return float(m['model'].predict(X.to_numpy())[0])


def corner_gaps(cand, ct, c, ctx):
    """Unrounded lengths behind the corner's leader for one candidate; ctx: per-race spread predictions / factors."""
    rel = ct - ct.min()
    if cand == 'C0':
        return LPS * rel
    if cand == 'C3':
        return ctx['k3'][c] * LPS * rel
    S = max(0.0, ctx['S1' if cand == 'C1' else 'S2'][c])
    a = anchor_of(ct)
    return S * rel / a if a > 1e-6 else LPS * rel


def identity_finish(gap_last, last3):
    f = gap_last + LPS * (last3 - last3.min())
    return f - f.min()


def monotone_violation(values, order):
    """Largest drop of `values` along `order` (0 if never decreasing)."""
    x = values[order]
    return float(np.max(np.maximum.accumulate(x) - x))


def any_monotone_feasible(ct_last, T, last3):
    """Whether ANY corner-4 gaps non-decreasing down v3's corner order satisfy condition 3 with v3's last3 and finish order
    (difference constraints u_b >= u_a + w; a positive cycle = infeasible; Bellman-Ford)."""
    n = len(T)
    oc, of = np.argsort(ct_last, kind='stable'), np.argsort(T, kind='stable')
    E = [(a, b, 0.0) for a, b in zip(oc[:-1], oc[1:])] + [(a, b, LPS * (last3[a] - last3[b])) for a, b in zip(of[:-1], of[1:])]
    u = np.zeros(n)
    for _ in range(n + 1):
        changed = False
        for a, b, w in E:
            if u[a] + w > u[b] + TOL:
                u[b] = u[a] + w
                changed = True
        if not changed:
            return True
    return False


def scale_interval(ct_last, T, last3):
    """Range of a race-level factor s on v3's last-corner time gaps for which condition 3 holds (empty: lo > hi)."""
    of = np.argsort(T, kind='stable')
    lo, hi = 0.0, np.inf
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = of[i], of[j]
            dct, dl = ct_last[b] - ct_last[a], last3[a] - last3[b]
            if dct > 0:
                lo = max(lo, dl / dct)
            elif dct < 0:
                hi = min(hi, dl / dct)
            elif dl > 0:
                lo = np.inf
    return lo, hi


def finish_lengths(T, k_fin):
    """The hybrid's finish margin (lengths behind the winner): v3's finish-time gap x the factor x 5 lengths/s."""
    return k_fin * LPS * (T - T.min())


def race_ctx(feats, c2, k3):
    """Per-race inputs of the candidates: C1 v2.0's spread predictions, C2 the simple spread model, C3 the factors."""
    return {'k3': {c: k3[c]['k'] for c in (1, 2, 3, 4)}, 'S1': {c: float(feats.get(f'spread{c}', np.nan)) for c in (1, 2, 3, 4)},
            'S2': {c: predict_c2(c2, c, {**feats, 'pace_pred': float(feats['pace'])}) for c in (1, 2, 3, 4)}}


def race_forecast(v, cand, ctx, k_fin):
    """THE hybrid's point forecast of one race (used by evaluate and forecast alike).
    v: v3's deterministic run of the race (rows indexed by horse number, in output order).
    Corner order, closing 3F and finish order are v3's as they are; the gaps are v3's shape x the chosen candidate's spread;
    the finish margin is v3's finish-time gap x k_fin. The pace is not here: it is v2.0's, attached by the caller."""
    nums = [int(u) for u in v.index]
    ctm = {c: v[f'ct{c}'].to_numpy() for c in (1, 2, 3, 4)}
    corners = [c for c in (1, 2, 3, 4) if not np.isnan(ctm[c]).any()]
    T = v['T'].to_numpy()
    order = sorted(range(len(nums)), key=lambda i: (T[i], nums[i]))
    fin_rank = np.empty(len(nums), int)
    fin_rank[order] = np.arange(1, len(nums) + 1)
    return {'numbers': nums, 'corners': corners, 'ct': ctm, 'T': T, 'last3': v['last3'].to_numpy(),
            'rank': {c: int_rank(ctm[c]) for c in corners}, 'gap': {c: corner_gaps(cand, ctm[c], c, ctx) for c in corners},
            'finRank': fin_rank, 'fin': finish_lengths(T, k_fin)}


def int_rank(ct):
    """Whole places 1..n from times, ties by order (s4.int_rank)."""
    return np.argsort(np.argsort(ct, kind='stable'), kind='stable') + 1


def consistency(f):
    """Conditions (1) and (3) of the revised gate on one race forecast: the largest violation in lengths (0 = none)."""
    mono = max(monotone_violation(f['gap'][c], np.argsort(f['rank'][c], kind='stable')) for c in f['corners'])
    order = monotone_violation(f['fin'], np.argsort(f['finRank'], kind='stable'))
    return mono, order


def cmd_candidates(args):
    """The spread candidates on the selection races. --finish time (the owner's revision, 2026-11-23): finish margin =
    v3's finish-time gap x k_fin (fitted like C3 on the validation races), conditions (1) and (3). --finish identity (the
    first gate): margin recomputed with the published identity, conditions (1)-(3)."""
    import eval_standard as es
    act = ev.actuals()
    res = {'finish': args.finish, 'seeds': {}, 'c3': {}, 'c2': {}}
    for seed in SEEDS:
        parts = pd.read_pickle(OUT / f'hybrid_v20_sel_seed{seed}.pkl')
        base = pd.read_pickle(OUT / f'v20_sel_seed{seed}.pkl')
        v3 = pd.read_pickle(OUT / f's4_point_sel_seed{seed}.pkl')
        k3 = fit_c3(pd.read_pickle(OUT / f'hybrid_valruns_sel_seed{seed}.pkl'), act)
        k_fin = k3['fin']['k']
        c2 = fit_c2(parts['train'])
        res['c3'][seed] = k3
        res['c2'][seed] = {c: {k: m[k] for k in ('rows', 'coef', 'intercept')} for c, m in c2.items()}
        print('seed', seed, 'C3 factors', {c: round(v['k'], 3) for c, v in k3.items()}, flush=True)
        tab = es.RaceTable()
        names = CANDIDATES + ('C0_S4',)
        cons = {cand: {'monotone': 0, 'identity': 0, 'order': 0, 'races': 0} for cand in names}
        worst = {cand: [] for cand in names}
        spread_err = {cand: {c: [] for c in (1, 2, 3, 4)} for cand in CANDIDATES + ('v20',)}
        empty_scale, infeasible, races_n, fin_spread = 0, 0, 0, []
        common = sorted(set(base.RACE_CODE) & set(v3.RACE_CODE))
        for code in common:
            b = base[base.RACE_CODE == code].set_index('UMABAN')
            v = v3[v3.RACE_CODE == code].set_index('UMABAN')
            nums = [n for n in b.index if n in v.index]
            a = act.loc[[(code, u) for u in nums]].reset_index(level=0, drop=True)
            keys = list(nums)
            ctx = race_ctx(parts['races'].loc[code].to_dict(), c2, k3)
            ctm = {c: v.loc[nums, f'ct{c}'].to_numpy() for c in (1, 2, 3, 4)}
            shown = [c for c in (1, 2, 3, 4) if not np.isnan(ctm[c]).any()]     # the corners the forecast publishes
            T, l3 = v.loc[nums, 'T'].to_numpy(), v.loc[nums, 'last3'].to_numpy()
            last = shown[-1]
            races_n += 1
            lo, hi = scale_interval(ctm[last], T, l3)
            empty_scale += int(lo > hi)
            infeasible += int(not any_monotone_feasible(ctm[last], T, l3))
            normal = a.normal.to_numpy() & a.fin.notna().to_numpy()
            gap_fin = a.finish_gap.to_numpy()
            ok_fin = normal & ~np.isnan(gap_fin)
            tab.add('finGap', 'v20', code, np.where(ok_fin, np.abs(b.loc[nums, 'finSeconds'].to_numpy() - gap_fin), np.nan), keys)
            by_t = np.argsort(T, kind='stable')
            for cand in CANDIDATES:
                g = {c: corner_gaps(cand, ctm[c], c, ctx) for c in shown}
                mono = max(monotone_violation(g[c], np.argsort(ctm[c], kind='stable')) for c in shown)
                if args.finish == 'identity':
                    fin = identity_finish(g[last], l3)
                else:
                    fin = finish_lengths(T, k_fin)
                order = monotone_violation(fin, by_t)
                cons[cand]['races'] += 1
                cons[cand]['monotone'] += int(mono > TOL)
                cons[cand]['order'] += int(order > TOL)
                worst[cand].append(order)
                tab.add('finGap', cand, code, np.where(ok_fin, np.abs(fin / LPS - gap_fin), np.nan), keys)
                for c in b.corners.iloc[0]:
                    if c not in shown or not labelled(a, c):
                        continue
                    tab.add(f'g{c}', cand, code, np.abs(g[c] - a[f'g{c}'].to_numpy()), keys)
                    spread_err[cand][c].append(abs(float(np.quantile(g[c], SPREAD_Q)) - float(np.quantile(a[f'g{c}'].to_numpy(), SPREAD_Q))))
            # C0 as S4 published it: finish = 5 x finish-time gap
            fin_s4 = LPS * (T - T.min())
            g0 = LPS * (ctm[last] - ctm[last].min())
            cons['C0_S4']['races'] += 1
            cons['C0_S4']['identity'] += int(np.max(np.abs(fin_s4 - identity_finish(g0, l3))) > TOL)
            cons['C0_S4']['order'] += int(monotone_violation(fin_s4, by_t) > TOL)
            worst['C0_S4'].append(float(np.max(np.abs(fin_s4 - identity_finish(g0, l3)))))
            tab.add('finGap', 'C0_S4', code, np.where(ok_fin, np.abs(fin_s4 / LPS - gap_fin), np.nan), keys)
            if ok_fin.sum() >= 3:
                q = lambda z: float(np.quantile(z[ok_fin], SPREAD_Q))  # noqa: E731
                fin_spread.append((q(gap_fin), q(T - T.min()), q(b.loc[nums, 'finSeconds'].to_numpy())))
            for c in b.corners.iloc[0]:
                if c in shown and labelled(a, c):
                    tab.add(f'g{c}', 'v20', code, np.abs(b.loc[nums, f'gap{c}'].to_numpy() - a[f'g{c}'].to_numpy()), keys)
                    spread_err['v20'][c].append(abs(float(np.quantile(b.loc[nums, f'gap{c}'].to_numpy(), SPREAD_Q))
                                                    - float(np.quantile(a[f'g{c}'].to_numpy(), SPREAD_Q))))
        out = {'races': races_n, 'kFinish': k_fin, 'scaleIntervalEmpty': empty_scale, 'anyMonotoneInfeasible': infeasible,
               'consistency': cons,
               'orderViolationLengths': {k: [float(q) for q in np.quantile(w, [.5, .75, .9, 1])] for k, w in worst.items()},
               'mae': {}, 'spreadMae': {k: {c: float(np.mean(x)) if x else None for c, x in v.items()} for k, v in spread_err.items()},
               'finishSpread80': dict(zip(('actual', 'v3', 'v20'), map(float, np.median(np.array(fin_spread), 0))))}
        for m in ('g1', 'g2', 'g3', 'g4', 'finGap'):
            out['mae'][m] = {}
            for s in ('v20',) + CANDIDATES + (('C0_S4',) if m == 'finGap' else ()):
                out['mae'][m][s] = {'mae': tab.mean(m, s), 'rows': tab.count(m, s)}
                if s != 'v20':
                    out['mae'][m][s]['vsV20'] = tab.diff(m, s, 'v20')
                    if s != 'C0':
                        out['mae'][m][s]['vsC0'] = tab.diff(m, s, 'C0')
        res['seeds'][seed] = out
        print(f'== seed {seed} ({args.finish}): {races_n} races; k_fin {k_fin:.3f}')
        for m, row in out['mae'].items():
            print('  ', m, '  '.join(f'{s} {v["mae"]:.3f}' for s, v in row.items()))
        for cand, cc in cons.items():
            print('   consistency', cand, cc)
    (OUT / f'hybrid_candidates_{args.finish}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


SELECTION = OUT / 'hybrid_selection.json'


def cmd_select(args):
    """§5's rule on the selection window (revised gate): eligible = no violation of (1) or (3) on either seed; chosen = the
    smallest mean of the corner 1-4 gap MAEs on both seeds (if the seeds disagree, the smallest mean over the two seeds).
    Written once, before the report window is run; evaluate and forecast read it."""
    res = json.loads((OUT / 'hybrid_candidates_time.json').read_text(encoding='utf-8'))
    score, eligible = {}, []
    for cand in CANDIDATES:
        ok = all(res['seeds'][str(s)]['consistency'][cand]['monotone'] == 0 and res['seeds'][str(s)]['consistency'][cand]['order'] == 0
                 for s in SEEDS)
        score[cand] = {str(s): float(np.mean([res['seeds'][str(s)]['mae'][f'g{c}'][cand]['mae'] for c in (1, 2, 3, 4)])) for s in SEEDS}
        if ok:
            eligible.append(cand)
    if not eligible:
        raise SystemExit('no eligible candidate: (1) or (3) violated (implementation error under the revised definition)')
    best = {str(s): min(eligible, key=lambda k: score[k][str(s)]) for s in SEEDS}
    chosen = best['42'] if len(set(best.values())) == 1 else min(eligible, key=lambda k: np.mean(list(score[k].values())))
    out = {'candidate': chosen, 'eligible': eligible, 'meanGapMae': score, 'bestPerSeed': best, 'finish': 'time',
           'kFinishSelection': {str(s): res['seeds'][str(s)]['kFinish'] for s in SEEDS},
           'recipe': {'spread': 'C2: QuantileRegressor(0.5, alpha 1e-4) per corner on field, km, surface, pace (standardised) and '
                                'venue x surface, fitted on the v2.0 spread-model training rows before the fit date',
                      'kFinish': "weighted median of official margin / v3 finish-time gap on the window model's validation races "
                                 '(the 90 days before the fit date), normal finishers'},
           'fixedAt': datetime.now().isoformat(timespec='seconds'), 'reportWindowRun': False}
    SELECTION.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


# --- the single path: v3 runs, evaluation, forecast ---------------------------------------------------------------
def cmd_runs(args):
    """v3's deterministic run of the window's races, made again by the hybrid's path (s4.point_runs on the same model)."""
    import torch
    import s4
    import train as tr
    tr.DEVICE = torch.device(args.device)
    races, r = tr.load_tables()
    codes = s4.window_codes(races, args.window)
    data, X, C = s4.make_data(races, r, codes)
    point = s4.point_runs(s4.load_model(f'{s4.POINT}_{args.window}_seed{args.seed}', X, C), data)
    point.to_pickle(OUT / f'hybrid_point_{args.window}_seed{args.seed}.pkl')
    print(f'{args.window} seed {args.seed}: {len(codes)} races, {len(point)} horses ({args.device})', flush=True)


def inherited_check(f, rec, nums):
    """Per-horse comparison of the hybrid's v3 outputs with the S4 record of the same race."""
    r = rec.loc[nums]
    T = r['T'].to_numpy()
    order = sorted(range(len(nums)), key=lambda i: (T[i], nums[i]))
    fr = np.empty(len(nums), int)
    fr[order] = np.arange(1, len(nums) + 1)
    out = {'rankMismatch': 0, 'finRankMismatch': int((fr != f['finRank']).sum()),
           'last3MaxDiff': float(np.max(np.abs(r['last3'].to_numpy() - f['last3']))),
           'last3RoundedMismatch': int((np.round(r['last3'].to_numpy(), 1) != np.round(f['last3'], 1)).sum())}
    for c in f['corners']:
        out['rankMismatch'] += int((int_rank(r[f'ct{c}'].to_numpy()) != f['rank'][c]).sum())
    return out


def cmd_evaluate(args):
    import eval_standard as es
    import s4
    choice = json.loads(SELECTION.read_text(encoding='utf-8'))
    cand = choice['candidate']
    act = ev.actuals()
    races = pd.read_pickle(OUT / 'races.pkl')
    pa, norms = s4.load_pace()
    res = {'candidate': cand, 'selection': choice, 'windows': {}}
    for window in ('sel', 'rep'):
        for seed in SEEDS:
            key = f'{window}/seed{seed}'
            parts = pd.read_pickle(OUT / f'hybrid_v20_{window}_seed{seed}.pkl')
            k3 = fit_c3(pd.read_pickle(OUT / f'hybrid_valruns_{window}_seed{seed}.pkl'), act)
            k_fin = k3['fin']['k']
            c2 = fit_c2(parts['train'])
            point = pd.read_pickle(OUT / f'hybrid_point_{window}_seed{seed}.pkl')
            rec = pd.read_pickle(OUT / f's4_point_{window}_seed{seed}.pkl')
            base = pd.read_pickle(OUT / f'v20_{window}_seed{seed}.pkl')
            common = sorted(set(base.RACE_CODE) & set(point.RACE_CODE))
            tab = es.RaceTable()
            chk = {'rankMismatch': 0, 'finRankMismatch': 0, 'last3MaxDiff': 0.0, 'last3RoundedMismatch': 0, 'fieldMismatch': 0,
                   'paceMaxDiff': 0.0, 'paceClassMismatch': 0, 'monotone': 0, 'order': 0, 'horses': 0}
            for code in common:
                b = base[base.RACE_CODE == code].set_index('UMABAN')
                v = point[point.RACE_CODE == code].set_index('UMABAN')
                feats = parts['races'].loc[code].to_dict()
                f = race_forecast(v, cand, race_ctx(feats, c2, k3), k_fin)
                mono, order = consistency(f)
                chk['monotone'] += int(mono > TOL)
                chk['order'] += int(order > TOL)
                nums = [n for n in b.index if n in v.index]
                chk['fieldMismatch'] += int(sorted(nums) != sorted(f['numbers']))
                pos = [f['numbers'].index(n) for n in nums]
                ic = inherited_check(f, rec[rec.RACE_CODE == code].set_index('UMABAN'), f['numbers'])
                for k in ('rankMismatch', 'finRankMismatch', 'last3RoundedMismatch'):
                    chk[k] += ic[k]
                chk['last3MaxDiff'] = max(chk['last3MaxDiff'], ic['last3MaxDiff'])
                chk['horses'] += len(f['numbers'])
                # pace: v2.0's recomputed value, carried as it is
                pace_h = float(feats['pace'])
                chk['paceMaxDiff'] = max(chk['paceMaxDiff'], abs(pace_h - float(b.pace.iloc[0])))
                chk['paceClassMismatch'] += int(s4.pace_class(pace_h) != s4.pace_class(float(b.pace.iloc[0])))
                # E0's measures: v2.0, the hybrid, and S4's record (gaps x 5, finish = finish-time gap, v3's own pace)
                n = len(nums)
                a = act.loc[[(code, u) for u in nums]].reset_index(level=0, drop=True)
                keys = list(nums)
                r4 = rec[rec.RACE_CODE == code].set_index('UMABAN').loc[nums]
                normal = a.normal.to_numpy() & a.fin.notna().to_numpy()
                l3 = a.last3.to_numpy()
                ok3 = normal & ~np.isnan(l3)
                gapf, place = a.finish_gap.to_numpy(), a.fin.to_numpy()
                T4 = r4['T'].to_numpy()
                sysv = {'v20': {'last3': b.loc[nums, 'last3'].to_numpy(), 'fin': b.loc[nums, 'finSeconds'].to_numpy()},
                        'hyb': {'last3': f['last3'][pos], 'fin': f['fin'][pos] / LPS},
                        's4': {'last3': r4['last3'].to_numpy(), 'fin': T4 - T4.min()}}
                for s, p in sysv.items():
                    tab.add('last3', s, code, np.where(ok3, np.abs(p['last3'] - l3), np.nan), keys)
                    tab.add('finGap', s, code, np.where(normal & ~np.isnan(gapf), np.abs(p['fin'] - gapf), np.nan), keys)
                    if normal.sum() >= 3:
                        tab.add_value('finTau', s, code, es.kendall(p['fin'][normal], place[normal]))
                        tab.add_value('winner', s, code, float(place[normal][np.argmin(p['fin'][normal])] == 1))
                for c in b.corners.iloc[0]:
                    if c not in f['corners'] or not labelled(a, c):
                        continue
                    y, g = a[f'r{c}'].to_numpy(), a[f'g{c}'].to_numpy()
                    ct4 = r4[f'ct{c}'].to_numpy()
                    ranks = {'v20': (b.loc[nums, f'rank{c}'].to_numpy() - 1) / (n - 1), 'hyb': (f['rank'][c][pos] - 1) / (n - 1),
                             's4': (int_rank(ct4) - 1) / (n - 1)}
                    gaps = {'v20': b.loc[nums, f'gap{c}'].to_numpy(), 'hyb': f['gap'][c][pos], 's4': (ct4 - ct4.min()) * LPS}
                    for s in ranks:
                        tab.add_value(f'r{c}.tau', s, code, es.kendall(ranks[s], y))
                        tab.add(f'r{c}.integer', s, code, np.abs(ranks[s] - y), keys)
                        tab.add(f'r{c}.places', s, code, np.abs(ranks[s] - y) * (n - 1), keys)
                        tab.add(f'g{c}', s, code, np.abs(gaps[s] - g), keys)
                if code in pa.index and not np.isnan(pa.at[code, 'pace']):
                    rc = races.loc[code]
                    pact = float(pa.at[code, 'pace'])
                    pv = {'v20': float(b.pace.iloc[0]), 'hyb': pace_h,
                          's4': float(r4.lead600.iloc[0]) - s4.norm_of(norms, rc.venue, rc.track, rc.distance, rc.surface)}
                    for s, p in pv.items():
                        tab.add_value('pace.mae', s, code, abs(p - pact))
                        tab.add_value('pace.hit', s, code, float(s4.pace_class(p) == s4.pace_class(pact)))
            out = {'races': len(common), 'kFinish': k_fin, 'checks': chk}
            for m in sorted({k[0] for k in tab.rows}):
                out[m] = {'v20': tab.mean(m, 'v20'), 'hyb': tab.mean(m, 'hyb'), 's4': tab.mean(m, 's4'),
                          'vsV20': tab.diff(m, 'hyb', 'v20'), 'vsS4': tab.diff(m, 'hyb', 's4')}
            res['windows'][key] = out
            print(f'== {key}: {len(common)} races, k_fin {k_fin:.3f}, checks {chk}', flush=True)
            for m in MAIN_H:
                if m in out:
                    d, d4 = out[m]['vsV20'], out[m]['vsS4']
                    print(f'  {m:12} v2.0 {out[m]["v20"]:.3f}  hyb {out[m]["hyb"]:.3f}  S4 {out[m]["s4"]:.3f}  '
                          f'vs v2.0 {d["diff"]:+.3f} [{d["lo"]:+.3f}, {d["hi"]:+.3f}]  vs S4 {d4["diff"]:+.3f}', flush=True)
    res['verdict'] = verdict(res)
    s4_eval = OUT / 's4_eval.json'
    if s4_eval.exists():   # the ranges are S4's (no range depends on the gaps, the finish margin or the pace: §7)
        res['coverageS4'] = json.loads(s4_eval.read_text(encoding='utf-8')).get('coverage', {})
    (OUT / 'hybrid_eval.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    print('verdict', json.dumps(res['verdict'], indent=1, default=float))


MAIN_H = ([f'r{c}.tau' for c in (1, 2, 3, 4)] + [f'r{c}.integer' for c in (1, 2, 3, 4)] + [f'g{c}' for c in (1, 2, 3, 4)]
          + ['last3', 'pace.mae', 'finTau', 'finGap', 'pace.hit', 'winner'])
IMPROVE = ['g1', 'g2', 'g3', 'g4', 'finGap']   # §3: lower is better, point estimate on both seeds of the report window


def verdict(res):
    out = {'improve': {}, 'inherited': {}, 'consistency': {}}
    for m in IMPROVE:
        ds = {s: res['windows'][f'rep/seed{s}'][m]['vsV20']['diff'] for s in SEEDS}
        out['improve'][m] = {'diff': ds, 'pass': all(d < 0 for d in ds.values())}
    for w in ('sel', 'rep'):
        for s in SEEDS:
            c = res['windows'][f'{w}/seed{s}']['checks']
            out['inherited'][f'{w}/seed{s}'] = bool(c['rankMismatch'] == 0 and c['finRankMismatch'] == 0 and c['last3MaxDiff'] <= 1e-3
                                                    and c['last3RoundedMismatch'] == 0 and c['paceMaxDiff'] <= 1e-9
                                                    and c['paceClassMismatch'] == 0 and c['fieldMismatch'] == 0)
            out['consistency'][f'{w}/seed{s}'] = bool(c['monotone'] == 0 and c['order'] == 0)
    out['pass'] = (all(v['pass'] for v in out['improve'].values()) and all(out['inherited'].values())
                   and all(out['consistency'].values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['v20parts', 'valruns', 'candidates', 'select', 'runs', 'evaluate'])
    ap.add_argument('--window', choices=['sel', 'rep'], default='sel')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--finish', choices=['time', 'identity'], default='time')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = ap.parse_args()
    {'v20parts': cmd_v20parts, 'valruns': cmd_valruns, 'candidates': cmd_candidates, 'select': cmd_select, 'runs': cmd_runs,
     'evaluate': cmd_evaluate}[args.mode](args)


if __name__ == '__main__':
    main()
