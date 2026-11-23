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


def cmd_candidates(args):
    import eval_standard as es
    act = ev.actuals()
    res = {'seeds': {}, 'c3': {}, 'c2': {}}
    for seed in SEEDS:
        parts = pd.read_pickle(OUT / f'hybrid_v20_sel_seed{seed}.pkl')
        base = pd.read_pickle(OUT / f'v20_sel_seed{seed}.pkl')
        v3 = pd.read_pickle(OUT / f's4_point_sel_seed{seed}.pkl')
        k3 = fit_c3(pd.read_pickle(OUT / f'hybrid_valruns_sel_seed{seed}.pkl'), act)
        c2 = fit_c2(parts['train'])
        res['c3'][seed] = k3
        res['c2'][seed] = {c: {k: m[k] for k in ('rows', 'coef', 'intercept')} for c, m in c2.items()}
        print('seed', seed, 'C3 factors', {c: round(v['k'], 3) for c, v in k3.items()}, flush=True)
        tab = es.RaceTable()
        cons = {cand: {'monotone': 0, 'identity': 0, 'order': 0, 'races': 0} for cand in CANDIDATES + ('C0_S4',)}
        worst = {cand: [] for cand in CANDIDATES + ('C0_S4',)}
        spread_err = {cand: {c: [] for c in (1, 2, 3, 4)} for cand in CANDIDATES + ('v20',)}
        empty_scale, infeasible, races_n, fin_spread = 0, 0, 0, []
        common = sorted(set(base.RACE_CODE) & set(v3.RACE_CODE))
        for code in common:
            b = base[base.RACE_CODE == code].set_index('UMABAN')
            v = v3[v3.RACE_CODE == code].set_index('UMABAN')
            nums = [n for n in b.index if n in v.index]
            a = act.loc[[(code, u) for u in nums]].reset_index(level=0, drop=True)
            keys = list(nums)
            pr = parts['races'].loc[code]
            ctx = {'k3': {c: k3[c]['k'] for c in k3}, 'S1': {c: float(pr[f'spread{c}']) for c in (1, 2, 3, 4)},
                   'S2': {c: predict_c2(c2, c, {**pr.to_dict(), 'pace_pred': float(pr['pace'])}) for c in (1, 2, 3, 4)}}
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
            for cand in CANDIDATES:
                g = {c: corner_gaps(cand, ctm[c], c, ctx) for c in shown}
                mono = max(monotone_violation(g[c], np.argsort(ctm[c], kind='stable')) for c in shown)
                fin = identity_finish(g[last], l3)
                ident = float(np.max(np.abs(fin - identity_finish(g[last], l3))))   # 0 by construction (recomputed)
                order = monotone_violation(fin, np.argsort(T, kind='stable'))
                cons[cand]['races'] += 1
                cons[cand]['monotone'] += int(mono > TOL)
                cons[cand]['identity'] += int(ident > TOL)
                cons[cand]['order'] += int(order > TOL)
                worst[cand].append(order)
                tab.add('finGap', cand, code, np.where(ok_fin, np.abs(fin / LPS - gap_fin), np.nan), keys)
                for c in b.corners.iloc[0]:
                    if c not in shown or not labelled(a, c):
                        continue
                    tab.add(f'g{c}', cand, code, np.abs(g[c] - a[f'g{c}'].to_numpy()), keys)
                    spread_err[cand][c].append(abs(float(np.quantile(g[c], SPREAD_Q)) - float(np.quantile(a[f'g{c}'].to_numpy(), SPREAD_Q))))
            # C0 as S4 published it: finish = 5 x finish-time gap (condition 3 holds, the identity does not)
            fin_s4 = LPS * (T - T.min())
            g0 = LPS * (ctm[last] - ctm[last].min())
            cons['C0_S4']['races'] += 1
            cons['C0_S4']['identity'] += int(np.max(np.abs(fin_s4 - identity_finish(g0, l3))) > TOL)
            cons['C0_S4']['order'] += int(monotone_violation(fin_s4, np.argsort(T, kind='stable')) > TOL)
            worst['C0_S4'].append(float(np.max(np.abs(fin_s4 - identity_finish(g0, l3)))))
            tab.add('finGap', 'C0_S4', code, np.where(ok_fin, np.abs(fin_s4 / LPS - gap_fin), np.nan), keys)
            tab.add('finGap', 'F_k', code, np.where(ok_fin, np.abs(k3['fin']['k'] * (T - T.min()) - gap_fin), np.nan), keys)
            if ok_fin.sum() >= 3:
                q = lambda z: float(np.quantile(z[ok_fin], SPREAD_Q))  # noqa: E731
                fin_spread.append((q(gap_fin), q(T - T.min()), q(b.loc[nums, 'finSeconds'].to_numpy())))
            for c in b.corners.iloc[0]:
                if c in shown and labelled(a, c):
                    tab.add(f'g{c}', 'v20', code, np.abs(b.loc[nums, f'gap{c}'].to_numpy() - a[f'g{c}'].to_numpy()), keys)
                    spread_err['v20'][c].append(abs(float(np.quantile(b.loc[nums, f'gap{c}'].to_numpy(), SPREAD_Q))
                                                    - float(np.quantile(a[f'g{c}'].to_numpy(), SPREAD_Q))))
        out = {'races': races_n, 'scaleIntervalEmpty': empty_scale, 'anyMonotoneInfeasible': infeasible, 'consistency': cons,
               'orderViolationLengths': {k: [float(q) for q in np.quantile(w, [.5, .75, .9, 1])] for k, w in worst.items()},
               'mae': {}, 'spreadMae': {k: {c: float(np.mean(x)) if x else None for c, x in v.items()} for k, v in spread_err.items()},
               'finishSpread80': dict(zip(('actual', 'v3', 'v20'), map(float, np.median(np.array(fin_spread), 0))))}
        for m in ('g1', 'g2', 'g3', 'g4', 'finGap'):
            out['mae'][m] = {}
            for s in ('v20',) + CANDIDATES + (('C0_S4', 'F_k') if m == 'finGap' else ()):
                out['mae'][m][s] = {'mae': tab.mean(m, s), 'rows': tab.count(m, s)}
                if s != 'v20':
                    out['mae'][m][s]['vsV20'] = tab.diff(m, s, 'v20')
                    if s != 'C0':
                        out['mae'][m][s]['vsC0'] = tab.diff(m, s, 'C0')
        res['seeds'][seed] = out
        print(f'== seed {seed}: {races_n} races; scale interval empty {empty_scale}; no monotone corner-4 gaps at all {infeasible}')
        for m, row in out['mae'].items():
            print('  ', m, '  '.join(f'{s} {v["mae"]:.3f}' for s, v in row.items()))
        for cand, cc in cons.items():
            print('   consistency', cand, cc)
    (OUT / 'hybrid_candidates.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['v20parts', 'valruns', 'candidates'])
    ap.add_argument('--window', choices=['sel', 'rep'], default='sel')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = ap.parse_args()
    {'v20parts': cmd_v20parts, 'valruns': cmd_valruns, 'candidates': cmd_candidates}[args.mode](args)


if __name__ == '__main__':
    main()
