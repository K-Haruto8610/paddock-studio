"""v3 S1: train the solo-run simulator on real races, or run the ability-recovery test on synthetic ones.

  python -I scripts/v3/train.py real --window sel|rep --seed 42|7 --pool named|all
      fit on complete races with laps before the window's fit date (the last 90 days before it held out for early
      stopping), predict the window's races -> outputs/v3/s1_<pool>_<window>_seed<seed>.pkl (+ .json log)
  python -I scripts/v3/train.py synthetic --seed 42
      the recovery test (design S1): horses with known abilities run synthetic races through the same simulator;
      the encoder is trained only on the records and its abilities are compared with the truth -> outputs/v3/s1_recovery.json

pool: where training races and history come from. named = named races only (v2.0's rule); all = every race (unnamed
ordinary races too, about 3.7x the rows; design section 9: include them only if S1 shows it helps).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import courses as cs  # noqa: E402
import features as ft  # noqa: E402
import sim  # noqa: E402

ROOT = cs.ROOT
OUT = ROOT / 'outputs' / 'v3'
H = 18
WINDOWS = {'sel': ('20250601', '20250601', '20250831'), 'rep': ('20250901', '20250901', '20251231')}
torch.set_num_threads(6)
DEVICE = torch.device('cpu')  # --device cuda: the RTX 3070 (torch 2.15.0+cu132)


class Data:
    """Per-race padded arrays for a set of races and their rows of `r` (the run table with observations)."""

    def __init__(self, races, r, codes, X, M, C, rowpos):
        races = races.loc[codes]
        R = len(codes)
        self.codes = np.array(codes)
        self.steps = (races.distance.to_numpy() / sim.DX).astype(int)
        self.distance = races.distance.to_numpy(np.float32)
        self.course = ft.course_index(races.venue, races.kind)
        self.surface = races.surface.to_numpy()
        self.going = races.going.fillna(1).to_numpy()
        self.cls = races.cls.fillna(2).to_numpy(np.float32)
        self.log_dist = np.log(self.distance / 1600).astype(np.float32)
        self.slot = np.full((R, H), -1)
        self.umaban = np.zeros((R, H), int)
        g = r.reset_index(drop=True).groupby('RACE_CODE').indices
        for k, code in enumerate(codes):
            idx = g[code][:H]
            self.slot[k, :len(idx)] = idx
            self.umaban[k, :len(idx)] = r.UMABAN.to_numpy()[idx]
        present = self.slot >= 0
        si = np.where(present, self.slot, 0)
        get = lambda col: np.where(present, r[col].to_numpy(np.float32)[si], np.nan)
        normal = np.where(present, r.normal.to_numpy()[si], False)
        self.y_T, self.y_t600, self.y_last3 = get('soha'), get('t600'), get('last3')
        for a in ('y_T', 'y_t600', 'y_last3'):
            setattr(self, a, np.where(normal, getattr(self, a), np.nan))
        self.y_corner = np.stack([get(f'r{c}') for c in (1, 2, 3, 4)], -1)
        complete_c = (np.where(present[:, :, None], ~np.isnan(self.y_corner), True)).all(1)  # [R, 4]
        self.m_corner = ~np.isnan(self.y_corner) & complete_c[:, None, :]
        self.corner_idx = np.full((R, 4), -1)
        marks, mlen = [], []
        for k, (v, tr, dist) in enumerate(zip(races.venue, races.track, races.distance)):
            rc = cs.race_course(v, tr, dist)
            for c, xpos in rc.corner_x.items():
                self.corner_idx[k, c - 1] = int(round(xpos / sim.DX))
            n = int(np.ceil(dist / 200))
            marks.append([int(round((dist - 200 * (n - 1 - i)) / sim.DX)) for i in range(n)])
        self.m_corner &= (self.corner_idx >= 0)[:, None, :]
        Mx = max(len(m) for m in marks)
        self.mark_idx = np.zeros((R, Mx), int)
        self.y_lead = np.full((R, Mx), np.nan, np.float32)
        for k, (m, lead) in enumerate(zip(marks, races.lead)):
            self.mark_idx[k, :len(m)] = m
            if isinstance(lead, np.ndarray) and len(lead) == len(m):
                self.y_lead[k, :len(m)] = lead
        self.present = present
        self.X, self.M, self.C, self.rowpos = X, M, C, rowpos

    def batch(self, ks):
        slot = self.slot[ks]
        present = slot >= 0
        rows = self.rowpos[slot[present]]
        tt = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
        b = {'slot': tt(slot, torch.long), 'present': tt(present, torch.bool), 'flat': tt(np.flatnonzero(present.ravel()), torch.long),
             'X': torch.from_numpy(self.X[rows]), 'M': torch.from_numpy(self.M[rows]), 'C': torch.from_numpy(self.C[rows]),
             'steps': tt(self.steps[ks], torch.long), 'distance': tt(self.distance[ks]), 'course': tt(self.course[ks], torch.long),
             'surface': tt(self.surface[ks], torch.long), 'going': tt(self.going[ks], torch.long), 'cls': tt(self.cls[ks]),
             'log_dist': tt(self.log_dist[ks]), 'corner_idx': tt(self.corner_idx[ks], torch.long), 'mark_idx': tt(self.mark_idx[ks], torch.long)}
        for name in ('T', 't600', 'last3'):
            y = getattr(self, f'y_{name}')[ks]
            b[f'm_{name}'] = tt(~np.isnan(y), torch.bool)
            b[f'y_{name}'] = tt(np.nan_to_num(y))
        b['y_corner'], b['m_corner'] = tt(np.nan_to_num(self.y_corner[ks], nan=9)), tt(self.m_corner[ks], torch.bool)
        y = self.y_lead[ks]
        b['y_lead'], b['m_lead'] = tt(np.nan_to_num(y)), tt(~np.isnan(y) & (self.mark_idx[ks] > 0), torch.bool)
        return {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}

    def batches(self, size, rng=None):
        order = np.argsort(self.steps, kind='stable')
        chunks = [order[i:i + size] for i in range(0, len(order), size)]
        if rng is not None:
            rng.shuffle(chunks)
        return chunks


def fit(model, train, val, seed, max_epochs=40, patience=8, lr=2e-3, size=32, log=None):
    """Adam; the learning rate halves after 2 epochs without a better validation loss; stops after `patience`; keeps the best."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        t0, tot = time.time(), []
        for ks in train.batches(size, rng):
            b = train.batch(ks)
            t, _ = model(b)
            loss, parts = sim.losses(t, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot.append(float(loss.detach()))
        v = evaluate_loss(model, val, 96)
        sched.step(v[0])
        msg = f'epoch {ep} train {np.mean(tot):.4f} val {v[0]:.4f} {json.dumps({k: round(x, 4) for k, x in v[1].items()})} {time.time() - t0:.0f}s'
        print(msg, flush=True)
        if log is not None:
            log.append(msg)
        if v[0] < best - 1e-4:
            best, bad = v[0], 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best


@torch.no_grad()
def evaluate_loss(model, data, size):
    model.eval()
    tot, parts, n = 0.0, {}, 0
    for ks in data.batches(size):
        b = data.batch(ks)
        t, _ = model(b)
        loss, p = sim.losses(t, b)
        tot += float(loss) * len(ks)
        for k, x in p.items():
            parts[k] = parts.get(k, 0) + x * len(ks)
        n += len(ks)
    return tot / n, {k: x / n for k, x in parts.items()}


@torch.no_grad()
def predict(model, data, size=96):
    model.eval()
    rows = []
    for ks in data.batches(size):
        b = data.batch(ks)
        t, ab = model(b)
        T, t600, last3, corners, lead = sim.read_out(t, b)
        for bi, k in enumerate(ks):
            for h in np.flatnonzero(data.present[k]):
                r = {'RACE_CODE': data.codes[k], 'UMABAN': int(data.umaban[k, h]), 'T': float(T[bi, h]), 't600': float(t600[bi, h]),
                     'last3': float(last3[bi, h])}
                for c in range(4):
                    r[f'ct{c + 1}'] = float(corners[bi, h, c]) if data.corner_idx[k, c] >= 0 else np.nan
                for a in sim.ABILITIES:
                    r[f'ab_{a}'] = float(ab[a][bi, h])
                rows.append(r)
    return pd.DataFrame(rows)


def load_tables():
    races = pd.read_pickle(OUT / 'races.pkl')
    runs = pd.read_pickle(OUT / 'runs.pkl')
    r = ft.run_observations(races, runs).reset_index(drop=True)
    return races, r


def cmd_real(args):
    torch.manual_seed(args.seed)
    races, r = load_tables()
    fit_end, start, end = WINDOWS[args.window]
    pool = r.named.to_numpy() if args.pool == 'named' else np.ones(len(r), bool)
    pool = pd.Series(pool, index=r.index)
    ok_race = races.complete & races.lead.notna() & (races.kind != 'straight')
    cand = races[ok_race & (races.named if args.pool == 'named' else True)]
    val_start = (pd.Timestamp(fit_end) - pd.Timedelta(days=90)).strftime('%Y%m%d')
    train_codes = cand[cand.date < val_start].index.tolist()
    val_codes = races[ok_race & races.named & races.date.between(val_start, fit_end, inclusive='left')].index.tolist()
    # the window's races: every complete named race on a course with corners (evaluate.py pairs them with v2.0's)
    eval_codes = races[races.complete & races.named & (races.kind != 'straight') & races.date.between(start, end)].index.tolist()
    target_rows = np.flatnonzero(r.RACE_CODE.isin(set(train_codes) | set(val_codes) | set(eval_codes)).to_numpy())
    t0 = time.time()
    X, M = ft.history_tensor(r, pool, target_rows)
    C = ft.condition_matrix(r.iloc[target_rows])
    rowpos = np.full(len(r), -1)
    rowpos[target_rows] = np.arange(len(target_rows))
    print(f'history tensor {X.shape} in {time.time() - t0:.0f}s; train {len(train_codes)} val {len(val_codes)} eval {len(eval_codes)} races', flush=True)
    mk = lambda codes: Data(races, r, codes, X, M, C, rowpos)
    train, val, ev = mk(train_codes), mk(val_codes), mk(eval_codes)
    model = sim.Model(X.shape[2], C.shape[1]).to(DEVICE)
    log = []
    best = fit(model, train, val, args.seed, log=log, max_epochs=args.epochs, size=args.batch)
    pred = predict(model, ev)
    name = f's1_{args.pool}{args.tag}_{args.window}_seed{args.seed}'
    pred.to_pickle(OUT / f'{name}.pkl')
    torch.save(model.state_dict(), OUT / f'{name}.pt')
    ph = model.physics
    info = {'best_val': best, 'log': log, 'train_races': len(train_codes), 'val_races': len(val_codes), 'eval_races': len(eval_codes),
            'physics': {'relax_m': float(torch.exp(ph.relax)), 'recovery': float(torch.sigmoid(ph.recovery)),
                        'fatigue_ms': float(torch.exp(ph.fatigue)), 'e_soft_m': float(torch.exp(ph.e_soft)),
                        'dash_ms2': float(3 + 5 * torch.sigmoid(ph.dash)), 'abilitySet': sim.ABILITY_SET, 'loss': sim.LOSS,
                        'dist_slope': ph.dist_slope.tolist(), 'surface0': ph.surface0.tolist(), 'cls_slope': float(ph.cls_slope)}}
    (OUT / f'{name}.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    print('wrote', name, flush=True)


# --- recovery test --------------------------------------------------------------------------------------------
def cmd_synthetic(args):
    """Known abilities -> synthetic records -> encoder trained on records only -> compare with the truth (held-out horses)."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    races, r = load_tables()
    # course templates: real (venue, track, distance) combinations of named races
    tmpl = races[races.named & races.complete & (races.kind != 'straight')][['venue', 'track', 'kind', 'distance', 'surface']].drop_duplicates()
    n_horses, per_horse, field = args.horses, 8, 12
    true_z = rng.normal(0, 1, (n_horses, len(sim.ABILITIES))).astype(np.float32)
    true_z[:, 0] *= 0.7
    # each "meeting": `field` random horses, a random course; every horse runs per_horse meetings
    starts = np.repeat(np.arange(n_horses), per_horse)
    rng.shuffle(starts)
    n_races = len(starts) // field
    rows, rl = [], []
    phys = sim.Physics()
    for k in range(n_races):
        c = tmpl.iloc[rng.integers(len(tmpl))]
        rl.append({'RACE_CODE': f'S{k:06d}', 'date': (pd.Timestamp('2020-01-01') + pd.Timedelta(days=int(k * 3650 / n_races))).strftime('%Y%m%d'),
                   'venue': int(c.venue), 'track': int(c.track), 'kind': c.kind, 'distance': float(c.distance), 'field': field,
                   'surface': int(c.surface), 'going': 1.0, 'cls': 2.0, 'named': True, 'complete': True})
        for j, h in enumerate(starts[k * field:(k + 1) * field]):
            rows.append({'RACE_CODE': f'S{k:06d}', 'horse': f'H{h:05d}', 'UMABAN': j + 1, 'hid': h})
    syn_races = pd.DataFrame(rl).set_index('RACE_CODE')
    syn = pd.DataFrame(rows)
    # run the truth through the simulator (par = the untrained physics' par: known, the same for the fitted model)
    with torch.no_grad():
        dist = torch.tensor(syn_races.distance.to_numpy(np.float32))
        par = phys.par(torch.tensor(ft.course_index(syn_races.venue, syn_races.kind)), torch.tensor(syn_races.surface.to_numpy()),
                       torch.tensor(syn_races.going.to_numpy()), torch.log(dist / 1600), torch.tensor(syn_races.cls.to_numpy(np.float32)))
        Z = torch.tensor(true_z[syn.hid.to_numpy()].reshape(n_races, field, -1))
        ab = sim.physical(Z, par[:, None], phys)
        t = sim.run(ab, phys, dist, int(dist.max() / sim.DX))
        steps = (dist / sim.DX).long()
        ar = torch.arange(n_races)
        noise = lambda: torch.randn(n_races, field) * 0.15
        T = (t[ar, :, steps] + noise()).numpy()
        t600 = (t[ar, :, steps - 24] + noise()).numpy()
        marks = []
        for k in range(n_races):
            dd = syn_races.distance.iat[k]
            n = int(np.ceil(dd / 200))
            idx = [int(round((dd - 200 * (n - 1 - i)) / sim.DX)) for i in range(n)]
            marks.append(t[k, :, idx].min(0).values.numpy() + rng.normal(0, .05, n))
        corner_t = {}
        for k in range(n_races):
            rc = cs.race_course(syn_races.venue.iat[k], syn_races.track.iat[k], syn_races.distance.iat[k])
            corner_t[k] = {c: t[k, :, int(round(x / sim.DX))].numpy() for c, x in rc.corner_x.items()}
    syn_races['lead'] = marks
    syn['soha'] = T.ravel()
    syn['t600'] = t600.ravel()
    syn['last3'] = syn.soha - syn.t600
    syn['normal'] = True
    syn['fin'] = pd.Series(syn.soha.to_numpy()).groupby(syn.RACE_CODE.to_numpy()).rank().to_numpy()
    syn['finish_gap'] = syn.soha - syn.groupby('RACE_CODE').soha.transform('min')
    for c in (1, 2, 3, 4):
        vals = np.full(len(syn), np.nan)
        for k in range(n_races):
            if c in corner_t[k]:
                vals[k * field:(k + 1) * field] = (np.argsort(np.argsort(corner_t[k][c])) / (field - 1))
        syn[f'r{c}'] = vals
    for col, v in (('waku', 4), ('age', 4), ('sex', 1), ('weight', 55), ('blinker', 0), ('rating_pre', np.nan)):
        syn[col] = v
    syn['date'] = syn.RACE_CODE.map(syn_races.date)
    rr = ft.run_observations(syn_races, syn).reset_index(drop=True)
    pool = pd.Series(True, index=rr.index)
    target_rows = np.arange(len(rr))
    X, M = ft.history_tensor(rr, pool, target_rows)
    C = ft.condition_matrix(rr)
    rowpos = np.arange(len(rr))
    # held-out horses: their races are the test set; train on races without them
    test_h = set(range(int(n_horses * 0.8), n_horses))
    has_test = syn.groupby('RACE_CODE').hid.agg(lambda s: bool(set(s) & test_h))
    codes = syn_races.index.to_numpy()
    tr = [c for c in codes if not has_test[c]]
    rng.shuffle(tr)
    nval = len(tr) // 10
    mk = lambda cc: Data(syn_races, rr, list(cc), X, M, C, rowpos)
    train, val, test = mk(tr[nval:]), mk(tr[:nval]), mk([c for c in codes if has_test[c]])
    model = sim.Model(X.shape[2], C.shape[1]).to(DEVICE)
    model.physics.load_state_dict(phys.state_dict())
    for p in model.physics.parameters():
        p.requires_grad_(False)   # mechanism known: the test is whether the abilities are identifiable from the records
    fit(model, train, val, args.seed, max_epochs=args.epochs, patience=12, size=args.batch)
    pred = predict(model, test)
    key = syn.set_index(['RACE_CODE', 'UMABAN']).hid
    pred['hid'] = [key[(a, b)] for a, b in zip(pred.RACE_CODE, pred.UMABAN)]
    with torch.no_grad():
        par_by_race = dict(zip(codes, par.cpu().numpy()))
        truth = sim.physical(torch.tensor(true_z[pred.hid.to_numpy()]), torch.tensor([par_by_race[c] for c in pred.RACE_CODE]), phys)
    # only rows with history (the first run of a horse has nothing to recover from)
    hist = np.array([test.M[test.rowpos[i]].any() for i in rr.set_index(['RACE_CODE', 'UMABAN']).index.get_indexer(list(zip(pred.RACE_CODE, pred.UMABAN)))])
    res = {'horses': n_horses, 'runsPerHorse': per_horse, 'testRows': int(hist.sum()), 'noise_s': 0.15, 'corr': {}}
    for a in sim.ABILITIES:
        if sim.ABILITY_SET == 'merged' and a == 'dash':
            continue
        tv = truth[a].numpy()[hist]
        pv = pred[f'ab_{a}'].to_numpy()[hist]
        if a == 'vc':
            pr = np.array([par_by_race[c] for c in pred.RACE_CODE])[hist]
            tv, pv = tv - pr, pv - pr
        res['corr'][a] = float(np.corrcoef(tv, pv)[0, 1])
        print(f'{a:8} corr(true, recovered) {res["corr"][a]:+.3f}')
    # pairwise: which abilities the records cannot tell apart (correlated errors)
    names = [a for a in sim.ABILITIES if not (sim.ABILITY_SET == 'merged' and a == 'dash')]
    errs = np.column_stack([(pred[f'ab_{a}'].to_numpy() - truth[a].numpy())[hist] for a in names])
    res['abilitySet'], res['errorCorrNames'] = sim.ABILITY_SET, names
    res['errorCorr'] = np.round(np.corrcoef(errs.T), 3).tolist()
    out = OUT / f's1_recovery{args.tag}_seed{args.seed}.json'
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['real', 'synthetic'])
    ap.add_argument('--window', choices=list(WINDOWS), default='sel')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pool', choices=['named', 'all'], default='all')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--horses', type=int, default=3000)
    ap.add_argument('--loss', choices=['absolute', 'centred'], default='centred')
    ap.add_argument('--abilities', choices=['six', 'merged'], default='six')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--batch', type=int, default=32, help='races per training batch')
    ap.add_argument('--tag', default='', help='suffix of the run name (outputs/v3/s1_<pool><tag>_...)')
    args = ap.parse_args()
    sim.LOSS, sim.ABILITY_SET = args.loss, args.abilities
    global DEVICE
    DEVICE = torch.device(args.device)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    {'real': cmd_real, 'synthetic': cmd_synthetic}[args.mode](args)


if __name__ == '__main__':
    main()
