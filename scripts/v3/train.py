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
        # S3: the turn of every step, the start positions from the gate, the passing-string lengths at each corner
        self.turn = np.zeros((R, int(self.steps.max())), np.float32)
        for k, (v, tr, dist) in enumerate(zip(races.venue, races.track, races.distance)):
            tn = cs.race_course(v, tr, dist, sim.DX).turn
            self.turn[k, :len(tn)] = tn
        self.y0 = np.where(present, (self.umaban - 1) * sim.STALL, 0).astype(np.float32)
        self.y_gap = np.stack([get(f'g{c}') for c in (1, 2, 3, 4)], -1)
        self.m_gap = ~np.isnan(self.y_gap) & self.m_corner
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
             'log_dist': tt(self.log_dist[ks]), 'corner_idx': tt(self.corner_idx[ks], torch.long), 'mark_idx': tt(self.mark_idx[ks], torch.long),
             'turn': tt(self.turn[ks][:, :int(self.steps[ks].max())]), 'y0': tt(self.y0[ks]),
             'y_gap': tt(np.nan_to_num(self.y_gap[ks])), 'm_gap': tt(self.m_gap[ks], torch.bool)}
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
        if getattr(model, 'frozen_encoder', False):
            model.encoder.eval()
        t0, tot = time.time(), []
        for ks in train.batches(size, rng):
            b = train.batch(ks)
            t, _ = model(b)
            loss, parts = sim.read_loss(t, b, model.K, model.physics)
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
    gen = torch.Generator(device=DEVICE).manual_seed(0)   # the same draws every epoch: early stopping not on noise
    for ks in data.batches(size):
        b = data.batch(ks)
        t, _ = model(b, gen=gen)
        loss, p = sim.read_loss(t, b, model.K, model.physics)
        tot += float(loss) * len(ks)
        for k, x in p.items():
            parts[k] = parts.get(k, 0) + x * len(ks)
        n += len(ks)
    return tot / n, {k: x / n for k, x in parts.items()}


PRED_SAMPLES = 1   # runs per race when predicting (32 with stochastic runs)


@torch.no_grad()
def predict(model, data, size=96):
    """Per horse: the read-outs averaged over the runs (sim.read_mean) and, in S3, the lane in the abreast group per
    run (evaluate.v3_lanes on that run's corner times and lateral positions), its median over the runs."""
    import evaluate as ev
    model.eval()
    rows = []
    K = PRED_SAMPLES
    gen = torch.Generator(device=DEVICE).manual_seed(12345)
    for ks in data.batches(max(8, size // K)):
        b = data.batch(ks)
        t, ab = model(b, samples=K, gen=gen)
        ro = sim.read_mean(t, b, K)
        T, t600, last3, corners, lead600 = ro['T'], ro['t600'], ro['last3'], ro['corners'], ro['lead600']
        lengths = ro['behind'] * torch.exp(model.physics.gap_scale)
        Y = model.y
        if Y is not None:
            bk = sim.repeat_batch(b, K)
            idx = bk['corner_idx'][:, None, :].expand(t.shape[0], t.shape[1], 4).clamp(min=0)
            ck = torch.gather(t, 2, idx).cpu().numpy()
            yk = torch.gather(Y, 2, idx).cpu().numpy()
            ycorner = yk.reshape(K, len(ks), *yk.shape[1:]).mean(0)
            yfin = Y[torch.arange(Y.shape[0], device=Y.device), :, bk['steps']].view(K, len(ks), -1).mean(0)
            lanes = np.full((len(ks), t.shape[1], 4), np.nan)
            for bi, k in enumerate(ks):
                hs = np.flatnonzero(data.present[k])
                for c in range(4):
                    if data.corner_idx[k, c] < 0:
                        continue
                    per = [ev.v3_lanes(ck[j * len(ks) + bi, hs, c], yk[j * len(ks) + bi, hs, c]) for j in range(K)]
                    lanes[bi, hs, c] = np.median(per, 0)
        for bi, k in enumerate(ks):
            for h in np.flatnonzero(data.present[k]):
                r = {'RACE_CODE': data.codes[k], 'UMABAN': int(data.umaban[k, h]), 'T': float(T[bi, h]), 't600': float(t600[bi, h]),
                     'last3': float(last3[bi, h]), 'lead600': float(lead600[bi])}
                for c in range(4):
                    r[f'ct{c + 1}'] = float(corners[bi, h, c]) if data.corner_idx[k, c] >= 0 else np.nan
                    r[f'gl{c + 1}'] = float(lengths[bi, h, c]) if data.corner_idx[k, c] >= 0 else np.nan
                    if Y is not None:
                        r[f'y{c + 1}'] = float(ycorner[bi, h, c]) if data.corner_idx[k, c] >= 0 else np.nan
                        r[f'lane{c + 1}'] = float(lanes[bi, h, c])
                if Y is not None:
                    r['yfin'] = float(yfin[bi, h])
                for a in sim.ABILITIES:
                    r[f'ab_{a}'] = float(ab[a][bi, h])
                rows.append(r)
    return pd.DataFrame(rows)


def load_tables():
    races = pd.read_pickle(OUT / 'races.pkl')
    runs = pd.read_pickle(OUT / 'runs.pkl')
    r = ft.run_observations(races, runs).reset_index(drop=True)
    return races, r


def prepare(args):
    """The window's train / validation / evaluation races and a maker of their Data (history tensors built once)."""
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
    return mk, train_codes, val_codes, eval_codes, X, C


def new_model(args, X, C):
    model = sim.Model(X.shape[2], C.shape[1]).to(DEVICE)
    if args.init:
        state = torch.load(OUT / f'{args.init}.pt', map_location=DEVICE)
        missing, _ = model.load_state_dict(state, strict=False)
        print(f'warm start from {args.init} (new: {missing})', flush=True)
    if args.train_only:
        keep = set(args.train_only.split(','))
        for name, p in model.named_parameters():
            p.requires_grad = name.split('.')[-1] in keep
        model.frozen_encoder = True   # its dropout stays off in fit (the abilities are the --init model's)
        print('training only', sorted(n for n, p in model.named_parameters() if p.requires_grad), flush=True)
    return model


def cmd_real(args):
    torch.manual_seed(args.seed)
    mk, train_codes, val_codes, eval_codes, X, C = prepare(args)
    train, val, ev = mk(train_codes), mk(val_codes), mk(eval_codes)
    model = new_model(args, X, C)
    log = []
    if args.epochs > 0:
        best = fit(model, train, val, args.seed, log=log, max_epochs=args.epochs, size=args.batch)
    else:
        best = evaluate_loss(model, val, 96)[0]   # --epochs 0: the --init model as it is (stochastic runs only)
    pred = predict(model, ev)
    name = f'{args.stage}_{args.pool}{args.tag}_{args.window}_seed{args.seed}'
    pred.to_pickle(OUT / f'{name}.pkl')
    torch.save(model.state_dict(), OUT / f'{name}.pt')
    ph = model.physics
    info = {'best_val': best, 'log': log, 'train_races': len(train_codes), 'val_races': len(val_codes), 'eval_races': len(eval_codes),
            'physics': {'relax_m': float(torch.exp(ph.relax)), 'recovery': float(torch.sigmoid(ph.recovery)),
                        'fatigue_ms': float(torch.exp(ph.fatigue)), 'e_soft_m': float(torch.exp(ph.e_soft)),
                        'dash_ms2': float(3 + 5 * torch.sigmoid(ph.dash)), 'stage': sim.STAGE, 'span': sim.CONTEND_SPAN, 'settle_m': float(torch.exp(ph.settle)), 'wish': sim.WISH, 'wish_scale': float(torch.exp(ph.wish_scale)),
                        'contend_ms': float(torch.nn.functional.softplus(ph.contend)), 'shelter_save': float(0.5 * torch.sigmoid(ph.shelter)), 'abilitySet': sim.ABILITY_SET, 'loss': sim.LOSS,
                        'hold': float(torch.sigmoid(ph.hold)), 'traffic': sim.TRAFFIC, 'init': args.init,
                        'samples': sim.SAMPLES, 'predSamples': PRED_SAMPLES, 'form': sim.FORM, 'start': sim.START, 'wishn': sim.WISHN, 'read': sim.READ,
                        'trainOnly': args.train_only, 'gap_scale': float(torch.exp(ph.gap_scale)),
                        'dist_slope': ph.dist_slope.tolist(), 'surface0': ph.surface0.tolist(), 'cls_slope': float(ph.cls_slope)}}
    (OUT / f'{name}.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    print('wrote', name, flush=True)


LPS = 5.0   # lengths per second for the corner width of the runs (the fixed conversion of the reports, not learnt)


@torch.no_grad()
def spread(model, data, K, size=48, solo=None):
    """Within-race spread of the runs and of the records, medians over races (normal finishers; corners: horses with a
    passing-string gap): sd of the 600 m-mark time and the finish time (s), sd at corners 3 and 4 (lengths; the runs'
    seconds behind the corner leader x LPS), and (S3, `solo` = {(code, umaban, corner): label}) the share of horses
    running alone (lane 0) at corners 3 and 4. Each run is measured on its own, then averaged over the K runs."""
    import evaluate as ev
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(777)
    out = {k: [] for k in ('p600', 'pT', 'a600', 'aT', 'p3', 'p4', 'a3', 'a4', 'z3', 'z4', 'y3', 'y4')}
    for ks in data.batches(size):
        b = data.batch(ks)
        t, _ = model(b, samples=K, gen=gen)
        bk = sim.repeat_batch(b, K)
        Tk, t6k, _, ck, _ = sim.read_out(t, bk)
        n = len(ks)
        Tk, t6k = Tk.view(K, n, -1).cpu().numpy(), t6k.view(K, n, -1).cpu().numpy()
        ck = ck.view(K, n, *ck.shape[1:]).cpu().numpy()
        yk = None
        if model.y is not None:
            idx = bk['corner_idx'][:, None, :].expand(t.shape[0], t.shape[1], 4).clamp(min=0)
            yk = torch.gather(model.y, 2, idx).view(K, n, t.shape[1], 4).cpu().numpy()
        m = (b['m_T'] & b['m_t600']).cpu().numpy()
        yT, y6 = b['y_T'].cpu().numpy(), b['y_t600'].cpu().numpy()
        yg, mg = b['y_gap'].cpu().numpy(), b['m_gap'].cpu().numpy()
        for bi, k in enumerate(ks):
            mm = m[bi]
            if mm.sum() >= 5:
                out['p600'].append(t6k[:, bi, mm].std(1, ddof=1).mean())
                out['pT'].append(Tk[:, bi, mm].std(1, ddof=1).mean())
                out['a600'].append(y6[bi, mm].std(ddof=1))
                out['aT'].append(yT[bi, mm].std(ddof=1))
            hs = np.flatnonzero(data.present[k])
            for c in (2, 3):
                if data.corner_idx[k, c] < 0:
                    continue
                g = mg[bi, hs, c]
                if g.sum() >= 5:
                    out[f'p{c + 1}'].append(np.mean([((ck[j, bi, hs, c] - ck[j, bi, hs, c].min()) * LPS)[g].std(ddof=1)
                                                     for j in range(K)]))
                    out[f'a{c + 1}'].append(yg[bi, hs, c][g].std(ddof=1))
                if yk is not None and solo is not None:
                    lab = np.array([solo.get((data.codes[k], int(data.umaban[k, h]), c + 1), np.nan) for h in hs])
                    ok = ~np.isnan(lab)
                    if ok.sum() >= 5:
                        runs = [ev.v3_lanes(ck[j, bi, hs, c], yk[j, bi, hs, c]) for j in range(K)]
                        out[f'z{c + 1}'].append(np.mean([np.mean(l[ok] == 0) for l in runs]))
                        out[f'y{c + 1}'].append(np.mean(lab[ok] == 0))
    med = lambda x: float(np.median(x)) if x else float('nan')
    mean = lambda x: float(np.mean(x)) if x else float('nan')
    return {'sd600': med(out['p600']), 'sdT': med(out['pT']), 'actual600': med(out['a600']), 'actualT': med(out['aT']),
            'c3': med(out['p3']), 'c4': med(out['p4']), 'actual3': med(out['a3']), 'actual4': med(out['a4']),
            'solo3': mean(out['z3']), 'solo4': mean(out['z4']),
            'actualSolo3': mean(out['y3']), 'actualSolo4': mean(out['y4'])}


def cmd_calibrate(args):
    """Noise sizes so the runs' within-race spread matches the records', on races before the window's fit date only.
    No score is used.
      --calib form  FORM x START grid on 1,500 training + the validation races; target: the sd at the 600 m mark and at
                    the finish (first stochastic runs, 2026-11-22)
      --calib wish  FORM and START as given (--form --start), WISHN grid on 600 training + the validation races; the
                    target adds the sd at corners 3 and 4 and (S3) the share running alone there"""
    torch.manual_seed(args.seed)
    mk, train_codes, val_codes, eval_codes, X, C = prepare(args)
    rng = np.random.default_rng(args.seed)
    n = 1500 if args.calib == 'form' else 600
    codes = list(rng.choice(train_codes, n, replace=False)) + list(val_codes)
    data = mk(codes)
    model = new_model(args, X, C)
    solo = None
    if args.calib == 'wish' and sim.STAGE == 's3':
        lab = pd.read_pickle(OUT / 'lanes.pkl')
        lab = lab[lab.RACE_CODE.isin(set(codes))]
        solo = {(a, int(u), c): v for c in (3, 4) for a, u, v in zip(lab.RACE_CODE, lab.UMABAN, lab[f'l{c}']) if v == v}
    rel = lambda p, a: (p / a - 1) ** 2
    if args.calib == 'form':
        cells = [(f, st, 0.0) for f in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4) for st in (0.0, 0.2, 0.4, 0.7, 1.0)]
    else:
        cells = [(sim.FORM, sim.START, w) for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]
    grid = []
    for form, start, wn in cells:
        sim.FORM, sim.START, sim.WISHN = form, start, wn
        r = spread(model, data, 8, solo=solo)
        err = rel(r['sd600'], r['actual600']) + rel(r['sdT'], r['actualT'])
        if args.calib == 'wish':
            err += rel(r['c3'], r['actual3']) + rel(r['c4'], r['actual4'])
            if solo is not None:
                err += rel(r['solo3'], r['actualSolo3']) + rel(r['solo4'], r['actualSolo4'])
        r.update({'form': form, 'start': start, 'wishn': wn, 'err': err})
        grid.append(r)
        print(' '.join(f'{k} {v:.3f}' for k, v in r.items()), flush=True)
    best = min(grid, key=lambda g: g['err'])
    print('best', best, flush=True)
    suffix = '' if args.calib == 'form' else '_wish'
    (OUT / f'calib{suffix}_{args.init}.json').write_text(json.dumps({'best': best, 'grid': grid, 'races': len(codes)},
                                                                    indent=1), encoding='utf-8')


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
        t, _ = sim.run(ab, phys, dist, int(dist.max() / sim.DX))
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
        if (sim.ABILITY_SET == 'merged' and a == 'dash') or (sim.STAGE == 's1' and a == 'wish'):
            continue
        if sim.WISH == 'intent' and a == 'wish':
            continue
        tv = truth[a].numpy()[hist]
        pv = pred[f'ab_{a}'].to_numpy()[hist]
        if a == 'vc':
            pr = np.array([par_by_race[c] for c in pred.RACE_CODE])[hist]
            tv, pv = tv - pr, pv - pr
        res['corr'][a] = float(np.corrcoef(tv, pv)[0, 1])
        print(f'{a:8} corr(true, recovered) {res["corr"][a]:+.3f}')
    # pairwise: which abilities the records cannot tell apart (correlated errors)
    names = [a for a in sim.ABILITIES if not ((sim.ABILITY_SET == 'merged' and a == 'dash') or (sim.STAGE == 's1' and a == 'wish') or (sim.WISH == 'intent' and a == 'wish'))]
    errs = np.column_stack([(pred[f'ab_{a}'].to_numpy() - truth[a].numpy())[hist] for a in names])
    res['abilitySet'], res['errorCorrNames'] = sim.ABILITY_SET, names
    res['errorCorr'] = np.round(np.corrcoef(errs.T), 3).tolist()
    out = OUT / f'{args.stage}_recovery{args.tag}_seed{args.seed}.json'
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['real', 'synthetic', 'calibrate'])
    ap.add_argument('--window', choices=list(WINDOWS), default='sel')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pool', choices=['named', 'all'], default='all')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--horses', type=int, default=3000)
    ap.add_argument('--loss', choices=['absolute', 'centred'], default='centred')
    ap.add_argument('--abilities', choices=['six', 'merged'], default='six')
    ap.add_argument('--stage', choices=['s1', 's2', 's3'], default='s2')
    ap.add_argument('--span', choices=['kick', 'early'], default='kick', help='S2 contention span (sim.CONTEND_SPAN)')
    ap.add_argument('--wish', choices=['free', 'intent'], default='free', help='S2 wish (sim.WISH)')
    ap.add_argument('--traffic', choices=['full', 'none', 'half'], default='full', help='S3 blocking (sim.TRAFFIC)')
    ap.add_argument('--init', default='', help='warm start from outputs/v3/<init>.pt (calibrate: the model to calibrate)')
    ap.add_argument('--samples', type=int, default=1, help='stochastic runs per race in training (sim.SAMPLES)')
    ap.add_argument('--pred-samples', type=int, default=0, help='runs per race when predicting (default 32 if stochastic, else 1)')
    ap.add_argument('--form', type=float, default=0.0, help='sd of the day form, m/s (sim.FORM)')
    ap.add_argument('--start', type=float, default=0.0, help='scale of the start delay, s (sim.START)')
    ap.add_argument('--wishn', type=float, default=0.0, help='sd of the wished place, logits (sim.WISHN)')
    ap.add_argument('--read', choices=['mean', 'run'], default='mean', help='training losses on averaged runs or per run (sim.READ)')
    ap.add_argument('--calib', choices=['form', 'wish'], default='form', help='calibrate: which noise sizes')
    ap.add_argument('--train-only', default='', help='comma list of parameter names to train, the rest frozen '
                                                     '(e.g. hold,gap_scale: S3 on a deterministic model)')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--batch', type=int, default=32, help='races per training batch')
    ap.add_argument('--tag', default='', help='suffix of the run name (outputs/v3/s1_<pool><tag>_...)')
    args = ap.parse_args()
    sim.LOSS, sim.ABILITY_SET, sim.STAGE, sim.CONTEND_SPAN, sim.WISH = args.loss, args.abilities, args.stage, args.span, args.wish
    sim.TRAFFIC = args.traffic
    sim.SAMPLES, sim.FORM, sim.START, sim.WISHN = args.samples, args.form, args.start, args.wishn
    sim.READ = args.read
    global PRED_SAMPLES
    PRED_SAMPLES = args.pred_samples or (32 if (args.form > 0 or args.start > 0 or args.wishn > 0) else 1)
    global DEVICE  # noqa
    DEVICE = torch.device(args.device)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    {'real': cmd_real, 'synthetic': cmd_synthetic, 'calibrate': cmd_calibrate}[args.mode](args)


if __name__ == '__main__':
    main()
