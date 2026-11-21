"""v3 S1: the solo-run race simulator and the ability encoder (docs/race-simulator-design.md 5.1-5.3, S1).

Every horse runs the race on the distance axis in steps of DX metres, on its own (S1: no interaction). State: speed v and
the anaerobic reserve E (metres that may still be run above the critical speed, the critical-speed model of exercise
physiology). Per step:

  wanted   = cruise target (vc + intent) before the kick point, top speed vmax after it (smooth switch)
  cap      = what the reserve allows: from vmax when E is full down to vc - fatigue when E is empty
  target   = smooth min(wanted, cap)
  v'       = towards target: speeding up no faster than the horse's acceleration (dash: the standing start and the kick),
             slowing down relaxed over RELAX metres
  E'       = E - (v - vc) dt above vc, + RECOVERY (vc - v) dt below vc, never above the horse's reserve W
  t'       = t + DX / mean(v, v')

Abilities of a horse in a race (from the encoder): vc (critical speed), W (reserve), vmax (top speed), dash
(acceleration), intent (early speed over vc) and kick (metres to go where the final effort starts). The race conditions
enter as a par speed (course x surface/distance x going x class) that vc is measured from. Every output is read from the
same run: the time at the 600 m-to-go mark, the finish time (so the closing 3F is their difference), the passing times
at the corners, and the leader's time at every 200 m mark (a smooth minimum over the field).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import features as ft

DX = 25.0          # metres per step (every JRA distance and every 200 m mark is a multiple of 25 m from the post)
ABILITIES = ('vc', 'reserve', 'vmax', 'dash', 'intent', 'kick')


def softmin(x, tau, dim, mask=None):
    if mask is not None:
        x = x.masked_fill(~mask, 1e4)
    return -tau * torch.logsumexp(-x / tau, dim=dim)


class Physics(nn.Module):
    """The mechanism's strengths, shared by every horse (learnt), and the par speed of the race conditions."""

    def __init__(self):
        super().__init__()
        n_course = ft.VENUES * len(ft.KINDS)
        self.course = nn.Embedding(n_course, 1)
        nn.init.zeros_(self.course.weight)
        self.going = nn.Embedding(2 * 4, 1)
        nn.init.zeros_(self.going.weight)
        self.dist_slope = nn.Parameter(torch.tensor([-0.9, -0.9]))   # m/s per log(distance / 1600), turf / dirt
        self.cls_slope = nn.Parameter(torch.tensor(0.05))
        self.surface0 = nn.Parameter(torch.tensor([16.6, 15.9]))     # par at 1600 m, good going, class 2
        self.relax = nn.Parameter(torch.tensor(math.log(60.0)))      # metres to close 63% of a speed gap
        self.recovery = nn.Parameter(torch.tensor(-1.0))             # logit of the reserve refilled below vc
        self.fatigue = nn.Parameter(torch.tensor(math.log(0.8)))     # m/s below vc when the reserve is empty
        self.e_soft = nn.Parameter(torch.tensor(math.log(15.0)))     # metres of reserve over which the cap falls
        self.kick_width = 40.0
        self.dash = nn.Parameter(torch.tensor(0.0))                  # logit of the shared acceleration (ABILITY_SET 'merged')

    def par(self, course, surface, going, log_dist, cls):
        s = surface.long()
        return (self.surface0[s] + self.course(course).squeeze(-1) + self.going(s * 4 + (going.long() - 1).clamp(0, 3)).squeeze(-1)
                + self.dist_slope[s] * log_dist + self.cls_slope * (cls - 2))


ABILITY_SET = 'six'  # adopted in S1 (chosen on the selection window); 'merged' kept for the identifiability follow-up
# 'six'    the six abilities of design 5.1, each from the encoder (first S1 runs).
# 'merged' after the S1 recovery test (reserve, dash and kick not recoverable from the records, 2026-11-22): dash is one
#          value shared by every horse (learnt, Physics.dash), and the kick starts where the reserve would just last at top
#          speed to the post (time W / (vmax - vc), so W vmax / (vmax - vc) metres to go): reserve and kick merged into one.
#          The encoder's outputs 3 and 5 are then unused.


def physical(z, par, physics=None):
    """Encoder output z [..., 6] -> named abilities in physical units (par [...] in m/s)."""
    vc = par + 0.6 * z[..., 0]
    # reserve scale: 'six' as first run; 'merged' 25 m at z = 0 (~43 m), so the derived kick (~650 m to go) sits inside
    # its 150-1500 m range instead of on its upper bound (the first merged test, 2026-11-22)
    W = (25 if ABILITY_SET == 'merged' else 60) * F.softplus(z[..., 1] + 1.5)
    vmax = vc + 0.3 + 1.2 * F.softplus(z[..., 2])
    if ABILITY_SET == 'merged':
        dash = (3 + 5 * torch.sigmoid(physics.dash)).expand_as(vc) if physics is not None else torch.full_like(vc, 5.5)
        kick = (W * vmax / (vmax - vc)).clamp(150, 1500)
    else:
        dash, kick = 3 + 5 * torch.sigmoid(z[..., 3]), 200 + 800 * torch.sigmoid(z[..., 5])
    return {'vc': vc, 'reserve': W, 'vmax': vmax, 'dash': dash, 'intent': 0.8 * torch.tanh(z[..., 4]), 'kick': kick}


def run(ab, physics, distance, steps):
    """Solo runs. ab: abilities [B, H] each; distance [B] (m); steps = max steps in the batch.
    Returns t [B, H, steps + 1]: elapsed time at every DX mark from the start."""
    B, H = ab['vc'].shape
    relax = torch.exp(physics.relax)
    keep = torch.exp(-DX / relax)
    rec = torch.sigmoid(physics.recovery)
    fat = torch.exp(physics.fatigue)
    esoft = torch.exp(physics.e_soft)
    vc, W, vmax, dash = ab['vc'], ab['reserve'], ab['vmax'], ab['dash']
    cruise = vc + ab['intent']
    kick_at = distance[:, None] - ab['kick']                      # metres from the start where the kick starts
    v = torch.zeros(B, H, device=vc.device)
    E = W
    t = [torch.zeros(B, H, device=vc.device)]
    for j in range(steps):
        x = (j + 0.5) * DX
        k = torch.sigmoid((x - kick_at) / physics.kick_width)
        wanted = cruise + k * (vmax - cruise)
        e_pos = F.softplus(E / 2) * 2
        cap = vc - fat + (vmax - vc + fat) * (1 - torch.exp(-e_pos / esoft))
        target = softmin(torch.stack([wanted, cap]), 0.1, 0)
        # speeding up is limited by the horse's acceleration (the standing start, the kick); slowing down relaxes
        up = softmin(torch.stack([target, torch.sqrt(v * v + 2 * dash * DX + 1e-6)]), 0.1, 0)
        down = target + (v - target) * keep
        v_new = torch.where(target > v, up, down)
        vm = 0.5 * (v + v_new)
        dt = DX / vm.clamp(min=1.0)
        above = vm - vc
        E = torch.minimum(E - F.relu(above) * dt + rec * F.relu(-above) * dt, W)
        t.append(t[-1] + dt)
        v = v_new
    return torch.stack(t, -1)


class Encoder(nn.Module):
    """Past runs (+ the target's conditions) -> 6 ability values (unconstrained; `physical` maps them)."""

    def __init__(self, f_hist, f_cond, hidden=64):
        super().__init__()
        self.run = nn.Sequential(nn.Linear(f_hist + f_cond, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.att = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden + f_cond + 1, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(ABILITIES)))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, X, M, C):
        N, K, _ = X.shape
        h = self.run(torch.cat([X, C[:, None, :].expand(N, K, C.shape[1])], -1))
        a = self.att(h).squeeze(-1).masked_fill(~M, -1e4)
        w = torch.softmax(a, -1) * M.float()
        pooled = (w[..., None] * h).sum(1)
        has = M.any(1, keepdim=True).float()
        return self.head(torch.cat([pooled * has, C, has], -1))


class Model(nn.Module):
    def __init__(self, f_hist, f_cond):
        super().__init__()
        self.encoder = Encoder(f_hist, f_cond)
        self.physics = Physics()

    def abilities(self, b):
        """b: a batch dict (see train.batch). Returns physical abilities [B, H] each."""
        z = self.encoder(b['X'], b['M'], b['C'])            # [R, 6] for the R real horse rows
        Bn, H = b['slot'].shape
        zz = torch.zeros(Bn * H, len(ABILITIES), device=z.device)
        zz[b['flat']] = z
        zz = zz.view(Bn, H, -1)
        par = self.physics.par(b['course'], b['surface'], b['going'], b['log_dist'], b['cls'])
        return physical(zz, par[:, None], self.physics)

    def forward(self, b):
        ab = self.abilities(b)
        return run(ab, self.physics, b['distance'], int(b['steps'].max())), ab


def read_out(t, b):
    """Times read from the runs: finish T, t600, last3, corner times [B, H, 4], leader marks [B, M]."""
    Bn, H, _ = t.shape
    ar = torch.arange(Bn, device=t.device)
    T = t[ar, :, b['steps']]
    t600 = t[ar, :, b['steps'] - int(600 / DX)]
    corners = torch.gather(t, 2, b['corner_idx'][:, None, :].expand(Bn, H, 4).clamp(min=0))
    marks = torch.gather(t, 2, b['mark_idx'][:, None, :].expand(Bn, H, b['mark_idx'].shape[1]).clamp(min=0))
    lead = softmin(marks, 0.05, 1, b['present'][:, :, None].expand_as(marks))
    return T, t600, T - t600, corners, lead


LOSS = 'centred'   # 'absolute' (first S1 run): finish and 600 m times as they are; 'centred': within-race differences
#                    for the horses plus the race mean on its own, so the day's track speed does not drown the horse terms
WEIGHTS = {'absolute': {'T': 1.0, 't600': 1.0, 'last3': 2.0, 'lead': 0.5, 'order': 0.5, 'corner': 0.5},
           'centred': {'T': 1.0, 't600': 1.0, 'mean': 0.5, 'last3': 2.0, 'lead': 0.5, 'order': 1.0, 'corner': 0.5}}


def centred(p, y, m):
    """Huber of the within-race deviations (from the race mean of the horses in m) and the race means themselves."""
    n = m.sum(1).clamp(min=1)
    pm = (p * m).sum(1) / n
    ym = (y * m).sum(1) / n
    has = m.sum(1) > 0
    dev = F.huber_loss((p - pm[:, None])[m], (y - ym[:, None])[m], delta=0.5) if m.any() else p.sum() * 0
    return dev, (F.huber_loss(pm[has], ym[has], delta=0.5) if has.any() else p.sum() * 0)


def losses(t, b, w=None):
    w = w or WEIGHTS[LOSS]
    T, t600, last3, corners, lead = read_out(t, b)
    hub = lambda p, y, m: (F.huber_loss(p[m], y[m], delta=0.5) if m.any() else p.sum() * 0)
    if LOSS == 'centred':
        dT, mT = centred(T, b['y_T'], b['m_T'])
        d6, m6 = centred(t600, b['y_t600'], b['m_t600'])
        out = {'T': dT, 't600': d6, 'mean': 0.5 * (mT + m6), 'last3': hub(last3, b['y_last3'], b['m_last3'])}
    else:
        out = {'T': hub(T, b['y_T'], b['m_T']), 't600': hub(t600, b['y_t600'], b['m_t600']), 'last3': hub(last3, b['y_last3'], b['m_last3'])}
    out['lead'] = hub(lead, b['y_lead'], b['m_lead'])
    # finishing order: every pair of normal finishers, smooth on the time difference
    out['order'] = pair_loss(T, b['y_T'], b['m_T'], 0.2)
    cl = []
    for c in range(4):
        m = b['m_corner'][:, :, c]
        if m.any():
            cl.append(pair_loss(corners[:, :, c], b['y_corner'][:, :, c], m, 0.15))
    out['corner'] = torch.stack(cl).mean() if cl else T.sum() * 0
    total = sum(w[k] * v for k, v in out.items())
    return total, {k: float(v.detach()) for k, v in out.items()}


def pair_loss(pred, actual, mask, tau):
    """Mean over pairs (i ahead of j in `actual`) of softplus(-(pred_j - pred_i) / tau)."""
    ahead = (actual[:, :, None] < actual[:, None, :]) & mask[:, :, None] & mask[:, None, :]
    if not ahead.any():
        return pred.sum() * 0
    diff = pred[:, None, :] - pred[:, :, None]   # [b, i, j] = pred_j - pred_i
    return F.softplus(-diff[ahead] / tau).mean() * tau
