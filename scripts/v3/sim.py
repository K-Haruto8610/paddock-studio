"""v3: the race simulator and the ability encoder (docs/race-simulator-design.md 5.1-5.3; S1 solo runs, S2 pace interaction).

Every horse runs the race on the distance axis in steps of DX metres (S1: on its own; S2: see STAGE). State: speed v and
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
ABILITIES = ('vc', 'reserve', 'vmax', 'dash', 'intent', 'kick', 'wish')
STAGE = 's2'
# 's1' solo runs (wish unused). 's2' (design S2, pace interaction): every step each horse sees the field at the same point
#      of the course (the distance axis: who passed it earlier is ahead):
#      contention  before its kick a horse behind the place it wishes (wish, 0 = the lead, 1 = last) runs faster than
#                  its cruise, one ahead of it eases: + CONTEND x tanh((share of the field ahead - wish) / 0.25) m/s,
#                  so several horses wanting the lead push the pace and spend their reserve (M2 should follow);
#      shelter     a horse running up to SHELTER_GAP s behind another drains its reserve less (x (1 - SHELTER)); the
#                  leader has none. How much shelter is worth is learnt (design 5.2: "効くかどうかは学習に任せる").
SHELTER_GAP = 0.3   # seconds behind the nearest horse ahead (about 1.5 lengths)
CONTEND_SPAN = 'kick'
WISH = 'free'
# 'free'   wish is its own encoder output (first S2 runs; the S2 recovery test could not recover it, 0.13, and intent
#          fell from 0.80 to 0.51: both set the early position).
# 'intent' wish follows intent: sigmoid(-intent / Physics.wish_scale) (a horse that runs its early part faster wants to
#          be further forward); the encoder's output 6 is then unused.
# 'kick'  contention acts until the horse's kick (first S2 runs: ranks fell against S1, most at the 4th corner).
# 'early' contention acts only through the position-taking stretch after the start (Physics.settle, learnt, 600 m at
#         first); after it a horse holds its cruise (positions are taken early and then kept).
# 's3' (design S3, lanes and traffic; everything of 's2' too): every horse also has a lateral position y (metres outside
#      the innermost path). The rules are geometry and fixed, not learnt (design 8, M3: lanes are not learnable from the
#      records); y is computed without gradients:
#      start       y = (UMABAN - 1) x STALL (horse 1 is on the inside)
#      drift       each step a horse moves DRIFT towards the rail ...
#      no overlap  ... but stays SEP outside any horse alongside it (|time gap| < ALONG) that is further in
#      blocked     a horse up to BLOCK s behind another within SEP sideways is held to that horse's speed (how fully is
#                  learnt: Physics.hold); a held horse that wants to go faster (by WANT_OUT m/s) moves MOVE_OUT outwards
#                  unless a horse alongside is there (boxed in)
#      distance    a step at y through a turn of a radians runs DX + a y metres (radius + y), so running wide costs
#                  exactly the geometry's ground (M3)
#      Corner gaps in lengths are read as Physics.gap_scale x seconds behind the leader (learnt, the 'gap' loss).
STALL, DRIFT, SEP, ALONG, BLOCK, WANT_OUT, MOVE_OUT = 1.0, 1.0, 1.1, 0.12, 0.2, 0.15, 0.8
TRAFFIC = 'full'
# Stochastic runs (design 5.3, brought forward from S4 after the first S3 runs, 2026-11-22): each race is run SAMPLES
# times. In every run each horse has its own day's form, one draw for the whole run (not noise per step): vc and vmax
# both shifted by FORM x N(0, 1) m/s; and its own start, a delay of START x |N(0, 1)| s at the gate. FORM and START are
# calibrated so the runs' within-race spread matches the measured one (train.py calibrate: median sd of the 600 m-mark
# time and of the finish time among normal finishers), never on a score. Everything (traffic rules of S3 included) runs
# on each sampled run; the read-outs (times, corner times, gaps behind the corner leader, the leader's marks) are taken
# per run and only then averaged. SAMPLES 1 and FORM = START = 0 are the deterministic runs of S1-S3.
SAMPLES, FORM, START = 1, 0.0, 0.0
# 'full'  the rules above (first S3 runs, 2026-11-22: lanes 3-5x the constant-0 error and the 4th-corner order -0.09 tau
#         against S2; the point forecast's field is 0.35-0.6 of the real spread, so the rules fire far too often)
# 'none'  no blocking: lanes, no overlap and the corner distance only (is the geometry alone worth anything?)
# 'half'  ALONG and BLOCK halved, for the tighter predicted field


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
        self.contend = nn.Parameter(torch.tensor(math.log(math.e ** 0.3 - 1)))  # softplus -> m/s per unit of wish gap (S2)
        self.shelter = nn.Parameter(torch.tensor(-1.0))              # logit; drain saved in shelter = 0.5 sigmoid (S2)
        self.settle = nn.Parameter(torch.tensor(math.log(600.0)))    # metres of position taking (CONTEND_SPAN 'early')
        self.wish_scale = nn.Parameter(torch.tensor(math.log(0.2)))  # m/s of intent per logit of wish (WISH 'intent')
        self.hold = nn.Parameter(torch.tensor(1.0))                  # logit of how fully a blocked horse is held (S3)
        self.gap_scale = nn.Parameter(torch.tensor(math.log(5.0)))   # passing-string lengths per second (S3)

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
    intent = 0.8 * torch.tanh(z[..., 4])
    if WISH == 'intent':
        wish = torch.sigmoid(-intent / (torch.exp(physics.wish_scale) if physics is not None else 0.2))
    else:
        wish = torch.sigmoid(z[..., 6])
    return {'vc': vc, 'reserve': W, 'vmax': vmax, 'dash': dash, 'intent': intent, 'kick': kick, 'wish': wish}


def field_view(t, present):
    """At the current point: share of the (present) field ahead of each horse, and the time gap to the nearest horse
    ahead (large for the leader). t [B, H] times at this point; present [B, H] bool."""
    d = t[:, :, None] - t[:, None, :]                       # [b, i, j] > 0: j passed earlier (ahead of i)
    other = present[:, None, :] & present[:, :, None] & ~torch.eye(t.shape[1], dtype=torch.bool, device=t.device)
    ahead = torch.sigmoid(d / 0.1) * other
    n = (other.sum(2) .clamp(min=1)).float()
    share = ahead.sum(2) / n
    gap = d + 10 * (1 - torch.sigmoid(d / 0.02))            # horses behind count as far away
    gap = softmin(gap, 0.02, 2, other)
    return share, gap


@torch.no_grad()
def lanes_step(y, t, v, wanted, present):
    """S3 lateral rules for one step (no gradients). y, t, v, wanted [B, H] at the current mark. Returns the new y and
    the blocking horse of each (index into H; -1 = free)."""
    H = y.shape[1]
    f = 0.5 if TRAFFIC == 'half' else 1.0
    other = present[:, None, :] & present[:, :, None] & ~torch.eye(H, dtype=torch.bool, device=y.device)
    d = t[:, :, None] - t[:, None, :]                        # [b, i, j] > 0: j ahead of i
    dy = y[:, None, :] - y[:, :, None]                        # [b, i, j] = y_j - y_i
    along = other & (d.abs() < ALONG * f)
    near = other & (d > 0) & (d < BLOCK * f) & (dy.abs() < SEP)
    dd = d.masked_fill(~near, 1e4)
    nearest = dd.argmin(2)
    blocked = near.any(2)
    nearest = torch.where(blocked, nearest, torch.full_like(nearest, -1))
    v_ahead = torch.gather(v, 1, nearest.clamp(min=0))
    wants_out = blocked & (wanted > v_ahead + WANT_OUT) & (TRAFFIC != 'none')
    out_y = y + MOVE_OUT
    # boxed in: a horse alongside already there
    boxed = (along & (dy > 0) & ((y[:, None, :] - out_y[:, :, None]).abs() < SEP)).any(2)
    want = torch.where(wants_out & ~boxed, out_y, torch.where(wants_out, y, (y - DRIFT).clamp(min=0)))
    new = want
    for _ in range(3):                                        # stay SEP outside horses alongside further in
        inside = along & (y[:, None, :] < y[:, :, None] - 1e-3)
        floor = (new[:, None, :] + SEP).masked_fill(~inside, 0).amax(2)
        new = torch.maximum(want, floor)
    return torch.where(present, new, y), nearest


def run(ab, physics, distance, steps, present=None, lanes=None, t0=None):
    """Runs of a field. ab: abilities [B, H] each; distance [B] (m); steps = max steps in the batch; present [B, H] (S2:
    who is in the field); lanes (S3): {'turn': [B, steps] radians per step, 'y0': [B, H] start positions}.
    Returns t [B, H, steps + 1]: elapsed time at every DX mark from the start, and (S3) y [B, H, steps + 1]."""
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
    s2 = STAGE in ('s2', 's3')
    s3 = STAGE == 's3' and lanes is not None
    if s2:
        present = torch.ones(B, H, dtype=torch.bool, device=vc.device) if present is None else present
        contend = F.softplus(physics.contend)
        save = 0.5 * torch.sigmoid(physics.shelter)
    t = [torch.zeros(B, H, device=vc.device) if t0 is None else t0]
    if s3:
        hold = torch.sigmoid(physics.hold)
        y = lanes['y0'].clone()
        ys = [y]
    for j in range(steps):
        x = (j + 0.5) * DX
        k = torch.sigmoid((x - kick_at) / physics.kick_width)
        base = cruise
        if s2:
            share, gap = field_view(t[-1], present)
            push = contend * torch.tanh((share - ab['wish']) / 0.25)
            if CONTEND_SPAN == 'early':
                push = push * torch.sigmoid((torch.exp(physics.settle) - x) / 50)
            base = cruise + push
            drain = 1 - save * torch.sigmoid((SHELTER_GAP - gap) / 0.05)
        wanted = base + k * (vmax - base)
        e_pos = F.softplus(E / 2) * 2
        cap = vc - fat + (vmax - vc + fat) * (1 - torch.exp(-e_pos / esoft))
        target = softmin(torch.stack([wanted, cap]), 0.1, 0)
        if s3:
            y_new, ahead = lanes_step(y, t[-1].detach(), v.detach(), wanted.detach(), present)
            held = (ahead >= 0) & (TRAFFIC != 'none')
            v_ahead = torch.gather(v, 1, ahead.clamp(min=0))
            target = target - hold * held.float() * F.softplus((target - v_ahead) / 0.1) * 0.1
        # speeding up is limited by the horse's acceleration (the standing start, the kick); slowing down relaxes
        up = softmin(torch.stack([target, torch.sqrt(v * v + 2 * dash * DX + 1e-6)]), 0.1, 0)
        down = target + (v - target) * keep
        v_new = torch.where(target > v, up, down)
        vm = 0.5 * (v + v_new)
        if s3:
            ym = 0.5 * (y + y_new)
            path = (DX + lanes['turn'][:, j, None] * ym).clamp(min=0.5 * DX)
            y = y_new
            ys.append(y)
        else:
            path = DX
        dt = path / vm.clamp(min=1.0)
        above = vm - vc
        E = torch.minimum(E - F.relu(above) * dt * (drain if s2 else 1) + rec * F.relu(-above) * dt, W)
        t.append(t[-1] + dt)
        v = v_new
    return torch.stack(t, -1), (torch.stack(ys, -1) if s3 else None)


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

    def forward(self, b, samples=None, gen=None):
        """Runs SAMPLES (or `samples`) stochastic runs of every race: t [K B, H, steps + 1] (run k of race i at k B + i);
        self.y the lateral positions (S3) in the same layout, self.K = K."""
        ab = self.abilities(b)
        K = samples or SAMPLES
        B, H = ab['vc'].shape
        abk = {a: x.repeat(K, 1) for a, x in ab.items()}
        t0 = None
        if FORM > 0 or START > 0:
            dev = ab['vc'].device
            eps = torch.randn(K * B, H, device=dev, generator=gen)
            delay = torch.randn(K * B, H, device=dev, generator=gen).abs()
            abk['vc'] = abk['vc'] + FORM * eps
            abk['vmax'] = abk['vmax'] + FORM * eps
            t0 = START * delay
        lanes = {'turn': b['turn'].repeat(K, 1), 'y0': b['y0'].repeat(K, 1)} if STAGE == 's3' else None
        t, self.y = run(abk, self.physics, b['distance'].repeat(K), int(b['steps'].max()), b['present'].repeat(K, 1), lanes, t0)
        self.K = K
        return t, ab


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
           'centred': {'T': 1.0, 't600': 1.0, 'mean': 0.5, 'last3': 2.0, 'lead': 0.5, 'order': 1.0, 'corner': 0.5, 'gap': 0.1}}


def centred(p, y, m):
    """Huber of the within-race deviations (from the race mean of the horses in m) and the race means themselves."""
    n = m.sum(1).clamp(min=1)
    pm = (p * m).sum(1) / n
    ym = (y * m).sum(1) / n
    has = m.sum(1) > 0
    dev = F.huber_loss((p - pm[:, None])[m], (y - ym[:, None])[m], delta=0.5) if m.any() else p.sum() * 0
    return dev, (F.huber_loss(pm[has], ym[has], delta=0.5) if has.any() else p.sum() * 0)


def repeat_batch(b, K):
    """The batch entries read_out needs, repeated for K runs per race."""
    return {k: b[k].repeat(K, *([1] * (b[k].dim() - 1))) for k in ('steps', 'corner_idx', 'mark_idx', 'present')}


def read_mean(t, b, K):
    """Read-outs of every run, then averaged over the K runs of each race: T, t600, last3, corners [B, H, 4], lead
    [B, M], behind (seconds behind the corner's leader [B, H, 4]), lead600 (the leader's time 600 m from the start)."""
    bk = repeat_batch(b, K)
    T, t600, last3, corners, lead = read_out(t, bk)
    behind = corners - softmin(corners, 0.02, 1, bk['present'][:, :, None].expand_as(corners))[:, None, :]
    lead600 = softmin(t[:, :, int(600 / DX)], 0.05, 1, bk['present'])
    m = lambda x: x.view(K, -1, *x.shape[1:]).mean(0)
    return {'T': m(T), 't600': m(t600), 'last3': m(last3), 'corners': m(corners), 'lead': m(lead), 'behind': m(behind),
            'lead600': m(lead600)}


def corner_lengths(corners, present, physics):
    """Corner times [B, H, 4] -> lengths behind the leader of each corner (S3)."""
    lead = softmin(corners, 0.02, 1, present[:, :, None].expand_as(corners))
    return (corners - lead[:, None, :]) * torch.exp(physics.gap_scale)


def losses(ro, b, w=None, physics=None):
    """ro: read_mean's averaged read-outs."""
    w = w or WEIGHTS[LOSS]
    T, t600, last3, corners, lead = ro['T'], ro['t600'], ro['last3'], ro['corners'], ro['lead']
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
    if STAGE == 's3':
        # passing-string lengths behind the leader at each corner (design 5.3), lengths = gap_scale x seconds
        out['gap'] = hub(ro['behind'] * torch.exp(physics.gap_scale), b['y_gap'], b['m_gap'])
    total = sum(w[k] * v for k, v in out.items())
    return total, {k: float(v.detach()) for k, v in out.items()}


def pair_loss(pred, actual, mask, tau):
    """Mean over pairs (i ahead of j in `actual`) of softplus(-(pred_j - pred_i) / tau)."""
    ahead = (actual[:, :, None] < actual[:, None, :]) & mask[:, :, None] & mask[:, None, :]
    if not ahead.any():
        return pred.sum() * 0
    diff = pred[:, None, :] - pred[:, :, None]   # [b, i, j] = pred_j - pred_i
    return F.softplus(-diff[ahead] / tau).mean() * tau
