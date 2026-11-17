// Fits a venue's digitized inner/outer rails with "line + clothoid + arc" elements.
// Usage: node scripts/fit_alignment.mjs 08        (writes outputs/alignment-08.json)
//
// Layout (travel order; both courses start at the goal on the home straight):
//   S        : shared. home straight, corners 1-2, back straight, up to the divergence point D
//   tail     : D -> corner 3-4 -> exit onto the home straight (own elements per course)
//   approach : straight ending at the goal (the official final-straight length of the course)
// Both approaches lie on one line, so the rails are the same geometry wherever they coincide.
// Official numbers (lap, final straight) are exact; the plan only decides the shape.
import fs from 'node:fs';
import * as T from 'three';

const venue = process.argv[2] ?? '08';
const SPEC = {
  '08': {inner: {lap: 1782.8, straight: 328.4}, outer: {lap: 1894.3, straight: 403.7}, right: true},
  '09': {inner: {lap: 1689, straight: 356.5}, outer: {lap: 2089, straight: 473.6}, right: true},
  '06': {inner: {lap: 1667.1, straight: 310}, outer: {lap: 1839.7, straight: 310}, right: true},
};
const spec = SPEC[venue];
const src = fs.readFileSync('app/courseData.ts', 'utf8');
const data = JSON.parse(src.slice(src.indexOf('=', src.indexOf('COURSE_DATA')) + 1).trim().replace(/;\s*$/, ''));
const drawing = data[venue];
const [cx, cy] = drawing.center;

// Same shared plan scale as app/trackGeometry.ts: the inner rail is calibrated to its official lap.
const innerRaw = new T.CatmullRomCurve3(drawing.rails.turf.map(([x, z]) => new T.Vector3(x - cx, 0, z - cy)), true, 'centripetal');
innerRaw.arcLengthDivisions = 4096;
const scale = spec.inner.lap / innerRaw.getLength();
const plan = {
  inner: drawing.rails.turf.map(([x, z]) => [(x - cx) * scale, (z - cy) * scale]),
  outer: drawing.rails.outer.map(([x, z]) => [(x - cx) * scale, (z - cy) * scale]),
};
console.log('scale m/px', scale.toFixed(4));

// ---- integration -------------------------------------------------------------------------
// element: [kind, length, kEnd]; 'L' line, 'S' spiral (curvature ramps to kEnd), 'A' arc (curvature kept).
function walk(pose, elements, step, emit) {
  let {x, z, h} = pose;
  let k = 0;
  let travelled = 0;
  for (const [kind, len, kEnd] of elements) {
    const k0 = kind === 'L' ? 0 : k;
    const k1 = kind === 'L' ? 0 : kind === 'A' ? k : kEnd;
    const n = Math.max(1, Math.ceil(len / step));
    const ds = len / n;
    for (let i = 0; i < n; i++) {
      const km = k0 + (k1 - k0) * ((i + 0.5) / n);
      const hm = h + (km * ds) / 2;
      x += Math.cos(hm) * ds;
      z += Math.sin(hm) * ds;
      h += km * ds;
      travelled += ds;
      emit?.(x, z, h, travelled);
    }
    k = k1;
  }
  return {x, z, h};
}

const sgn = spec.right ? 1 : -1;
const GOAL = {x: plan.inner[0][0], z: plan.inner[0][1]};
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const softAbs = (r, d) => d * Math.sqrt(2 * (Math.sqrt(1 + (r / d) ** 2) - 1));
function distToPolyline(pts, [px, pz]) {
  let best = Infinity;
  for (let i = 0; i + 1 < pts.length; i++) {
    const [ax, az] = pts[i], [bx, bz] = pts[i + 1];
    const dx = bx - ax, dz = bz - az;
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / (dx * dx + dz * dz)));
    const d = Math.hypot(ax + dx * t - px, az + dz * t - pz);
    if (d < best) best = d;
  }
  return best;
}
const STEP = 2;
const trace = (pose, elements) => {
  const pts = [[pose.x, pose.z]];
  const end = walk(pose, elements, STEP, (x, z) => pts.push([x, z]));
  return {pts, end};
};
const clampK = (v) => sgn * Math.max(1 / 700, Math.min(1 / 70, sgn * v));
const SMAX = +(process.env.SMAX ?? 300), KLO = +(process.env.KLO ?? 0.7), KHI = +(process.env.KHI ?? 1.45), T2 = +(process.env.T2 ?? 30);
const clampS = (v) => Math.max(40, Math.min(SMAX, v));
const clampL = (v) => Math.max(0, Math.min(3000, v));
const bound = (v, lo, hi) => 3 * (Math.max(0, lo - v) + Math.max(0, v - hi));
const boundK = (v) => 3e4 * bound(sgn * v, 1 / 700, 1 / 70);

// ---- Levenberg-Marquardt -------------------------------------------------------------------
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[piv][i])) piv = j;
    [M[i], M[piv]] = [M[piv], M[i]];
    for (let j = i + 1; j < n; j++) {
      const f = M[j][i] / M[i][i];
      for (let k = i; k <= n; k++) M[j][k] -= f * M[i][k];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k];
    x[i] = s / M[i][i];
  }
  return x;
}
const cost = (r) => r.reduce((a, v) => a + v * v, 0);
function fit(p, residuals, unit, iterations = 80) {
  let lambda = 1;
  let r = residuals(p);
  let c = cost(r);
  for (let it = 0; it < iterations; it++) {
    const n = p.length;
    const J = Array.from({length: r.length}, () => new Float64Array(n));
    for (let j = 0; j < n; j++) {
      const h = unit[j] * 1e-4;
      const q = [...p];
      q[j] += h;
      const rr = residuals(q);
      for (let i = 0; i < r.length; i++) J[i][j] = ((rr[i] - r[i]) / h) * unit[j];
    }
    const A = Array.from({length: n}, () => new Float64Array(n));
    const g = new Float64Array(n);
    for (let i = 0; i < r.length; i++)
      for (let a = 0; a < n; a++) {
        g[a] += J[i][a] * r[i];
        for (let b = 0; b < n; b++) A[a][b] += J[i][a] * J[i][b];
      }
    let improved = false;
    for (let tries = 0; tries < 12 && !improved; tries++) {
      const M = A.map((row, a) => Array.from(row, (v, b) => (a === b ? v * (1 + lambda) + 1e-9 : v)));
      const du = solve(M, Array.from(g, (v) => -v));
      const q = p.map((v, i) => v + du[i] * unit[i]);
      const rq = residuals(q);
      const cq = cost(rq);
      if (cq < c) {
        p = q; r = rq; c = cq; lambda = Math.max(1e-6, lambda / 3); improved = true;
      } else lambda *= 4;
    }
    if (!improved) break;
  }
  return {p, c};
}
let seed = 7;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
function multi(init, residuals, unit, tries = 6, keepFirst = false) {
  let best = null;
  for (let t = 0; t < tries; t++) {
    const start = init.map((v, i) => (t === 0 || (keepFirst && i === 0) ? v : v * (0.8 + 0.4 * rnd())));
    const res = fit(start, residuals, unit);
    if (!best || res.c < best.c) best = res;
  }
  return best;
}

// ---- stages --------------------------------------------------------------------------------
const N = plan.inner.length;
// Plan indices (the rail lists start at the goal), from the shared-distance scan of the outer rail.
const IDX = {
  '08': {sharedHead: [0, 70], approach: [160, N - 1], tailIn: [76, 160], tailOut: [76, 166], D: 73},
  '09': {sharedHead: [0, 66], approach: [142, N - 1], tailIn: [74, 142], tailOut: [74, 150], D: 70},
}[venue];
const range = (a, b) => Array.from({length: b - a + 1}, (_, i) => a + i);

// Stage 1: shared S. p = [gh, a, s1, k1, c1, s3, b]
const sElems = (p) => [['L', clampL(p[1])], ['S', clampS(p[2]), clampK(p[3])], ['A', clampL(p[4])], ['S', clampS(p[5]), 0], ['L', clampL(p[6])]];
function stage1(p) {
  const gh = p[0];
  const {pts, end} = trace({x: GOAL.x, z: GOAL.z, h: gh}, sElems(p));
  const line = trace({x: GOAL.x - Math.cos(gh) * 340, z: GOAL.z - Math.sin(gh) * 340, h: gh}, [['L', 340]]).pts;
  const r = [];
  for (const i of range(...IDX.sharedHead))
    r.push(softAbs(Math.min(distToPolyline(pts, plan.inner[i]), distToPolyline(pts, plan.outer[i])), 4));
  for (const i of range(...IDX.approach)) r.push(softAbs(distToPolyline(line, plan.inner[i]), 4));
  // The back straight ends at the plan's divergence point.
  r.push(30 * (end.x - plan.outer[IDX.D][0]), 30 * (end.z - plan.outer[IDX.D][1]));
  r.push(bound(p[1], 0, 3000), bound(p[6], 0, 3000), boundK(p[3]), bound(p[2], 40, 400), bound(p[5], 40, 400), bound(p[4], 0, 3000));
  return r;
}
// Stages 2 and 3: a tail from D to the start of the approach straight. p = [t1, k1, c1, t3];
// the straight ahead of the corner takes whatever length the official lap leaves.
const sLength = (S) => S.reduce((a, e) => a + e[1], 0);
function tailElems(p, kind, Slen) {
  const o = spec[kind];
  const own = [['S', clampS(p[0]), clampK(p[1])], ['A', clampL(p[2])], ['S', clampS(p[3]), 0]];
  const used = own[0][1] + own[1][1] + own[2][1];
  const lead = o.lap - o.straight - Slen - used;
  return {elems: [['L', Math.max(0, lead)], ...own], lead};
}
function stageTail(kind, S, dPose, gh) {
  const o = spec[kind];
  const Slen = sLength(S);
  const target = {x: GOAL.x - Math.cos(gh) * o.straight, z: GOAL.z - Math.sin(gh) * o.straight};
  const span = kind === 'inner' ? IDX.tailIn : IDX.tailOut;
  return (p) => {
    const {elems, lead} = tailElems(p, kind, Slen);
    const {pts, end} = trace(dPose, elems);
    const r = [];
    for (const i of range(...span)) r.push(softAbs(distToPolyline(pts, plan[kind][i]), 4));
    r.push(80 * (end.x - target.x), 80 * (end.z - target.z), 8000 * wrap(end.h - gh - 2 * Math.PI * sgn));
    r.push(5 * Math.max(0, -lead), boundK(p[1]), bound(p[0], 40, 400), bound(p[3], 40, 400), bound(p[2], 0, 3000));
    return r;
  };
}

const turn1 = (172 * Math.PI) / 180;
const s1 = multi([Math.PI, 110, 60, sgn / 140, 140 * turn1 - 60, 60, 160], stage1, [0.1, 50, 50, 0.01, 50, 50, 50], 6, true);
console.log('stage1 cost', s1.c.toFixed(1), s1.p.map((v) => +v.toFixed(4)).join(' '));
const S = sElems(s1.p);
let jointGh = null;
const gh0 = s1.p[0];
const gh = gh0;
const dPose = {...walk({x: GOAL.x, z: GOAL.z, h: gh}, S, STEP)};
const result = {};
for (const kind of ['inner', 'outer']) {
  const R = kind === 'inner' ? 150 : 180;
  const turn2 = 2 * Math.PI - turn1;
  const best = multi([70, sgn / R, R * turn2 - 70, 70], stageTail(kind, S, dPose, gh), [50, 0.01, 50, 50], 8);
  console.log(kind, 'tail cost', best.c.toFixed(1), best.p.map((v) => +v.toFixed(4)).join(' '));
  result[kind] = {...tailElems(best.p, kind, sLength(S)), p: best.p};
}

// Joint refinement with a richer model: compound corners (two arcs), and an outer course whose
// own final straight sits a little outside the inner one and merges onto it with a gentle S-bend.
// p: [gh | S: a s1 k1 c1 s2 k2 c2 s3 b (1..9) | inner: t1 k1 c1 t2 k2 c2 t3 (10..16) | outer: same (17..23), m1 u kb mid (24..27)]
const cornerElems = (t1, k1, c1, t2, k2, c2, t3) =>
  [['S', clampS(t1), clampK(k1)], ['A', clampL(c1)], ['S', Math.max(T2, Math.min(200, t2)), clampK(Math.max(KLO * clampK(k1), Math.min(KHI * clampK(k1), k2)))], ['A', clampL(c2)], ['S', clampS(t3), 0]];
const clampSharedS = (p) => [['L', clampL(p[1])], ...cornerElems(p[2], p[3], p[4], p[5], p[6], p[7], p[8]), ['L', clampL(p[9])]];
function elemsFor(p, kind) {
  const S2 = clampSharedS(p);
  const Slen = sLength(S2);
  const o = spec[kind];
  if (kind === 'inner') {
    const own = cornerElems(...p.slice(10, 17));
    const lead = o.lap - o.straight - Slen - sLength(own);
    return {S2, lead, dm: o.straight, elems: [...S2, ['L', Math.max(0, lead)], ...own, ['L', o.straight]]};
  }
  const own = cornerElems(...p.slice(17, 24));
  const [m1, u, kb, mid] = [clampL(p[24]), Math.max(15, Math.min(90, p[25])), sgn * Math.max(5e-4, Math.min(0.012, sgn * p[26])), clampL(p[27])];
  const dm = o.straight - m1 - 4 * u - mid;
  const lead = o.lap - o.straight - Slen - sLength(own);
  const bend = [['L', m1], ['S', u, kb], ['S', u, 0], ['L', mid], ['S', u, -kb], ['S', u, 0], ['L', Math.max(0, dm)]];
  return {S2, lead, dm, elems: [...S2, ['L', Math.max(0, lead)], ...own, ...bend]};
}
// The plan is drawn from a bitmap; its polyline is a little longer than the smooth rail (digitising
// jitter), so the metres-per-pixel is fitted too: the official lap fixes the size, the plan the shape.
function jointResiduals(p) {
  const r = [];
  const gh2 = p[0];
  const ratio = p[28];
  const gx = GOAL.x * ratio, gz = GOAL.z * ratio;
  for (const kind of ['inner', 'outer']) {
    const {elems, lead, dm} = elemsFor(p, kind);
    const {pts, end} = trace({x: gx, z: gz, h: gh2}, elems);
    for (let i = 0; i < N; i++) r.push(softAbs(distToPolyline(pts, [plan[kind][i][0] * ratio, plan[kind][i][1] * ratio]), 4));
    r.push(80 * (end.x - gx), 80 * (end.z - gz), 8000 * wrap(end.h - gh2 - 2 * Math.PI * sgn));
    r.push(100 * Math.max(0, -lead), 50 * Math.max(0, -dm));
  }
  r.push(0.2 * (p[9] - s1.p[6]), bound(p[28], 0.97, 1.05) * 1e3);
  for (const i of [3, 6, 11, 14, 18, 21]) r.push(boundK(p[i]));
  for (const i of [2, 8, 10, 16, 17, 23]) r.push(bound(p[i], 40, SMAX));
  for (const i of [5, 13, 20]) r.push(bound(p[i], T2, 200));
  for (const i of [4, 7, 12, 15, 19, 22]) r.push(bound(p[i], 0, 3000));
  r.push(bound(p[25], 15, 90), bound(p[24], 0, 3000), bound(p[27], 0, 3000), bound(sgn * p[26], 5e-4, 0.02) * 1e3);
  // Gentle regularisation: the two arcs of a corner should differ by less than a factor of ~2.
  for (const [i, j] of [[3, 6], [11, 14], [18, 21]]) r.push(300 * Math.max(0, Math.abs(p[i] - p[j]) - 0.004));
  return r;
}
jointGh = gh0;
let jointRatio = 1;
{
  const [t1i, k1i, c1i, t3i] = result.inner.p, [t1o, k1o, c1o, t3o] = result.outer.p;
  const sp = s1.p;
  const init = [gh0, sp[1], sp[2], sp[3], sp[4] * 0.98, 10, sp[3], 2, sp[5], sp[6],
    t1i, k1i, c1i * 0.98, 10, k1i, 2, t3i, t1o, k1o, c1o * 0.98, 10, k1o, 2, t3o, 90, 40, sgn * 0.004, 20, 1.01];
  const unit = [0.1, 50, 50, 0.01, 50, 50, 0.01, 50, 50, 50, 50, 0.01, 50, 50, 0.01, 50, 50, 50, 0.01, 50, 50, 0.01, 50, 50, 50, 10, 0.01, 30, 0.01];
  console.log('joint init cost', cost(jointResiduals(init)).toFixed(0));
  let best = {c: Infinity};
  for (let t = 0; t < 10; t++) {
    const start = init.map((v, i) => (t === 0 || i === 0 ? v : v * (0.85 + 0.3 * rnd())));
    const j = fit(start, jointResiduals, unit, 250);
    console.log(' try', t, j.c.toFixed(1));
    if (j.c < best.c) best = j;
  }
  console.log('joint cost', best.c.toFixed(1));
  jointGh = best.p[0];
  jointRatio = best.p[28];
  S.length = 0;
  S.push(...clampSharedS(best.p));
  for (const kind of ['inner', 'outer']) {
    const e = elemsFor(best.p, kind);
    result[kind] = {elems: e.elems.slice(S.length), lead: e.lead};
  }
  console.log('outer bend: m1,u,kb,mid,dm =', [best.p[24], best.p[25], best.p[26], best.p[27], elemsFor(best.p, 'outer').dm].map((v) => +v.toFixed(4)).join(' '));
}

// ---- report --------------------------------------------------------------------------------
const report = {};
for (const kind of ['inner', 'outer']) {
  const o = spec[kind];
  const elems = [...S, ...result[kind].elems];
  const {pts, end} = trace({x: GOAL.x * jointRatio, z: GOAL.z * jointRatio, h: jointGh ?? gh}, elems);
  const dev = plan[kind].map((q) => distToPolyline(pts, [q[0] * jointRatio, q[1] * jointRatio]) / (scale * jointRatio));
  const total = elems.reduce((a, e) => a + e[1], 0);
  console.log(kind, 'length', total.toFixed(2), 'official', o.lap,
    'closure m', Math.hypot(end.x - GOAL.x * jointRatio, end.z - GOAL.z * jointRatio).toFixed(3),
    'heading err deg', ((wrap(end.h - (jointGh ?? gh) - 2 * Math.PI * sgn) * 180) / Math.PI).toFixed(3),
    'lead', result[kind].lead.toFixed(1),
    '| plan px rms', Math.sqrt(dev.reduce((a, d) => a + d * d, 0) / dev.length).toFixed(2),
    'max', Math.max(...dev).toFixed(2), '>3px', dev.filter((d) => d > 3).length);
  if (process.env.PROFILE) console.log(kind, dev.map((d, i) => (i % 6 === 0 ? i + ':' + d.toFixed(0) : '')).filter(Boolean).join(' '));
  report[kind] = {elems, lap: o.lap, straight: o.straight};
}
const file = process.env.FIT_OUT ?? 'outputs/alignment-' + venue + '.json';
fs.writeFileSync(file, JSON.stringify({venue, scale: scale * jointRatio, planRatio: jointRatio, pose: {x: GOAL.x * jointRatio, z: GOAL.z * jointRatio, h: jointGh ?? gh}, S, courses: report}, null, 1));
console.log('wrote', file);
