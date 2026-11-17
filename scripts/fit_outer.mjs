// Fits a venue's outer course onto an already fitted inner course (outputs/alignment-<venue>-inner.json,
// written by fit_simple.mjs). The outer starts as an exact copy of the inner chain up to a cut point
// (so shared rails coincide with 0 deviation), then runs its own corners and joins the same final straight.
// usage: node scripts/fit_outer.mjs 09     (writes outputs/alignment-09.json)
import fs from 'node:fs';

const venue = process.argv[2];
const SPEC = {
  // turns: corners after the cut. cutIdx: plan index (rail lists start at the goal) where the outer leaves the inner.
  '09': {right: true, lap: 2089, straight: 473.6, innerLap: 1689, turns: 1, cutIdx: 66},
  '06': {right: true, lap: 1839.7, straight: 310, innerLap: 1667.1, turns: 2, cutIdx: 24},
}[venue];
const sgn = SPEC.right ? 1 : -1;
const inner = JSON.parse(fs.readFileSync(`outputs/alignment-${venue}-inner.json`, 'utf8'));
const innerElems = inner.courses.turf.elems, scale = inner.scale, pose = inner.pose;
const cd = fs.readFileSync('app/courseData.ts', 'utf8');
const data = JSON.parse(cd.slice(cd.indexOf('{', cd.indexOf('COURSE_DATA')), cd.lastIndexOf('}') + 1))[venue];
const [cx, cy] = data.center;
const plan = data.rails.outer.map(([x, y]) => [(x - cx) * scale, (y - cy) * scale]);

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
function walk(p0, elems, step, emit) {
  let {x, z, h} = p0, k = 0, s = 0;
  for (const [kind, len, kEnd = 0] of elems) {
    const k0 = kind === 'L' ? 0 : k, k1 = kind === 'L' ? 0 : kind === 'A' ? k : kEnd;
    const n = Math.max(1, Math.ceil(len / step)), ds = len / n;
    for (let i = 0; i < n; i++) {
      const km = k0 + (k1 - k0) * ((i + 0.5) / n), hm = h + (km * ds) / 2;
      x += Math.cos(hm) * ds; z += Math.sin(hm) * ds; h += km * ds; s += ds;
      emit?.(x, z, h, s);
    }
    k = k1;
  }
  return {x, z, h, k};
}
// inner chain cut at arclength c (an element is split where the cut falls)
function prefix(c) {
  const out = []; let left = c, k = 0;
  for (const [kind, len, kEnd = 0] of innerElems) {
    const k0 = kind === 'L' ? 0 : k, k1 = kind === 'L' ? 0 : kind === 'A' ? k : kEnd;
    if (left <= 0) break;
    if (left >= len) { out.push([kind, len, kEnd]); left -= len; }
    else {
      const f = left / len;
      out.push(kind === 'S' ? ['S', left, k0 + (k1 - k0) * f] : [kind, left, kEnd]);
      left = 0;
    }
    k = k1;
  }
  return out;
}
const SMID = 30;
const turn = (sIn, ka, aa, kb, ab, sOut) => [['S', sIn, sgn * ka], ['A', aa], ['S', SMID, sgn * kb], ['A', ab], ['S', sOut, 0]];
// p = [c, (turn1: sIn ka aa kb ab sOut)?, turn2: sIn ka aa kb ab sOut]
function chain(p) {
  const c = p[0];
  const turns = []; let used = SPEC.straight;
  for (let t = 0; t < SPEC.turns; t++) {
    const q = p.slice(1 + 6 * t, 7 + 6 * t);
    turns.push(turn(...q));
    used += q[0] + q[2] + SMID + q[4] + q[5];
  }
  const pre = prefix(c);
  // lengths of the turns after the first sit between straights; the remaining length is the back straight
  const L1 = SPEC.lap - c - used - (SPEC.turns === 2 ? 0 : 0);
  const body = SPEC.turns === 2 ? [...turns[0], ['L', Math.max(0, L1)], ...turns[1]] : [['L', Math.max(0, L1)], ...turns[0]];
  return {elems: [...pre, ...body, ['L', SPEC.straight]], L1, pre};
}
const goal = {x: pose.x, z: pose.z, h: pose.h};
function dist(pts, q) {
  let best = Infinity;
  for (let i = 0; i + 1 < pts.length; i++) {
    const [ax, az] = pts[i], [bx, bz] = pts[i + 1];
    const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz || 1;
    const t = Math.max(0, Math.min(1, ((q[0] - ax) * dx + (q[1] - az) * dz) / l2));
    best = Math.min(best, Math.hypot(ax + dx * t - q[0], az + dz * t - q[1]));
  }
  return best;
}
const softAbs = (v, c) => Math.sqrt(v * v + c * c) - c;
const bound = (v, lo, hi) => 200 * (Math.max(0, lo - v) + Math.max(0, v - hi));
const nT = SPEC.turns, nP = 1 + 6 * nT;
function residual(p) {
  const {elems, L1} = chain(p);
  const pts = [[goal.x, goal.z]];
  const end = walk(goal, elems, 3, (x, z) => pts.push([x, z]));
  const r = [];
  for (const q of plan) r.push(softAbs(dist(pts, q), 4));
  r.push(80 * (end.x - goal.x), 80 * (end.z - goal.z), 8000 * wrap(end.h - goal.h - sgn * 2 * Math.PI));
  r.push(bound(p[0], 20, SPEC.innerLap - SPEC.straight - 200), bound(L1, 0, 1500));
  for (let t = 0; t < nT; t++) {
    const o = 1 + 6 * t;
    for (const i of [1, 3]) r.push(bound(1 / p[o + i], 60, 500));
    for (const i of [0, 5]) r.push(bound(p[o + i], 25, 300));
    for (const i of [2, 4]) r.push(bound(p[o + i], 0, 900));
  }
  return r;
}
const cost = (p) => residual(p).reduce((a, v) => a + v * v, 0);
function lm(p, iters = 250) {
  let lambda = 1, c = cost(p);
  for (let it = 0; it < iters; it++) {
    const r0 = residual(p), n = p.length, m = r0.length;
    const J = Array.from({length: m}, () => new Float64Array(n));
    for (let j = 0; j < n; j++) {
      const d = Math.max(1e-6, Math.abs(p[j]) * 1e-5), q = p.slice(); q[j] += d;
      const r1 = residual(q);
      for (let i = 0; i < m; i++) J[i][j] = (r1[i] - r0[i]) / d;
    }
    const JtJ = Array.from({length: n}, (_, a) => Array.from({length: n}, (_, b) => J.reduce((s, row) => s + row[a] * row[b], 0)));
    const Jtr = Array.from({length: n}, (_, a) => J.reduce((s, row, i) => s + row[a] * r0[i], 0));
    let improved = false;
    for (let tries = 0; tries < 8 && !improved; tries++) {
      const M = JtJ.map((row, a) => [...row.map((v, b) => v + (a === b ? lambda * (v + 1e-9) : 0)), -Jtr[a]]);
      let ok = true;
      for (let i = 0; i < n && ok; i++) {
        let piv = i;
        for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
        if (Math.abs(M[piv][i]) < 1e-14) { ok = false; break; }
        [M[i], M[piv]] = [M[piv], M[i]];
        for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
      }
      if (!ok) { lambda *= 10; continue; }
      const dx = new Array(n).fill(0);
      for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * dx[j]; dx[i] = s / M[i][i]; }
      const q = p.map((v, i) => v + dx[i]), cq = cost(q);
      if (cq < c) { p = q; c = cq; lambda = Math.max(1e-6, lambda / 3); improved = true; } else lambda *= 5;
    }
    if (!improved) break;
  }
  return {p, c};
}
const rng = (() => { let s = 4242; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
const c0 = (SPEC.cutIdx / plan.length) * SPEC.lap;
let best = null;
for (let attempt = 0; attempt < 16; attempt++) {
  const R = 100 + rng() * 150, k = 1 / R, init = [c0 * (0.9 + 0.2 * rng())];
  for (let t = 0; t < nT; t++) init.push(80, k, (Math.PI / nT) / k * 0.8, k, (Math.PI / nT) / k * 0.15, 80);
  const p = attempt ? init.map((v, i) => (i === 0 ? v : v * (0.8 + rng() * 0.4))) : init;
  const res = lm(p);
  console.log('try', attempt, res.c.toFixed(1));
  if (!best || res.c < best.c) best = res;
}
const {elems, L1} = chain(best.p);
const pts = [[goal.x, goal.z]];
const end = walk(goal, elems, 1, (x, z) => pts.push([x, z]));
const dev = plan.map((q) => dist(pts, q) / scale);
console.log('outer length', elems.reduce((s, e) => s + e[1], 0).toFixed(2), 'official', SPEC.lap, 'closure m', Math.hypot(end.x - goal.x, end.z - goal.z).toFixed(3),
  'heading err deg', ((wrap(end.h - goal.h - sgn * 2 * Math.PI) * 180) / Math.PI).toFixed(4), 'cut', best.p[0].toFixed(1), 'L1', L1.toFixed(1),
  '| plan px rms', Math.sqrt(dev.reduce((a, d) => a + d * d, 0) / dev.length).toFixed(2), 'max', Math.max(...dev).toFixed(2), '>3px', dev.filter((d) => d > 3).length, '/', dev.length);
const out = {venue, scale: inner.scale, pose: inner.pose, courses: {turf: inner.courses.turf, outer: {elems}}};
fs.writeFileSync(`outputs/alignment-${venue}.json`, JSON.stringify(out, null, 1));
console.log('wrote outputs/alignment-' + venue + '.json');
