// Fits a venue whose courses are independent closed loops (Tokyo: turf and dirt) with
// "line, easing curve, arcs, easing curve, line ..." chains. Official lap and final straight are exact;
// the digitised plan only decides the shape.   usage: node scripts/fit_simple.mjs 05
import fs from 'node:fs';
import * as T from 'three';

const venue = process.argv[2] ?? '05';
const SPEC = {
  '05': {left: true, courses: {turf: {lap: 2083.1, straight: 525.9}, dirt: {lap: 1899, straight: 501.6}}},
  '09': {left: false, courses: {turf: {lap: 1689, straight: 356.5}}},
  '06': {left: false, courses: {turf: {lap: 1667.1, straight: 310}}},
  '07': {left: true, courses: {turf: {lap: 1705.9, straight: 412.5}, dirt: {lap: 1530, straight: 410.7}}},
};
const spec = SPEC[venue];
const sgn = spec.left ? -1 : 1;

// ---- plan data ----------------------------------------------------------------------------------
const tg = fs.readFileSync('app/trackGeometry.ts', 'utf8');
const grab = (name) => JSON.parse(tg.slice(tg.indexOf('[', tg.indexOf(name + ' =')), tg.indexOf('];', tg.indexOf(name + ' =')) + 1).replace(/,\s*\]/g, ']'));
let plans, cx, cy;
if (venue === '05') { plans = {turf: grab('TOKYO_TURF_PIXELS'), dirt: grab('TOKYO_DIRT_PIXELS')}; [cx, cy] = [280, 170]; }
else {
  const cd = fs.readFileSync('app/courseData.ts', 'utf8');
  const data = JSON.parse(cd.slice(cd.indexOf('{', cd.indexOf('COURSE_DATA')), cd.lastIndexOf('}') + 1))[venue];
  plans = {turf: data.rails.turf, dirt: data.rails.dirt, outer: data.rails.outer}; [cx, cy] = data.center;
}

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
function walk(pose, elems, step, emit) {
  let {x, z, h} = pose, k = 0, s = 0;
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
  return {x, z, h};
}
function distToPolyline(pts, q) {
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

// Parameter vector -> chain. Two turns (egg shaped: each may use two radii), a back straight between them.
const SMID = 30;
function chain(p, straight, lap) {
  const [L0, sIn1, k1a, a1a, k1b, a1b, sOut1, sIn2, k2a, a2a, k2b, a2b, sOut2] = p;
  const turn = (sIn, ka, aa, kb, ab, sOut) => [['S', sIn, sgn * ka], ['A', aa], ['S', SMID, sgn * kb], ['A', ab], ['S', sOut, 0]];
  const fixed = L0 + sIn1 + a1a + SMID + a1b + sOut1 + sIn2 + a2a + SMID + a2b + sOut2 + straight;
  const L1 = lap - fixed;
  return {L1, elems: [['L', Math.max(0, L0)], ...turn(sIn1, k1a, a1a, k1b, a1b, sOut1), ['L', Math.max(0, L1)], ...turn(sIn2, k2a, a2a, k2b, a2b, sOut2), ['L', straight]]};
}

function fitCourse(name, scale, fixRatio) {
  const {lap, straight} = spec.courses[name];
  const plan = plans[name].map(([x, y]) => [x - cx, y - cy]);
  let ratio0 = 1;
  if (!fixRatio) {
    const raw = new T.CatmullRomCurve3(plan.map(([x, z]) => new T.Vector3(x, 0, z)), true, 'centripetal');
    raw.arcLengthDivisions = 4096;
    scale = lap / raw.getLength();
  }
  const h0 = Math.atan2(plan[1][1] - plan[0][1], plan[1][0] - plan[0][0]);
  const toM = (r) => plan.map(([x, z]) => [x * scale * r, z * scale * r]);
  const R = (ratio) => ({x: plan[0][0] * scale * ratio, z: plan[0][1] * scale * ratio});
  const residual = (p) => {
    const ratio = fixRatio ? 1 : p[13], goal = R(ratio);
    const {L1, elems} = chain(p, straight, lap);
    const pts = [[goal.x, goal.z]];
    const end = walk({x: goal.x, z: goal.z, h: p[14]}, elems, 3, (x, z) => pts.push([x, z]));
    const r = [];
    for (const q of toM(ratio)) r.push(softAbs(distToPolyline(pts, q), 4));
    r.push(80 * (end.x - goal.x), 80 * (end.z - goal.z), 8000 * wrap(end.h - p[14] - sgn * 2 * Math.PI));
    // Keep the shape sane: radii, easing lengths and the back straight within bounds.
    const bound = (v, lo, hi) => 200 * (Math.max(0, lo - v) + Math.max(0, v - hi));
    for (const i of [2, 4, 8, 10]) r.push(bound(1 / p[i], 60, 420));
    for (const i of [1, 6, 7, 12]) r.push(bound(p[i], 25, 260));
    for (const i of [3, 5, 9, 11]) r.push(bound(p[i], 0, 800));
    r.push(bound(L1, 60, 1200), bound(p[0], 0, 300));
    if (!fixRatio) r.push(bound(ratio, 0.95, 1.05) * 50);
    return r;
  };
  const cost = (p) => residual(p).reduce((a, v) => a + v * v, 0);
  // initial guess: two half-circles
  const Rad = (lap - 2 * straight) / (2 * Math.PI) * 0.9;
  const make = (R0) => {
    const k = 1 / R0, s = 80, a = (Math.PI - k * (s + SMID + s) / 2) / k / 2 * 0.98;
    return [30, s, k, a, k, a, s, s, k, a, k, a, s, 1, h0];
  };
  let bestP = null, bestC = Infinity;
  const rng = (() => { let s = 12345; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
  for (let attempt = 0; attempt < 14; attempt++) {
    let p = make((120 + rng() * 120));
    if (attempt) for (let i = 0; i < p.length - 2; i++) p[i] *= 0.8 + rng() * 0.4;
    p = p.slice(0, 15);
    let lambda = 1, c = cost(p);
    for (let it = 0; it < 220; it++) {
      const r0 = residual(p), n = p.length, m = r0.length;
      const J = Array.from({length: m}, () => new Float64Array(n));
      for (let j = 0; j < n; j++) {
        if (fixRatio && j === 13) continue;
        const d = Math.max(1e-6, Math.abs(p[j]) * 1e-5), q = p.slice();
        q[j] += d;
        const r1 = residual(q);
        for (let i = 0; i < m; i++) J[i][j] = (r1[i] - r0[i]) / d;
      }
      const JtJ = Array.from({length: n}, (_, a) => Array.from({length: n}, (_, b) => J.reduce((s, row) => s + row[a] * row[b], 0)));
      const Jtr = Array.from({length: n}, (_, a) => J.reduce((s, row, i) => s + row[a] * r0[i], 0));
      let improved = false;
      for (let tries = 0; tries < 8 && !improved; tries++) {
        const A = JtJ.map((row, a) => row.map((v, b) => v + (a === b ? lambda * (v + 1e-9) : 0)));
        const b = Jtr.map((v) => -v);
        // gaussian elimination
        const M = A.map((row, i) => [...row, b[i]]);
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
        const q = p.map((v, i) => v + dx[i]);
        const cq = cost(q);
        if (cq < c) { p = q; c = cq; lambda = Math.max(1e-6, lambda / 3); improved = true; } else lambda *= 5;
      }
      if (!improved) break;
    }
    console.log(name, 'try', attempt, c.toFixed(1));
    if (c < bestC) { bestC = c; bestP = p; }
  }
  const p = bestP, ratio = fixRatio ? 1 : p[13], goal = R(ratio);
  const {L1, elems} = chain(p, straight, lap);
  const pts = [[goal.x, goal.z]], headings = [];
  const end = walk({x: goal.x, z: goal.z, h: p[14]}, elems, 1, (x, z, h) => { pts.push([x, z]); headings.push(h); });
  const dev = toMetres(plan, scale * ratio).map((q) => distToPolyline(pts, q) / (scale * ratio));
  console.log(name, 'length', elems.reduce((s, e) => s + e[1], 0).toFixed(2), 'official', lap, 'closure m', Math.hypot(end.x - goal.x, end.z - goal.z).toFixed(3),
    'heading err deg', ((wrap(end.h - p[14] - sgn * 2 * Math.PI) * 180) / Math.PI).toFixed(4), 'L1', L1.toFixed(1), 'ratio', ratio.toFixed(4),
    '| plan px rms', Math.sqrt(dev.reduce((a, d) => a + d * d, 0) / dev.length).toFixed(2), 'max', Math.max(...dev).toFixed(2), '>3px', dev.filter((d) => d > 3).length, '/', dev.length);
  return {scale: scale * ratio, pose: {x: goal.x, z: goal.z, h: p[14]}, elems, dev};
}
const toMetres = (plan, s) => plan.map(([x, z]) => [x * s, z * s]);

const turf = fitCourse('turf', 0, false);
const dirt = spec.courses.dirt ? fitCourse('dirt', turf.scale, false) : null;
// the dirt plan is drawn on the same sheet, so it uses the turf's scale; its goal is its own plan point
const out = {venue, scale: turf.scale, pose: turf.pose, courses: {turf: {elems: turf.elems}, ...(dirt ? {dirt: {elems: dirt.elems, pose: dirt.pose}} : {})}};
fs.writeFileSync(`outputs/alignment-${venue}.json`, JSON.stringify(out, null, 1));
console.log('wrote outputs/alignment-' + venue + '.json');
