// Kyoto trial plots (SVG): alignment over the digitised plan, and curvature against distance.
import fs from 'node:fs';
const a = JSON.parse(fs.readFileSync('outputs/alignment-08-wide.json', 'utf8'));
const src = fs.readFileSync('app/courseData.ts', 'utf8');
const data = JSON.parse(src.slice(src.indexOf('=', src.indexOf('COURSE_DATA')) + 1).trim().replace(/;\s*$/, ''));
const d = data['08'], [cx, cy] = d.center;
const toM = ([x, y]) => [(x - cx) * a.scale, (y - cy) * a.scale];
const walk = (elems) => {
  let {x, z, h} = a.pose, k = 0, s = 0; const pts = [[x, z]], kap = [[0, 0]];
  for (const [kind, len, kEnd = 0] of elems) {
    const k0 = kind === 'L' ? 0 : k, k1 = kind === 'L' ? 0 : kind === 'A' ? k : kEnd;
    const n = Math.max(1, Math.ceil(len / 1)), ds = len / n;
    for (let i = 0; i < n; i++) {
      const km = k0 + (k1 - k0) * ((i + .5) / n), hm = h + km * ds / 2;
      x += Math.cos(hm) * ds; z += Math.sin(hm) * ds; h += km * ds; s += ds;
      pts.push([x, z]); kap.push([s, k0 + (k1 - k0) * ((i + 1) / n)]);
    }
    k = k1;
  }
  return {pts, kap};
};
const inner = walk(a.courses.inner.elems), outer = walk(a.courses.outer.elems);
const plan = {inner: d.rails.turf.map(toM), outer: d.rails.outer.map(toM)};
fs.mkdirSync('docs/visual-baseline/kyoto-trial', {recursive: true});
const all = [...plan.inner, ...plan.outer, ...inner.pts, ...outer.pts];
const xs = all.map((p) => p[0]), zs = all.map((p) => p[1]);
const [x0, x1, z0, z1] = [Math.min(...xs) - 30, Math.max(...xs) + 30, Math.min(...zs) - 30, Math.max(...zs) + 30];
const poly = (pts, color, w, dash = '', close = false) =>
  `<polyline fill="none" stroke="${color}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ''} points="${pts.map((p) => p.map((v) => v.toFixed(1)).join(',')).join(' ')}${close ? ' ' + pts[0].join(',') : ''}"/>`;
const overlay = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${x0} ${z0} ${x1 - x0} ${z1 - z0}" width="1400" style="background:#fff">
<g>${poly(plan.inner, '#e0732a', 1.6, '', true)}${poly(plan.outer, '#2a7de0', 1.6, '', true)}
${poly(inner.pts, '#111', 1)}${poly(outer.pts, '#c01040', 1, '5 3')}</g>
<g font-family="sans-serif" font-size="14" fill="#222"><text x="${x0 + 20}" y="${z0 + 24}">orange/blue = plan (inner/outer); black = new inner; red dashed = new outer (m)</text></g></svg>`;
fs.writeFileSync('docs/visual-baseline/kyoto-trial/overlay.svg', overlay);
const curv = (name, c, lap, color, y0) => {
  const W = 1300, H = 200, kmax = 0.013;
  const px = (s) => 60 + (s / 1900) * W, py = (k) => y0 + H / 2 - (k / kmax) * (H / 2);
  const line = c.kap.map(([s, k]) => `${px(s).toFixed(1)},${py(k).toFixed(1)}`).join(' ');
  return `<g><rect x="60" y="${y0}" width="${W}" height="${H}" fill="none" stroke="#bbb"/>
<line x1="60" x2="${60 + W}" y1="${py(0)}" y2="${py(0)}" stroke="#bbb"/>
<polyline fill="none" stroke="${color}" stroke-width="2" points="${line}"/>
<text x="64" y="${y0 + 18}" font-size="16" font-family="sans-serif">${name} (lap ${lap} m; curvature 1/R, +: clockwise; ${(kmax * 1000).toFixed(0)}/1000 at edge = R${(1 / kmax).toFixed(0)}m)</text>
<text x="64" y="${y0 + H + 16}" font-size="13" font-family="sans-serif">0 m = winning post, distance along the race direction</text></g>`;
};
fs.writeFileSync('docs/visual-baseline/kyoto-trial/curvature.svg', `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 520" width="1400" style="background:#fff">${curv('Inner (turf)', inner, 1782.8, '#111', 20)}${curv('Outer', outer, 1894.3, '#c01040', 280)}</svg>`);
const kmaxI = Math.max(...inner.kap.map((p) => Math.abs(p[1]))), kmaxO = Math.max(...outer.kap.map((p) => Math.abs(p[1])));
console.log('max curvature', kmaxI, kmaxO, 'R min', 1 / kmaxI, 1 / kmaxO);

// Plan deviation: per digitised point, distance to the new rail (in plan pixels), grouped into runs over 3 px.
const near = (pts, q) => { let b = Infinity, bi = 0; for (let i = 0; i < pts.length; i++) { const e = Math.hypot(pts[i][0] - q[0], pts[i][1] - q[1]); if (e < b) { b = e; bi = i; } } return [b, bi]; };
const rows = [];
for (const [name, w] of [['inner', inner], ['outer', outer]]) {
  const dev = plan[name].map((q) => near(w.pts, q)).map(([e, i]) => ({px: e / a.scale, s: w.kap[Math.min(i, w.kap.length - 1)][0]}));
  let run = null;
  dev.forEach((v, i) => {
    if (v.px > 3) { if (!run) run = {from: v.s, to: v.s, max: 0, i0: i}; run.to = v.s; run.max = Math.max(run.max, v.px); run.i1 = i; }
    else if (run) { rows.push({course: name, ...run}); run = null; }
  });
  if (run) rows.push({course: name, ...run});
  const sorted = dev.map((v) => v.px).sort((x, y) => x - y);
  console.log(name, 'points', dev.length, 'mean', (sorted.reduce((x, y) => x + y, 0) / sorted.length).toFixed(1), 'p90', sorted[Math.floor(sorted.length * .9)].toFixed(1), 'max', sorted.at(-1).toFixed(1), 'within3px', sorted.filter((x) => x <= 3).length);
}
for (const r of rows) console.log(`${r.course} s=${r.from.toFixed(0)}..${r.to.toFixed(0)} m  max ${r.max.toFixed(1)} px (${(r.max * a.scale).toFixed(0)} m)`);
