// Tokyo trial plots (SVG/PNG): alignment over the digitised plan, curvature against distance, plan deviations.
import fs from 'node:fs';
import sharp from 'sharp';
const a = JSON.parse(fs.readFileSync('outputs/alignment-05.json', 'utf8'));
const tg = fs.readFileSync('app/trackGeometry.ts', 'utf8');
const grab = (name) => JSON.parse(tg.slice(tg.indexOf('[', tg.indexOf(name + ' =')), tg.indexOf('];', tg.indexOf(name + ' =')) + 1).replace(/,\s*\]/g, ']'));
const planPx = {turf: grab('TOKYO_TURF_PIXELS'), dirt: grab('TOKYO_DIRT_PIXELS')};
const [cx, cy] = [280, 170];
const toM = ([x, y]) => [(x - cx) * a.scale, (y - cy) * a.scale];
const walk = (pose, elems) => {
  let {x, z, h} = pose, k = 0, s = 0; const pts = [[x, z]], kap = [[0, 0]];
  for (const [kind, len, kEnd = 0] of elems) {
    const k0 = kind === 'L' ? 0 : k, k1 = kind === 'L' ? 0 : kind === 'A' ? k : kEnd;
    const n = Math.max(1, Math.ceil(len)), ds = len / n;
    for (let i = 0; i < n; i++) {
      const km = k0 + (k1 - k0) * ((i + .5) / n), hm = h + km * ds / 2;
      x += Math.cos(hm) * ds; z += Math.sin(hm) * ds; h += km * ds; s += ds;
      pts.push([x, z]); kap.push([s, k0 + (k1 - k0) * ((i + 1) / n)]);
    }
    k = k1;
  }
  return {pts, kap};
};
const rails = {turf: walk(a.pose, a.courses.turf.elems), dirt: walk(a.courses.dirt.pose, a.courses.dirt.elems)};
const plan = {turf: planPx.turf.map(toM), dirt: planPx.dirt.map(toM)};
const all = [...plan.turf, ...plan.dirt, ...rails.turf.pts, ...rails.dirt.pts];
const xs = all.map((p) => p[0]), zs = all.map((p) => p[1]);
const [x0, x1, z0, z1] = [Math.min(...xs) - 30, Math.max(...xs) + 30, Math.min(...zs) - 40, Math.max(...zs) + 30];
const poly = (pts, color, w, dash = '') => `<polyline fill="none" stroke="${color}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ''} points="${[...pts, pts[0]].map((p) => p.map((v) => v.toFixed(1)).join(',')).join(' ')}"/>`;
const dots = (pts, color) => pts.map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.2" fill="${color}"/>`).join('');
const dir = 'docs/visual-baseline/tokyo/';
fs.mkdirSync(dir, {recursive: true});
fs.writeFileSync(dir + 'overlay.svg', `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${x0} ${z0} ${x1 - x0} ${z1 - z0}" width="1400" style="background:#fff">${poly(plan.turf, '#e0732a', 1.4)}${dots(plan.turf, '#e0732a')}${poly(plan.dirt, '#2a7de0', 1.4)}${dots(plan.dirt, '#2a7de0')}${poly(rails.turf.pts, '#111', 1)}${poly(rails.dirt.pts, '#c01040', 1, '5 3')}<text x="${x0 + 20}" y="${z0 + 24}" font-family="sans-serif" font-size="14">orange/blue dots = digitised plan (turf/dirt); black = new turf rail; red dashed = new dirt rail (m)</text></svg>`);
const curv = (name, c, lap, color, y0) => {
  const W = 1300, H = 200, kmax = 0.01, px = (s) => 60 + (s / 2100) * W, py = (k) => y0 + H / 2 - (k / kmax) * (H / 2);
  const line = c.kap.map(([s, k]) => `${px(s).toFixed(1)},${py(-k).toFixed(1)}`).join(' ');
  return `<g><rect x="60" y="${y0}" width="${W}" height="${H}" fill="none" stroke="#bbb"/><line x1="60" x2="${60 + W}" y1="${py(0)}" y2="${py(0)}" stroke="#bbb"/><polyline fill="none" stroke="${color}" stroke-width="2" points="${line}"/><text x="64" y="${y0 + 18}" font-size="16" font-family="sans-serif">${name} (lap ${lap} m; curvature 1/R, up = left turn; edge = R${(1 / kmax).toFixed(0)} m)</text><text x="64" y="${y0 + H + 16}" font-size="13" font-family="sans-serif">0 m = winning post, distance along the race direction</text></g>`;
};
fs.writeFileSync(dir + 'curvature.svg', `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 520" width="1400" style="background:#fff">${curv('Turf', rails.turf, 2083.1, '#111', 20)}${curv('Dirt', rails.dirt, 1899, '#c01040', 280)}</svg>`);
for (const n of ['overlay', 'curvature']) await sharp(dir + n + '.svg', {density: 96}).png().toFile(dir + n + '.png');
const near = (pts, q) => { let b = Infinity, bi = 0; for (let i = 0; i < pts.length; i++) { const e = Math.hypot(pts[i][0] - q[0], pts[i][1] - q[1]); if (e < b) { b = e; bi = i; } } return [b, bi]; };
const rows = [];
for (const name of ['turf', 'dirt']) {
  const dev = plan[name].map((q) => near(rails[name].pts, q)).map(([e, i]) => ({px: e / a.scale, s: rails[name].kap[Math.min(i, rails[name].kap.length - 1)][0]}));
  let run = null;
  dev.forEach((v) => { if (v.px > 3) { if (!run) run = {from: v.s, to: v.s, max: 0}; run.to = v.s; run.max = Math.max(run.max, v.px); } else if (run) { rows.push({name, ...run}); run = null; } });
  if (run) rows.push({name, ...run});
  const sorted = dev.map((v) => v.px).sort((x, y) => x - y);
  console.log(name, 'points', dev.length, 'mean', (sorted.reduce((x, y) => x + y, 0) / sorted.length).toFixed(1), 'max', sorted.at(-1).toFixed(1), '>3px', sorted.filter((x) => x > 3).length);
  const kmax = Math.max(...rails[name].kap.map((p) => Math.abs(p[1]))); console.log(name, 'min radius', (1 / kmax).toFixed(0));
}
for (const r of rows) console.log(`${r.name} s=${r.from.toFixed(0)}..${r.to.toFixed(0)} m max ${r.max.toFixed(1)} px (${(r.max * a.scale).toFixed(0)} m)`);
