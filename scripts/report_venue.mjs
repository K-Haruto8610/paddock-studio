// Per-venue report for the B-展開 venues (alignment fit result vs the digitised plan, plus camera shots).
// usage: node scripts/report_venue.mjs 07 [races...]   -> docs/visual-baseline/<slug>/ (overlay.svg, metrics.json), outputs/<slug>/shots.json
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const venue = process.argv[2];
const slug = {'06': 'nakayama', '07': 'chukyo', '09': 'hanshin'}[venue];
const tmp = await fs.mkdtemp(path.resolve('outputs', slug + '-tmp-'));
try {
  for (const f of await fs.readdir('app')) {
    if (!f.endsWith('.ts')) continue;
    const js = ts.transpileModule(await fs.readFile('app/' + f, 'utf8'), {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g, "from '$1.js'");
    await fs.writeFile(path.join(tmp, f.replace(/\.ts$/, '.js')), js);
  }
  await fs.writeFile(path.join(tmp, 'package.json'), '{"type":"module"}');
  const imp = (n) => import(pathToFileURL(path.join(tmp, n + '.js')));
  const {makeTrack, drawingFor} = await imp('trackGeometry');
  const {courseFor} = await imp('courses');
  const forecasts = JSON.parse(await fs.readFile('docs/visual-baseline/forecasts.json', 'utf8'));
  const races = forecasts.races.filter((r) => r.venueCode === venue);
  const drawing = drawingFor(venue);
  const kindsSeen = new Map();
  for (const r of races) kindsSeen.set(courseFor(r).kind, r);
  for (const k of Object.keys(drawing.rails)) if (!kindsSeen.has(k)) kindsSeen.set(k, {...races[0], surface: k === 'dirt' ? 'ダート' : '芝', trackCode: k === 'dirt' ? 23 : k === 'outer' ? 18 : 17, synthetic: true});
  const dist = (p, pts) => { let best = Infinity; for (let i = 0; i + 1 < pts.length; i++) { const [ax, az] = pts[i], [bx, bz] = pts[i + 1]; const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz || 1; const t = Math.max(0, Math.min(1, ((p[0] - ax) * dx + (p[1] - az) * dz) / l2)); best = Math.min(best, Math.hypot(ax + dx * t - p[0], az + dz * t - p[1])); } return best; };
  const out = {venue, courses: {}};
  const svg = [];
  const color = {turf: ['#e0732a', '#111'], outer: ['#2a7de0', '#c01040'], dirt: ['#2a9d4a', '#7a3']};
  for (const [kind, race] of kindsSeen) {
    const t = makeTrack(race, kind);
    const pts = []; for (let d = 0; d < t.lap; d += 1) { const p = t.sample(d).point; pts.push([p.x, p.z]); } pts.push(pts[0]);
    const plan = drawing.rails[kind].map(([x, y]) => { const p = t.pixel(x, y); return [p.x, p.z]; });
    const dev = plan.map((q) => dist(q, pts) / t.scale);
    out.courses[kind] = {lap: t.lap, railLength: +t.curve.getLength().toFixed(2), planMean: +(dev.reduce((a, b) => a + b, 0) / dev.length).toFixed(2), planMax: +Math.max(...dev).toFixed(2), over3px: dev.filter((d) => d > 3).length, points: dev.length, scale: +t.scale.toFixed(4)};
    const [a, b] = color[kind];
    svg.push(`<polyline fill="none" stroke="${a}" stroke-width="1.4" points="${plan.map((p) => p.join(',')).join(' ')}"/>`, `<polyline fill="none" stroke="${b}" stroke-width="1" points="${pts.map((p) => p.join(',')).join(' ')}"/>`);
  }
  const all = svg.join('').match(/-?\d+\.?\d*,-?\d+\.?\d*/g).map((s) => s.split(',').map(Number));
  const [x0, x1] = [Math.min(...all.map((p) => p[0])) - 20, Math.max(...all.map((p) => p[0])) + 20], [z0, z1] = [Math.min(...all.map((p) => p[1])) - 20, Math.max(...all.map((p) => p[1])) + 20];
  const dir = `docs/visual-baseline/${slug}/`;
  await fs.mkdir(dir, {recursive: true});
  await fs.writeFile(dir + 'overlay.svg', `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${x0} ${z0} ${x1 - x0} ${z1 - z0}" width="1400" style="background:#fff">${svg.join('')}</svg>`);
  await fs.writeFile(dir + 'metrics.json', JSON.stringify(out, null, 1));
  // camera shots: whole view and a low view down each course's final straight
  const shots = [];
  for (const [kind, race] of kindsSeen) {
    const t = makeTrack(race, kind), L = t.lap;
    const P = (d, lane = 0, up = 0) => { const p = t.sample(d, lane).point; return [+p.x.toFixed(1), +(p.y + up).toFixed(1), +p.z.toFixed(1)]; };
    shots.push({name: `whole-${kind}`, race: race.id, fixed: {position: [0, 480, 1050], target: [0, 0, 0]}, scale: undefined});
    shots.push({name: `straight-low-${kind}`, race: race.id, fixed: {position: P(L - 520, 12, 2.4), target: P(L - 60, 12, 1.2)}, scale: undefined});
  }
  await fs.mkdir(`outputs/${slug}`, {recursive: true});
  await fs.writeFile(`outputs/${slug}/shots.json`, JSON.stringify(shots, null, 1));
  console.log(JSON.stringify(out.courses, null, 1));
} finally { await fs.rm(tmp, {recursive: true, force: true}); }
