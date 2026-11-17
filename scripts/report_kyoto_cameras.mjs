// World-space camera placements for the Kyoto trial images (outer course, Queen Elizabeth Cup race).
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const tmp = await fs.mkdtemp(path.resolve('outputs', 'kyoto-tmp-'));
for (const f of await fs.readdir('app')) {
  if (!f.endsWith('.ts')) continue;
  const js = ts.transpileModule(await fs.readFile('app/' + f, 'utf8'), {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g, "from '$1.js'");
  await fs.writeFile(path.join(tmp, f.replace(/\.ts$/, '.js')), js);
}
await fs.writeFile(path.join(tmp, 'package.json'), '{"type":"module"}');
const {makeTrack} = await import(pathToFileURL(path.join(tmp, 'trackGeometry.js')));
const data = JSON.parse(await fs.readFile('docs/visual-baseline/forecasts.json', 'utf8'));
const race = data.races.find((r) => r.id === '2026111508050411');
const outer = makeTrack(race), inner = makeTrack({...race, trackCode: 17, distance: 2000});
const P = (t, d, lane = 0, up = 0) => { const p = t.sample(d, lane).point; return [+p.x.toFixed(1), +(p.y + up).toFixed(1), +p.z.toFixed(1)]; };
// where the outer rail leaves / rejoins the inner one
const gapAt = (d) => { const p = outer.sample(d, 0).point; let b = 1e9; for (let s = 0; s < inner.lap; s += 1) { const q = inner.sample(s, 0).point; b = Math.min(b, Math.hypot(p.x - q.x, p.z - q.z)); } return b; };
let leave = null, rejoin = null;
for (let d = 0; d < outer.lap; d += 4) { const g = gapAt(d); if (g > 1.5 && leave === null && d > 100) leave = d; if (leave !== null && g < 1.5 && d > leave) { rejoin = d; break; } }
const L = outer.lap;
const out = {lap: L, leave, rejoin,
  whole: {position: [0, 330, 1000], target: [0, 0, 0]},
  straightLow: {position: P(outer, L - 345, 14, 2.4), target: P(outer, L - 60, 14, 1.2)},
  slopeSide: {position: P(outer, L - 1000 + 0, 75, 5), target: P(outer, L - 1000, 0, 1)},
  leaveClose: {position: P(outer, leave - 25, 38, 14), target: P(outer, leave + 5, 6, 0)},
  rejoinClose: {position: P(outer, rejoin - 20, 40, 14), target: P(outer, rejoin + 5, 6, 0)},
  bend: {position: P(outer, L - 110, 45, 16), target: P(outer, L - 70, 8, 0)},
};
console.log(JSON.stringify(out));
await fs.writeFile('outputs/kyoto-trial/cameras.json', JSON.stringify(out, null, 1));
await fs.rm(tmp, {recursive: true, force: true});
