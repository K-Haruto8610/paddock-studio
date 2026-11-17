// Camera shots for the Tokyo images: [{name, race, fixed, scale}] -> outputs/tokyo/shots.json
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const tmp = await fs.mkdtemp(path.resolve('outputs', 'tokyo-tmp-'));
for (const f of await fs.readdir('app')) {
  if (!f.endsWith('.ts')) continue;
  const js = ts.transpileModule(await fs.readFile('app/' + f, 'utf8'), {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g, "from '$1.js'");
  await fs.writeFile(path.join(tmp, f.replace(/\.ts$/, '.js')), js);
}
await fs.writeFile(path.join(tmp, 'package.json'), '{"type":"module"}');
const imp = (n) => import(pathToFileURL(path.join(tmp, n + '.js')));
const {makeTrack} = await imp('trackGeometry');
const {makeRoute} = await imp('raceRoutes');
const data = JSON.parse(await fs.readFile('docs/visual-baseline/forecasts.json', 'utf8'));
const find = (id) => data.races.find((r) => r.id === id);
const AUTUMN = '2026110105041111', FEB = '2026022205010811', DERBY = '2026053105021211';
const turf = makeTrack(find(AUTUMN)), dirt = makeTrack(find(FEB));
const P = (t, d, lane = 0, up = 0) => { const p = t.sample(d, lane).point; return [+p.x.toFixed(1), +(p.y + up).toFixed(1), +p.z.toFixed(1)]; };
const L = turf.lap;
const r2000 = makeRoute(find(AUTUMN), turf);
const r1800 = makeRoute({...find(AUTUMN), distance: 1800}, turf);
const rDirt = makeRoute(find(FEB), dirt);
const endOf = (r) => r.chute.at(-1).toArray().map((v) => +v.toFixed(1));
const at = (pt, dx, dy, dz) => [pt[0] + dx, (pt[1] ?? 0) + dy, pt[2] + dz];
const shots = [];
const add = (name, race, position, target, scale = 8) => shots.push({name, race, fixed: {position, target}, scale});
add('whole', AUTUMN, [0, 480, 1050], [0, 0, 0]);
add('whole-dirt', FEB, [0, 480, 1050], [0, 0, 0]);
{ const e = endOf(r2000); add('chute2000-join', AUTUMN, at(e, 40, 30, 80), at(e, 0, 0, 0)); }
{ const e = endOf(r1800); add('chute1800-join', AUTUMN, at(e, 40, 30, 80), at(e, 0, 0, 0)); }
{ const e = endOf(rDirt); add('chuteDirt-join', FEB, at(e, 40, 30, 70), at(e, 0, 0, 0)); }
add('straight-low', DERBY, P(turf, L - 540, 18, 2.4), P(turf, L - 60, 18, 1.2));
add('straight-low-dirt', FEB, P(dirt, dirt.lap - 520, 12, 2.4), P(dirt, dirt.lap - 60, 12, 1.2));
add('slope-side-x8', DERBY, [P(turf, L - 800, 0)[0] + 0, 5, P(turf, L - 800, 0)[2] + 300], [P(turf, L - 800, 0)[0], 6, P(turf, L - 800, 0)[2] - 60]);
add('corner-close', DERBY, P(turf, 220, 70, 30), P(turf, 220, 6, 0));
add('apron', FEB, [215, 40, 380], [215, 0, 250]);
await fs.mkdir('outputs/tokyo', {recursive: true});
await fs.writeFile('outputs/tokyo/shots.json', JSON.stringify(shots, null, 1));
console.log(JSON.stringify(shots.map((s) => [s.name, s.race, s.fixed.position, s.fixed.target])));
await fs.rm(tmp, {recursive: true, force: true});
