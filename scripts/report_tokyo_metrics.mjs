// Kyoto metrics for the trial report: lap, corner positions, start-mark error per Kyoto G1 route.
// usage: node scripts/report_kyoto_metrics.mjs <app dir> <out json>   (run on HEAD and on the working tree to compare)
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';
import {G1_RACES} from './g1_races.mjs';
const [appDir = 'app', outFile = 'outputs/tokyo-trial/metrics.json'] = process.argv.slice(2);
const tmp = await fs.mkdtemp(path.resolve('outputs', 'tokyo-tmp-'));
for (const f of await fs.readdir(appDir)) {
  if (!f.endsWith('.ts')) continue;
  const src = await fs.readFile(path.join(appDir, f), 'utf8');
  const js = ts.transpileModule(src, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g, "from '$1.js'");
  await fs.writeFile(path.join(tmp, f.replace(/\.ts$/, '.js')), js);
}
await fs.writeFile(path.join(tmp, 'package.json'), '{"type":"module"}');
const imp = (n) => import(pathToFileURL(path.join(tmp, n + '.js')));
const {makeTrack} = await imp('trackGeometry');
const {makeRoute} = await imp('raceRoutes');
const {courseFor, cornerPositions} = await imp('courses');
const data = JSON.parse(await fs.readFile('public/data/forecasts.json', 'utf8'));
const base = data.races.find((r) => r.venueCode === '05');
const out = {courses: {}, g1: []};
for (const [kind, surface, trackCode] of [['turf', '芝', 11], ['dirt', 'ダート', 23]]) {
  const race = {...base, surface, trackCode, distance: 1600, name: kind};
  const t = makeTrack(race);
  out.courses[kind] = {lap: t.lap, railLength: t.curve.getLength()};
}
for (const [venueCode, surface, trackCode, distance, source, name] of G1_RACES) {
  if (venueCode !== '05') continue;
  const race = {...base, surface, trackCode, distance, name, horses: base.horses};
  const t = makeTrack(race), r = makeRoute(race, t);
  out.g1.push({name, surface, trackCode, distance, source, lap: t.lap, markError: r.markError});
}
await fs.mkdir(path.dirname(outFile), {recursive: true});
await fs.writeFile(outFile, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1).slice(0, 3000));
