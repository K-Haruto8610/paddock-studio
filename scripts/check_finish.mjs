import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const out=path.resolve('outputs/finish-check');await fs.mkdir(out,{recursive:true});
for(const name of ['courseData','alignmentData','alignment','courses','raceLayout','playback','closingKick']){
 const src=await fs.readFile(`app/${name}.ts`,'utf8');
 const js=ts.transpileModule(src,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g,"from '$1.js'");
 await fs.writeFile(path.join(out,name+'.js'),js);
}
const {buildPlayback,playbackFrame,playbackEndProgress}=await import(pathToFileURL(path.join(out,'playback.js')));
const {HORSE_LENGTH,HORSE_WIDTH}=await import(pathToFileURL(path.join(out,'raceLayout.js')));
const data=JSON.parse(await fs.readFile('public/data/forecasts.json','utf8')),results=JSON.parse(await fs.readFile('public/data/replays.json','utf8'));
for (const source of data.races) {
 for (const count of [source.horses.length, 4]) {
  const race = {...source, horses: source.horses.slice(0, count)};
  const plans = [buildPlayback(race, 'forecast')];
  if (results.races[race.id]) plans.push(buildPlayback(race, 'replay', results.races[race.id]));
  for (const plan of plans) {
   const end = playbackEndProgress(race, plan);
   const frame = playbackFrame(race, plan, end);
   const finished = frame.placements.filter(p => frame.travel - p.gap > race.distance);
   assert.ok(end > 1, 'Playback continues beyond the leader reaching the finish');
   assert.ok(finished.length >= Math.min(6, race.horses.length), 'Top six (or smaller field) cross before stopping');
   assert.deepEqual(playbackFrame(race, plan, end + 1).placements, frame.placements);
   assert.equal(playbackFrame(race, plan, end + 1).travel, frame.travel, 'Clamp at the extended end');
  }
 }
}
console.log('PASS: top-six finish and smaller-field completion for playback and recording timeline');
