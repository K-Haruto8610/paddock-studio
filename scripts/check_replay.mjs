import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const out=path.resolve('outputs/replay-check');await fs.mkdir(out,{recursive:true});
for(const name of ['courseData','alignmentData','alignment','courses','raceLayout','playback','closingKick']){
 const src=await fs.readFile(`app/${name}.ts`,'utf8');
 const js=ts.transpileModule(src,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g,"from '$1.js'");
 await fs.writeFile(path.join(out,name+'.js'),js);
}
const {buildPlayback,playbackFrame}=await import(pathToFileURL(path.join(out,'playback.js')));
const {HORSE_LENGTH,HORSE_WIDTH}=await import(pathToFileURL(path.join(out,'raceLayout.js')));
const data=JSON.parse(await fs.readFile('public/data/forecasts.json','utf8')),results=JSON.parse(await fs.readFile('public/data/replays.json','utf8'));
for(const r of data.races)if(r.historical!==false&&!results.races[r.id])console.warn(`WARN no replay (forecast only): ${r.id} ${r.name}`);
for(const race of data.races.filter(r=>results.races[r.id])){
 const replay=results.races[race.id],plan=buildPlayback(race,'replay',replay);
 assert.ok(plan.available);
 assert.deepEqual(plan.checkpoints.filter(c=>c.corner).map(c=>c.corner),replay.corners,'Omit nonexistent corners');
 assert.ok(plan.knots.every((v,i)=>i===0||v>plan.knots[i-1]),'Monotonic checkpoints');
 for(const c of plan.checkpoints.filter(c=>c.observed)){
  const frame=playbackFrame(race,plan,c.distance/race.distance);
  assert.equal(frame.exact,c,'Checkpoint seek is exact');
  for(let i=0;i<race.horses.length;i++){
   const source=replay.horses[String(race.horses[i].number)];
   assert.equal(c.ranks[i],c.corner?source.corners[c.corner-1]:source.finish,'Match recorded rank, including tied groups');
   for(let j=i+1;j<race.horses.length;j++){
    if(c.ranks[i]!==c.ranks[j])assert.equal(Math.sign(frame.placements[i].gap-frame.placements[j].gap),Math.sign(c.ranks[i]-c.ranks[j]),'3D order matches source');
   }
  }
 }
 const fourth=plan.checkpoints.find(c=>c.corner===4),at4=playbackFrame(race,plan,fourth.distance/race.distance),atFinish=playbackFrame(race,plan,1);
 assert.notDeepEqual(at4.order.map(h=>h.horse.number),atFinish.order.map(h=>h.horse.number),'Straight changes order until finish');
 const forecast=buildPlayback(race,'forecast',replay),withoutResults=buildPlayback(race,'forecast');
 assert.deepEqual(forecast,withoutResults,'Forecast must never consume observed results');
 assert.notDeepEqual(forecast.layouts.at(-1),forecast.layouts.at(-2),'Last3F prediction animates the straight');
 for(let n=0;n<=200;n++){
  const frame=playbackFrame(race,plan,n/200);
  assert.ok(frame.placements.every(p=>Number.isFinite(p.gap)&&Number.isFinite(p.lane)&&p.lane>0&&p.lane<35));
  for(let a=0;a<frame.placements.length;a++)for(let b=a+1;b<frame.placements.length;b++){
   const x=frame.placements[a],y=frame.placements[b];assert.ok(Math.abs(x.gap-y.gap)>=HORSE_LENGTH*.95||Math.abs(x.lane-y.lane)>=HORSE_WIDTH*.6,'No overlapping silhouettes');
  }
 }
 console.log(`${race.name}: observed checkpoints=${plan.checkpoints.slice(1).map(c=>c.label).join(',')} PASS`);
}
const derby=data.races.find(r=>r.name==='東京優駿'),p=buildPlayback(derby,'replay',results.races[derby.id]),idx=derby.horses.findIndex(h=>h.number===5);
assert.deepEqual(p.checkpoints.slice(1).map(c=>c.ranks[idx]),[18,18,2,2,3],'Regression: Bastele makes a mid-race move');
const absent=buildPlayback(derby,'replay');assert.equal(absent.available,false,'Missing results must not fall back to predictions');
console.log('PASS: actual-rank parity, mid-race movement, finish movement, missing corners, isolated prediction input, clearance');
