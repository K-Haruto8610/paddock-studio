import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const out=path.resolve('outputs/geometry-check');await fs.mkdir(out,{recursive:true});
for(const name of ['courses','trackGeometry','raceLayout']){
 const src=await fs.readFile(`app/${name}.ts`,'utf8');
 const js=ts.transpileModule(src,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g,"from '$1.js'");
 await fs.writeFile(path.join(out,name+'.js'),js);
}
const {makeTrack,makeRoute,TURF_HEIGHT,linearProfile}=await import(pathToFileURL(path.join(out,'trackGeometry.js')));
const {buildLayouts,sampleLayouts}=await import(pathToFileURL(path.join(out,'raceLayout.js')));
const data=JSON.parse(await fs.readFile('public/data/forecasts.json','utf8'));
const races=data.races.filter(r=>r.venueCode==='05');
for(const distance of [1800,2000])races.push({...races[0],distance,name:`Tokyo turf ${distance}m chute`});
for(const race of races){
 const track=makeTrack(race,race.surface==='ダート'),route=makeRoute(race,track);
 assert.ok(Math.abs(track.curve.getLength()-track.lap)<.1,'Calibrated rail length');
 assert.ok(track.sample(0).point.distanceTo(track.sample(track.lap).point)<.001,'Closed rail');
 assert.ok(route.sample(race.distance).point.distanceTo(track.sample(0).point)<.05,'Finish at winning post');
 assert.ok(Math.abs(track.sample(400,10).point.distanceTo(track.sample(400,0).point)-10)<1e-6,'Lanes use metres');
 const layouts=buildLayouts(race);const knots=route.phaseDistances;
 assert.ok(knots.every((v,i)=>i===0||v>knots[i-1]),'Monotonic phase distances');
 let maxSpan=0;for(let i=0;i<=500;i++){
  const travel=i/500*race.distance,places=sampleLayouts(layouts,travel,knots);
  maxSpan=Math.max(maxSpan,...places.map(p=>p.gap));
  assert.ok(places.every(p=>Number.isFinite(p.gap)&&p.lane>0&&p.lane<track.width-1),'In-bounds finite placements');
  for(let a=0;a<places.length;a++)for(let b=a+1;b<places.length;b++)assert.ok(Math.abs(places[a].gap-places[b].gap)>=2.4||Math.abs(places[a].lane-places[b].lane)>=1,'No horse silhouette overlap');
  if(i>0){const prev=route.sample(travel-race.distance/500).point,now=route.sample(travel).point;assert.ok(prev.distanceTo(now)<race.distance/500*2.5,'Continuous route, including chute join');}
 }
 assert.ok(maxSpan>25&&maxSpan<150,'Field has a plausible visible longitudinal extent');
 console.log(`${race.name}: rail=${track.curve.getLength().toFixed(1)}m field span=${maxSpan.toFixed(1)}m, route continuity PASS`);
}
const heights=Array.from({length:2084},(_,d)=>linearProfile(TURF_HEIGHT,d));
assert.ok(Math.abs(Math.max(...heights)-Math.min(...heights)-2.7)<.01,'Official turf elevation range');
assert.ok(Math.abs(linearProfile(TURF_HEIGHT,460)+2)<.01&&linearProfile(TURF_HEIGHT,300)===0,'Home straight uphill 460m to 300m');
console.log('PASS: physical scale, source profile, track closure, starts/finish, chute continuity, field layout');
