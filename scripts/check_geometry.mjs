import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const out=path.resolve('outputs/geometry-check');await fs.mkdir(out,{recursive:true});
for(const name of ['courseData','courses','trackGeometry','raceLayout']){
 const src=await fs.readFile(`app/${name}.ts`,'utf8');
 const js=ts.transpileModule(src,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g,"from '$1.js'");
 await fs.writeFile(path.join(out,name+'.js'),js);
}
const {makeTrack,makeRoute,makeTerrain,TURF_HEIGHT,linearProfile,VERTICAL_EXAGGERATION}=await import(pathToFileURL(path.join(out,'trackGeometry.js')));
const {courseFor,G1_COURSES}=await import(pathToFileURL(path.join(out,'courses.js')));
const {buildLayouts,sampleLayouts,HORSE_LENGTH,HORSE_WIDTH}=await import(pathToFileURL(path.join(out,'raceLayout.js')));
const data=JSON.parse(await fs.readFile('public/data/forecasts.json','utf8'));
const races=[...data.races];const tokyo=races.find(r=>r.venueCode==='05'&&r.surface==='芝');
for(const distance of [1800,2000])races.push({...tokyo,distance,name:`Tokyo turf ${distance}m chute`});
// Every GⅠ course of every venue, including ones no published race uses yet (e.g. Nakayama outer).
for(const base of [...new Map(data.races.map(r=>[r.venueCode,r])).values()]){
 races.push({...base,surface:'芝',trackCode:17,name:`${base.venue} turf inner`});
 if(G1_COURSES[base.venueCode].includes('outer'))races.push({...base,surface:'芝',trackCode:18,name:`${base.venue} turf outer`});
 if(G1_COURSES[base.venueCode].includes('dirt'))
 races.push({...base,surface:'ダート',trackCode:24,distance:1800,horses:base.horses.slice(0,16),name:`${base.venue} dirt`}); // JRA dirt fields are at most 16
}
for(const race of races){
 const course=courseFor(race),track=makeTrack(race),route=makeRoute(race,track);
 assert.equal(track.kind,course.kind,'Race runs on its own rail (inner/outer/dirt)');
 // Lanes grow away from the infield on both left- and right-handed courses.
 const centre=track.curve.getPoints(200).reduce((c,p)=>c.add(p.clone().multiplyScalar(1/201)),track.sample(0).point.clone().multiplyScalar(0));
 for(let i=0;i<40;i++){const d=i/40*track.lap,a=track.sample(d,0).point,b=track.sample(d,10).point;assert.ok(Math.hypot(b.x-centre.x,b.z-centre.z)>Math.hypot(a.x-centre.x,a.z-centre.z)-1,'Positive lanes point outward');}
 assert.equal(track.lap,course.lap,'Official lap for the rail');
 // Section graph: digitized relief matches the official elevation difference.
 const relief=Array.from({length:400},(_,i)=>track.profile(i/400*track.lap));
 assert.ok(Math.abs(Math.max(...relief)-Math.min(...relief)-course.rise)<.4,`Official elevation difference ${course.rise}m`);
 const drawn=Array.from({length:400},(_,i)=>track.height(i/400*track.lap));
 assert.ok(Math.abs((Math.max(...drawn)-Math.min(...drawn))-(Math.max(...relief)-Math.min(...relief))*VERTICAL_EXAGGERATION)<1e-6,'Relief is drawn exaggerated');
 // Inner turf (and Tokyo) is calibrated exactly; other rails share the venue's plan
 // scale so coinciding rails coincide in 3D, and stay within 5% of the official lap.
 if(track.kind==='turf'||race.venueCode==='05')assert.ok(Math.abs(track.curve.getLength()-track.lap)<.1,'Calibrated rail length');
 else assert.ok(Math.abs(track.curve.getLength()/track.lap-1)<.05,'Shared-scale rail close to official lap');
 // Level stretches and straight slopes only: no stair steps in the section graph.
 {const breaks=[];let last=null;for(let d=0;d<=track.lap;d+=1){const s=Math.round((track.profile(track.lap-d-1)-track.profile(track.lap-d))*1e6);if(last!==null&&s!==last)breaks.push(d);last=s;}
  if(track.kind!=='outer')assert.ok(breaks.length<60,`Few slope breaks (${breaks.length})`);}
 // Where the outer course runs on the inner rail there is no ledge between them.
 if(track.kind==='outer'){const inner=makeTrack(race,'turf');let shared=0;for(let d=0;d<track.lap;d+=10){const p=track.sample(d).point;for(let e=0;e<inner.lap;e+=1){const q=inner.sample(e).point;if(Math.hypot(p.x-q.x,p.z-q.z)<1){shared++;assert.ok(Math.abs(p.y-q.y)<.25,`No ledge on shared rail at ${d}m`);break;}}}assert.ok(shared>20,'Outer course shares the inner rail');}
 assert.ok(track.sample(0).point.distanceTo(track.sample(track.lap).point)<.001,'Closed rail');
 assert.ok(route.sample(race.distance).point.distanceTo(track.sample(0).point)<.05,'Finish at winning post');
 assert.ok(Math.abs(track.sample(400,10).point.distanceTo(track.sample(400,0).point)-10)<1e-6,'Lanes use metres');
 const layouts=buildLayouts(race);const knots=route.phaseDistances;
 assert.ok(knots.every((v,i)=>i===0||v>knots[i-1]),'Monotonic phase distances');
 let maxSpan=0;for(let i=0;i<=500;i++){
  const travel=i/500*race.distance,places=sampleLayouts(layouts,travel,knots);
  maxSpan=Math.max(maxSpan,...places.map(p=>p.gap));
  assert.ok(places.every(p=>Number.isFinite(p.gap)&&p.lane>0&&p.lane<track.width-.5),'In-bounds finite placements');
  for(let a=0;a<places.length;a++)for(let b=a+1;b<places.length;b++)assert.ok(Math.abs(places[a].gap-places[b].gap)>=HORSE_LENGTH*.95||Math.abs(places[a].lane-places[b].lane)>=HORSE_WIDTH*.6,'No horse silhouette overlap');
  if(i>0){const prev=route.sample(travel-race.distance/500).point,now=route.sample(travel).point;assert.ok(prev.distanceTo(now)<race.distance/500*2.5,'Continuous route, including chute join');}
 }
 assert.ok(maxSpan>25&&maxSpan<160,'Field has a plausible visible longitudinal extent');
 // Smooth rails: the outer edge (lane = track width) must not zigzag. Digitizing
 // noise is magnified by the lane offset and showed up as jagged course edges.
 {let prevH=null,prevTurn=null,prevP=null;for(let d=0;d<=track.lap;d+=2){const q=track.sample(d,track.width).point;if(prevP){const len=Math.hypot(q.x-prevP.x,q.z-prevP.z);assert.ok(len>.5&&len<3.6,'Outer edge does not fold back');const h=Math.atan2(q.z-prevP.z,q.x-prevP.x);if(prevH!==null){let a=h-prevH;a=Math.atan2(Math.sin(a),Math.cos(a))*180/Math.PI;if(prevTurn!==null)assert.ok(Math.abs(a-prevTurn)<1.5,`Smooth outer edge at ${d}m`);prevTurn=a;}prevH=h;}prevP=q;}}
 // Terrain never covers the racing surface (horses were buried on Tokyo's slopes).
 const terrain=makeTerrain([track]);
 for(let i=0;i<120;i++){const d=i/120*track.lap;for(const lane of [1,track.width/2,track.width-1]){const p=track.sample(d,lane).point;assert.ok(terrain.heightAt(p.x,p.z)<p.y-.1,'Terrain stays below the track');}}
 console.log(`${race.name}: ${course.kind} rail=${track.curve.getLength().toFixed(1)}m relief=${(Math.max(...relief)-Math.min(...relief)).toFixed(2)}m field span=${maxSpan.toFixed(1)}m PASS`);
}
const heights=Array.from({length:2084},(_,d)=>linearProfile(TURF_HEIGHT,d));
assert.ok(Math.abs(Math.max(...heights)-Math.min(...heights)-2.7)<.01,'Official turf elevation range');
assert.ok(Math.abs(linearProfile(TURF_HEIGHT,460)+2)<.01&&linearProfile(TURF_HEIGHT,300)===0,'Home straight uphill 460m to 300m');
console.log('PASS: all venues on JRA plans, official relief, exaggerated heights, terrain clearance, smooth rails, track closure, starts/finish, chute continuity, field layout');
