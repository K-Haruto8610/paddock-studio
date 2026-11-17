// Dumps every GⅠ route (and the plan start mark) in plan pixels to outputs/route-check/routes.json.
// scripts/draw_route_overlays.py draws them on the JRA plan images for visual review.
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {G1_RACES} from './g1_races.mjs';
const out=path.resolve('outputs/geometry-check');await fs.mkdir(out,{recursive:true});
for(const name of ['courseData','alignmentData','alignment','courses','trackGeometry','raceRoutes','raceLayout']){
 const src=await fs.readFile(`app/${name}.ts`,'utf8');
 const js=ts.transpileModule(src,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText.replace(/from '(\.\/[^']+)'/g,"from '$1.js'");
 await fs.writeFile(path.join(out,name+'.js'),js);
}
const {makeTrack,drawingFor}=await import(pathToFileURL(path.join(out,'trackGeometry.js')));
const {makeRoute,ROUTE_SPECS}=await import(pathToFileURL(path.join(out,'raceRoutes.js')));
const {courseFor}=await import(pathToFileURL(path.join(out,'courses.js')));
const data=JSON.parse(await fs.readFile('public/data/forecasts.json','utf8'));
const result=[];
for(const [venueCode,surface,trackCode,distance,,name] of G1_RACES){
 const base=data.races.find(r=>r.venueCode===venueCode);
 const race={...base,surface,trackCode,distance,name};
 const track=makeTrack(race),route=makeRoute(race,track),turf=track.kind==='turf'?track:makeTrack(race,'turf');
 const [cx,cy]=drawingFor(venueCode).center,px=p=>[p.x/turf.scale+cx,p.z/turf.scale+cy];
 const spec=ROUTE_SPECS[`${venueCode}:${courseFor(race).kind}:${distance}`];
 const N=600,path=Array.from({length:N+1},(_,i)=>px(route.sample(i/N*distance).point));
 result.push({venueCode,name,surface,distance,kind:track.kind,source:route.source,markError:route.markError,mark:spec.mark,plan:spec.plan,path,chute:route.chute.map(px)});
}
await fs.mkdir('outputs/route-check',{recursive:true});
await fs.writeFile('outputs/route-check/routes.json',JSON.stringify(result));
console.log(`wrote ${result.length} routes`);
