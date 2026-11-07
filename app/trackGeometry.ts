import * as T from 'three';
import type {Race} from './types';
import {courseFor} from './courses';

// Digitized INNER RAIL from JRA's 568 x 385 Tokyo plan (not an aerial survey).
// Origin is the winning post. Order follows the official left-handed direction.
export const TOKYO_TURF_PIXELS = [
 [399,270],[428,267],[452,257],[471,239],[484,211],[491,182],
 [489,154],[480,128],[465,107],[445,90],[422,79],[395,72],[366,69],
 [306,68],[246,68],[186,68],[139,68],[112,70],[89,77],
 [70,91],[55,112],[44,139],[38,165],[36,190],[39,215],
 [48,237],[63,255],[84,266],[108,272],[150,273],[220,272],[290,271],[355,270]
];
const TOKYO_DIRT_PIXELS = [
 [399,253],[425,249],[445,237],[459,218],[470,192],[476,169],
 [472,145],[461,123],[443,106],[422,95],[398,88],[367,85],
 [305,84],[245,84],[185,84],[139,84],[114,86],[93,94],[78,109],
 [65,132],[57,157],[53,181],[53,202],[60,222],[74,240],[94,251],
 [117,255],[165,255],[230,254],[300,254],[355,253]
];
// Remaining distance -> relative height, digitized from the official section graphs.
export const TURF_HEIGHT = [[0,0],[300,0],[460,-2],[650,-2.2],[880,-2.5],[1050,-1.1],[1190,-.3],[1260,-.3],[1330,-1.7],[1600,-1],[1800,-.5],[1950,.2],[2083.1,0]];
const DIRT_HEIGHT = [[0,0],[200,0],[500,-2.4],[800,-2.5],[1000,-1],[1090,-.9],[1200,-1.7],[1500,-1],[1800,-.3],[1899,0]];
export function linearProfile(points:number[][],x:number){for(let i=1;i<points.length;i++){if(x<=points[i][0]){const [a,b]=[points[i-1],points[i]];return T.MathUtils.lerp(a[1],b[1],(x-a[0])/(b[0]-a[0]));}}return points.at(-1)![1];}
const mod=(v:number,n:number)=>(v%n+n)%n;
export type TrackPoint={point:T.Vector3;tangent:T.Vector3;outward:T.Vector3};
export type TrackModel={lap:number;width:number;scale:number;curve:T.CatmullRomCurve3;sample:(distance:number,lane?:number)=>TrackPoint;pixel:(x:number,y:number)=>T.Vector3;height:(distance:number)=>number};
export function makeTrack(race:Race,dirt=false):TrackModel{
 const c=courseFor(race),tokyo=race.venueCode==='05';
 const lap=tokyo?(dirt?1899:2083.1):c.lap;
 const pixels=tokyo?(dirt?TOKYO_DIRT_PIXELS:TOKYO_TURF_PIXELS):Array.from({length:48},(_,i)=>{const a=i/48*Math.PI*2;return [280+200*Math.cos(a),170+100*Math.sin(a)];});
 const raw=new T.CatmullRomCurve3(pixels.map(([x,z])=>new T.Vector3(x-280,0,z-170)),true,'centripetal');raw.arcLengthDivisions=4096;
 const scale=lap/raw.getLength();
 const curve=new T.CatmullRomCurve3(raw.points.map(p=>p.clone().multiplyScalar(scale)),true,'centripetal');curve.arcLengthDivisions=4096;
 const height=(distance:number)=>tokyo?linearProfile(dirt?DIRT_HEIGHT:TURF_HEIGHT,lap-mod(distance,lap)):0;
 const width=tokyo?(dirt?25:36):30;
 const sample=(distance:number,lane=0):TrackPoint=>{const u=mod(distance,lap)/lap;const point=curve.getPointAt(u);const tangent=curve.getTangentAt(u);const outward=new T.Vector3(-tangent.z,0,tangent.x).normalize();point.addScaledVector(outward,lane);point.y=height(distance);return {point,tangent,outward};};
 return {lap,width,scale,curve,sample,height,pixel:(x,y)=>new T.Vector3((x-280)*scale,0,(y-170)*scale)};
}
export type RaceRoute={sample:(travel:number,lane?:number)=>TrackPoint;start:T.Vector3;chute:T.Vector3[];phaseDistances:number[]};
export function makeRoute(race:Race,track:TrackModel):RaceRoute{
 // Standard starts are placed by official remaining distance along the inner rail.
 const startDistance=track.lap-race.distance;
 const standard=(travel:number,lane=0)=>track.sample(startDistance+travel,lane);
 let chute:T.Vector3[]=[];
 if(race.venueCode==='05'&&race.surface==='芝'&&race.distance===2000){
  chute=[[540,231],[540,192],[537,165],[526,134],[509,109],[487,90],[461,78],[429,72]].map(([x,y])=>track.pixel(x,y));
 }
 if(race.venueCode==='05'&&race.surface==='芝'&&race.distance===1800){
  chute=[[530,153],[505,131],[481,110],[452,91],[422,79]].map(([x,y])=>track.pixel(x,y));
 }
 if(race.venueCode==='05'&&race.surface==='ダート'&&race.distance===1600){
  // Turf start, then merge into the dirt back straight. Coordinates share the plan's turf scale.
  const turf=makeTrack(race,false);chute=[[470,77],[445,82],[421,87],[398,88]].map(([x,y])=>turf.pixel(x,y));
 }
 let sample=standard;
 if(chute.length){
  const last=chute.at(-1)!;let best=0,error=Infinity;
  for(let i=0;i<4096;i++){const p=track.sample(i/4096*track.lap).point;const e=Math.hypot(p.x-last.x,p.z-last.z);if(e<error){error=e;best=i/4096*track.lap;}}
  chute[chute.length-1]=track.sample(best).point;
  const branch=new T.CatmullRomCurve3(chute,false,'centripetal');branch.arcLengthDivisions=1024;
  const branchRaceDistance=race.distance-(track.lap-best);
  sample=(travel,lane=0)=>{if(travel>=branchRaceDistance)return track.sample(best+travel-branchRaceDistance,lane);const u=T.MathUtils.clamp(travel/branchRaceDistance,0,1);const point=branch.getPointAt(u),tangent=branch.getTangentAt(u),outward=new T.Vector3(-tangent.z,0,tangent.x).normalize();point.addScaledVector(outward,lane);point.y=track.height(best);return {point,tangent,outward};};
 }
 // Last-lap checkpoints: no nonexistent 1/2 corners in Tokyo mile races.
 const lastLapStart=race.distance-track.lap;
 const straight=courseFor(race).straight,approach=race.distance-straight;
 const phaseDistances=race.venueCode==='05'
  ? [0,race.distance<=1800?race.distance*.12:Math.max(150,lastLapStart+200),race.distance<=1800?race.distance*.28:Math.max(350,lastLapStart+650),race.distance-950,approach]
  : [0,approach*.18,approach*.48,approach*.82,approach];
 return {sample,start:sample(0).point,chute,phaseDistances};
}
