import type {Race} from './types';
import {COURSE_DATA,type Pixel} from './courseData';
// Official numerical specifications, retrieved 2026-11-08. Geometry is a schematic,
// not survey-grade coordinates. Inner/outer and rail configurations are explicit.
const specs:Record<string,{slug:string;left:boolean;turf:number[];outer?:number[];dirt:number[]}>={
 '05':{slug:'tokyo',left:true,turf:[2083.1,525.9,2.7],dirt:[1899,501.6,2.5]},
 '06':{slug:'nakayama',left:false,turf:[1667.1,310,5.3],outer:[1839.7,310,5.3],dirt:[1493,308,4.5]},
 '07':{slug:'chukyo',left:true,turf:[1705.9,412.5,3.5],dirt:[1530,410.7,3.4]},
 '08':{slug:'kyoto',left:false,turf:[1782.8,328.4,3.1],outer:[1894.3,403.7,4.3],dirt:[1607.6,329.1,3]},
 '09':{slug:'hanshin',left:false,turf:[1689,356.5,1.9],outer:[2089,473.6,2.4],dirt:[1517.6,352.7,1.6]}
};
// JV-Data track codes whose final lap is on the outer turf course:
// 12 芝左外, 13 芝左内→外, 16 芝左外2周, 18 芝右外, 19 芝右内→外, 22 芝右外2周.
// (17 is 芝右 = inner; 14/20 are 外→内 and finish on the inner course.)
const OUTER_TRACK_CODES=[12,13,16,18,19,22];
export type CourseKind='turf'|'outer'|'dirt';
// Courses drawn per venue: only those JRA flat GⅠ races use (see digitize_jra_courses.py).
export const G1_COURSES:Record<string,CourseKind[]>={'05':['turf','dirt'],'06':['turf','outer'],'07':['turf','dirt'],'08':['turf','outer'],'09':['turf','outer']};
export function courseFor(r:Race){const s=specs[r.venueCode]; const outer=OUTER_TRACK_CODES.includes(r.trackCode)&&!!s.outer; const kind:CourseKind=r.surface==='ダート'?'dirt':outer?'outer':'turf'; const [lap,straight,rise]=kind==='dirt'?s.dirt:kind==='outer'?s.outer!:s.turf;
return {...s,kind,lap,straight,rise,outer,profileImage:`/courses/${s.slug}-${kind}.png`,source:`https://www.jra.go.jp/facilities/race/${s.slug}/course/index.html`,label:r.surface==='ダート'?'ダート':`${s.outer?(outer?'外回り・':'内回り・'):''}Aコース基準`};}
export function lapFor(venueCode:string,kind:CourseKind){const s=specs[venueCode];return (kind==='dirt'?s.dirt:kind==='outer'&&s.outer?s.outer:s.turf)[0];}

// Corner positions (metres from the winning post along the race direction) from
// the digitized rail. A lap turns 360°; 1角〜4角 are taken where the accumulated
// turn since the winning post reaches 45°, 135°, 225° and 315°. Robust to the
// small kinks of a digitized schematic, unlike a local-curvature threshold.
function cornersFromRail(pixels:Pixel[],lap:number){
 const n=pixels.length,seg:number[]=[],turned=[0];
 for(let i=0;i<n;i++){const a=pixels[i],b=pixels[(i+1)%n];seg.push(Math.hypot(b[0]-a[0],b[1]-a[1]));}
 const heading=(i:number)=>{const a=pixels[(i+n)%n],b=pixels[(i+1+n)%n];return Math.atan2(b[1]-a[1],b[0]-a[0]);};
 for(let i=1;i<=n;i++){let d=heading(i)-heading(i-1);d=Math.atan2(Math.sin(d),Math.cos(d));turned.push(turned[i-1]+d);}
 const total=seg.reduce((a,b)=>a+b,0),cum=[0];for(let i=0;i<n;i++)cum.push(cum[i]+seg[i]);
 const full=turned[n];
 return [1,3,5,7].map(k=>{const target=full*k/8;let i=1;while(i<n&&Math.abs(turned[i])<Math.abs(target))i++;
  const t=(target-turned[i-1])/((turned[i]-turned[i-1])||1);return (cum[i-1]+seg[i-1]*Math.max(0,Math.min(1,t)))/total*lap;});
}

export function cornerPositions(race:Race){
 const c=courseFor(race),rails=COURSE_DATA[race.venueCode]?.rails;
 if(!rails)return null;
 // Courses without a drawn rail (non-GⅠ) fall back to the inner turf rail.
 return cornersFromRail(rails[c.kind]??rails.turf,c.lap);
}

export function checkpointDistances(race:Race){
 const c=courseFor(race),lastLapStart=race.distance-c.lap,approach=race.distance-c.straight;
 if(race.venueCode==='05')
  return [0,race.distance<=1800?race.distance*.12:Math.max(150,lastLapStart+200),race.distance<=1800?race.distance*.28:Math.max(350,lastLapStart+650),race.distance-950,approach];
 const corners=cornerPositions(race);
 if(!corners)return [0,approach*.18,approach*.48,approach*.82,approach];
 const travel=corners.map(s=>lastLapStart+s);
 // Corners the race never reaches become early/middle phases before 3角.
 if(travel[1]<=60){travel[0]=travel[2]*.3;travel[1]=travel[2]*.65;}
 else if(travel[0]<=30)travel[0]=travel[1]*.45;
 return [0,...travel];
}
export function cornerLabels(race:Race){
 const passes=race.venueCode==='05'?race.distance>1800:(()=>{const c=cornerPositions(race);return !c||race.distance-courseFor(race).lap+c[1]>60;})();
 return passes?['1角','2角','3角','4角']:['序盤','道中','3角','4角'];
}
