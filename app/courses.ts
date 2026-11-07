import type {Race} from './types';
// Official numerical specifications, retrieved 2026-11-08. Geometry is a schematic,
// not survey-grade coordinates. Inner/outer and rail configurations are explicit.
const specs:Record<string,{slug:string;left:boolean;turf:number[];outer?:number[];dirt:number[]}>={
 '05':{slug:'tokyo',left:true,turf:[2083.1,525.9,2.7],dirt:[1899,501.6,2.5]},
 '06':{slug:'nakayama',left:false,turf:[1667.1,310,5.3],outer:[1839.7,310,5.3],dirt:[1493,308,4.5]},
 '07':{slug:'chukyo',left:true,turf:[1705.9,412.5,3.5],dirt:[1530,410.7,3.4]},
 '08':{slug:'kyoto',left:false,turf:[1782.8,328.4,3.1],outer:[1894.3,403.7,4.3],dirt:[1607.6,329.1,3]},
 '09':{slug:'hanshin',left:false,turf:[1689,356.5,1.9],outer:[2089,473.6,2.4],dirt:[1517.6,352.7,1.6]}
};
export function courseFor(r:Race){const s=specs[r.venueCode]; const outer=[12,14,17,19,20,21,22].includes(r.trackCode)&&!!s.outer; const [lap,straight,rise]=r.surface==='ダート'?s.dirt:outer?s.outer!:s.turf;
return {...s,lap,straight,rise,outer,source:`https://www.jra.go.jp/facilities/race/${s.slug}/course/index.html`,label:r.surface==='ダート'?'ダート':`${outer?'外回り':'内回り'}・Aコース基準`};}
