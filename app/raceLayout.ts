import type {Race} from './types';
export type Placement={gap:number;lane:number};
export function buildLayouts(race:Race):Placement[][]{
 const pitch=(race.surface==='ダート'?21:30)/Math.max(1,race.horses.length-1);
 const starts=race.horses.map((h,i)=>({gap:0,lane:1.8+i*pitch}));
 const stages=[starts];
 for(let phase=0;phase<4;phase++){
  const order=race.horses.map((h,i)=>({h,i,score:h.positions[phase]})).sort((a,b)=>a.score-b.score||a.h.number-b.h.number);
  const min=order[0].score, result:Placement[]=Array(order.length),placed:Placement[]=[];
  for(const [rank,{h,i,score}] of order.entries()){
   // Distances are visualization assumptions: neither time gaps nor lateral lanes are observed in JV-Data.
   const gap=(score-min)*110+rank*1.1;
   const preferred=1.8+Math.min(5,h.gate-1)*1.4+(phase>=2&&h.positions[phase]<h.positions[0]?2:0);
   const candidates=Array.from({length:9},(_,j)=>1.8+j*1.8).sort((a,b)=>Math.abs(a-preferred)-Math.abs(b-preferred));
   const lane=candidates.find(l=>placed.every(p=>Math.abs(p.gap-gap)>=3.5||Math.abs(p.lane-l)>=1.7))??candidates.at(-1)!;
   result[i]={gap,lane};placed.push(result[i]);
  }
  stages.push(result);
 }
 return stages;
}
export function sampleLayouts(layouts:Placement[][],travel:number,knots:number[]):Placement[]{
 let segment=0;while(segment<knots.length-2&&travel>knots[segment+1])segment++;
 const t=Math.max(0,Math.min(1,(travel-knots[segment])/(knots[segment+1]-knots[segment])));
 const smooth=t*t*(3-2*t);
 const result=layouts[segment].map((a,i)=>({gap:a.gap+(layouts[segment+1][i].gap-a.gap)*smooth,lane:a.lane+(layouts[segment+1][i].lane-a.lane)*smooth}));
 // Keep silhouettes separate while passing. The elliptical clearance grows smoothly
 // as two horses draw level, rather than snapping the field into fixed columns.
 for(let pass=0;pass<32;pass++){
  for(let i=0;i<result.length;i++)for(let j=i+1;j<result.length;j++){
   const a=result[i],b=result[j],dx=Math.abs(a.gap-b.gap);
   if(dx>=4)continue;
   const clearance=1.4*Math.sqrt(1-(dx/4)**2),dy=b.lane-a.lane;
   if(Math.abs(dy)<clearance){const shift=(clearance-Math.abs(dy)+.001)/2,sign=dy>=0?1:-1;a.lane-=shift*sign;b.lane+=shift*sign;}
  }
 }
 return result;
}
