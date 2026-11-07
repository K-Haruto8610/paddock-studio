import type {Race,ReplayRace,PlaybackMode} from './types';
import {checkpointDistances,cornerLabels,courseFor} from './courses';
import {buildLayouts,sampleLayouts,laneSlots,HORSE_LENGTH,LANE_PITCH,type Placement} from './raceLayout';

export type Checkpoint={label:string;distance:number;observed:boolean;ranks:(number|null)[];corner?:number};
export type PlaybackPlan={layouts:Placement[][];knots:number[];checkpoints:Checkpoint[];mode:PlaybackMode;available:boolean};

function observedLayout(race:Race,ranks:number[],previous?:Placement[]):Placement[]{
 const minimum=Math.min(...ranks),out:Placement[]=Array(ranks.length),placed:Placement[]=[];
 for(const {rank,i} of ranks.map((rank,i)=>({rank,i})).sort((a,b)=>a.rank-b.rank||a.i-b.i)){
  const gap=(rank-minimum)*HORSE_LENGTH*1.3;
  // Lanes/gaps are illustrative; only ranks come from the recorded results.
  const advancing=previous&&gap<previous[i].gap;
  const preferred=1.8+Math.min(4,race.horses[i].gate-1)*LANE_PITCH*.8+(advancing?LANE_PITCH*1.5:0);
  const lanes=laneSlots(10).sort((a,b)=>Math.abs(a-preferred)-Math.abs(b-preferred));
  const lane=lanes.find(l=>placed.every(p=>Math.abs(p.gap-gap)>=HORSE_LENGTH*1.3||Math.abs(p.lane-l)>=LANE_PITCH*.9))??lanes.at(-1)!;
  out[i]={gap,lane};placed.push(out[i]);
 }
 return out;
}

export function buildPlayback(race:Race,mode:PlaybackMode,replay?:ReplayRace):PlaybackPlan{
 const forecast=buildLayouts(race),distances=checkpointDistances(race);
 const start:Checkpoint={label:'スタート',distance:0,observed:false,ranks:race.horses.map(()=>null)};
 if(mode==='replay'){
  if(!replay)return {mode,available:false,layouts:[forecast[0],forecast[0]],knots:[0,race.distance],checkpoints:[start,{...start,label:'実績データなし',distance:race.distance}]};
  const checkpoints=[start],layouts=[forecast[0]];
  for(const corner of replay.corners){
   const ranks=race.horses.map(h=>replay.horses[String(h.number)]?.corners[corner-1]??null);
   if(ranks.some(r=>r===null))continue;
   layouts.push(observedLayout(race,ranks as number[],layouts.at(-1)));
   checkpoints.push({label:`${corner}角`,distance:distances[corner],observed:true,ranks,corner});
  }
  if(replay.completeFinish){
   const ranks=race.horses.map(h=>replay.horses[String(h.number)]?.finish??null);
   if(ranks.every(r=>r!==null)){layouts.push(observedLayout(race,ranks as number[],layouts.at(-1)));checkpoints.push({label:'ゴール',distance:race.distance,observed:true,ranks});}
  }
  // A missing terminal observation is explicitly held, never replaced by prediction results.
  if(checkpoints.at(-1)!.distance<race.distance){layouts.push(layouts.at(-1)!);checkpoints.push({label:'入線データなし',distance:race.distance,observed:false,ranks:race.horses.map(()=>null)});}
  return {mode,available:checkpoints.some(c=>c.observed),layouts,knots:checkpoints.map(c=>c.distance),checkpoints};
 }
 const labels=cornerLabels(race),checkpoints=[start,...[1,2,3,4].map(c=>({label:labels[c-1],distance:distances[c],observed:false,ranks:race.horses.map(h=>h.positions[c-1]*(race.horses.length-1)+1)}))];
 // Existing last-3F prediction now affects the straight instead of freezing the field.
 // This is a relative-motion estimate, not an observed or separately trained finishing rank.
 const ratio=Math.min(1,courseFor(race).straight/600),best=Math.min(...race.horses.map(h=>h.last3));
 const finalGaps=forecast[4].map((p,i)=>p.gap+(race.horses[i].last3-best)*16*ratio);
 const first=Math.min(...finalGaps),finish=forecast[4].map((p,i)=>({gap:finalGaps[i]-first,lane:p.lane}));
 const order=finish.map((v,i)=>({gap:v.gap,i})).sort((a,b)=>a.gap-b.gap),ranks=Array(race.horses.length);order.forEach((v,i)=>ranks[v.i]=i+1);
 checkpoints.push({label:'ゴール推定',distance:race.distance,observed:false,ranks});
 return {mode,available:true,layouts:[...forecast,finish],knots:[...distances,race.distance],checkpoints};
}

export function playbackFrame(race:Race,plan:PlaybackPlan,progress:number){
 const travel=Math.max(0,Math.min(1,progress))*race.distance;
 const placements=sampleLayouts(plan.layouts,travel,plan.knots);
 const exact=plan.checkpoints.find(c=>Math.abs(c.distance-travel)<.05);
 let previous=0;for(let i=0;i<plan.checkpoints.length;i++)if(plan.checkpoints[i].distance<=travel)previous=i;
 const next=Math.min(previous+1,plan.checkpoints.length-1);
 const label=exact?exact.label:`${plan.checkpoints[previous].label} → ${plan.checkpoints[next].label}`;
 const order=placements.map((p,i)=>({horse:race.horses[i],gap:p.gap,index:i})).sort((a,b)=>a.gap-b.gap||a.horse.number-b.horse.number);
 return {travel,placements,exact,label,order,previous:plan.checkpoints[previous]};
}
