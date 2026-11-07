'use client';
import {useEffect,useRef,useState} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Race} from './types';
import {makeTrack,makeRoute,makeTerrain,availableKinds,type TrackModel} from './trackGeometry';
import {sampleLayouts,startLanes} from './raceLayout';
import type {PlaybackPlan} from './playback';
import {plush,label} from './plushHorse';

// The plush model is 4 units long: scale 1.0 draws a ~4m horse (≈1.7x life size)
// so the field stays legible against the true-scale course. Never enlarged on selection.
const HORSE_SCALE=1.0;
function material(color:string,lift=0){return new T.MeshStandardMaterial({color,roughness:.96,side:T.DoubleSide,polygonOffset:lift!==0,polygonOffsetFactor:-lift,polygonOffsetUnits:-lift});}
function mesh(geometry:T.BufferGeometry,color:string,lift=0){return new T.Mesh(geometry,material(color,lift));}
function ribbon(track:TrackModel,inner:number,outer:number,color:string,lift=1){
 const positions:number[]=[],indices:number[]=[];
 for(let i=0;i<=1024;i++){for(const lane of [inner,outer]){const p=track.sample(i/1024*track.lap,lane).point;positions.push(p.x,p.y+.02,p.z);}if(i<1024){const k=i*2;indices.push(k,k+1,k+2,k+1,k+3,k+2);}}
 const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(positions,3));geo.setIndex(indices);geo.computeVertexNormals();const m=mesh(geo,color,lift);m.receiveShadow=true;return m;
}
function rail(track:TrackModel,lane:number){
 const group=new T.Group();const points:T.Vector3[]=[];
 for(let i=0;i<=600;i++){const p=track.sample(i/600*track.lap,lane).point;p.y+=1.1;points.push(p);}
 group.add(mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points,true),1200,.10,5,true),'#f2f1de'));
 const postGeo=new T.CylinderGeometry(.075,.075,1.2,5),postMat=new T.MeshStandardMaterial({color:'#eaeedc'});
 const count=Math.ceil(track.lap/8),posts=new T.InstancedMesh(postGeo,postMat,count),dummy=new T.Object3D();
 for(let i=0;i<count;i++){dummy.position.copy(track.sample(i/count*track.lap,lane).point);dummy.position.y+=.55;dummy.updateMatrix();posts.setMatrixAt(i,dummy.matrix);}group.add(posts);return group;
}
export default function TrackScene({race,plan,progress,selected,followKey,onSelect}:{race:Race;plan:PlaybackPlan;progress:number;selected:number|null;followKey:number;onSelect:(n:number)=>void}){
 const host=useRef<HTMLDivElement>(null),state=useRef({progress,selected,followKey}),select=useRef(onSelect);state.current={progress,selected,followKey};select.current=onSelect;const [error,setError]=useState('');
 useEffect(()=>{
 const el=host.current!;let renderer:T.WebGLRenderer;try{renderer=new T.WebGLRenderer({antialias:true});setError('');}catch{setError('3D表示にはWebGL対応ブラウザーが必要です。');return;}
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;renderer.setClearColor('#e8eddf');el.appendChild(renderer.domElement);
 const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.3,6000);camera.position.set(0,800,800);
 scene.fog=new T.Fog('#e8eddf',500,1600);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=12;controls.maxDistance=900;controls.maxPolarAngle=Math.PI*.48;
 scene.add(new T.HemisphereLight('#fffbed','#809177',2));const sun=new T.DirectionalLight('#fff6e6',2);sun.position.set(-200,500,180);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-110;sun.shadow.camera.right=110;sun.shadow.camera.top=110;sun.shadow.camera.bottom=-110;sun.shadow.camera.far=1800;sun.shadow.bias=-.0003;scene.add(sun);scene.add(sun.target);
 const tokyo=race.venueCode==='05',track=makeTrack(race),route=makeRoute(race,track);
 const tracks=availableKinds(race).map(k=>k===track.kind?track:makeTrack(race,k));
 const turf=tracks.find(t=>t.kind==='turf')!,dirt=tracks.find(t=>t.kind==='dirt'),outer=tracks.find(t=>t.kind==='outer');

 // Ground follows the (exaggerated) relief of every track and stays below each surface.
 const terrain=makeTerrain(tracks);
 const box=new T.Box3().setFromPoints(terrain.samples.map(s=>new T.Vector3(s.x,0,s.z))).expandByScalar(160);
 const size=box.getSize(new T.Vector3()),cell=5,nx=Math.ceil(size.x/cell),nz=Math.ceil(size.z/cell);
 const ground=new T.PlaneGeometry(size.x,size.z,nx,nz);ground.rotateX(-Math.PI/2);ground.translate(box.min.x+size.x/2,0,box.min.z+size.z/2);
 const gp=ground.attributes.position as T.BufferAttribute;for(let i=0;i<gp.count;i++)gp.setY(i,terrain.heightAt(gp.getX(i),gp.getZ(i)));ground.computeVertexNormals();
 const groundMesh=mesh(ground,'#cbd4bb');groundMesh.receiveShadow=true;scene.add(groundMesh);
 const drape=(points:number[][],color:string,lift:number)=>{const shape=new T.Shape();points.forEach(([x,z],i)=>{const p=turf.pixel(x,z);if(i===0)shape.moveTo(p.x,-p.z);else shape.lineTo(p.x,-p.z);});shape.closePath();const geo=new T.ShapeGeometry(shape);geo.rotateX(-Math.PI/2);const pos=geo.attributes.position as T.BufferAttribute;for(let i=0;i<pos.count;i++)pos.setY(i,terrain.heightAt(pos.getX(i),pos.getZ(i))+lift);geo.computeVertexNormals();const m=mesh(geo,color,1);m.receiveShadow=true;return m;};

 // Outer course first (lowest priority), then the inner turf, then dirt on top.
 if(outer)scene.add(ribbon(outer,0,outer.width,'#7faf66',1));
 scene.add(ribbon(turf,0,turf.width,'#7faf66',2));
 if(dirt)scene.add(ribbon(dirt,0,dirt.width,'#ceb28a',3));
 if(tokyo&&dirt){
  // Turf chutes from the main plan, draped onto the ground instead of floating flat.
  scene.add(drape([[425,49],[457,51],[487,63],[513,85],[535,116],[548,147],[551,173],[549,238],[532,242],[533,164],[520,133],[500,106],[474,85],[443,72],[420,68]],'#7faf66',.3));
  scene.add(drape([[427,71],[451,79],[480,96],[508,121],[533,145],[533,164],[508,143],[483,121],[456,101],[432,88]],'#7faf66',.3));
  scene.add(drape([[367,69],[414,68],[455,73],[490,84],[479,94],[440,84],[401,82],[366,83]],'#7faf66',.3));
  scene.add(drape([[229,317],[460,298],[485,327],[462,352],[410,356],[268,356],[230,347]],'#bdc4be',5));
  scene.add(drape([[160,322],[225,316],[230,354],[171,361]],'#bfc7c0',4));
 }
 // Rails of the course this race is run on.
 scene.add(rail(track,0));scene.add(rail(track,track.width));
 if(track!==turf)scene.add(rail(turf,turf.width));

 const mark=(text:string,at:T.Vector3,w:number)=>{const s=label(text,256);s.position.copy(at);s.position.y+=w/4+1.5;s.scale.set(w,w/4,1);scene.add(s);return s;};
 const finish=track.sample(0);
 const finishLine=mesh(new T.PlaneGeometry(.7,track.width),'#fff9df',4);finishLine.rotation.x=-Math.PI/2;finishLine.rotation.z=Math.atan2(finish.tangent.z,finish.tangent.x);finishLine.position.copy(track.sample(0,track.width/2).point);finishLine.position.y+=.1;scene.add(finishLine);
 mark('GOAL',track.sample(0,track.width+8).point,16);
 // Remaining-distance boards along the last lap, sized to be readable from the pack camera.
 for(let remaining=200;remaining<track.lap;remaining+=200){mark(`${remaining}m`,track.sample(track.lap-remaining,track.width+6).point,11);}
 mark(`START ${race.distance}m`,route.sample(0,track.width+9).point,24);
 // Gate stalls between the starting lanes.
 const lanes=startLanes(race),gateGroup=new T.Group();
 for(let i=0;i<=lanes.length;i++){const lane=i===0?lanes[0]-.8:i===lanes.length?lanes.at(-1)!+.8:(lanes[i-1]+lanes[i])/2;const p=route.sample(0,lane);const gate=mesh(new T.BoxGeometry(2.6,2.9,.14),'#386d62');gate.position.copy(p.point);gate.position.y+=1.45;gate.rotation.y=-Math.atan2(p.tangent.z,p.tangent.x);gateGroup.add(gate);}
 scene.add(gateGroup);

 const layouts=plan.layouts,knots=plan.knots;
 const horses=race.horses.map(h=>{const g=plush(h);g.scale.setScalar(HORSE_SCALE);g.rotation.order='YZX';g.userData.number=h.number;const name=g.getObjectByName('nameplate')!;name.visible=false;const number=g.getObjectByName('numberplate') as T.Sprite;number.position.set(0,4.4,0);number.scale.set(2.2,2.2,1);scene.add(g);return g;});
 const ring=mesh(new T.RingGeometry(2.4,2.75,40),'#f4cf65',5);ring.rotation.x=-Math.PI/2;scene.add(ring);ring.visible=false;
 const ray=new T.Raycaster(),mouse=new T.Vector2();let down=[0,0];
 const pd=(e:PointerEvent)=>{down=[e.clientX,e.clientY];};
 const click=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;const b=el.getBoundingClientRect();mouse.set((e.clientX-b.left)/b.width*2-1,-(e.clientY-b.top)/b.height*2+1);ray.setFromCamera(mouse,camera);const hit=ray.intersectObjects(horses,true)[0];if(hit){let o:T.Object3D=hit.object;while(o.parent&&!o.userData.number)o=o.parent;if(o.userData.number)select.current(o.userData.number);}};
 renderer.domElement.addEventListener('pointerdown',pd);renderer.domElement.addEventListener('pointerup',click);
 const resize=()=>{camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();renderer.setSize(el.clientWidth,el.clientHeight);};const observer=new ResizeObserver(resize);observer.observe(el);resize();
 // The camera always frames the pack. Dragging takes manual control until the
 // "back to the pack" button bumps followKey.
 // Wheel zoom keeps following (it scales the pack distance); only a drag hands over control.
 let frame=0,following=true,lastFollowKey=-1,first=true,zoom=1,dragging=false;
 const onControlStart=()=>{if(dragging)following=false;};controls.addEventListener('start',onControlStart);
 const onDown=()=>{dragging=true;},onUp=()=>{dragging=false;};renderer.domElement.addEventListener('pointerdown',onDown);window.addEventListener('pointerup',onUp);
 const onWheel=(e:WheelEvent)=>{if(!following)return;e.preventDefault();zoom=T.MathUtils.clamp(zoom*Math.exp(e.deltaY*.001),.35,3);};renderer.domElement.addEventListener('wheel',onWheel,{passive:false});
 let lastTime=performance.now(),lastProgress=state.current.progress;
 function draw(now=performance.now()){
  const dt=Math.min((now-lastTime)/1000,.1);lastTime=now;const {progress:p,selected,followKey}=state.current;const travel=p*race.distance;
  if(followKey!==lastFollowKey){following=true;lastFollowKey=followKey;}
  const placements=sampleLayouts(layouts,travel,knots);ring.visible=selected!==null;
  horses.forEach((g,i)=>{const {gap,lane}=placements[i],at=Math.max(0,travel-gap),here=route.sample(at,lane);
   // Pitch with the slope so hooves stay on the exaggerated relief.
   const ahead=route.sample(at+2,lane).point,behind=route.sample(Math.max(0,at-2),lane).point;
   const pitch=Math.atan2(ahead.y-behind.y,Math.hypot(ahead.x-behind.x,ahead.z-behind.z));
   g.position.copy(here.point);g.rotation.set(0,-Math.atan2(here.tangent.z,here.tangent.x),pitch);
   g.getObjectByName('nameplate')!.visible=selected===race.horses[i].number;
   if(selected===race.horses[i].number){ring.position.copy(g.position);ring.position.y+=.12;}});
  const horseBox=new T.Box3().setFromPoints(horses.map(g=>g.position)),packCenter=horseBox.getCenter(new T.Vector3());
  const fieldSpan=Math.max(...placements.map(v=>v.gap))+8;
  controls.enableZoom=!following;
  const closeDistance=Math.max(30,fieldSpan/(Math.max(camera.aspect,.7)*.8))*zoom;
  const heading=route.sample(Math.max(0,travel-fieldSpan/2));
  const packOffset=heading.outward.clone().multiplyScalar(closeDistance).addScaledVector(heading.tangent,closeDistance*.12);packOffset.y=closeDistance*.62;
  if(following){const alpha=first||Math.abs(p-lastProgress)>.03?1:1-Math.exp(-dt*6);first=false;controls.target.lerp(packCenter,alpha);camera.position.lerp(packCenter.clone().add(packOffset),alpha);}
  lastProgress=p;
  sun.target.position.copy(packCenter);sun.position.copy(packCenter).add(new T.Vector3(-200,500,180));
  controls.update();renderer.render(scene,camera);frame=requestAnimationFrame(draw);
 }
 draw();
 return()=>{cancelAnimationFrame(frame);observer.disconnect();controls.removeEventListener('start',onControlStart);renderer.domElement.removeEventListener('pointerdown',onDown);window.removeEventListener('pointerup',onUp);renderer.domElement.removeEventListener('wheel',onWheel);controls.dispose();renderer.domElement.removeEventListener('pointerdown',pd);renderer.domElement.removeEventListener('pointerup',click);scene.traverse(o=>{if(o instanceof T.Mesh)o.geometry.dispose();if(o instanceof T.Mesh||o instanceof T.Sprite){for(const m of Array.isArray(o.material)?o.material:[o.material]){if('map' in m&&m.map)m.map.dispose();m.dispose();}}});renderer.dispose();renderer.domElement.remove();};
 },[race,plan]);
 return <div className="scene" ref={host} aria-label={`${race.name}の3D展開。馬群を自動で追尾。ドラッグで回転、スクロールで拡大`}>{error&&<p className="scene-error">{error}</p>}</div>;
}
