'use client';
import {useEffect,useRef,useState} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Race} from './types';
import {makeTrack,makeRoute,type TrackModel} from './trackGeometry';
import {buildLayouts,sampleLayouts} from './raceLayout';
import type {PlaybackPlan} from './playback';
import {plush,label} from './plushHorse';

const HORSE_SCALE=.60; // 4m model length -> ~2.4m physical silhouette; never enlarged on selection.
function mesh(geometry:T.BufferGeometry,color:string){return new T.Mesh(geometry,new T.MeshStandardMaterial({color,roughness:.96,side:T.DoubleSide}));}
function ribbon(track:TrackModel,inner:number,outer:number,color:string){
 const positions:number[]=[],indices:number[]=[];
 for(let i=0;i<=1024;i++){for(const lane of [inner,outer]){const p=track.sample(i/1024*track.lap,lane).point;positions.push(p.x,p.y,p.z);}if(i<1024){const k=i*2;indices.push(k,k+1,k+2,k+1,k+3,k+2);}}
 const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(positions,3));geo.setIndex(indices);geo.computeVertexNormals();const m=mesh(geo,color);m.receiveShadow=true;return m;
}
function polygon(track:TrackModel,points:number[][],color:string,height=.02){const shape=new T.Shape();points.forEach(([x,z],i)=>{const p=track.pixel(x,z);if(i===0)shape.moveTo(p.x,-p.z);else shape.lineTo(p.x,-p.z);});shape.closePath();const geo=new T.ShapeGeometry(shape);geo.rotateX(-Math.PI/2);const m=mesh(geo,color);m.position.y=height;return m;}
function rail(track:TrackModel,lane:number){
 const group=new T.Group();const points:T.Vector3[]=[];
 for(let i=0;i<=600;i++){const p=track.sample(i/600*track.lap,lane).point;p.y+=1.1;points.push(p);}
 group.add(mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points),1200,.10,5,false),'#f2f1de'));
 const postGeo=new T.CylinderGeometry(.075,.075,1.2,5),postMat=new T.MeshStandardMaterial({color:'#eaeedc'});
 const count=Math.ceil(track.lap/8),posts=new T.InstancedMesh(postGeo,postMat,count),dummy=new T.Object3D();
 for(let i=0;i<count;i++){dummy.position.copy(track.sample(i/count*track.lap,lane).point);dummy.position.y+=.55;dummy.updateMatrix();posts.setMatrixAt(i,dummy.matrix);}group.add(posts);return group;
}
export default function TrackScene({race,plan,progress,selected,view,onSelect}:{race:Race;plan:PlaybackPlan;progress:number;selected:number|null;view:string;onSelect:(n:number)=>void}){
 const host=useRef<HTMLDivElement>(null),state=useRef({progress,selected,view}),select=useRef(onSelect);state.current={progress,selected,view};select.current=onSelect;const [error,setError]=useState('');
 useEffect(()=>{
 const el=host.current!;let renderer:T.WebGLRenderer;try{renderer=new T.WebGLRenderer({antialias:true});setError('');}catch{setError('3D表示にはWebGL対応ブラウザーが必要です。');return;}
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;renderer.setClearColor('#e8eddf');el.appendChild(renderer.domElement);
 const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.3,6000);camera.position.set(0,800,800);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=18;controls.maxDistance=2400;controls.maxPolarAngle=Math.PI*.48;
 scene.add(new T.HemisphereLight('#fffbed','#809177',2));const sun=new T.DirectionalLight('#fff6e6',2);sun.position.set(-200,500,180);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-160;sun.shadow.camera.right=160;sun.shadow.camera.top=160;sun.shadow.camera.bottom=-160;sun.shadow.camera.far=1800;sun.shadow.bias=-.0003;scene.add(sun);scene.add(sun.target);
 const tokyo=race.venueCode==='05',turf=makeTrack(race,false),dirt=makeTrack(race,true),track=race.surface==='ダート'?dirt:turf,route=makeRoute(race,track);
 const base=mesh(new T.BoxGeometry(1050,6,620),'#cbd4bb');base.position.set(0,-6,15);base.receiveShadow=true;scene.add(base);
 scene.add(ribbon(turf,0,turf.width,'#7faf66'));scene.add(rail(turf,0));scene.add(rail(turf,turf.width));
 if(tokyo){scene.add(ribbon(dirt,0,25,'#ceb28a'));scene.add(rail(dirt,0));
 // Infield steeplechase lane visible in the official plan, for spatial context.
 scene.add(ribbon(dirt,-22,-3,'#91b4bf'));
 // Turf chutes: same pixel-to-metre transform as the main plan; deliberately not an ellipse extension.
 scene.add(polygon(turf,[[425,49],[457,51],[487,63],[513,85],[535,116],[548,147],[551,173],[549,238],[532,242],[533,164],[520,133],[500,106],[474,85],[443,72],[420,68]],'#7faf66',-.2));
 scene.add(polygon(turf,[[427,71],[451,79],[480,96],[508,121],[533,145],[533,164],[508,143],[483,121],[456,101],[432,88]],'#7faf66',-.12));
 scene.add(polygon(turf,[[367,69],[414,68],[455,73],[490,84],[479,94],[440,84],[401,82],[366,83]],'#7faf66',-.10));
 // Main and memorial stands, traced as subdued context rather than exaggerated landmarks.
 scene.add(polygon(turf,[[229,317],[460,298],[485,327],[462,352],[410,356],[268,356],[230,347]],'#bdc4be',5));
 scene.add(polygon(turf,[[160,322],[225,316],[230,354],[171,361]],'#bfc7c0',4));
 }
 const majorLabels:T.Sprite[]=[];
 const mark=(text:string,at:T.Vector3,w=28)=>{const s=label(text,256);s.position.copy(at);s.position.y+=5;s.scale.set(w,w/4,1);scene.add(s);majorLabels.push(s);return s;};
 mark(`${race.venue}競馬場`,new T.Vector3(-15,0,-10),105);
 const finish=track.sample(0);const finishLine=mesh(new T.PlaneGeometry(.7,track.width),'#fff9df');finishLine.rotation.x=-Math.PI/2;finishLine.rotation.z=Math.atan2(finish.tangent.z,finish.tangent.x);finishLine.position.copy(track.sample(0,track.width/2).point);finishLine.position.y+=.08;scene.add(finishLine);mark('GOAL',track.sample(0,track.width+12).point,24);
 // Official distance markers along the last lap.
 for(let remaining=200;remaining<track.lap;remaining+=200){const p=track.sample(track.lap-remaining,track.width+7).point;mark(`${remaining}m`,p,18);}
 mark(`START ${race.distance}m`,route.sample(0,track.width+9).point,40);
 if(tokyo){for(const [text,x,z] of [['1角',471,239],['2角',465,107],['3角',70,91],['4角',63,255]] as [string,number,number][]){const p=turf.pixel(x,z);p.y=4;const s=mark(text,p,18);s.position.z+=(text==='1角'||text==='4角'?35:-35);}}
 // Gate stalls use the same 1.65m spacing as each horse's starting lane.
 const gateGroup=new T.Group();for(const [i] of race.horses.entries()){const p=route.sample(0,buildLayouts(race)[0][i].lane);const gate=mesh(new T.BoxGeometry(.18,2.9,1.3),'#386d62');gate.position.copy(p.point);gate.position.y+=1.4;gate.rotation.y=-Math.atan2(p.tangent.z,p.tangent.x);gateGroup.add(gate);}scene.add(gateGroup);
 const layouts=plan.layouts;
 // Start -> early -> middle -> final corner -> final straight. Distances are route-specific.
 const knots=plan.knots;
 const horses=race.horses.map(h=>{const g=plush(h);g.scale.setScalar(HORSE_SCALE);g.userData.number=h.number;const name=g.getObjectByName('nameplate')!;name.visible=false;const number=g.getObjectByName('numberplate') as T.Sprite;number.position.set(0,4,0);number.scale.set(3.6,3.6,1);scene.add(g);return g;});
 const ring=mesh(new T.RingGeometry(1.5,1.7,40),'#f4cf65');ring.rotation.x=-Math.PI/2;scene.add(ring);ring.visible=false;
 const ray=new T.Raycaster(),mouse=new T.Vector2();let down=[0,0];const pd=(e:PointerEvent)=>{down=[e.clientX,e.clientY];};const click=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;const b=el.getBoundingClientRect();mouse.set((e.clientX-b.left)/b.width*2-1,-(e.clientY-b.top)/b.height*2+1);ray.setFromCamera(mouse,camera);const hit=ray.intersectObjects(horses,true)[0];if(hit){let o:T.Object3D=hit.object;while(o.parent&&!o.userData.number)o=o.parent;if(o.userData.number)select.current(o.userData.number);}};
 renderer.domElement.addEventListener('pointerdown',pd);renderer.domElement.addEventListener('pointerup',click);
 const resize=()=>{camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();renderer.setSize(el.clientWidth,el.clientHeight);};const observer=new ResizeObserver(resize);observer.observe(el);resize();
 let frame=0,lastView='',following=true;const onControlStart=()=>{following=false;};controls.addEventListener('start',onControlStart);
 const bbox=new T.Box3().setFromPoints(turf.curve.getPoints(256));const size=bbox.getSize(new T.Vector3()),center=bbox.getCenter(new T.Vector3());
 const overviewDistance=()=>Math.max((size.x+120)/camera.aspect,size.z+170)/(2*Math.tan(T.MathUtils.degToRad(20)))*1.12;
 let lastTime=performance.now(),lastProgress=state.current.progress;
 function draw(now=performance.now()){
  const dt=Math.min((now-lastTime)/1000,.1);lastTime=now;const {progress:p,selected,view}=state.current;const travel=p*race.distance;
  const placements=sampleLayouts(layouts,travel,knots);ring.visible=selected!==null;
  horses.forEach((g,i)=>{const {gap,lane}=placements[i];const at=route.sample(Math.max(0,travel-gap),lane);g.position.copy(at.point);g.rotation.y=-Math.atan2(at.tangent.z,at.tangent.x);g.getObjectByName('nameplate')!.visible=selected===race.horses[i].number;if(selected===race.horses[i].number){ring.position.copy(g.position);ring.position.y+=.10;}});
  const horseBox=new T.Box3().setFromPoints(horses.map(g=>g.position)),packCenter=horseBox.getCenter(new T.Vector3()),packSize=horseBox.getSize(new T.Vector3());
  const fieldSpan=Math.max(...placements.map(v=>v.gap))+8;
  const closeDistance=Math.max(40,fieldSpan/(Math.max(camera.aspect,.7)*.73));
  const heading=route.sample(Math.max(0,travel-fieldSpan/2));
  const packOffset=heading.outward.clone().multiplyScalar(closeDistance).addScaledVector(heading.tangent,closeDistance*.12);packOffset.y=closeDistance*.8;
  if(view!==lastView){following=true;lastView=view;if(view==='top'){controls.target.copy(center);camera.position.copy(center).add(new T.Vector3(0,overviewDistance(),.01));}else if(view==='orbit'){controls.target.copy(center);camera.position.copy(center).add(new T.Vector3(0,overviewDistance()*.78,overviewDistance()*.80));}else {controls.target.copy(packCenter);camera.position.copy(packCenter).add(packOffset);}}
  if(view==='pack'&&following){const alpha=Math.abs(p-lastProgress)>.03?1:1-Math.exp(-dt*6);controls.target.lerp(packCenter,alpha);camera.position.lerp(packCenter.clone().add(packOffset),alpha);}lastProgress=p;
  majorLabels.forEach(s=>{s.visible=view!=='pack';const pixels=s===majorLabels[0]?42:28;const h=camera.position.distanceTo(s.position)*2*Math.tan(T.MathUtils.degToRad(20))*pixels/el.clientHeight;s.scale.set(h*2,h,1);});
  const overview=view!=='pack';horses.forEach((g,i)=>{(g.getObjectByName('numberplate') as T.Sprite).visible=!overview||selected===race.horses[i].number;});
  sun.target.position.copy(packCenter);sun.position.copy(packCenter).add(new T.Vector3(-200,500,180));
  controls.update();renderer.render(scene,camera);frame=requestAnimationFrame(draw);
 }
 draw();return()=>{cancelAnimationFrame(frame);observer.disconnect();controls.removeEventListener('start',onControlStart);controls.dispose();renderer.domElement.removeEventListener('pointerdown',pd);renderer.domElement.removeEventListener('pointerup',click);scene.traverse(o=>{if(o instanceof T.Mesh)o.geometry.dispose();if(o instanceof T.Mesh||o instanceof T.Sprite){for(const m of Array.isArray(o.material)?o.material:[o.material]){if('map' in m&&m.map)m.map.dispose();m.dispose();}}});renderer.dispose();renderer.domElement.remove();};
 },[race,plan]);
 return <div className="scene" ref={host} aria-label={`${race.name}の3D展開予想。ドラッグで回転、スクロールで拡大`}>{error&&<p className="scene-error">{error}</p>}</div>;
}
