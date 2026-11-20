'use client';

import Link from 'next/link';
import {useEffect, useRef, useState} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {useHorseModelReady} from '../blenderHorse';
import {plush, posePlush} from '../plushHorse';
import {disposeScene, HORSE_SCALE} from '../sceneBuilders';
import {dirtTexture, turfTexture, SURFACE_TILE} from '../scenery';
import {bladeMaterial, bladeMesh, dirtReliefMap, flatBladeField, grassUniforms, shellGrass, DEFAULT_BLADE_HEIGHT, RELIEF_TILE, type GrassUniforms} from '../surfaceDetail';
import {addDirtTrail, type DirtTrail} from '../dirtTrail';
import type {Ground} from '../courseGround';
import {NO_KICK} from '../closingKick';
import type {Horse} from '../types';
import './surface.css';

/** The patch of course on show, in metres: x runs along the course. */
const LENGTH = 60;
const WIDTH = 30;
const MAX_BLADES = 800_000;
const SHELL_LAYERS = 24;
/** With blades standing above it, the shell layer only fills in the base. */
const SHELL_UNDER_BLADES = 0.55;
const RUN_SPEED = 16;

const turfLooks = {texture: 'テクスチャのみ', shell: '層を重ねる', blades: '葉を1本ずつ', both: '層＋葉'} as const;
const dirtLooks = {texture: 'テクスチャのみ', relief: '凹凸をつける'} as const;
type TurfLook = keyof typeof turfLooks;
type DirtLook = keyof typeof dirtLooks;

const views: Record<string, {position: [number, number, number]; target: [number, number, number]}> = {
  // Framed on the horse, so it comes closer with the horse's drawn size; the ground views stay put.
  '中継カメラ': {position: [9 * HORSE_SCALE, 3.4 * HORSE_SCALE, 13 * HORSE_SCALE], target: [0, 1.5 * HORSE_SCALE, 0]},
  '俯瞰': {position: [0, 30, 26], target: [0, 0, 0]},
  '地面すれすれ': {position: [4.2, 0.45, 5.2], target: [0, 0.9, 0]},
  '足もと': {position: [1.6, 0.9, 2.6], target: [0, 0.25, 0]},
};

type Settings = {
  surface: 'turf' | 'dirt'; turf: TurfLook; dirt: DirtLook;
  blades: number; height: number; wind: number; stripes: boolean; relief: number; sun: number; moving: boolean; trail: boolean;
};
const START: Settings = {
  surface: 'turf', turf: 'both', dirt: 'relief',
  blades: 400_000, height: DEFAULT_BLADE_HEIGHT, wind: 0.5, stripes: true, relief: 1.2, sun: 38, moving: false, trail: true,
};

type Rig = {
  uniforms: GrassUniforms;
  turfGround: T.Mesh<T.PlaneGeometry, T.MeshStandardMaterial>;
  dirtGround: T.Mesh<T.PlaneGeometry, T.MeshStandardMaterial>;
  shell: T.InstancedMesh;
  blades: T.Mesh<T.InstancedBufferGeometry>;
  relief: T.Texture;
  trail: DirtTrail;
  camera: T.PerspectiveCamera;
  controls: OrbitControls;
};

/** The flat patch as the hooves see it. */
const patch: Ground = (x, z, out) => {
  out.y = 0;
  out.normal.set(0, 1, 0);
  out.dirt = true;
  return Math.abs(x) <= LENGTH / 2 && Math.abs(z) <= WIDTH / 2;
};

function ground(map: T.Texture) {
  map.repeat.set(LENGTH / SURFACE_TILE, WIDTH / SURFACE_TILE);
  const mesh = new T.Mesh(new T.PlaneGeometry(LENGTH, WIDTH), new T.MeshStandardMaterial({map, roughness: 0.96}));
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  return mesh;
}

export default function SurfacePreview() {
  const host = useRef<HTMLDivElement>(null);
  const status = useRef<HTMLParagraphElement>(null);
  const rig = useRef<Rig | null>(null);
  const live = useRef(START);
  const horseReady = useHorseModelReady();
  const [s, setS] = useState(START);
  const [view, setView] = useState('中継カメラ');
  const set = (patch: Partial<Settings>) => setS((old) => ({...old, ...patch}));

  useEffect(() => {
    if (!horseReady) return;
    const el = host.current!;
    const renderer = new T.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    el.appendChild(renderer.domElement);
    const scene = new T.Scene();
    scene.background = new T.Color('#dfe9ee');
    const camera = new T.PerspectiveCamera(40, 1, 0.05, 400);
    camera.position.set(...views['中継カメラ'].position);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...views['中継カメラ'].target);
    controls.minDistance = 0.6;
    controls.maxDistance = 90;
    controls.maxPolarAngle = Math.PI * 0.495;

    scene.add(new T.HemisphereLight('#f4f8ff', '#8f8a70', 1.9));
    const sun = new T.DirectionalLight('#fff3dc', 3.2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = sun.shadow.camera.bottom = -7;
    sun.shadow.camera.right = sun.shadow.camera.top = 7;
    sun.shadow.camera.far = 120;
    sun.shadow.normalBias = 0.03;
    scene.add(sun, sun.target);

    const uniforms = grassUniforms();
    const turfGround = ground(turfTexture());
    const dirtGround = ground(dirtTexture());
    const relief = dirtReliefMap();
    relief.repeat.set(LENGTH / RELIEF_TILE, WIDTH / RELIEF_TILE);
    const shellMap = turfTexture();
    shellMap.repeat.set(LENGTH / SURFACE_TILE, WIDTH / SURFACE_TILE);
    const shell = shellGrass(LENGTH, WIDTH, SHELL_LAYERS, shellMap, uniforms);
    const blades = bladeMesh(flatBladeField(LENGTH, WIDTH, MAX_BLADES), bladeMaterial(uniforms));
    scene.add(turfGround, dirtGround, shell, blades);

    const horse: Horse = {number: 1, name: 'ぬいぐるみ', coat: '', coatCode: '03', blinker: false, gate: 1, jockey: '', age: 4, weight: 500, positions: [], last3: 0, history: []};
    const model = plush(horse, {bridle: true});
    model.scale.setScalar(HORSE_SCALE);
    model.getObjectByName('nameplate')!.visible = false;
    model.getObjectByName('numberplate')!.visible = false;
    scene.add(model);

    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const trail = addDirtTrail(scene, patch, () => motion.matches);
    trail.show(false);
    const field = {horses: [model], at: [0], kick: [NO_KICK]};
    rig.current = {uniforms, turfGround, dirtGround, shell, blades, relief, trail, camera, controls};

    const resize = () => {
      renderer.setSize(el.clientWidth, el.clientHeight);
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    let frame = 0, last = 0, distance = 0, frames = 0, since = 0;
    const draw = (now: number) => {
      const dt = last ? Math.min((now - last) / 1000, 0.05) : 0;
      last = now;
      const cfg = live.current;
      if (!motion.matches) uniforms.uTime.value += dt;
      if (cfg.moving) {
        distance += dt * RUN_SPEED;
        const before = model.position.x;
        model.position.x = ((distance + LENGTH / 2 - 4) % (LENGTH - 8)) - (LENGTH / 2 - 4);
        // The camera rides along, so the look of the ground under a moving shot can be judged.
        const dx = model.position.x - before;
        camera.position.x += dx;
        controls.target.x += dx;
      }
      posePlush(model, distance, motion.matches);
      // The preview's horse runs on the spot in a loop, so distance (not position) is the race it has run.
      field.at[0] = distance;
      if (cfg.surface === 'dirt' && cfg.trail) trail.update(field, distance, false, null);
      const lift = T.MathUtils.degToRad(cfg.sun);
      sun.target.position.copy(model.position);
      sun.position.copy(model.position).add(new T.Vector3(-0.75 * Math.cos(lift), Math.sin(lift), 0.66 * Math.cos(lift)).multiplyScalar(45));
      controls.update();
      renderer.render(scene, camera);
      frames++;
      if (now - since > 500) {
        const tris = renderer.info.render.triangles;
        if (status.current) status.current.textContent = `${Math.round(frames * 1000 / (now - since))} fps ・ 三角形 ${(tris / 1e6).toFixed(2)}M ・ 描画 ${renderer.info.render.calls}回`;
        frames = 0;
        since = now;
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      relief.dispose();
      trail.dispose();
      disposeScene(scene);
      renderer.dispose();
      renderer.domElement.remove();
      rig.current = null;
    };
  }, [horseReady]);

  useEffect(() => {
    live.current = s;
    const r = rig.current;
    if (!r) return;
    const turf = s.surface === 'turf';
    const withBlades = turf && (s.turf === 'blades' || s.turf === 'both');
    const withShell = turf && (s.turf === 'shell' || s.turf === 'both');
    r.turfGround.visible = turf && !withShell;
    // Under bare blades the ground is the shaded soil and thatch seen between them.
    r.turfGround.material.color.set(s.turf === 'blades' ? '#6f7f62' : '#ffffff');
    r.shell.visible = withShell;
    r.blades.visible = withBlades;
    r.blades.geometry.instanceCount = s.blades;
    r.uniforms.uHeight.value = s.height;
    r.uniforms.uShellHeight.value = s.height * (s.turf === 'both' ? SHELL_UNDER_BLADES : 0.8);
    r.uniforms.uWind.value = s.wind;
    r.uniforms.uStripe.value = s.stripes ? 1 : 0;
    r.dirtGround.visible = !turf;
    const bumpy = s.dirt === 'relief';
    if ((r.dirtGround.material.normalMap !== null) !== bumpy) {
      r.dirtGround.material.normalMap = bumpy ? r.relief : null;
      r.dirtGround.material.needsUpdate = true;
    }
    r.dirtGround.material.normalScale.setScalar(s.relief);
    const marks = !turf && s.trail;
    r.trail.show(marks);
    if (!marks) r.trail.clear();
  }, [s]);

  const pickView = (name: string) => {
    setView(name);
    const r = rig.current;
    if (!r) return;
    // Views are framed on the horse, wherever along the patch it has run to.
    const shift = r.controls.target.x - views[view].target[0];
    const at = s.moving ? shift : 0;
    r.camera.position.set(...views[name].position).setX(views[name].position[0] + at);
    r.controls.target.set(...views[name].target).setX(views[name].target[0] + at);
  };
  const turf = s.surface === 'turf';
  const grown = turf && s.turf !== 'texture';

  return <main className="surface-editor">
    <nav><Link href="/">← レース画面に戻る</Link></nav>
    <h1>馬場の質感を確かめる</h1>
    <p>コースの一部（{LENGTH}m × {WIDTH}m）を切り出しています。ドラッグで向き、ホイールで距離を変えられます。</p>
    <div className="surface-controls">
      <div className="surface-group"><span>馬場</span>
        <button aria-pressed={turf} onClick={() => set({surface: 'turf'})}>芝</button>
        <button aria-pressed={!turf} onClick={() => set({surface: 'dirt'})}>ダート</button>
      </div>
      <div className="surface-group"><span>表現</span>
        {turf
          ? (Object.keys(turfLooks) as TurfLook[]).map((k) => <button key={k} aria-pressed={s.turf === k} onClick={() => set({turf: k})}>{turfLooks[k]}</button>)
          : (Object.keys(dirtLooks) as DirtLook[]).map((k) => <button key={k} aria-pressed={s.dirt === k} onClick={() => set({dirt: k})}>{dirtLooks[k]}</button>)}
      </div>
      <div className="surface-group"><span>カメラ</span>
        {Object.keys(views).map((name) => <button key={name} aria-pressed={view === name} onClick={() => pickView(name)}>{name}</button>)}
      </div>
      <button aria-pressed={s.moving} onClick={() => set({moving: !s.moving})}>{s.moving ? '止める' : '走らせる'}</button>
      {!turf && <button aria-pressed={s.trail} onClick={() => set({trail: !s.trail})}>ひづめ跡と砂煙</button>}
    </div>
    <div className="surface-controls">
      {turf && (s.turf === 'blades' || s.turf === 'both') && <label>葉の本数
        <input type="range" min={50_000} max={MAX_BLADES} step={50_000} value={s.blades} onChange={(e) => set({blades: +e.target.value})} />
        <output>{s.blades / 10_000}万本</output>
      </label>}
      {grown && <label>草丈
        <input type="range" min={0.06} max={0.5} step={0.01} value={s.height} onChange={(e) => set({height: +e.target.value})} />
        <output>{Math.round(s.height * 100)}cm</output>
      </label>}
      {grown && <label>風
        <input type="range" min={0} max={1.5} step={0.05} value={s.wind} onChange={(e) => set({wind: +e.target.value})} />
        <output>{s.wind.toFixed(2)}</output>
      </label>}
      {grown && <label><input type="checkbox" checked={s.stripes} onChange={(e) => set({stripes: e.target.checked})} /> 刈り目で葉を寝かせる</label>}
      {!turf && s.dirt === 'relief' && <label>凹凸の強さ
        <input type="range" min={0.2} max={3} step={0.1} value={s.relief} onChange={(e) => set({relief: +e.target.value})} />
        <output>{s.relief.toFixed(1)}</output>
      </label>}
      <label>日差しの高さ
        <input type="range" min={8} max={80} step={1} value={s.sun} onChange={(e) => set({sun: +e.target.value})} />
        <output>{s.sun}°</output>
      </label>
    </div>
    <div ref={host} className="surface-canvas" aria-label="馬場の3Dプレビュー" />
    <p ref={status} className="surface-status" role="status" aria-live="off">計測中…</p>
  </main>;
}
