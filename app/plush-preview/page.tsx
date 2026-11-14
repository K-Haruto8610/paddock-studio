'use client';

import Link from 'next/link';
import {useEffect, useRef, useState} from 'react';
import * as T from 'three';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {plush, posePlush, PREVIOUS_HEAD, CURRENT_HEAD, SAVED_HEAD, normalizeHead, type HeadSettings} from '../plushHorse';
import './preview.css';
import {disposeScene} from '../sceneBuilders';
import type {Horse} from '../types';
import {KICK_CADENCE, STRIDE_LENGTH, type Kick} from '../closingKick';

const sliders: {key: keyof HeadSettings; label: string; left: string; right: string}[] = [
  {key: 'headWidth', label: '頭の幅', left: '細い 0.60倍', right: '変更前 1.00倍'},
  {key: 'noseSize', label: '鼻の大きさ', left: '変更前の大きさ', right: '小さく'},
  {key: 'noseTaper', label: '鼻先の先細り', left: '丸い鼻先', right: '先を細く'},
  {key: 'foreheadSlope', label: '額の傾斜', left: '丸い額', right: 'なだらかな額'},
  {key: 'faceBlend', label: '頭と鼻のつながり', left: '輪郭を残す', right: '滑らかに一体化'},
  {key: 'chestSlope', label: '胸元から鼻への傾斜', left: '変更前の胸元', right: '傾斜でつなぐ'},
  {key: 'noseColor', label: '鼻先の色', left: '明るい鼻先', right: '胴体と同じ毛色'},
  {key: 'eyeScale', label: '目の大きさ・位置', left: '変更前', right: '前回変更後'},
  {key: 'maneHeight', label: 'たてがみの高さ', left: '変更前', right: '前回変更後'},
];
const storageKey = 'paddock-plush-head-v1';

export default function PlushPreview() {
  const host = useRef<HTMLDivElement>(null);
  const [coatCode, setCoatCode] = useState('07');
  const [settings, setSettings] = useState<HeadSettings>({...SAVED_HEAD});
  const [renderSettings, setRenderSettings] = useState(settings);
  const [blinker, setBlinker] = useState(false);
  const [bridle, setBridle] = useState(true);
  const [view, setView] = useState('斜め');
  const [moving, setMoving] = useState(false);
  const [kick, setKick] = useState(0);
  const [saved, setSaved] = useState('');
  const movingRef = useRef(moving);
  const kickRef = useRef(kick);
  const sceneRef = useRef<T.Scene | null>(null);
  const modelRef = useRef<T.Group | null>(null);
  const cameraRef = useRef<T.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  useEffect(() => {movingRef.current = moving;}, [moving]);
  useEffect(() => {kickRef.current = kick;}, [kick]);
  // Restore after mount: reading localStorage during render would mismatch the server HTML.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    try {const value = localStorage.getItem(storageKey); if (value) setSettings(normalizeHead(JSON.parse(value)));} catch {}
  }, []);
  useEffect(() => {
    const timer = setTimeout(() => setRenderSettings(settings), 140);
    return () => clearTimeout(timer);
  }, [settings]);
  useEffect(() => {
    const el = host.current!;
    const renderer = new T.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    el.appendChild(renderer.domElement);
    const scene = new T.Scene();
    sceneRef.current = scene;
    scene.background = new T.Color('#e8e4dc');
    // Broad studio reflections soften fibre highlights without external HDR assets.
    const pmrem = new T.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    const environment = pmrem.fromScene(room, 0.06);
    scene.environment = environment.texture;
    scene.environmentIntensity = 0.65;
    room.dispose();
    pmrem.dispose();
    const camera = new T.PerspectiveCamera(36, 1, 0.1, 100);
    camera.position.set(6, 3.9, 7);
    cameraRef.current = camera;
    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;
    controls.target.set(0.15, 1.6, 0);
    controls.minDistance = 4;
    controls.maxDistance = 16;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.enableDamping = true;
    scene.add(new T.HemisphereLight('#fffaf0', '#8e8272', 0.8));
    const key = new T.DirectionalLight('#fff4e4', 2.4);
    key.position.set(3, 7, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 3;
    key.shadow.camera.left = key.shadow.camera.bottom = -4;
    key.shadow.camera.right = key.shadow.camera.top = 4;
    key.shadow.normalBias = 0.025;
    scene.add(key);
    const rim = new T.DirectionalLight('#e4edff', 1.2);
    rim.position.set(-4, 5, -3);
    scene.add(rim);
    const floor = new T.Mesh(new T.PlaneGeometry(200, 200), new T.MeshStandardMaterial({color: '#e8e4dc', roughness: 1}));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.015;
    floor.receiveShadow = true;
    scene.add(floor);
    const resize = () => {
      renderer.setSize(el.clientWidth, el.clientHeight);
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    let frame = 0;
    let distance = 0;
    // The preview holds the kick level fixed, so the faster cadence is accumulated here as the slider moves
    // (the race derives it in closed form from the distance instead).
    let strides = 0;
    let last = performance.now();
    const draw = (now: number) => {
      if (movingRef.current) {
        const step = Math.min((now - last) / 1000, 0.05) * 32;
        distance += step;
        strides += step * KICK_CADENCE * kickRef.current / STRIDE_LENGTH;
      }
      last = now;
      const state: Kick = {level: kickRef.current, strides};
      if (modelRef.current) posePlush(modelRef.current, distance, motion.matches, state);
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      disposeScene(scene);
      environment.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      sceneRef.current = null; modelRef.current = null; cameraRef.current = null; controlsRef.current = null;
    };
  }, []);
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (modelRef.current) {
      scene.remove(modelRef.current);
      const discarded = new T.Scene();
      discarded.add(modelRef.current);
      disposeScene(discarded);
    }
    const horse: Horse = {number: 10, name: 'ぬいぐるみ', coat: '', coatCode, blinker, gate: 1, jockey: '', age: 4, weight: 500, positions: [], last3: 0, history: []};
    const model = plush(horse, {...renderSettings, bridle});
    model.getObjectByName('nameplate')!.visible = false;
    model.getObjectByName('numberplate')!.visible = false;
    modelRef.current = model;
    scene.add(model);
  }, [coatCode, blinker, bridle, renderSettings]);
  useEffect(() => {
    const views: Record<string, [number, number, number]> = {'斜め': [6, 3.9, 7], '横': [0, 2.8, 9], '正面': [9, 2.8, 0], '後ろ': [-7, 3, 4]};
    cameraRef.current?.position.set(...views[view]);
    controlsRef.current?.target.set(0.15, 1.6, 0);
    controlsRef.current?.update();
  }, [view]);
  const update = (key: keyof HeadSettings, value: number) => {setSettings(s => ({...s, [key]: value})); setSaved('');};
  return <main className="plush-editor">
    <Link href="/">← レースに戻る</Link>
    {' '}<Link href="/plush-preview/bridle" className="plush-bridle-link">頭絡に書き込む →</Link>
    {' '}<Link href="/plush-preview/fabric" className="plush-bridle-link">布の質感を比べる →</Link>
    <h1>ぬいぐるみの形を調整</h1>
    <p>変更前の丸い頭を起点に、各項目を少しずつ調整できます。耳は現在の形、脚は0.8倍で固定です。</p>
    <div className="plush-workspace">
      <section className="plush-settings" aria-label="頭部の調整">
        <div className="plush-actions">
          <button onClick={() => {setSettings({...PREVIOUS_HEAD}); setSaved('');}}>頭を変更前に戻す</button>
          <button onClick={() => {setSettings({...CURRENT_HEAD}); setSaved('');}}>前回変更後を表示</button>
          <button onClick={() => {setSettings({...SAVED_HEAD}); setSaved('');}}>採用中の形</button>
        </div>
        <p className="plush-hint">0%＝変更前、100%＝前回の変更を最大限反映。頭の幅は倍率です。</p>
        {sliders.map(({key, label, left, right}) => <label className="plush-slider" key={key}>
          <span><strong>{label}</strong><output>{key === 'headWidth' ? `${settings[key].toFixed(2)}倍` : `${Math.round(settings[key] * 100)}%`}</output></span>
          <input type="range" aria-label={label} min={key === 'headWidth' ? 0.6 : 0} max={1} step={0.01} value={settings[key]} onChange={e => update(key, Number(e.target.value))} />
          <small><span>{left}</span><span>{right}</span></small>
        </label>)}
        <button className="plush-save" onClick={() => {
          try {localStorage.setItem(storageKey, JSON.stringify(settings)); setSaved('このブラウザーに保存しました。');}
          catch {setSaved('保存できませんでした。下の調整値をコピーしてください。');}
        }}>この調整を保存</button>
        <p role="status">{saved}</p>
        <details><summary>調整値を確認・コピー</summary><p>この値をチャットに送ると、選んだ形を本体に反映できます。</p><textarea aria-label="調整値" readOnly value={JSON.stringify(settings, null, 2)} rows={11} /></details>
      </section>
      <section className="plush-viewer">
        <div className="plush-actions">
          <label>毛色 <select aria-label="毛色" value={coatCode} onChange={e => setCoatCode(e.target.value)}><option value="07">芦毛</option><option value="01">栗毛</option><option value="03">鹿毛</option><option value="05">青毛</option><option value="11">白毛</option></select></label>
          <label><input type="checkbox" checked={blinker} onChange={e => setBlinker(e.target.checked)} /> メンコ</label>
          <label><input type="checkbox" checked={bridle} onChange={e => setBridle(e.target.checked)} /> 頭絡</label>
          {['斜め', '横', '正面', '後ろ'].map(angle => <button key={angle} aria-pressed={view === angle} onClick={() => setView(angle)}>{angle}</button>)}
          <button onClick={() => setMoving(!moving)}>{moving ? '一時停止' : '動きを見る'}</button>
          <label>上がりの走り <input type="range" aria-label="上がりの走り" min={0} max={1} step={0.01} value={kick} onChange={e => setKick(Number(e.target.value))} /> {Math.round(kick * 100)}%</label>
        </div>
        <div ref={host} aria-label="馬のぬいぐるみの3Dプレビュー" className="plush-canvas" />
        <p className="plush-hint">ドラッグで回転・スクロールで拡大。調整しても視点を保ちます。</p>
      </section>
    </div>
  </main>;
}
