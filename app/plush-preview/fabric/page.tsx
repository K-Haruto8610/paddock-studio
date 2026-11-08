'use client';

import Link from 'next/link';
import {useEffect, useRef, useState} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {plush, posePlush, FABRIC_LABELS, type FabricChoice, type FabricKind} from '../../plushHorse';
import {disposeScene} from '../../sceneBuilders';
import type {Horse} from '../../types';
import './fabric.css';

type Part = keyof FabricChoice;
const parts: {key: Part; label: string}[] = [
  {key: 'body', label: '胴体・頭'},
  {key: 'mane', label: 'たてがみ・尾'},
  {key: 'menko', label: 'メンコ'},
  {key: 'bridle', label: '頭絡'},
];
const kinds = Object.keys(FABRIC_LABELS) as FabricKind[];
type Card = Required<FabricChoice>;
const START: Card[] = [
  {body: 'current', mane: 'current', menko: 'current', bridle: 'current'},
  {body: 'minky', mane: 'boa', menko: 'felt', bridle: 'leather'},
  {body: 'velour', mane: 'velour', menko: 'felt', bridle: 'leather'},
  {body: 'boa', mane: 'boa', menko: 'felt', bridle: 'leather'},
  {body: 'terry', mane: 'felt', menko: 'felt', bridle: 'leather'},
  {body: 'knit', mane: 'knit', menko: 'felt', bridle: 'leather'},
];
const storageKey = 'paddock-plush-fabric-v1';
const views: Record<string, {position: [number, number, number]; target: [number, number, number]}> = {
  '斜め': {position: [6, 3.9, 7], target: [0.15, 1.6, 0]},
  '横': {position: [0, 2.8, 9], target: [0.15, 1.6, 0]},
  '正面': {position: [9, 2.8, 0], target: [0.15, 1.6, 0]},
  '顔アップ': {position: [3.4, 2.9, 3], target: [1, 2.1, 0]},
};

/** Camera and motion shared by every card, so all horses are seen from the same angle at the same moment. */
type Shared = {
  position: T.Vector3; target: T.Vector3; owner: number;
  moving: boolean; distance: number; last: number;
};

function FabricView({index, choice, coatCode, blinker, bridle, shared}: {
  index: number; choice: Card; coatCode: string; blinker: boolean; bridle: boolean; shared: React.RefObject<Shared>;
}) {
  const host = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<T.Scene | null>(null);
  const modelRef = useRef<T.Group | null>(null);
  useEffect(() => {
    const el = host.current!;
    const renderer = new T.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    el.appendChild(renderer.domElement);
    const scene = new T.Scene();
    sceneRef.current = scene;
    scene.background = new T.Color('#e8e4dc');
    const camera = new T.PerspectiveCamera(36, 1, 0.1, 100);
    camera.position.copy(shared.current.position);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(shared.current.target);
    controls.minDistance = 2.5;
    controls.maxDistance = 16;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.addEventListener('start', () => {shared.current.owner = index;});
    scene.add(new T.HemisphereLight('#fffaf0', '#8e8272', 2.2));
    const key = new T.DirectionalLight('#fff4e4', 3);
    key.position.set(3, 7, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = key.shadow.camera.bottom = -4;
    key.shadow.camera.right = key.shadow.camera.top = 4;
    key.shadow.normalBias = 0.025;
    scene.add(key);
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
    const draw = (now: number) => {
      const s = shared.current;
      // The first card to draw in a frame advances the shared clock; the others reuse it.
      if (now > s.last) {
        if (s.moving) s.distance += Math.min((now - s.last) / 1000, 0.05) * 32;
        s.last = now;
      }
      if (s.owner === index) {
        controls.update();
        s.position.copy(camera.position);
        s.target.copy(controls.target);
      } else {
        camera.position.copy(s.position);
        controls.target.copy(s.target);
        camera.lookAt(s.target);
      }
      if (modelRef.current) posePlush(modelRef.current, s.distance, motion.matches);
      renderer.render(scene, camera);
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      disposeScene(scene);
      renderer.dispose();
      renderer.domElement.remove();
      sceneRef.current = null;
      modelRef.current = null;
    };
  }, [index, shared]);
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (modelRef.current) {
      scene.remove(modelRef.current);
      const discarded = new T.Scene();
      discarded.add(modelRef.current);
      disposeScene(discarded);
    }
    const horse: Horse = {number: index + 1, name: 'ぬいぐるみ', coat: '', coatCode, blinker, gate: 1, jockey: '', age: 4, weight: 500, positions: [], last3: 0, history: []};
    const model = plush(horse, {bridle, fabric: choice});
    model.getObjectByName('nameplate')!.visible = false;
    model.getObjectByName('numberplate')!.visible = false;
    modelRef.current = model;
    scene.add(model);
  }, [index, choice, coatCode, blinker, bridle]);
  return <div ref={host} className="fabric-canvas" aria-label={`見本${index + 1}の3Dプレビュー`} />;
}

export default function FabricCompare() {
  const [cards, setCards] = useState<Card[]>(START);
  const [chosen, setChosen] = useState<number | null>(null);
  const [coatCode, setCoatCode] = useState('03');
  const [blinker, setBlinker] = useState(true);
  const [bridle, setBridle] = useState(true);
  const [view, setView] = useState('斜め');
  const [moving, setMoving] = useState(false);
  const [copied, setCopied] = useState('');
  const shared = useRef<Shared>({
    position: new T.Vector3(...views['斜め'].position), target: new T.Vector3(...views['斜め'].target),
    owner: 0, moving: false, distance: 0, last: 0,
  });
  // Restore after mount: reading localStorage during render would mismatch the server HTML.
  useEffect(() => {
    try {
      const value = JSON.parse(localStorage.getItem(storageKey) ?? 'null');
      if (Array.isArray(value?.cards) && value.cards.length === START.length) {
        const valid = (k: unknown): k is FabricKind => typeof k === 'string' && k in FABRIC_LABELS;
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setCards(value.cards.map((c: Card, i: number) => Object.fromEntries(parts.map(({key}) => [key, valid(c?.[key]) ? c[key] : START[i][key]])) as Card));
        if (typeof value.chosen === 'number') setChosen(value.chosen);
      }
    } catch {}
  }, []);
  useEffect(() => {
    try {localStorage.setItem(storageKey, JSON.stringify({cards, chosen}));} catch {}
  }, [cards, chosen]);
  useEffect(() => {shared.current.moving = moving;}, [moving]);
  const pickView = (name: string) => {
    setView(name);
    shared.current.position.set(...views[name].position);
    shared.current.target.set(...views[name].target);
    shared.current.owner = -1;
  };
  const setPart = (i: number, part: Part, kind: FabricKind) => setCards(all => all.map((c, j) => j === i ? {...c, [part]: kind} : c));
  const chosenJson = chosen === null ? '' : JSON.stringify(cards[chosen], null, 2);
  return <main className="fabric-editor">
    <nav><Link href="/plush-preview">← 形の調整に戻る</Link></nav>
    <h1>布の質感を比べる</h1>
    <p>6体を同じ角度で並べています。どれか1体をドラッグすると、全員の向きがそろって変わります。部位ごとに布を選び替えられます。</p>
    <div className="fabric-actions">
      <label>毛色 <select aria-label="毛色" value={coatCode} onChange={e => setCoatCode(e.target.value)}><option value="07">芦毛</option><option value="01">栗毛</option><option value="03">鹿毛</option><option value="05">青毛</option><option value="11">白毛</option></select></label>
      <label><input type="checkbox" checked={blinker} onChange={e => setBlinker(e.target.checked)} /> メンコ</label>
      <label><input type="checkbox" checked={bridle} onChange={e => setBridle(e.target.checked)} /> 頭絡</label>
      {Object.keys(views).map(name => <button key={name} aria-pressed={view === name} onClick={() => pickView(name)}>{name}</button>)}
      <button onClick={() => setMoving(!moving)}>{moving ? '一時停止' : '動きを見る'}</button>
      <button onClick={() => {setCards(START); setChosen(null);}}>最初の組み合わせに戻す</button>
    </div>
    <div className="fabric-grid">
      {cards.map((card, i) => <section key={i} className={`fabric-card${chosen === i ? ' is-chosen' : ''}`} aria-label={`見本${i + 1}`}>
        <FabricView index={i} choice={card} coatCode={coatCode} blinker={blinker} bridle={bridle} shared={shared} />
        <div className="fabric-parts">
          {parts.map(({key, label}) => <label key={key}>
            <span>{label}</span>
            <select value={card[key]} onChange={e => setPart(i, key, e.target.value as FabricKind)}>
              {kinds.map(k => <option key={k} value={k}>{FABRIC_LABELS[k]}</option>)}
            </select>
          </label>)}
        </div>
        <button className="fabric-pick" aria-pressed={chosen === i} onClick={() => {setChosen(chosen === i ? null : i); setCopied('');}}>
          {chosen === i ? '選択中' : 'これにする'}
        </button>
      </section>)}
    </div>
    {chosen !== null && <section className="fabric-result" aria-label="選んだ組み合わせ">
      <p>見本{chosen + 1}を選びました。下の値をチャットに貼ると、レース画面の全馬に反映できます。</p>
      <textarea readOnly rows={6} value={chosenJson} aria-label="選んだ組み合わせの値" />
      <button onClick={() => navigator.clipboard.writeText(chosenJson).then(() => setCopied('コピーしました。'), () => setCopied('コピーできませんでした。上の値を選んでコピーしてください。'))}>値をコピー</button>
      <span role="status">{copied}</span>
    </section>}
  </main>;
}
