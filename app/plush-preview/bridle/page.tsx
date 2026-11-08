'use client';

import Link from 'next/link';
import {useEffect, useRef, useState} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {plush} from '../../plushHorse';
import {disposeScene} from '../../sceneBuilders';
import type {Horse} from '../../types';
import '../preview.css';
import './bridle.css';

type Point = [number, number, number];
type Mode = 'rotate' | 'pen' | 'pin';
/** One instruction. Points are in the head group's own coordinates, the same frame the bridle code uses. */
type Note = {id: number; kind: 'pen' | 'pin'; part: string; points: Point[]; comment: string};

const storageKey = 'paddock-bridle-notes-v1';
const partNames: Record<string, string> = {
  'bridle-noseband': '鼻革', 'bridle-headstall': '項革・頬革', 'bridle-browband': '額革', 'bridle-joint': '革のつなぎ目', 'bridle-ring': '金具の輪',
  'seamless-body-and-feet': '顔・体の表面', 'cupped-cloth-ear': '耳', 'felt-zigzag-mane': 'たてがみ',
};
const colors = ['#d23c3c', '#2c6fd1', '#1f9c5a', '#c98a12', '#8a45c4', '#d1489a'];
const colorOf = (id: number) => colors[(id - 1) % colors.length];
const round = (v: T.Vector3): Point => [+v.x.toFixed(3), +v.y.toFixed(3), +v.z.toFixed(3)];

function partOf(object: T.Object3D, local: T.Vector3) {
  let o: T.Object3D | null = object;
  while (o && !partNames[o.name]) o = o.parent;
  const name = o ? partNames[o.name] : 'その他（目・鼻など）';
  const side = Math.abs(local.z) < 0.08 ? '中央' : local.z < 0 ? '左側' : '右側';
  return `${name}・${side}`;
}

function numberSprite(text: string, color: string) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(32, 32, 28, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 34px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 34);
  const map = new T.CanvasTexture(canvas);
  map.colorSpace = T.SRGBColorSpace;
  const sprite = new T.Sprite(new T.SpriteMaterial({map, depthTest: false}));
  sprite.scale.setScalar(0.11);
  sprite.renderOrder = 10;
  return sprite;
}

function noteObject(note: Note) {
  const color = colorOf(note.id);
  const group = new T.Group();
  const material = new T.MeshBasicMaterial({color, depthTest: false, transparent: true, opacity: 0.9});
  const points = note.points.map(p => new T.Vector3(...p));
  if (note.kind === 'pin' || points.length < 2) {
    const dot = new T.Mesh(new T.SphereGeometry(0.022, 12, 8), material);
    dot.position.copy(points[0]);
    dot.renderOrder = 9;
    group.add(dot);
  } else {
    const tube = new T.Mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points), Math.min(points.length * 2, 400), 0.012, 6, false), material);
    tube.renderOrder = 9;
    group.add(tube);
  }
  const tag = numberSprite(String(note.id), color);
  tag.position.copy(points[0]).add(new T.Vector3(0, 0.09, 0));
  group.add(tag);
  return group;
}

export default function BridleNotes() {
  const host = useRef<HTMLDivElement>(null);
  const [notes, setNotes] = useState<Note[]>([]);
  const [mode, setMode] = useState<Mode>('pen');
  const [blinker, setBlinker] = useState(false);
  const [coatCode, setCoatCode] = useState('07');
  const [general, setGeneral] = useState('');
  const [status, setStatus] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [showNotes, setShowNotes] = useState(true);
  const modeRef = useRef(mode);
  const notesRef = useRef(notes);
  const sceneRef = useRef<{scene: T.Scene; renderer: T.WebGLRenderer; camera: T.PerspectiveCamera; controls: OrbitControls} | null>(null);
  const modelRef = useRef<T.Group | null>(null);
  const layerRef = useRef(new T.Group());
  useEffect(() => {modeRef.current = mode;}, [mode]);
  useEffect(() => {notesRef.current = notes;}, [notes]);

  // Restore after mount: reading localStorage during render would mismatch the server HTML.
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (saved?.notes) {setNotes(saved.notes); setGeneral(saved.general || '');}
    } catch {}
    setLoaded(true);
  }, []);
  useEffect(() => {
    if (!loaded) return;
    try {localStorage.setItem(storageKey, JSON.stringify({notes, general}));} catch {}
  }, [notes, general, loaded]);

  useEffect(() => {
    const el = host.current!;
    const renderer = new T.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    el.appendChild(renderer.domElement);
    const scene = new T.Scene();
    scene.background = new T.Color('#e8e4dc');
    const camera = new T.PerspectiveCamera(32, 1, 0.05, 100);
    camera.position.set(4.4, 3.1, 3.9);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(1, 2.1, 0);
    controls.minDistance = 1.2;
    controls.maxDistance = 12;
    controls.enableDamping = true;
    scene.add(new T.HemisphereLight('#fffaf0', '#8e8272', 2.2));
    const key = new T.DirectionalLight('#fff4e4', 2.6);
    key.position.set(3, 7, 5);
    scene.add(key);
    sceneRef.current = {scene, renderer, camera, controls};
    const resize = () => {
      renderer.setSize(el.clientWidth, el.clientHeight);
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    const raycaster = new T.Raycaster();
    const hit = (event: PointerEvent) => {
      const model = modelRef.current;
      if (!model) return null;
      const rect = renderer.domElement.getBoundingClientRect();
      const pointer = new T.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const targets: T.Object3D[] = [];
      model.traverse(o => {if (o instanceof T.Mesh && o.visible && !layerRef.current.getObjectById(o.id)) targets.push(o);});
      const first = raycaster.intersectObjects(targets, false)[0];
      if (!first) return null;
      const head = (model.userData.plushRig as {head: T.Group}).head;
      // Lift slightly off the surface along the normal so the ink is not buried.
      const normal = first.face ? first.face.normal.clone().transformDirection(first.object.matrixWorld) : new T.Vector3();
      const world = first.point.clone().addScaledVector(normal, 0.01);
      return {local: head.worldToLocal(world), object: first.object};
    };
    let drawing: {points: T.Vector3[]; part: string; line: T.Line} | null = null;
    const nextId = () => notesRef.current.reduce((max, n) => Math.max(max, n.id), 0) + 1;
    const down = (event: PointerEvent) => {
      if (modeRef.current === 'rotate') return;
      const result = hit(event);
      if (!result) return;
      if (modeRef.current === 'pin') {
        const id = nextId();
        setNotes(list => [...list, {id, kind: 'pin', part: partOf(result.object, result.local), points: [round(result.local)], comment: ''}]);
        return;
      }
      renderer.domElement.setPointerCapture(event.pointerId);
      const line = new T.Line(new T.BufferGeometry().setFromPoints([result.local]), new T.LineBasicMaterial({color: colorOf(nextId()), depthTest: false}));
      line.renderOrder = 9;
      layerRef.current.add(line);
      drawing = {points: [result.local], part: partOf(result.object, result.local), line};
    };
    const move = (event: PointerEvent) => {
      if (!drawing) return;
      const result = hit(event);
      if (!result || result.local.distanceTo(drawing.points[drawing.points.length - 1]) < 0.015) return;
      drawing.points.push(result.local);
      drawing.line.geometry.dispose();
      drawing.line.geometry = new T.BufferGeometry().setFromPoints(drawing.points);
    };
    const up = () => {
      if (!drawing) return;
      const {points, part, line} = drawing;
      drawing = null;
      layerRef.current.remove(line);
      line.geometry.dispose();
      (line.material as T.Material).dispose();
      const id = nextId();
      setNotes(list => [...list, {id, kind: points.length > 1 ? 'pen' : 'pin', part, points: points.map(round), comment: ''}]);
    };
    const canvas = renderer.domElement;
    canvas.addEventListener('pointerdown', down);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);

    let frame = 0;
    const draw = () => {
      controls.enabled = modeRef.current === 'rotate';
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointercancel', up);
      controls.dispose();
      disposeScene(scene);
      renderer.dispose();
      canvas.remove();
      sceneRef.current = null;
      modelRef.current = null;
    };
  }, []);

  // Rebuild the horse when coat or menko changes; the notes layer moves to the new head.
  useEffect(() => {
    const ctx = sceneRef.current;
    if (!ctx) return;
    if (modelRef.current) {
      modelRef.current.userData.plushRig.head.remove(layerRef.current);
      ctx.scene.remove(modelRef.current);
      const discarded = new T.Scene();
      discarded.add(modelRef.current);
      disposeScene(discarded);
    }
    const horse: Horse = {number: 10, name: '', coat: '', coatCode, blinker, gate: 1, jockey: '', age: 4, weight: 500, positions: [], last3: 0, history: []};
    const model = plush(horse, {bridle: true});
    model.getObjectByName('nameplate')!.visible = false;
    model.getObjectByName('numberplate')!.visible = false;
    model.userData.plushRig.head.add(layerRef.current);
    modelRef.current = model;
    ctx.scene.add(model);
  }, [coatCode, blinker]);

  // Redraw the ink and pins whenever the list changes.
  useEffect(() => {
    const layer = layerRef.current;
    for (const child of [...layer.children]) {
      if (child instanceof T.Line) continue;
      layer.remove(child);
      const discarded = new T.Scene();
      discarded.add(child);
      disposeScene(discarded);
    }
    for (const note of notes) layer.add(noteObject(note));
  }, [notes]);

  useEffect(() => {layerRef.current.visible = showNotes;}, [showNotes]);

  const setComment = (id: number, comment: string) => setNotes(list => list.map(n => n.id === id ? {...n, comment} : n));
  const exportText = JSON.stringify({general, notes}, null, 1);
  const copy = async () => {
    try {await navigator.clipboard.writeText(exportText); setStatus('コピーしました。チャットに貼り付けてください。');}
    catch {setStatus('コピーできませんでした。下の欄から選択してコピーしてください。');}
  };
  const savePng = () => {
    const ctx = sceneRef.current;
    if (!ctx) return;
    const a = document.createElement('a');
    a.href = ctx.renderer.domElement.toDataURL('image/png');
    a.download = 'bridle-notes.png';
    a.click();
  };
  const view = (position: Point) => {
    const ctx = sceneRef.current;
    if (!ctx) return;
    ctx.camera.position.set(...position);
    ctx.controls.target.set(1, 2.1, 0);
    ctx.controls.update();
  };

  return <main className="plush-editor">
    <Link href="/plush-preview">← 形の調整に戻る</Link>
    <h1>頭絡への書き込み</h1>
    <p>モデルの表面に直接描いたりピンを立てたりして、番号ごとに指示を書いてください。書いた内容はこのブラウザーに自動で残ります。</p>
    <div className="plush-workspace">
      <section className="plush-settings" aria-label="指示の一覧">
        <label className="bridle-general"><strong>全体への指示</strong>
          <textarea value={general} onChange={e => setGeneral(e.target.value)} rows={3} placeholder="例：全体をもう少し細く、色は黒に" />
        </label>
        {notes.length === 0 && <p className="plush-hint">まだ書き込みはありません。右の画面でペンかピンを選んでモデルに描いてください。</p>}
        <ol className="bridle-notes">
          {notes.map(note => <li key={note.id} style={{borderColor: colorOf(note.id)}}>
            <div className="bridle-note-head">
              <span className="bridle-badge" style={{background: colorOf(note.id)}}>{note.id}</span>
              <span>{note.kind === 'pen' ? '線' : 'ピン'}・{note.part}</span>
              <button aria-label={`${note.id}を削除`} onClick={() => setNotes(list => list.filter(n => n.id !== note.id))}>削除</button>
            </div>
            <textarea value={note.comment} onChange={e => setComment(note.id, e.target.value)} rows={2} placeholder="この部分への指示" />
          </li>)}
        </ol>
        <div className="plush-actions">
          <button className="plush-save" onClick={copy} disabled={notes.length === 0 && !general}>指示をコピー</button>
          <button onClick={savePng}>画面を画像で保存</button>
          <button onClick={() => {if (confirm('書き込みをすべて消しますか？')) {setNotes([]); setGeneral('');}}} disabled={notes.length === 0 && !general}>すべて消す</button>
        </div>
        <p role="status" className="plush-hint">{status}</p>
        <details><summary>書き込みの中身を確認</summary><textarea aria-label="書き込みの中身" readOnly value={exportText} rows={10} /></details>
      </section>
      <section className="plush-viewer">
        <div className="plush-actions">
          <div className="bridle-modes" role="group" aria-label="操作">
            {([['pen', 'ペン'], ['pin', 'ピン'], ['rotate', '回転・拡大']] as [Mode, string][]).map(([m, text]) =>
              <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>{text}</button>)}
          </div>
          <button onClick={() => setNotes(list => list.slice(0, -1))} disabled={notes.length === 0}>ひとつ戻す</button>
          <label>毛色 <select aria-label="毛色" value={coatCode} onChange={e => setCoatCode(e.target.value)}><option value="07">芦毛</option><option value="01">栗毛</option><option value="03">鹿毛</option><option value="05">青毛</option><option value="11">白毛</option></select></label>
          <label><input type="checkbox" checked={blinker} onChange={e => setBlinker(e.target.checked)} /> メンコ</label>
          <label><input type="checkbox" checked={showNotes} onChange={e => setShowNotes(e.target.checked)} /> 書き込みを表示</label>
          {([['斜め', [4.4, 3.1, 3.9]], ['左', [1, 2.4, -4.2]], ['右', [1, 2.4, 4.2]], ['正面', [5, 2.4, 0]], ['上', [0.6, 5.5, 0.01]]] as [string, Point][]).map(([text, p]) =>
            <button key={text} onClick={() => view(p)}>{text}</button>)}
        </div>
        <div ref={host} className={`plush-canvas bridle-canvas mode-${mode}`} aria-label="頭絡を付けた馬のぬいぐるみ。ペンやピンで書き込めます" />
        <p className="plush-hint">ペン・ピンのときはモデルの上だけに書けます。向きを変えるときは「回転・拡大」に切り替えてください。</p>
      </section>
    </div>
  </main>;
}
