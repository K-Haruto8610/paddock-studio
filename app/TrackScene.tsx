'use client';
import {useEffect, useRef, useSyncExternalStore} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Race} from './types';
import {makeTrack, availableKinds} from './trackGeometry';
import {makeRoute, type RaceRoute} from './raceRoutes';
import {sampleLayouts} from './raceLayout';
import type {PlaybackPlan} from './playback';
import {createPackCamera} from './packCamera';
import {
  addCourseSurfaces,
  addGround,
  addHorses,
  addLights,
  addMarkers,
  addRails,
  addSelectionRing,
  buildCourseSet,
  disposeScene,
} from './sceneBuilders';

type LiveState = {progress: number; selected: number | null; followKey: number};

// WebGL support is a property of the browser, not of the race: check it once, without
// touching React state from inside the scene effect.
let webglSupport: boolean | undefined;
function hasWebGL() {
  if (webglSupport === undefined) {
    try {
      const canvas = document.createElement('canvas');
      webglSupport = !!(canvas.getContext('webgl2') || canvas.getContext('webgl'));
    } catch {
      webglSupport = false;
    }
  }
  return webglSupport;
}
const subscribeNever = () => () => {};

/** Moves every horse to its place on the route for this travel distance. */
function placeHorses(
  horses: T.Group[],
  race: Race,
  route: RaceRoute,
  ring: T.Mesh,
  placements: {gap: number; lane: number}[],
  travel: number,
  selected: number | null,
) {
  ring.visible = selected !== null;
  horses.forEach((g, i) => {
    const {gap, lane} = placements[i];
    const at = Math.max(0, travel - gap);
    const here = route.sample(at, lane);
    // Pitch with the slope so hooves stay on the exaggerated relief.
    const ahead = route.sample(at + 2, lane).point;
    const behind = route.sample(Math.max(0, at - 2), lane).point;
    const pitch = Math.atan2(
      ahead.y - behind.y,
      Math.hypot(ahead.x - behind.x, ahead.z - behind.z),
    );
    g.position.copy(here.point);
    g.rotation.set(0, -Math.atan2(here.tangent.z, here.tangent.x), pitch);
    g.getObjectByName('nameplate')!.visible = selected === race.horses[i].number;
    if (selected === race.horses[i].number) {
      ring.position.copy(g.position);
      ring.position.y += 0.12;
    }
  });
}

export default function TrackScene({
  race,
  plan,
  progress,
  selected,
  followKey,
  onSelect,
}: {
  race: Race;
  plan: PlaybackPlan;
  progress: number;
  selected: number | null;
  followKey: number;
  onSelect: (n: number) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  // The animation loop lives outside React's render cycle: it reads the latest props from refs.
  const state = useRef<LiveState>({progress, selected, followKey});
  const select = useRef(onSelect);
  useEffect(() => {
    state.current = {progress, selected, followKey};
    select.current = onSelect;
  });
  const webgl = useSyncExternalStore(subscribeNever, hasWebGL, () => true);

  useEffect(() => {
    if (!webgl) return;
    const el = host.current!;
    let renderer: T.WebGLRenderer;
    try {
      renderer = new T.WebGLRenderer({antialias: true});
    } catch {
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    renderer.setClearColor('#e8eddf');
    el.appendChild(renderer.domElement);

    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(40, 1, 0.3, 6000);
    camera.position.set(0, 800, 800);
    scene.fog = new T.Fog('#e8eddf', 500, 1600);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 12;
    controls.maxDistance = 900;
    controls.maxPolarAngle = Math.PI * 0.48;
    const sun = addLights(scene);

    // Course geometry and everything drawn on it.
    const tokyo = race.venueCode === '05';
    const track = makeTrack(race);
    const route = makeRoute(race, track);
    const courses = buildCourseSet(race, track, availableKinds(race));
    const terrain = addGround(scene, courses.tracks, route);
    addCourseSurfaces(scene, courses, terrain, tokyo, route);
    addRails(scene, track, courses.turf);
    addMarkers(scene, race, track, route);

    // Horses and click-to-select.
    const {layouts, knots} = plan;
    const horses = addHorses(scene, race);
    const ring = addSelectionRing(scene);
    const ray = new T.Raycaster();
    const mouse = new T.Vector2();
    let down = [0, 0];
    const onPointerDown = (e: PointerEvent) => {
      down = [e.clientX, e.clientY];
    };
    const onPointerUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const b = el.getBoundingClientRect();
      mouse.set(
        ((e.clientX - b.left) / b.width) * 2 - 1,
        -((e.clientY - b.top) / b.height) * 2 + 1,
      );
      ray.setFromCamera(mouse, camera);
      const hit = ray.intersectObjects(horses, true)[0];
      if (hit) {
        let o: T.Object3D = hit.object;
        while (o.parent && !o.userData.number) o = o.parent;
        if (o.userData.number) select.current(o.userData.number);
      }
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);

    const resize = () => {
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    const packCamera = createPackCamera(camera, controls, renderer.domElement);
    let frame = 0;
    let lastTime = performance.now();
    function draw(now = performance.now()) {
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      const {progress: p, selected, followKey} = state.current;
      const travel = p * race.distance;
      const placements = sampleLayouts(layouts, travel, knots);
      placeHorses(horses, race, route, ring, placements, travel, selected);

      const packCenter = new T.Box3()
        .setFromPoints(horses.map((g) => g.position))
        .getCenter(new T.Vector3());
      const fieldSpan = Math.max(...placements.map((v) => v.gap)) + 8;
      const heading = route.sample(Math.max(0, travel - fieldSpan / 2));
      packCamera.update({dt, progress: p, followKey, packCenter, fieldSpan, heading});
      sun.target.position.copy(packCenter);
      sun.position.copy(packCenter).add(new T.Vector3(-200, 500, 180));
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(draw);
    }
    draw();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      packCamera.dispose();
      controls.dispose();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      disposeScene(scene);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [race, plan, webgl]);

  return (
    <div
      className="scene"
      ref={host}
      aria-label={`${race.name}の3D展開。馬群を自動で追尾。ドラッグで回転、スクロールで拡大`}
    >
      {!webgl && <p className="scene-error">3D表示にはWebGL対応ブラウザーが必要です。</p>}
    </div>
  );
}
