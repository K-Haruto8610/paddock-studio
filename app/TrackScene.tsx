'use client';
import {useEffect, useRef, useSyncExternalStore} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Race} from './types';
import {makeTrack, availableKinds} from './trackGeometry';
import {makeRoute, type RaceRoute} from './raceRoutes';
import {interpolateLayouts, separateLayouts, type Placement} from './raceLayout';
import {clampPlacement} from './manualEdit';
import type {PlaybackPlan} from './playback';
import {createPackCamera} from './packCamera';
import {posePlush} from './plushHorse';
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
const Z_AXIS = new T.Vector3(0, 0, 1);

type LiveState = {
  progress: number;
  rate: number;
  selected: number | null;
  followKey: number;
  plan: PlaybackPlan;
  /** Manual edit mode: distance of the edit point the race is paused on (null elsewhere). */
  editAt: number | null;
};

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
  reducedMotion: boolean,
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
    // The number floats straight above the horse: undo the slope pitch so it does not sway.
    const number = g.getObjectByName('numberplate')!;
    number.position.set(0, 4.4, 0).applyAxisAngle(Z_AXIS, -pitch);
    posePlush(g, at, reducedMotion);
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
  rate = 0,
  selected,
  followKey,
  onSelect,
  editAt = null,
  onEdit,
}: {
  race: Race;
  plan: PlaybackPlan;
  progress: number;
  /** Progress per millisecond while playing (0 when paused). The scene advances on its own frame clock. */
  rate?: number;
  selected: number | null;
  followKey: number;
  onSelect: (n: number) => void;
  editAt?: number | null;
  /** Called when a dragged horse is let go at an edit point: its new place in plan units. */
  onEdit?: (index: number, placement: Placement) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  // The animation loop lives outside React's render cycle: it reads the latest props from refs.
  const state = useRef<LiveState>({progress, rate, selected, followKey, plan, editAt});
  const select = useRef(onSelect);
  const edit = useRef(onEdit);
  useEffect(() => {
    state.current = {progress, rate, selected, followKey, plan, editAt};
    select.current = onSelect;
    edit.current = onEdit;
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
    addRails(scene, track, courses.turf, route, race.distance);
    const gate = addMarkers(scene, race, track, route);
    const gateBox = new T.Box3().setFromObject(gate);
    const frustum = new T.Frustum();
    const view = new T.Matrix4();

    // Horses and click-to-select; in edit mode, drag a horse to move it.
    const horses = addHorses(scene, race);
    const ring = addSelectionRing(scene);
    const ray = new T.Raycaster();
    const mouse = new T.Vector2();
    let down = [0, 0];
    const aim = (e: PointerEvent) => {
      const b = el.getBoundingClientRect();
      mouse.set(
        ((e.clientX - b.left) / b.width) * 2 - 1,
        -((e.clientY - b.top) / b.height) * 2 + 1,
      );
      ray.setFromCamera(mouse, camera);
    };
    const horseAt = (e: PointerEvent) => {
      aim(e);
      const hit = ray.intersectObjects(horses, true)[0];
      if (!hit) return null;
      let o: T.Object3D = hit.object;
      while (o.parent && !o.userData.number) o = o.parent;
      return (o.userData.number as number | undefined) ?? null;
    };
    // A drag moves one horse on a level plane through it: along the course changes its gap,
    // across the course changes its lane.
    let drag: {
      index: number;
      start: Placement;
      from: T.Vector3;
      plane: T.Plane;
      tangent: T.Vector3;
      outward: T.Vector3;
      now: Placement;
      pointer: number;
    } | null = null;
    const hitPlane = (plane: T.Plane) => ray.ray.intersectPlane(plane, new T.Vector3());
    const onPointerDown = (e: PointerEvent) => {
      down = [e.clientX, e.clientY];
      const {editAt, plan: live} = state.current;
      if (editAt === null || !edit.current || e.button !== 0) return;
      const number = horseAt(e);
      if (number === null) return;
      const k = live.knots.findIndex((d) => Math.abs(d - editAt) < 0.05);
      const index = race.horses.findIndex((h) => h.number === number);
      if (k < 0 || index < 0) return;
      const start = live.layouts[k][index];
      const at = Math.max(0, editAt - start.gap);
      const here = route.sample(at, start.lane);
      const outward = route.sample(at, start.lane + 1).point.clone().sub(here.point).setY(0);
      const tangent = here.tangent.clone().setY(0).normalize();
      const plane = new T.Plane(new T.Vector3(0, 1, 0), -horses[index].position.y);
      const from = hitPlane(plane);
      if (!from) return;
      // Take the pointer before OrbitControls sees it, so the camera stays put while dragging.
      controls.enabled = false;
      e.stopPropagation();
      el.setPointerCapture(e.pointerId);
      drag = {index, start, from, plane, tangent, outward, now: start, pointer: e.pointerId};
      select.current(number);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.pointer) return;
      aim(e);
      const p = hitPlane(drag.plane);
      if (!p) return;
      const d = p.sub(drag.from);
      const per = drag.outward.lengthSq() || 1;
      drag.now = clampPlacement({
        gap: drag.start.gap - d.dot(drag.tangent),
        lane: drag.start.lane + d.dot(drag.outward) / per,
      });
    };
    const endDrag = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.pointer) return false;
      if (drag.now !== drag.start) edit.current?.(drag.index, drag.now);
      drag = null;
      controls.enabled = true;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
      return true;
    };
    const onPointerUp = (e: PointerEvent) => {
      if (endDrag(e)) return;
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const number = horseAt(e);
      if (number !== null) select.current(number);
    };
    const onPointerCancel = (e: PointerEvent) => {
      if (drag && e.pointerId === drag.pointer) drag.now = drag.start;
      endDrag(e);
    };
    // Capture phase on the host runs before OrbitControls' own listener on the canvas.
    el.addEventListener('pointerdown', onPointerDown, true);
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerup', onPointerUp);
    el.addEventListener('pointercancel', onPointerCancel);

    const resize = () => {
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    const packCamera = createPackCamera(camera, controls, renderer.domElement);
    const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let frame = 0;
    let lastTime = performance.now();
    let shown = state.current.progress;
    function draw(now = performance.now()) {
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      const {progress: target, rate, selected, followKey, plan: live} = state.current;
      // React state updates on its own clock, so its steps don't line up with this frame's time.
      // Advance by this frame's own time and only ease towards the React value, so motion stays even.
      // A seek (a large difference) jumps straight there.
      if (rate > 0 && Math.abs(target - shown) < 0.01) {
        shown = Math.min(1, shown + rate * dt * 1000);
        shown += (target - shown) * (1 - Math.exp(-dt * 0.5));
      } else {
        shown = target;
      }
      const p = shown;
      const travel = p * race.distance;
      const raw = interpolateLayouts(live.layouts, travel, live.knots);
      if (drag) raw[drag.index] = drag.now;
      const placements = separateLayouts(raw);
      placeHorses(horses, race, route, ring, placements, travel, selected, motionPreference.matches);

      const packCenter = new T.Box3()
        .setFromPoints(horses.map((g) => g.position))
        .getCenter(new T.Vector3());
      const gaps = placements.map((v) => v.gap);
      const leaderAt = travel - Math.min(...gaps);
      const meanLane = placements.reduce((sum, v) => sum + v.lane, 0) / placements.length;
      packCamera.update({
        dt,
        progress: p,
        followKey,
        spread: Math.max(...gaps) - Math.min(...gaps),
        behindLeader: (metres) => route.sample(Math.max(0, leaderAt - metres), meanLane),
      });
      // The starting gate goes once the race is under way and the camera has left it behind.
      if (travel < 1) gate.visible = true;
      else if (gate.visible) {
        camera.updateMatrixWorld();
        frustum.setFromProjectionMatrix(
          view.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
        );
        if (!frustum.intersectsBox(gateBox)) gate.visible = false;
      }
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
      el.removeEventListener('pointerdown', onPointerDown, true);
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerup', onPointerUp);
      el.removeEventListener('pointercancel', onPointerCancel);
      disposeScene(scene);
      renderer.dispose();
      renderer.domElement.remove();
    };
    // The plan is read live from state, so a hand edit does not rebuild the scene or move the camera.
  }, [race, webgl]);

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
