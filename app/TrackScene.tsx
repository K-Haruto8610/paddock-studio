'use client';
import {useEffect, useRef, useSyncExternalStore} from 'react';
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Race} from './types';
import {makeTrack, availableKinds} from './trackGeometry';
import {makeRoute, type RaceRoute} from './raceRoutes';
import {interpolateLayouts, sampleLayouts, separateLayouts, type Placement} from './raceLayout';
import {clampPlacement} from './manualEdit';
import type {PlaybackPlan} from './playback';
import {createPackCamera} from './packCamera';
import type {CameraMode} from './raceCamera';
import {completeTick, outputSeconds, RACE_FPS} from './raceClock';
import {cameraFrame, createSceneTimeline} from './sceneTimeline';
import {createRaceRenderer, PREVIEW_QUALITY, type SceneQuality} from './sceneQuality';
import {addHorseLabels} from './horseLabels';
import {addRaceLandscape, addRaceSky, skyEnvironment} from './raceLandscape';
import {courseFor} from './courses';
import {posePlush} from './plushHorse';
import {kickAt, NO_KICK, type Kick, type KickPlan} from './closingKick';
import {addFarGround} from './scenery';
import {
  addCourseSurfaces,
  tokyoChutes,
  addGround,
  addHorses,
  addContactShadows,
  addLights,
  addMarkers,
  addRails,
  addSelectionRing,
  buildCourseSet,
  disposeScene,
} from './sceneBuilders';
import {addDistantGrass, addRaceGrass} from './raceGrass';
import {addDirtTrail} from './dirtTrail';
import {raceSunOffset} from './raceLighting';
import {dayOfYear} from './raceSeason';
import {addPrecipitation, GOING_LABEL, WEATHER_LABEL, WEATHER_LOOKS, wetSurfaces, type Weather} from './raceWeather';
import {turfWearAmount} from './turfWear';
/** Beyond this distance from the field a blade of grass is smaller than a pixel, so none are drawn. */
const GRASS_SIGHT = 320;

export type FixedCamera = {position: [number, number, number]; target: [number, number, number]};
export type VerifyFrame = (input: {
  progress: number;
  speed: number;
  cameraMode: CameraMode;
  selected: number | null;
  width: number;
  height: number;
  reduced?: boolean;
  fixed?: FixedCamera;
  history?: number[];
  outputTime?: number;
  sequential?: boolean;
}) => Promise<HTMLCanvasElement>;
type VerificationRef = {current: VerifyFrame | null};

type LiveState = {
  progress: number;
  playing: boolean;
  speed: number;
  selected: number | null;
  followKey: number;
  cameraMode: CameraMode;
  plan: PlaybackPlan;
  /** Manual edit mode: distance of the edit point the race is paused on (null elsewhere). */
  editAt: number | null;
  suspended: boolean;
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
  kicks: KickPlan[],
  /** Filled with each horse's distance along the route and closing-kick state, for what the hooves leave behind. */
  pose: {at: number[]; kick: Kick[]},
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
    // The closing kick depends on the horse's own distance, so seeking and recording reproduce the same form.
    const kick = reducedMotion ? NO_KICK : kickAt(at, kicks[i]);
    posePlush(g, at, reducedMotion, kick);
    pose.at[i] = at;
    pose.kick[i] = kick;
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
  playing = false,
  speed = 2,
  selected,
  followKey,
  cameraMode = 'pack',
  onSelect,
  editAt = null,
  onEdit,
  onRendered,
  recordingSize,
  quality = PREVIEW_QUALITY,
  verification,
  suspended = false,
  elevationScale,
  weather = 'clear',
  going = null,
  courseDay = null,
}: {
  race: Race;
  plan: PlaybackPlan;
  progress: number;
  playing?: boolean;
  speed?: number;
  selected: number | null;
  followKey: number;
  cameraMode?: CameraMode;
  onSelect: (n: number) => void;
  editAt?: number | null;
  /** Called when a dragged horse is let go at an edit point: its new place in plan units. */
  onEdit?: (index: number, placement: Placement) => void;
  /** Called right after each frame renders, while the canvas still holds it (video recording). */
  onRendered?: (canvas: HTMLCanvasElement) => void;
  recordingSize?: {width: number; height: number};
  quality?: SceneQuality;
  verification?: VerificationRef;
  suspended?: boolean;
  elevationScale?: number;
  /** Race-day weather: sky, light, haze, rain or snow, wet surfaces (app/raceWeather.ts). */
  weather?: Weather;
  /** Official going (1 良 ... 4 不良) for the infield board, when known. */
  going?: number | null;
  /** Turf race days on the present rail position: the inside wears as a meeting goes on (app/turfWear.ts). */
  courseDay?: number | null;
}) {
  const host = useRef<HTMLDivElement>(null);
  // The animation loop lives outside React's render cycle: it reads the latest props from refs.
  const state = useRef<LiveState>({progress, playing, speed, selected, followKey, cameraMode, plan, editAt, suspended});
  const select = useRef(onSelect);
  const edit = useRef(onEdit);
  const rendered = useRef(onRendered);
  const captureSize = useRef(recordingSize);
  useEffect(() => {
    state.current = {progress, playing, speed, selected, followKey, cameraMode, plan, editAt, suspended};
    select.current = onSelect;
    edit.current = onEdit;
    rendered.current = onRendered;
    captureSize.current = recordingSize;
  });
  const webgl = useSyncExternalStore(subscribeNever, hasWebGL, () => true);

  useEffect(() => {
    if (!webgl) return;
    const el = host.current!;
    let renderer: T.WebGLRenderer;
    try {
      renderer = createRaceRenderer(quality, !!verification);
    } catch {
      return;
    }
    el.appendChild(renderer.domElement);

    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(40, 1, 0.3, 6000);
    camera.position.set(0, 800, 800);
    scene.fog = new T.Fog(quality.clearColor, 500, 1600);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 12;
    controls.maxDistance = 900;
    controls.maxPolarAngle = Math.PI * 0.48;
    const sunOffset = raceSunOffset(race.date);
    const look = WEATHER_LOOKS[weather];
    const sun = addLights(scene, sunOffset, look);
    sun.shadow.mapSize.setScalar(quality.shadowSize);
    addRaceSky(scene, sunOffset, look);
    const environment = skyEnvironment(renderer, sunOffset, look);
    scene.environment = environment;
    scene.environmentIntensity = 0.32;

    // Course geometry and everything drawn on it.
    const tokyo = race.venueCode === '05';
    const track = makeTrack(race, undefined, elevationScale);
    const route = makeRoute(race, track);
    const straight = courseFor(race).straight;
    const courses = buildCourseSet(race, track, availableKinds(race));
    const terrain = addGround(scene, courses.tracks, route);
    // On soft ground the turf along the inside rail is cut up (ground and blades share the pattern).
    const wear = {amount: turfWearAmount(look, going, courseDay), wet: look.wet};
    const surface = addCourseSurfaces(scene, courses, terrain, tokyo, route, tokyo ? tokyoChutes(race) : [], wear);
    addRails(scene, courses, track.width, route, race.distance);
    const gate = addMarkers(scene, race, track, route);
    const floor = Math.min(...terrain.samples.map(s => s.y));
    addFarGround(scene, floor, look.snowCover);
    addRaceLandscape(scene, track, terrain, floor - 1, {
      day: dayOfYear(race.date), look,
      board: {
        title: race.name,
        lines: [`${race.venue} ${race.surface} ${race.distance}m`,
          `天候 ${WEATHER_LABEL[weather]}${going ? `　${race.surface === 'ダート' ? 'ダート' : '芝'} ${GOING_LABEL[going]}` : ''}`],
      },
    });
    // Blades of grass stand around the field when the race is run on turf (STYLE_GUIDE: the grass is grown blade by blade).
    const grass = track.kind === 'dirt' ? null : addRaceGrass(scene, route, track.width, surface, wear);
    const disposeDistantGrass = grass && quality.extendedGrass ? addDistantGrass(scene, track, surface, grass.uniforms, wear) : null;
    const gateBox = new T.Box3().setFromObject(gate);
    // Everything built so far is the course and its surroundings: wet it in the rain (not the horses, added next).
    wetSurfaces(scene, look.wet);
    const falling = addPrecipitation(scene, look);

    // Horses and click-to-select; in edit mode, drag a horse to move it.
    const horses = addHorses(scene, race, quality.pileLayers);
    const labels = addHorseLabels(scene, race, horses);
    const contact = addContactShadows(scene, horses);
    if (quality.fullAnisotropy) {
      const anisotropy = renderer.capabilities.getMaxAnisotropy();
      scene.traverse(object => {
        if (!(object instanceof T.Mesh || object instanceof T.Sprite)) return;
        for (const material of Array.isArray(object.material) ? object.material : [object.material])
          for (const value of Object.values(material)) if (value instanceof T.Texture) value.anisotropy = anisotropy;
      });
    }
    const pose = {at: horses.map(() => 0), kick: horses.map(() => NO_KICK)};

    const ring = addSelectionRing(scene);
    const ray = new T.Raycaster();
    const mouse = new T.Vector2();
    let down = [0, 0];
    const aim = (e: PointerEvent) => {
      const b = renderer.domElement.getBoundingClientRect();
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

    let sizeKey = '';
    let outputWidth = 1;
    let outputHeight = 1;
    const resize = (size?: {width: number; height: number}) => {
      const capture = size ?? captureSize.current;
      const width = capture?.width ?? Math.max(1, el.clientWidth);
      const height = capture?.height ?? Math.max(1, el.clientHeight);
      const pixelRatio = capture ? 1 : Math.min(devicePixelRatio, quality.maxPixelRatio);
      const key = `${width}:${height}:${pixelRatio}:${el.clientWidth}:${el.clientHeight}`;
      if (key === sizeKey) return;
      sizeKey = key;
      outputWidth = width;
      outputHeight = height;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(width, height, false);
      // Fit the recording preview without stretching it; pointer coordinates stay accurate.
      const scale = capture ? Math.min(el.clientWidth / width, el.clientHeight / height) : 1;
      renderer.domElement.style.width = `${width * scale}px`;
      renderer.domElement.style.height = `${height * scale}px`;
    };
    const observer = new ResizeObserver(() => resize());
    observer.observe(el);
    resize();

    const packCamera = createPackCamera(camera, controls, renderer.domElement);
    const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let verificationMotion: boolean | undefined;
    // Hoofprints and kicked-up sand on a dirt course.
    const trail = track.kind === 'dirt' ? addDirtTrail(scene, surface, () => verificationMotion ?? motionPreference.matches) : null;
    const timeline = createSceneTimeline(race, route, straight, gateBox, trail);
    let frame = 0;
    let previousTick = -1;
    let lastFollowKey = state.current.followKey;
    let lastMode = state.current.cameraMode;
    let previousTime = performance.now();
    let transition: {position: T.Vector3; target: T.Vector3; elapsed: number} | null = null;
    let lastInput: LiveState | undefined;
    let lastSize = '';
    let lastReduced: boolean | undefined;
    let lastDrag: Placement | undefined;
    let interactionChanged = true;
    const markInteraction = () => { interactionChanged = true; };
    controls.addEventListener('change', markInteraction);
    renderer.domElement.addEventListener('wheel', markInteraction);

    function update(input: LiveState, fixed?: FixedCamera, verificationFrame = false, prepare = true, outputTime?: number) {
      const {progress, speed, plan: live, selected, cameraMode, followKey, playing, editAt} = input;
      const travel = progress * race.distance;
      const seconds = outputSeconds(travel, speed);
      const tick = completeTick(seconds);
      const reduced = verificationMotion ?? motionPreference.matches;
      const canonical = timeline.advanceTo(tick, {speed, plan: live, aspect: camera.aspect, mode: cameraMode, reduced});
      if (!prepare) return;
      const editing = !verificationFrame && (drag || (editAt !== null && !playing));
      let placements: Placement[];
      if (editing) {
        const raw = interpolateLayouts(live.layouts, travel, live.knots);
        if (drag) raw[drag.index] = drag.now;
        placements = separateLayouts(raw);
      } else placements = sampleLayouts(live.layouts, travel, live.knots);
      placeHorses(horses, race, route, ring, placements, travel, selected, reduced, live.kicks, pose);

      if (fixed) {
        camera.position.fromArray(fixed.position);
        camera.lookAt(new T.Vector3().fromArray(fixed.target));
      } else if (verificationFrame || (packCamera.automatic && followKey === lastFollowKey && cameraMode === lastMode && !transition)) {
        camera.position.copy(canonical.position);
        camera.lookAt(canonical.target);
        if (!verificationFrame) {
          controls.target.copy(canonical.target);
          // Keep the interaction camera's wheel anchor current.
          packCamera.update({
            dt: 0, progress, followKey, mode: cameraMode,
            ...cameraFrame(route, race.distance, straight, travel, placements, horses.map(h => h.position)),
          });
          camera.position.copy(canonical.position);
          camera.lookAt(canonical.target);
        }
      } else {
        const changed = followKey !== lastFollowKey || cameraMode !== lastMode;
        packCamera.update({
          dt: !changed && playing ? Math.max(0, tick - previousTick) / RACE_FPS : 0, progress, followKey, mode: cameraMode,
          ...cameraFrame(route, race.distance, straight, travel, placements, horses.map(h => h.position)),
        });
        controls.update();
      }
      if (!verificationFrame && !fixed) {
        if (followKey !== lastFollowKey || cameraMode !== lastMode) {
          transition = {
            position: camera.position.clone().sub(canonical.position),
            target: controls.target.clone().sub(canonical.target), elapsed: 0,
          };
        }
        if (transition && packCamera.automatic) {
          const t = Math.min(1, transition.elapsed / 2.6);
          const remaining = 1 - t * t * t * (t * (t * 6 - 15) + 10);
          camera.position.copy(canonical.position).addScaledVector(transition.position, remaining);
          controls.target.copy(canonical.target).addScaledVector(transition.target, remaining);
          camera.lookAt(controls.target);
          if (t === 1) transition = null;
        } else if (!packCamera.automatic) transition = null;
        if (packCamera.automatic) {
          controls.maxPolarAngle = canonical.maxPolarAngle;
          controls.maxDistance = canonical.maxDistance;
        }
        lastFollowKey = followKey;
        lastMode = cameraMode;
      }

      const packCenter = new T.Box3().setFromPoints(horses.map(g => g.position)).getCenter(new T.Vector3());
      const gaps = placements.map(v => v.gap);
      labels.update(camera, outputWidth, outputHeight, travel, gaps, selected);
      gate.visible = canonical.gate;
      if (grass) {
        grass.uniforms.uTime.value = reduced ? 0 : outputTime ?? seconds;
        const ticks = verificationFrame ? 0 : Math.min(4, Math.max(0, tick - previousTick));
        grass.update(travel - Math.max(...gaps), travel - Math.min(...gaps),
          camera.position.distanceTo(packCenter) < GRASS_SIGHT, ticks);
      }
      trail?.setTime(travel, pose.at);
      falling?.update(camera, reduced ? 0 : outputTime ?? seconds, outputHeight * renderer.getPixelRatio());
      contact.place();
      // Keep the shadow map on the field: its texel size follows the pack's spread, not a fixed 220m.
      const shadowHalf = T.MathUtils.clamp(Math.max(...gaps) / 2 - Math.min(...gaps) / 2 + 55, 75, 110);
      if (sun.shadow.camera.right !== shadowHalf) {
        const shadowCamera = sun.shadow.camera;
        shadowCamera.left = -shadowHalf; shadowCamera.bottom = -shadowHalf;
        shadowCamera.right = shadowHalf; shadowCamera.top = shadowHalf;
        shadowCamera.updateProjectionMatrix();
      }
      sun.target.position.copy(packCenter);
      sun.position.copy(packCenter).add(sunOffset);
      if (!verificationFrame) previousTick = tick;
    }

    // A very faint darkening toward the corners, drawn over the finished picture.
    const vignette = {
      scene: new T.Scene(),
      camera: new T.OrthographicCamera(-1, 1, 1, -1, 0, 1),
      geometry: new T.PlaneGeometry(2, 2),
      material: new T.ShaderMaterial({
        transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
        vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
        fragmentShader: `varying vec2 vUv;
          void main() {
            vec2 d = (vUv - 0.5) * vec2(1.0, 0.9);
            gl_FragColor = vec4(0.04, 0.06, 0.05, 0.16 * smoothstep(0.28, 0.78, length(d)));
          }`,
      }),
    };
    vignette.scene.add(new T.Mesh(vignette.geometry, vignette.material));
    const render = () => {
      renderer.render(scene, camera);
      const autoClear = renderer.autoClear;
      renderer.autoClear = false;
      renderer.render(vignette.scene, vignette.camera);
      renderer.autoClear = autoClear;
    };
    // Read the single-sample framebuffer directly. Chrome's multisample resolve can
    // round individual edge pixels differently even with identical scene state.
    const readback = verification ? new T.WebGLRenderTarget(1, 1, {
      colorSpace: quality.colorSpace, internalFormat: 'RGBA8',
    }) : null;
    // Three r180 otherwise treats render targets as linear intermediate passes. Select
    // its display-target path so the same material-level tone mapping applies as on screen.
    if (readback) Object.assign(readback, {isXRRenderTarget: true});
    const downsample = verification && quality.renderScale > 1 ? {
      scene: new T.Scene(),
      camera: new T.OrthographicCamera(-1, 1, 1, -1, 0, 1),
      geometry: new T.PlaneGeometry(2, 2),
      target: new T.WebGLRenderTarget(1, 1, {depthBuffer: false, colorSpace: T.LinearSRGBColorSpace}),
      material: new T.RawShaderMaterial({
        uniforms: {picture: {value: readback!.texture}},
        vertexShader: `precision highp float;
          attribute vec3 position; attribute vec2 uv; varying vec2 vUv;
          void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
        fragmentShader: `precision highp float;
          uniform sampler2D picture; varying vec2 vUv;
          void main() { gl_FragColor = texture2D(picture, vUv); }`,
        toneMapped: false, depthTest: false,
      }),
    } : null;
    if (downsample) downsample.scene.add(new T.Mesh(downsample.geometry, downsample.material));
    function draw(now = performance.now()) {
      frame = requestAnimationFrame(draw);
      if (state.current.suspended) {
        previousTime = now;
        return;
      }
      resize();
      const input = state.current;
      if (input === lastInput && sizeKey === lastSize && lastReduced === motionPreference.matches &&
        lastDrag === drag?.now && !transition && !interactionChanged && !rendered.current &&
        packCamera.automatic) return;
      if (transition) transition.elapsed += Math.min((now - previousTime) / 1000, 0.1);
      previousTime = now;
      interactionChanged = false;
      update(input);
      lastInput = input;
      lastSize = sizeKey;
      lastReduced = motionPreference.matches;
      lastDrag = drag?.now;
      render();
      rendered.current?.(renderer.domElement);
    }

    let firstCapture = true;
    const copy = verification ? document.createElement('canvas') : null;
    let pixels: Uint8Array<ArrayBuffer>;
    let outputPixels: ImageData;
    if (verification) verification.current = async ({progress, cameraMode, selected, speed, width, height, reduced, fixed, history, outputTime, sequential}) => {
      resize({width, height});
      verificationMotion = reduced;
      try {
        if (!sequential || firstCapture) timeline.reset();
        const input = {...state.current, progress, cameraMode, selected, speed, playing: false};
        for (const at of history ?? []) update({...input, progress: at}, fixed, true, false);
        update(input, fixed, true, true, outputTime);
        const scale = quality.renderScale;
        readback!.setSize(width * scale, height * scale);
        renderer.setRenderTarget(readback);
        if (firstCapture) {
          await renderer.compileAsync(scene, camera);
          render();
          renderer.getContext().finish();
          firstCapture = false;
        }
        render();
        if (downsample) {
          // The high-resolution target contains display-space pixels. Average them once.
          if (downsample.target.width !== width || downsample.target.height !== height)
            downsample.target.setSize(width, height);
          renderer.setRenderTarget(downsample.target);
          renderer.render(downsample.scene, downsample.camera);
        }
        const row = width * 4;
        if (!outputPixels || outputPixels.width !== width || outputPixels.height !== height) {
          pixels = new Uint8Array(row * height);
          outputPixels = new ImageData(width, height);
        }
        renderer.readRenderTargetPixels(downsample?.target ?? readback!, 0, 0, width, height, pixels!);
        for (let y = 0; y < height; y++) {
          const offset = (height - 1 - y) * row;
          outputPixels.data.set(pixels!.subarray(offset, offset + row), y * row);
        }
        if (copy!.width !== width) copy!.width = width;
        if (copy!.height !== height) copy!.height = height;
        copy!.getContext('2d')!.putImageData(outputPixels, 0, 0);
        return copy!;
      } finally {
        renderer.setRenderTarget(null);
        verificationMotion = undefined;
      }
    };
    if (!verification) draw();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      if (verification) verification.current = null;
      timeline.dispose();
      packCamera.dispose();
      controls.removeEventListener('change', markInteraction);
      renderer.domElement.removeEventListener('wheel', markInteraction);
      controls.dispose();
      grass?.dispose();
      disposeDistantGrass?.();
      falling?.dispose();
      trail?.dispose();
      labels.dispose();
      el.removeEventListener('pointerdown', onPointerDown, true);
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerup', onPointerUp);
      el.removeEventListener('pointercancel', onPointerCancel);
      sun.shadow.dispose();
      environment.dispose();
      vignette.geometry.dispose();
      vignette.material.dispose();
      readback?.dispose();
      downsample?.geometry.dispose();
      downsample?.material.dispose();
      downsample?.target.dispose();
      disposeScene(scene);
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
    // The plan is read live from state, so a hand edit does not rebuild the scene or move the camera.
  }, [race, webgl, quality, verification, elevationScale, weather, going, courseDay]);

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
