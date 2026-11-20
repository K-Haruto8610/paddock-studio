import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {TrackPoint} from './trackGeometry';
import {createShotTransition, fitRaceShot, type CameraMode, type RaceShot} from './raceCamera';

/** Where the leading horse sits by default: this fraction of the way from the screen centre to the edge. */
const LEADER_AT_EDGE = 0.52;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 3;

/**
 * The camera follows the race so the leader is always on screen: a compact field is centred,
 * a strung-out field is framed from the leader backwards with the leader near the screen edge.
 * The wheel zooms towards the cursor while following. Dragging takes manual control until the
 * "follow the pack" button bumps followKey (which also resets zoom and the cursor offset).
 */
export function createPackCamera(
  camera: T.PerspectiveCamera,
  controls: OrbitControls,
  canvas?: HTMLCanvasElement,
) {
  let following = true;
  let lastFollowKey = -1;
  let first = true;
  let zoom = 1;
  let adjusted = false;
  let dragging = false;
  let lastProgress = Number.NaN;
  let lastMode: CameraMode = 'pack';
  // Cursor-zoom offset of the target from the automatic frame, along the course (tangent, outward).
  const offset = new T.Vector2();
  // Last automatic frame and target, used to turn a wheel step into an offset.
  let frame: TrackPoint | null = null;
  const desired = new T.Vector3();
  const ray = new T.Raycaster();
  const transition = createShotTransition();

  const onControlStart = () => {
    if (dragging) following = false;
  };
  const onDown = () => {
    dragging = true;
  };
  const onUp = () => {
    dragging = false;
  };
  const onWheel = (e: WheelEvent) => {
    if (!canvas || !following || !frame) return;
    e.preventDefault();
    adjusted = true;
    const next = T.MathUtils.clamp(zoom * Math.exp(e.deltaY * 0.001), MIN_ZOOM, MAX_ZOOM);
    const r = next / zoom;
    zoom = next;
    // Scaling the whole view about the point under the cursor keeps that point still on screen.
    const b = canvas.getBoundingClientRect();
    ray.setFromCamera(
      new T.Vector2(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1),
      camera,
    );
    const ground = new T.Plane(new T.Vector3(0, 1, 0), -desired.y);
    const cursor = ray.ray.intersectPlane(ground, new T.Vector3());
    if (!cursor) return;
    const moved = cursor.clone().add(desired.clone().sub(cursor).multiplyScalar(r));
    const auto = desired.clone().sub(fromOffset(frame));
    const shift = moved.sub(auto);
    offset.set(shift.dot(frame.tangent), shift.dot(frame.outward));
  };
  const fromOffset = (f: TrackPoint) =>
    f.tangent.clone().multiplyScalar(offset.x).addScaledVector(f.outward, offset.y);
  if (canvas) {
    controls.addEventListener('start', onControlStart);
    canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    canvas.addEventListener('wheel', onWheel, {passive: false});
  }
  controls.zoomToCursor = true;

  return {
    get automatic() {
      return following && !adjusted;
    },
    /** Moves the camera towards its frame for this frame (or leaves it alone when dragged away). */
    update(opts: {
      dt: number;
      progress: number;
      followKey: number;
      mode: CameraMode;
      shot: RaceShot;
      /** Horse positions ordered by current distance, leader first. */
      horses: T.Vector3[];
      leader: TrackPoint;
      /** Length of the field from leader to last horse, in metres. */
      spread: number;
      /** A course point this many metres behind the leader, on the field's mean lane. */
      behindLeader: (metres: number) => TrackPoint;
    }) {
      const {dt, progress, followKey, spread, behindLeader, mode, shot, horses, leader} = opts;
      if (followKey !== lastFollowKey || mode !== lastMode) {
        following = true;
        lastMode = mode;
        lastFollowKey = followKey;
        zoom = 1;
        adjusted = false;
        offset.set(0, 0);
      }
      if (Number.isNaN(lastProgress)) lastProgress = progress;
      controls.enableZoom = !following;
      const aspect = Math.max(camera.aspect, 0.7);
      // Fit the field, but never pull back so far that the horses become specks.
      // A field spread more across the course than along it (the start, the first bend) hides its
      // horses behind one another when seen from the side: it is then seen from behind instead.
      let across = 0;
      if (horses.length) {
        let low = Infinity,
          high = -Infinity;
        for (const h of horses) {
          const y = h.dot(leader.outward);
          low = Math.min(low, y);
          high = Math.max(high, y);
        }
        across = high - low;
      }
      const behind = T.MathUtils.smoothstep(across - spread, 2, 14);
      const fitted = T.MathUtils.clamp((Math.max(spread, behind * across) + 12) / (aspect * 0.8), 30, 76);
      const distance = fitted * zoom * (1 - 0.25 * behind);
      // The frame is measured unzoomed, so zooming only scales the view about the cursor.
      const halfWidth = fitted * Math.tan(T.MathUtils.degToRad(camera.fov / 2)) * aspect;
      const back = Math.min(spread / 2, halfWidth * LEADER_AT_EDGE);
      frame = behindLeader(back);
      // Back at full width, let a cursor offset drift home so the leader cannot be lost.
      if (zoom >= 0.999) offset.multiplyScalar(Math.exp(-dt * 1.5));
      const maxOffset = halfWidth * 1.2;
      if (offset.length() > maxOffset) offset.setLength(maxOffset);
      desired.copy(frame.point).add(fromOffset(frame));
      let packOffset = frame.outward
        .clone()
        .multiplyScalar(distance)
        .multiplyScalar(1 - 0.85 * behind)
        .addScaledVector(frame.tangent, distance * T.MathUtils.lerp(0.12, -1, behind));
      packOffset.y = distance * T.MathUtils.lerp(0.19, 0.34, 1 - T.MathUtils.smoothstep(progress, 0.001, 0.08)) * (1 + 0.2 * behind);
      controls.maxPolarAngle = mode === 'broadcast' ? Math.PI * 0.55 : Math.PI * 0.49;
      controls.maxDistance = mode === 'broadcast' ? 5000 : 900;
      if (mode === 'broadcast' && horses.length) {
        frame = leader;
        if (shot === 'corner') {
          desired.copy(horses[0]).add(new T.Vector3(0, 3.2, 0));
          packOffset = leader.outward.clone().multiplyScalar(18)
            .addScaledVector(leader.tangent, 9).setY(-1.8);
        } else {
          const fittedShot = fitRaceShot(shot === 'finish' ? horses.slice(0, 6) : horses,
            leader.outward, shot, camera.fov, camera.aspect);
          desired.copy(fittedShot.target);
          packOffset = fittedShot.offset;
        }
        desired.add(fromOffset(frame));
        packOffset.multiplyScalar(zoom);
      }
      if (following) {
        const snap = first || Math.abs(progress - lastProgress) > 0.03;
        const blended = transition(`${mode}:${mode === 'broadcast' ? shot : 'pack'}:${followKey}`,
          dt, snap, leader, controls.target, camera.position, desired, desired.clone().add(packOffset));
        if (blended.active) controls.maxPolarAngle = Math.PI * 0.55;
        const alpha = snap ? 1 : 1 - Math.exp(-dt * 6);
        first = false;
        controls.target.lerp(blended.target, alpha);
        camera.position.lerp(blended.position, alpha);
      }
      lastProgress = progress;
    },
    dispose() {
      controls.removeEventListener('start', onControlStart);
      if (canvas) {
        canvas.removeEventListener('pointerdown', onDown);
        window.removeEventListener('pointerup', onUp);
        canvas.removeEventListener('wheel', onWheel);
      }
    },
  };
}
