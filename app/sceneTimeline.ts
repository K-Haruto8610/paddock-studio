import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Race} from './types';
import type {RaceRoute} from './raceRoutes';
import type {PlaybackPlan} from './playback';
import {sampleLayouts} from './raceLayout';
import {kickAt, NO_KICK} from './closingKick';
import {createPackCamera} from './packCamera';
import {raceShot, type CameraMode} from './raceCamera';
import type {DirtTrail} from './dirtTrail';
import {FRAME_SECONDS, progressAtTime} from './raceClock';

const TRAIL_CHECKPOINT_INTERVAL = 180;

export function cameraFrame(route: RaceRoute, distance: number, straight: number, travel: number,
  placements: {gap: number; lane: number}[], points: T.Vector3[]) {
  const order = placements.map((_, i) => i).sort((a, b) => placements[a].gap - placements[b].gap);
  const first = placements[order[0]];
  const rearGap = placements[order[order.length - 1]].gap;
  const leaderAt = Math.max(0, travel - first.gap);
  const meanLane = placements.reduce((sum, v) => sum + v.lane, 0) / placements.length;
  return {
    shot: raceShot(route, leaderAt, distance, straight),
    horses: order.map((i) => points[i]),
    leader: route.sample(leaderAt, first.lane),
    spread: rearGap - first.gap,
    toFinish: distance - leaderAt,
    behindLeader: (metres: number) => route.sample(Math.max(0, leaderAt - metres), meanLane),
  };
}

type TimelineOptions = {
  plan: PlaybackPlan;
  speed: number;
  aspect: number;
  mode: CameraMode;
  reduced: boolean;
};

export function createSceneTimeline(race: Race, route: RaceRoute, straight: number,
  gateBox: T.Box3, trail: DirtTrail | null) {
  const camera = new T.PerspectiveCamera(40, 1, 0.3, 6000);
  const controls = new OrbitControls(camera);
  const view = new T.Matrix4();
  const frustum = new T.Frustum();
  const field = {
    horses: race.horses.map(() => ({position: new T.Vector3(), rotation: new T.Euler()})),
    at: race.horses.map(() => 0),
    kick: race.horses.map(() => NO_KICK),
  };
  let pack = createPackCamera(camera, controls);
  let options: TimelineOptions | undefined;
  let lastTick = -1;
  let gateVisible = true;
  // Retain camera poses for constant-time backwards seeks. Dirt is replayed below and bounded by its GPU buffers.
  const cameras: {position: T.Vector3; target: T.Vector3; gate: boolean; maxPolarAngle: number; maxDistance: number}[] = [];
  let trailTick = -1;
  const trailStates = new Map<number, ReturnType<DirtTrail['save']>>();

  const reset = () => {
    pack.dispose();
    camera.position.set(0, 800, 800);
    controls.target.set(0, 0, 0);
    pack = createPackCamera(camera, controls);
    lastTick = -1;
    cameras.length = 0;
    gateVisible = true;
    trailTick = -1;
    trailStates.clear();
    trail?.clear();
  };
  return {
    reset,
    advanceTo(tick: number, next: TimelineOptions) {
      if (!options || options.plan !== next.plan || options.speed !== next.speed ||
        options.aspect !== next.aspect || options.mode !== next.mode || options.reduced !== next.reduced) {
        reset();
        options = next;
        camera.aspect = next.aspect;
        camera.updateProjectionMatrix();
      }
      if (trail) {
        const checkpointTick = tick - tick % TRAIL_CHECKPOINT_INTERVAL;
        const checkpoint = trailStates.get(checkpointTick);
        if (checkpoint && (tick < trailTick || checkpointTick > trailTick)) {
          trail.restore(checkpoint);
          trailTick = checkpointTick;
        } else if (tick < trailTick) {
          trail.clear();
          trailTick = -1;
        }
      }
      const from = trail ? Math.min(lastTick, trailTick) : lastTick;
      for (let k = from + 1; k <= tick; k++) {
        const progress = progressAtTime(k * FRAME_SECONDS, race.distance, next.speed);
        const travel = progress * race.distance;
        const placements = sampleLayouts(next.plan.layouts, travel, next.plan.knots);
        placements.forEach(({gap, lane}, i) => {
          const at = Math.max(0, travel - gap);
          const here = route.sample(at, lane);
          field.horses[i].position.copy(here.point);
          field.horses[i].rotation.y = -Math.atan2(here.tangent.z, here.tangent.x);
          field.at[i] = at;
          field.kick[i] = next.reduced ? NO_KICK : kickAt(at, next.plan.kicks[i]);
        });
        if (k > lastTick) {
          pack.update({
            dt: k === 0 ? 0 : FRAME_SECONDS,
            progress,
            followKey: 0,
            mode: next.mode,
            ...cameraFrame(route, race.distance, straight, travel, placements, field.horses.map(h => h.position)),
          });
          controls.update();
          if (gateVisible && travel >= 1) {
            camera.updateMatrixWorld();
            frustum.setFromProjectionMatrix(view.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
            gateVisible = frustum.intersectsBox(gateBox);
          }
          cameras.push({
            position: camera.position.clone(), target: controls.target.clone(), gate: gateVisible,
            maxPolarAngle: controls.maxPolarAngle, maxDistance: controls.maxDistance,
          });
          lastTick = k;
        }
        if (trail && k > trailTick) {
          trail.update(field, travel, false, next.plan);
          trailTick = k;
          if (k % TRAIL_CHECKPOINT_INTERVAL === 0 && !trailStates.has(k)) trailStates.set(k, trail.save());
        }
      }
      return cameras[tick];
    },
    dispose() {
      pack.dispose();
      cameras.length = 0;
      trailStates.clear();
    },
  };
}
