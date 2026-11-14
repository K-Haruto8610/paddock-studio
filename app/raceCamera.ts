import * as T from 'three';
import type {RaceRoute} from './raceRoutes';
import type {TrackPoint} from './trackGeometry';

export type CameraMode = 'pack' | 'broadcast';
export type RaceShot = 'start' | 'corner' | 'wide' | 'finish';

/** Blend in the moving leader's frame, so a slow camera change does not leave the horses behind. */
export function createShotTransition() {
  let previousKey = '';
  let elapsed = 2.6;
  const fromTarget = new T.Vector3();
  const fromCamera = new T.Vector3();
  return (key: string, dt: number, snap: boolean, leader: TrackPoint,
    currentTarget: T.Vector3, currentCamera: T.Vector3, target: T.Vector3, position: T.Vector3) => {
    const tangent = leader.tangent.clone().setY(0).normalize();
    const outward = leader.outward.clone().setY(0).normalize();
    const local = (point: T.Vector3) => {
      const delta = point.clone().sub(leader.point);
      return new T.Vector3(delta.dot(tangent), delta.y, delta.dot(outward));
    };
    const world = (point: T.Vector3) => leader.point.clone().addScaledVector(tangent, point.x)
      .addScaledVector(outward, point.z).add(new T.Vector3(0, point.y, 0));
    if (snap) elapsed = 2.6;
    else if (key !== previousKey) {
      fromTarget.copy(local(currentTarget));
      fromCamera.copy(local(currentCamera));
      elapsed = 0;
    }
    previousKey = key;
    elapsed = Math.min(2.6, elapsed + dt);
    const t = elapsed / 2.6;
    // Smootherstep starts and ends with zero velocity and acceleration.
    const blend = t * t * t * (t * (t * 6 - 15) + 10);
    return {
      target: world(fromTarget).lerp(target, blend),
      position: world(fromCamera).lerp(position, blend),
      active: elapsed < 2.6,
    };
  };
}

/** Use the route's heading so short races, chutes and earlier laps follow their actual course. */
export function raceShot(route: RaceRoute, travel: number, distance: number, straight: number): RaceShot {
  if (travel < Math.min(100, distance * 0.08)) return 'start';
  if (travel >= distance - straight) return 'finish';
  const heading = route.sample(Math.min(distance, travel + 20)).point.clone()
    .sub(route.sample(Math.max(0, travel - 20)).point).setY(0).normalize();
  const home = route.sample(distance).tangent.clone().setY(0).normalize();
  return Math.abs(heading.dot(home)) < 0.88 ? 'corner' : 'wide';
}

/** Fit horse-sized bounds in camera space, including lane width and exaggerated terrain heights. */
export function fitRaceShot(points: T.Vector3[], outward: T.Vector3, shot: RaceShot, fov: number, aspect: number) {
  const bounds = new T.Box3().setFromPoints(points);
  const target = bounds.getCenter(new T.Vector3()).add(new T.Vector3(0, 2.5, 0));
  const direction = outward.clone().setY(shot === 'start' ? 1.8 : shot === 'finish' ? 0.22 : 0.48).normalize();
  const right = new T.Vector3(0, 1, 0).cross(direction).normalize();
  const up = direction.clone().cross(right).normalize();
  const tanY = Math.tan(T.MathUtils.degToRad(fov / 2)) * (shot === 'finish' ? 0.92 : 0.82);
  const tanX = tanY * Math.max(0.1, aspect);
  let distance = shot === 'start' ? 45 : shot === 'finish' ? 16 : 22;
  for (const point of points) {
    for (const x of [-4, 4]) for (const y of [0, 6]) for (const z of [-4, 4]) {
      const p = point.clone().add(new T.Vector3(x, y, z)).sub(target);
      distance = Math.max(distance, p.dot(direction) + Math.max(Math.abs(p.dot(right)) / tanX, Math.abs(p.dot(up)) / tanY));
    }
  }
  return {target, offset: direction.multiplyScalar(distance)};
}
