import * as T from 'three';
import type {Race} from './types';
import {makeTrack, makeTerrain, type TrackModel} from './trackGeometry';
import type {RaceRoute} from './raceRoutes';
import {startLanes} from './raceLayout';
import {plush, label} from './plushHorse';

// The plush model is 4 units long: scale 1.0 draws a ~4m horse (≈1.7x life size)
// so the field stays legible against the true-scale course. Never enlarged on selection.
export const HORSE_SCALE = 1.0;

export type Terrain = ReturnType<typeof makeTerrain>;

function material(color: string, lift = 0) {
  return new T.MeshStandardMaterial({
    color,
    roughness: 0.96,
    side: T.DoubleSide,
    polygonOffset: lift !== 0,
    polygonOffsetFactor: -lift,
    polygonOffsetUnits: -lift,
  });
}

export function mesh(geometry: T.BufferGeometry, color: string, lift = 0) {
  return new T.Mesh(geometry, material(color, lift));
}

/** Flat strip between two lanes of a track, following its (exaggerated) height. */
function ribbon(track: TrackModel, inner: number, outer: number, color: string, lift = 1) {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= 1024; i++) {
    for (const lane of [inner, outer]) {
      const p = track.sample((i / 1024) * track.lap, lane).point;
      positions.push(p.x, p.y + 0.02, p.z);
    }
    if (i < 1024) {
      const k = i * 2;
      indices.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
  }
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const m = mesh(geo, color, lift);
  m.receiveShadow = true;
  return m;
}

/** White rail: a tube along the lane plus a post every ~8m. */
function rail(track: TrackModel, lane: number) {
  const group = new T.Group();
  const points: T.Vector3[] = [];
  for (let i = 0; i <= 600; i++) {
    const p = track.sample((i / 600) * track.lap, lane).point;
    p.y += 1.1;
    points.push(p);
  }
  group.add(
    mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points, true), 1200, 0.1, 5, true), '#f2f1de'),
  );
  const postGeo = new T.CylinderGeometry(0.075, 0.075, 1.2, 5);
  const postMat = new T.MeshStandardMaterial({color: '#eaeedc'});
  const count = Math.ceil(track.lap / 8);
  const posts = new T.InstancedMesh(postGeo, postMat, count);
  const dummy = new T.Object3D();
  for (let i = 0; i < count; i++) {
    dummy.position.copy(track.sample((i / count) * track.lap, lane).point);
    dummy.position.y += 0.55;
    dummy.updateMatrix();
    posts.setMatrixAt(i, dummy.matrix);
  }
  group.add(posts);
  return group;
}

export type CourseSet = {
  tracks: TrackModel[];
  turf: TrackModel;
  dirt?: TrackModel;
  outer?: TrackModel;
};

/** The set of courses drawn for a race: its own track plus the venue's other GⅠ courses. */
export function buildCourseSet(
  race: Race,
  track: TrackModel,
  kinds: TrackModel['kind'][],
): CourseSet {
  const tracks = kinds.map((k) => (k === track.kind ? track : makeTrack(race, k)));
  return {
    tracks,
    turf: tracks.find((t) => t.kind === 'turf')!,
    dirt: tracks.find((t) => t.kind === 'dirt'),
    outer: tracks.find((t) => t.kind === 'outer'),
  };
}

export function addLights(scene: T.Scene) {
  scene.add(new T.HemisphereLight('#fffbed', '#809177', 2));
  const sun = new T.DirectionalLight('#fff6e6', 2);
  sun.position.set(-200, 500, 180);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -110;
  sun.shadow.camera.right = 110;
  sun.shadow.camera.top = 110;
  sun.shadow.camera.bottom = -110;
  sun.shadow.camera.far = 1800;
  sun.shadow.bias = -0.0003;
  scene.add(sun);
  scene.add(sun.target);
  return sun;
}

/**
 * Ground follows the (exaggerated) relief of every track and stays below each surface.
 * Returns the terrain model (used to drape extra surfaces onto the ground).
 */
export function addGround(scene: T.Scene, tracks: TrackModel[], route?: RaceRoute): Terrain {
  // The ground also stays below the race's chute (引込線), which is not part of any lap.
  const chute: T.Vector3[] = [];
  const surface = route?.chuteSurface;
  if (surface && tracks[0]) {
    for (let i = 0; i <= 48; i++)
      for (const lane of [-4, 0, tracks[0].width / 2, tracks[0].width, tracks[0].width + 4])
        chute.push(surface.sample(i / 48, lane).point);
  }
  const terrain = makeTerrain(tracks, chute);
  const box = new T.Box3()
    .setFromPoints(terrain.samples.map((s) => new T.Vector3(s.x, 0, s.z)))
    .expandByScalar(160);
  const size = box.getSize(new T.Vector3());
  const cell = 5;
  const nx = Math.ceil(size.x / cell);
  const nz = Math.ceil(size.z / cell);
  const ground = new T.PlaneGeometry(size.x, size.z, nx, nz);
  ground.rotateX(-Math.PI / 2);
  ground.translate(box.min.x + size.x / 2, 0, box.min.z + size.z / 2);
  const gp = ground.attributes.position as T.BufferAttribute;
  for (let i = 0; i < gp.count; i++) gp.setY(i, terrain.heightAt(gp.getX(i), gp.getZ(i)));
  ground.computeVertexNormals();
  const groundMesh = mesh(ground, '#cbd4bb');
  groundMesh.receiveShadow = true;
  scene.add(groundMesh);
  return terrain;
}

/** Polygon given in plan pixels, draped onto the terrain `lift` metres above the ground. */
function drape(
  turf: TrackModel,
  terrain: Terrain,
  points: number[][],
  color: string,
  lift: number,
) {
  const shape = new T.Shape();
  points.forEach(([x, z], i) => {
    const p = turf.pixel(x, z);
    if (i === 0) shape.moveTo(p.x, -p.z);
    else shape.lineTo(p.x, -p.z);
  });
  shape.closePath();
  const geo = new T.ShapeGeometry(shape);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as T.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, terrain.heightAt(pos.getX(i), pos.getZ(i)) + lift);
  }
  geo.computeVertexNormals();
  const m = mesh(geo, color, 1);
  m.receiveShadow = true;
  return m;
}

// Tokyo plan-pixel polygons (same pixel space as the digitized rails).
const TOKYO_CHUTE_A = [[425, 49], [457, 51], [487, 63], [513, 85], [535, 116], [548, 147], [551, 173], [549, 238], [532, 242], [533, 164], [520, 133], [500, 106], [474, 85], [443, 72], [420, 68]]; // prettier-ignore
const TOKYO_CHUTE_B = [[427, 71], [451, 79], [480, 96], [508, 121], [533, 145], [533, 164], [508, 143], [483, 121], [456, 101], [432, 88]]; // prettier-ignore
const TOKYO_CHUTE_C = [[367, 69], [414, 68], [455, 73], [490, 84], [479, 94], [440, 84], [401, 82], [366, 83]]; // prettier-ignore
const TOKYO_APRON_A = [[229, 317], [460, 298], [485, 327], [462, 352], [410, 356], [268, 356], [230, 347]]; // prettier-ignore
const TOKYO_APRON_B = [
  [160, 322],
  [225, 316],
  [230, 354],
  [171, 361],
];

/** Strip along a chute (引込線) from its start to the main rail, over its own flat height. */
function chuteRibbon(surface: NonNullable<RaceRoute['chuteSurface']>, width: number, color: string) {
  const positions: number[] = [];
  const indices: number[] = [];
  const N = 96;
  for (let i = 0; i <= N; i++) {
    for (const lane of [0, width]) {
      const p = surface.sample(i / N, lane).point;
      positions.push(p.x, p.y + 0.02, p.z);
    }
    if (i < N) {
      const k = i * 2;
      indices.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
  }
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const m = mesh(geo, color, 2);
  m.receiveShadow = true;
  return m;
}

/** Turf/dirt course surfaces, the race's chute, plus Tokyo's chutes and the dirt-start apron. */
export function addCourseSurfaces(
  scene: T.Scene,
  courses: CourseSet,
  terrain: Terrain,
  tokyo: boolean,
  route?: RaceRoute,
) {
  const {turf, dirt, outer} = courses;
  // Outer course first (lowest priority), then the inner turf, then dirt on top.
  if (outer) scene.add(ribbon(outer, 0, outer.width, '#7faf66', 1));
  scene.add(ribbon(turf, 0, turf.width, '#7faf66', 2));
  if (dirt) scene.add(ribbon(dirt, 0, dirt.width, '#ceb28a', 3));
  if (route?.chuteSurface) scene.add(chuteRibbon(route.chuteSurface, turf.width, '#7faf66'));
  if (tokyo && dirt) {
    const add = (points: number[][], color: string, lift: number) =>
      scene.add(drape(turf, terrain, points, color, lift));
    // Turf chutes from the main plan, draped onto the ground instead of floating flat.
    add(TOKYO_CHUTE_A, '#7faf66', 0.3);
    add(TOKYO_CHUTE_B, '#7faf66', 0.3);
    add(TOKYO_CHUTE_C, '#7faf66', 0.3);
    add(TOKYO_APRON_A, '#bdc4be', 5);
    add(TOKYO_APRON_B, '#bfc7c0', 4);
  }
}

/** Rails of the course this race is run on (plus the turf rail when racing on another surface). */
export function addRails(scene: T.Scene, track: TrackModel, turf: TrackModel) {
  scene.add(rail(track, 0));
  scene.add(rail(track, track.width));
  if (track !== turf) scene.add(rail(turf, turf.width));
}

/** Finish line, GOAL/START labels, remaining-distance boards and the starting gate stalls. */
export function addMarkers(scene: T.Scene, race: Race, track: TrackModel, route: RaceRoute) {
  const mark = (text: string, at: T.Vector3, w: number) => {
    const s = label(text, 256);
    s.position.copy(at);
    s.position.y += w / 4 + 1.5;
    s.scale.set(w, w / 4, 1);
    scene.add(s);
    return s;
  };
  const finish = track.sample(0);
  const finishLine = mesh(new T.PlaneGeometry(0.7, track.width), '#fff9df', 4);
  finishLine.rotation.x = -Math.PI / 2;
  finishLine.rotation.z = Math.atan2(finish.tangent.z, finish.tangent.x);
  finishLine.position.copy(track.sample(0, track.width / 2).point);
  finishLine.position.y += 0.1;
  scene.add(finishLine);
  mark('GOAL', track.sample(0, track.width + 8).point, 16);
  // Remaining-distance boards along the last lap, sized to be readable from the pack camera.
  for (let remaining = 200; remaining < track.lap; remaining += 200) {
    mark(`${remaining}m`, track.sample(track.lap - remaining, track.width + 6).point, 11);
  }
  mark(`START ${race.distance}m`, route.sample(0, track.width + 9).point, 24);
  addGates(scene, race, route);
}

/** Gate stalls between the starting lanes. */
function addGates(scene: T.Scene, race: Race, route: RaceRoute) {
  const lanes = startLanes(race);
  const gateGroup = new T.Group();
  for (let i = 0; i <= lanes.length; i++) {
    let lane: number;
    if (i === 0) lane = lanes[0] - 0.8;
    else if (i === lanes.length) lane = lanes.at(-1)! + 0.8;
    else lane = (lanes[i - 1] + lanes[i]) / 2;
    const p = route.sample(0, lane);
    const gate = mesh(new T.BoxGeometry(2.6, 2.9, 0.14), '#386d62');
    gate.position.copy(p.point);
    gate.position.y += 1.45;
    gate.rotation.y = -Math.atan2(p.tangent.z, p.tangent.x);
    gateGroup.add(gate);
  }
  scene.add(gateGroup);
}

export function addHorses(scene: T.Scene, race: Race) {
  return race.horses.map((h) => {
    const g = plush(h);
    g.scale.setScalar(HORSE_SCALE);
    g.rotation.order = 'YZX';
    g.userData.number = h.number;
    const name = g.getObjectByName('nameplate')!;
    name.visible = false;
    const number = g.getObjectByName('numberplate') as T.Sprite;
    number.position.set(0, 4.4, 0);
    number.scale.set(2.2, 2.2, 1);
    scene.add(g);
    return g;
  });
}

/** Ring drawn under the selected horse. */
export function addSelectionRing(scene: T.Scene) {
  const ring = mesh(new T.RingGeometry(2.4, 2.75, 40), '#f4cf65', 5);
  ring.rotation.x = -Math.PI / 2;
  scene.add(ring);
  ring.visible = false;
  return ring;
}

/** Releases every geometry, texture and material in the scene. */
export function disposeScene(scene: T.Scene) {
  scene.traverse((o) => {
    if (o instanceof T.Mesh) o.geometry.dispose();
    if (o instanceof T.Mesh || o instanceof T.Sprite) {
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
        if ('map' in m && m.map) m.map.dispose();
        m.dispose();
      }
    }
  });
}
