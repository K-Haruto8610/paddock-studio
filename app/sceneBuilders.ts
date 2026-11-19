import * as T from 'three';
import type {Race} from './types';
import {makeTrack, makeTerrain, type TrackModel} from './trackGeometry';
import {makeRoute, type RaceRoute} from './raceRoutes';
import {startLanes} from './raceLayout';
import {plush} from './plushHorse';
import {addGoalBoard} from './goalBoard';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';
import {framePalette} from './frameColors';
import {
  FAR_GROUND_DROP,
  GROUND_TILE,
  SURFACE_TILE,
  addFurlongPoles,
  apronTexture,
  dirtTexture,
  groundTexture,
  turfTexture,
  worldVariation,
} from './scenery';
import {dirtReliefMap, RELIEF_TILE} from './surfaceDetail';
import {FELT_TILE, feltMaterial, paintedMetal, postGeometry, railPaintMap, sandRoughnessMap} from './surfaceFinish';
import {groundOf, type GroundSurface} from './courseGround';

// The plush model is 4 units long: scale 1.0 draws a ~4m horse (≈1.7x life size)
// so the field stays legible against the true-scale course. Never enlarged on selection.
export const HORSE_SCALE = 1.0;

/** Strength of the dirt's relief (1 = the furrows as modelled). */
const DIRT_RELIEF = 1.2;

export type Terrain = ReturnType<typeof makeTerrain>;

function material(color: string, lift = 0, map: T.Texture | null = null) {
  return new T.MeshStandardMaterial({
    color,
    map,
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

/** Strip of `n` quads between two edges; `at(i, edge)` gives each vertex and its texture coordinate. */
function strip(
  n: number,
  at: (i: number, edge: 0 | 1) => {p: T.Vector3; uv: [number, number]},
  /** True for a quad that another surface already covers; `a` and `b` are its two edges' vertices. */
  skip?: (a: T.Vector3[], b: T.Vector3[]) => boolean,
) {
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const edges: T.Vector3[][] = [];
  for (let i = 0; i <= n; i++) {
    const pair: T.Vector3[] = [];
    for (const edge of [0, 1] as const) {
      const {p, uv} = at(i, edge);
      positions.push(p.x, p.y, p.z);
      uvs.push(uv[0], uv[1]);
      pair.push(p);
    }
    edges.push(pair);
    if (i < n) {
      const k = i * 2;
      indices.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
  }
  if (skip) {
    const kept: number[] = [];
    for (let i = 0; i < n; i++) if (!skip(edges[i], edges[i + 1])) kept.push(...indices.slice(i * 6, i * 6 + 6));
    indices.length = 0;
    indices.push(...kept);
  }
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new T.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/** Flat strip between two lanes of a track, following its (exaggerated) height. */
function ribbon(
  track: TrackModel,
  inner: number,
  outer: number,
  map: T.Texture,
  lift = 1,
  relief: T.Texture | null = null,
  dirtTone = false,
  covered?: (p: T.Vector3) => boolean,
) {
  const geo = strip(1024, (i, edge) => {
    const d = (i / 1024) * track.lap,
      lane = edge ? outer : inner,
      p = track.sample(d, lane).point;
    p.y += 0.02;
    return {p, uv: [d / SURFACE_TILE, lane / SURFACE_TILE]};
  }, covered && ((a, b) => [...a, ...b].every(covered)));
  // Turf is felt (matte, a fine nap); dirt is sand (relief, and roughness that varies from patch to patch).
  const felt = SURFACE_TILE / FELT_TILE;
  const surface = worldVariation(dirtTone ? material('#ffffff', lift, map) : feltMaterial(map, felt, felt, lift), dirtTone ? 0.07 : 0.09);
  if (relief) {
    surface.normalMap = relief;
    surface.normalScale.setScalar(DIRT_RELIEF);
  }
  if (dirtTone) {
    surface.roughnessMap = sandRoughnessMap();
    surface.roughnessMap.repeat.setScalar(SURFACE_TILE / 8);
  }
  const m = new T.Mesh(geo, surface);
  m.receiveShadow = true;
  return m;
}

/** Whether a point lies on the strip of `track` (between its rails, plus `margin` metres each side). */
function coveredBy(track: TrackModel, margin = 0.1) {
  const CELL = 48; // wider than half the strip plus the margin, so the nearest centre is always in reach
  const step = 1;
  type S = {p: T.Vector3; c: T.Vector3; t: T.Vector3; o: T.Vector3};
  const cells = new Map<string, S[]>();
  for (let d = 0; d < track.lap; d += step) {
    const s = track.sample(d, 0);
    const c = s.point.clone().addScaledVector(s.outward, track.width / 2);
    const k = `${Math.floor(c.x / CELL)},${Math.floor(c.z / CELL)}`;
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k)!.push({p: s.point, c, t: s.tangent, o: s.outward});
  }
  return (q: T.Vector3) => {
    const cx = Math.floor(q.x / CELL), cz = Math.floor(q.z / CELL);
    let best: S | undefined, bd = Infinity;
    for (let i = -1; i <= 1; i++)
      for (let j = -1; j <= 1; j++)
        for (const s of cells.get(`${cx + i},${cz + j}`) ?? []) {
          const d2 = (s.c.x - q.x) ** 2 + (s.c.z - q.z) ** 2;
          if (d2 < bd) { bd = d2; best = s; }
        }
    if (!best) return false;
    const dx = q.x - best.p.x, dz = q.z - best.p.z;
    const across = dx * best.o.x + dz * best.o.z;
    const along = (q.x - best.c.x) * best.t.x + (q.z - best.c.z) * best.t.z;
    return Math.abs(along) < step && across > -margin && across < track.width + margin;
  };
}

const VERGE_COLOR ='#a9bd8f';
/**
 * Grass bank falling away from one edge of a course (`away`: 1 outside the outer rail,
 * -1 inside the inner rail), so the raised surface meets the ground instead of floating over it.
 */
function verge(track: TrackModel, lane: number, away: 1 | -1) {
  const geo = strip(512, (i, edge) => {
    const p = track.sample((i / 512) * track.lap, lane + edge * away * 3.5).point;
    p.y -= edge * 1.4;
    return {p, uv: [0, 0]};
  });
  const m = mesh(geo, VERGE_COLOR);
  m.receiveShadow = true;
  return m;
}

/** Runs of rail points along a lane (one closed loop when nothing is open). */
function railRuns(track: TrackModel, lane: number, open?: (p: T.Vector3) => boolean) {
  const N = 600;
  const points: T.Vector3[] = [];
  const kept: boolean[] = [];
  for (let i = 0; i < N; i++) {
    const p = track.sample((i / N) * track.lap, lane).point;
    kept.push(!open?.(p));
    points.push(p);
  }
  const firstGap = kept.indexOf(false);
  if (firstGap < 0) return {closed: true, runs: [points]};
  // Walk the loop from the first gap, collecting each run of kept points.
  const runs: T.Vector3[][] = [];
  let run: T.Vector3[] = [];
  for (let k = 1; k <= N; k++) {
    const i = (firstGap + k) % N;
    if (kept[i]) run.push(points[i]);
    if ((!kept[i] || k === N) && run.length > 1) runs.push(run);
    if (!kept[i]) run = [];
  }
  return {closed: false, runs};
}

const UP = new T.Vector3(0, 1, 0);
/**
 * White rail: a tube along the lane, carried on posts every ~8m that stand `away` from the
 * course (-1 infield side, 1 outside) and lean in to it. Left open where `open(p)` is true.
 */
function rail(track: TrackModel, lane: number, away: 1 | -1, open?: (p: T.Vector3) => boolean) {
  const group = new T.Group();
  const {closed, runs} = railRuns(track, lane, open);
  // Painted steel with a little wear, round and smooth (never a faceted tube); the paint repeats every ~3 m along the rail.
  const postMat = paintedMetal(railPaintMap(1));
  for (const run of runs) {
    const pts = run.map((p) => p.clone().setY(p.y + 1.1));
    const curve = new T.CatmullRomCurve3(pts, closed);
    const tube = new T.Mesh(
      new T.TubeGeometry(curve, pts.length * 2, 0.16, 16, closed),
      paintedMetal(railPaintMap(curve.getLength() / 3)),
    );
    tube.castShadow = true;
    group.add(tube);
  }
  const count = Math.ceil(track.lap / 8);
  const posts = new T.InstancedMesh(postGeometry(), postMat, count);
  const dummy = new T.Object3D();
  const arm = new T.Vector3();
  let used = 0;
  for (let i = 0; i < count; i++) {
    const d = (i / count) * track.lap;
    const top = track.sample(d, lane).point;
    if (open?.(top)) continue;
    const foot = track.sample(d, lane + away * 0.55).point;
    top.y += 1.06;
    arm.subVectors(top, foot);
    dummy.position.copy(foot).addScaledVector(arm, 0.5);
    dummy.quaternion.setFromUnitVectors(UP, arm.clone().normalize());
    dummy.scale.set(1, arm.length(), 1);
    dummy.updateMatrix();
    posts.setMatrixAt(used++, dummy.matrix);
  }
  posts.count = used;
  posts.castShadow = true;
  group.add(posts);
  return group;
}

/**
 * True for rail points inside the race's running strip: across its chute (引込線),
 * and wherever its route runs over another course's rail (outer course crossing
 * the inner course's outer rail, 外→内 races). The race's own rails stay closed.
 */
function onRoute(route: RaceRoute | undefined, width: number, distance: number) {
  if (!route) return undefined;
  const centre: T.Vector3[] = [];
  for (let t = 0; t <= distance; t += 2) centre.push(route.sample(t, width / 2).point);
  const chute: T.Vector3[] = [];
  const surface = route.chuteSurface;
  if (surface) for (let i = 0; i <= 200; i++) chute.push(surface.sample(i / 200, width / 2).point);
  const inside = (width / 2 - 1.5) ** 2,
    reach = (width / 2 + 1.5) ** 2;
  const near = (pts: T.Vector3[], p: T.Vector3, r: number) =>
    pts.some((c) => (c.x - p.x) ** 2 + (c.z - p.z) ** 2 < r);
  // The chute's own reach applies to outer rails only: its lane 0 meets the inner rail.
  return {
    inner: (p: T.Vector3) => near(centre, p, inside),
    outer: (p: T.Vector3) => near(chute, p, reach) || near(centre, p, inside),
  };
}

/** The rails drawn for a race: its course's two rails, plus the turf outer rail on another course. */
function railSpecs(
  tracks: TrackModel[],
  width: number,
  route?: RaceRoute,
  distance = 0,
) {
  // Rails open where the race's own strip runs across them.
  const open = onRoute(route, width, distance);
  type Spec = {track: TrackModel; lane: number; away: 1 | -1; open?: (p: T.Vector3) => boolean};
  const specs: Spec[] = [];
  // Every drawn course gets both its inner and outer rail.
  for (const t of tracks) {
    specs.push({track: t, lane: 0, away: -1, open: open?.inner});
    specs.push({track: t, lane: t.width, away: 1, open: open?.outer});
  }
  return specs;
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
  const tracks = kinds.map((k) => (k === track.kind ? track : makeTrack(race, k, track.elevationScale)));
  return {
    tracks,
    turf: tracks.find((t) => t.kind === 'turf')!,
    dirt: tracks.find((t) => t.kind === 'dirt'),
    outer: tracks.find((t) => t.kind === 'outer'),
  };
}

export function addLights(scene: T.Scene, sunOffset: T.Vector3) {
  scene.add(new T.HemisphereLight('#b9d3ed', '#758565', 1.05));
  const sun = new T.DirectionalLight('#fff0d8', 2.45);
  sun.position.copy(sunOffset);
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
  // The last 45m ease down to the flat land beyond, so the ground has no visible border.
  const floor = Math.min(...terrain.samples.map((s) => s.y)) - FAR_GROUND_DROP + 0.05;
  for (let i = 0; i < gp.count; i++) {
    const x = gp.getX(i),
      z = gp.getZ(i),
      edge = Math.min(x - box.min.x, box.max.x - x, z - box.min.z, box.max.z - z);
    gp.setY(i, T.MathUtils.lerp(floor, terrain.heightAt(x, z), T.MathUtils.smoothstep(edge, 0, 45)));
  }
  ground.computeVertexNormals();
  const map = groundTexture();
  map.repeat.set(size.x / GROUND_TILE, size.z / GROUND_TILE);
  const groundMesh = new T.Mesh(ground, worldVariation(feltMaterial(map, size.x / FELT_TILE, size.z / FELT_TILE), 0.12));
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
  map: T.Texture | null = null,
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
  const uv = geo.attributes.uv as T.BufferAttribute;
  for (let i = 0; i < uv.count; i++)
    uv.setXY(i, pos.getX(i) / SURFACE_TILE, pos.getZ(i) / SURFACE_TILE);
  geo.computeVertexNormals();
  const m = new T.Mesh(geo, worldVariation(material(map ? '#ffffff' : color, 1, map), 0.08));
  m.receiveShadow = true;
  return m;
}

/**
 * Tokyo's three pockets (引込線) as the routes define them, so each one is built from the same
 * path the field runs and meets the main rail with the route's own join. They are drawn whichever
 * race is shown: 2,000m and 1,800m turf, and the dirt 1,600m.
 */
export function tokyoChutes(race: Race) {
  const own = (surface: string, trackCode: number, distance: number) => ({...race, surface, trackCode, distance});
  const kinds: [string, number, number, 'turf' | 'dirt'][] = [['芝', 11, 2000, 'turf'], ['芝', 11, 1800, 'turf'], ['ダート', 23, 1600, 'dirt']];
  return kinds.flatMap(([surface, trackCode, distance, kind]) => {
    const r = own(surface, trackCode, distance);
    const surfaceOf = makeRoute(r, makeTrack(r, kind)).chuteSurface;
    return surfaceOf ? [{surface: surfaceOf, onDirt: kind === 'dirt'}] : [];
  });
}

// Tokyo plan-pixel polygons (same pixel space as the digitized rails).
const TOKYO_APRON_A = [[229, 317], [460, 298], [485, 327], [462, 352], [410, 356], [268, 356], [230, 347]]; // prettier-ignore
const TOKYO_APRON_B = [
  [160, 322],
  [225, 316],
  [230, 354],
  [171, 361],
];

/** Strip along a chute (引込線) from its start to the main rail, over its own flat height. */
function chuteRibbon(
  surface: NonNullable<RaceRoute['chuteSurface']>,
  width: number,
  map: T.Texture,
  /** Quads touching this other surface are left to it (turf pocket beside the dirt course). */
  covered?: (a: T.Vector3[], b: T.Vector3[]) => boolean,
  lift = 2,
) {
  const N = 96;
  const geo = strip(N, (i, edge) => {
    const p = surface.sample(i / N, edge ? width : 0).point;
    p.y += 0.02;
    return {p, uv: [((i / N) * surface.length) / SURFACE_TILE, (edge * width) / SURFACE_TILE]};
  }, covered);
  const felt = SURFACE_TILE / FELT_TILE;
  const m = new T.Mesh(geo, worldVariation(feltMaterial(map, felt, felt, lift), 0.09));
  m.receiveShadow = true;
  return m;
}

/**
 * Turf/dirt course surfaces, the race's chute, plus Tokyo's chutes and the dirt-start apron.
 * Returns where the race surfaces are, by position, for whatever is laid on them (grass, hoofprints).
 */
export function addCourseSurfaces(
  scene: T.Scene,
  courses: CourseSet,
  terrain: Terrain,
  tokyo: boolean,
  route?: RaceRoute,
  chutes: {surface: NonNullable<RaceRoute['chuteSurface']>; onDirt: boolean}[] = [],
) {
  const {turf, dirt, outer} = courses;
  const surfaces: GroundSurface[] = [];
  const lay = (m: T.Mesh, lift: number, dirt = false) => {
    scene.add(m);
    surfaces.push({geometry: m.geometry, lift, dirt});
  };
  // Outer course first (lowest priority), then the inner turf, then dirt on top.
  const grass = turfTexture();
  // Where the outer course runs on the inner course's own strip it is the same surface: leave
  // that stretch to the inner ribbon so no two layers overlap there.
  if (outer) lay(ribbon(outer, 0, outer.width, grass, 1, null, false, coveredBy(turf)), 1);
  lay(ribbon(turf, 0, turf.width, grass, 2), 2);
  if (dirt) {
    // Harrowed sand: furrows along the course, laid over the same strip as the colour.
    const relief = dirtReliefMap();
    relief.repeat.setScalar(SURFACE_TILE / RELIEF_TILE);
    lay(ribbon(dirt, 0, dirt.width, dirtTexture(), 3, relief, true), 3, true);
  }
  for (const t of courses.tracks) scene.add(verge(t, 0, -1), verge(t, t.width, 1));
  // The pocket's turf follows the line the field runs, which leaves the plan's drawn pocket
  // to come onto the course smoothly.
  // Tokyo draws all three pockets from their routes, whichever race is shown.
  if (route?.chuteSurface && !tokyo) lay(chuteRibbon(route.chuteSurface, route.chuteSurface.width, grass), 2);
  // A pocket is one surface with the course it joins: where it lies on that course's strip it is
  // left out, and a pocket that only crosses the turf (Tokyo's dirt start) lies above it.
  const onTurf = coveredBy(turf), onDirt = dirt ? coveredBy(dirt, 3) : undefined;
  for (const {surface: c, onDirt: crossing} of chutes) {
    if (crossing) {
      // Over the turf strip the pocket is the turf itself, so only the part beside it is laid.
      lay(chuteRibbon(c, c.width, grass, (a, b) => [...a, ...b].every(onTurf) || (!!onDirt && [...a, ...b].some(onDirt)), 2.5), 2.5);
    } else {
      lay(chuteRibbon(c, c.width, grass, (a, b) => [...a, ...b].every(onTurf), 1), 1);
    }
  }
  if (tokyo && dirt) {
    const add = (points: number[][], color: string, lift: number, map: T.Texture | null = null) =>
      scene.add(drape(turf, terrain, points, color, lift, map));
    const apron = apronTexture();
    add(TOKYO_APRON_A, '#ffffff', 5, apron);
    add(TOKYO_APRON_B, '#ffffff', 4, apron);
  }
  return groundOf(surfaces);
}

/** Inner and outer rails of every course drawn for the race. */
export function addRails(
  scene: T.Scene,
  courses: CourseSet,
  width: number,
  route?: RaceRoute,
  distance = 0,
) {
  for (const r of railSpecs(courses.tracks, width, route, distance)) scene.add(rail(r.track, r.lane, r.away, r.open));
}

/** Finish line, course-side signs and the starting gate stalls (returned). */
export function addMarkers(scene: T.Scene, race: Race, track: TrackModel, route: RaceRoute) {
  const finish = track.sample(0);
  const finishLine = mesh(new T.PlaneGeometry(0.7, track.width), '#fff9df', 4);
  finishLine.rotation.x = -Math.PI / 2;
  finishLine.rotation.z = Math.atan2(finish.tangent.z, finish.tangent.x);
  finishLine.position.copy(track.sample(0, track.width / 2).point);
  finishLine.position.y += 0.1;
  scene.add(finishLine);
  addGoalBoard(scene, race, track);
  addFurlongPoles(scene, track);
  addStartSign(scene, route, race.distance, track.width);
  return addGates(scene, race, route);
}

function addStartSign(scene: T.Scene, route: RaceRoute, distance: number, width: number) {
  const sign = new T.Group();
  const at = route.sample(0, width + 5);
  sign.position.copy(at.point);
  sign.rotation.y = -Math.atan2(at.tangent.z, at.tangent.x);
  const board = mesh(new T.BoxGeometry(5.6, 1.5, 0.24), '#2f6a5c');
  board.position.y = 2.6;
  board.castShadow = true;
  sign.add(board);
  const postGeometry = new T.BoxGeometry(0.18, 4.5, 0.18);
  const postMaterial = material('#e7e0cd');
  for (const x of [-2.2, 2.2]) {
    const post = new T.Mesh(postGeometry, postMaterial);
    post.position.set(x, 0.8, 0);
    post.castShadow = true;
    sign.add(post);
  }
  const canvas = document.createElement('canvas');
  canvas.width = 448;
  canvas.height = 120;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#2f6a5c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#c3b683';
  ctx.lineWidth = 3;
  ctx.strokeRect(9, 9, canvas.width - 18, canvas.height - 18);
  ctx.fillStyle = '#fff8e2';
  ctx.font = 'bold 52px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`START  ${distance}m`, canvas.width / 2, canvas.height / 2, canvas.width - 48);
  const map = new T.CanvasTexture(canvas);
  map.colorSpace = T.SRGBColorSpace;
  const faceGeometry = new T.PlaneGeometry(5.45, 1.42);
  const faceMaterial = new T.MeshBasicMaterial({map, toneMapped: false});
  for (const side of [-1, 1]) {
    const face = new T.Mesh(faceGeometry, faceMaterial);
    face.rotation.y = side < 0 ? Math.PI : 0;
    face.position.set(0, 2.6, side * 0.125);
    sign.add(face);
  }
  scene.add(sign);
}

/** 発馬機: padded stall partitions between the starting lanes under a frame on wheels. */
function addGates(scene: T.Scene, race: Race, route: RaceRoute) {
  const lanes = startLanes(race);
  const edges = [lanes[0] - 0.8];
  for (let i = 1; i < lanes.length; i++) edges.push((lanes[i - 1] + lanes[i]) / 2);
  edges.push(lanes.at(-1)! + 0.8);
  const middle = (edges[0] + edges.at(-1)!) / 2;
  const span = edges.at(-1)! - edges[0];
  const at = route.sample(0, middle);
  const gateGroup = new T.Group();
  gateGroup.position.copy(at.point);
  gateGroup.rotation.y = -Math.atan2(at.tangent.z, at.tangent.x);
  // Local x runs with the race and local z to its left; lanes grow outward, on either side.
  const side = -at.tangent.z * at.outward.x + at.tangent.x * at.outward.z >= 0 ? 1 : -1;
  const body = material('#2f6a5c'),
    pad = material('#ece9d6'),
    frame = new T.MeshStandardMaterial({color: '#dfe3da', roughness: 0.5}),
    tyre = material('#3a3f3c');
  const LENGTH = 3.4,
    TOP = 3.5;
  const add = (geo: T.BufferGeometry, mat: T.Material, x: number, y: number, z: number) => {
    const m = new T.Mesh(geo, mat);
    m.position.set(x, y, z);
    m.castShadow = true;
    gateGroup.add(m);
    return m;
  };
  const panel = new RoundedBoxGeometry(LENGTH, 1.7, 0.16, 3, 0.06),
    cushion = new RoundedBoxGeometry(LENGTH, 0.5, 0.26, 3, 0.1),
    post = new T.CylinderGeometry(0.07, 0.07, TOP, 8),
    tie = new T.BoxGeometry(LENGTH, 0.12, 0.12);
  for (const lane of edges) {
    const z = (lane - middle) * side;
    add(panel, body, 0, 1.05, z);
    add(cushion, pad, 0, 2.1, z);
    for (const x of [-LENGTH / 2, LENGTH / 2]) add(post, frame, x, TOP / 2, z);
    add(tie, frame, 0, TOP, z);
  }
  // Each stall carries its frame number on the beam, readable from behind and from the front.
  const plateGeometry = new T.PlaneGeometry(1, 1);
  race.horses.forEach((horse, i) => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d')!;
    const {background, color} = framePalette(horse.gate);
    ctx.fillStyle = background;
    ctx.beginPath();
    ctx.roundRect(4, 4, 120, 120, 22);
    ctx.fill();
    ctx.strokeStyle = 'rgba(30, 37, 34, .45)';
    ctx.lineWidth = 6;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = 'bold 88px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(horse.gate), 64, 70);
    const map = new T.CanvasTexture(canvas);
    map.colorSpace = T.SRGBColorSpace;
    const plate = new T.MeshBasicMaterial({map, toneMapped: false, side: T.DoubleSide});
    const z = ((edges[i] + edges[i + 1]) / 2 - middle) * side;
    for (const x of [-LENGTH / 2 - 0.12, LENGTH / 2 + 0.12]) {
      const m = new T.Mesh(plateGeometry, plate);
      m.rotation.y = Math.PI / 2;
      m.scale.setScalar(0.95);
      m.position.set(x, TOP - 0.62, z);
      gateGroup.add(m);
    }
  });
  const beam = new T.BoxGeometry(0.2, 0.28, span + 0.5);
  for (const x of [-LENGTH / 2, LENGTH / 2]) add(beam, frame, x, TOP, 0);
  const wheel = new T.CylinderGeometry(0.55, 0.55, 0.3, 16);
  for (const x of [-1.1, 1.1])
    for (const z of [-span / 2 - 0.4, span / 2 + 0.4])
      add(wheel, tyre, x, 0.55, z).rotation.x = Math.PI / 2;
  gateGroup.name = 'starting-gate';
  scene.add(gateGroup);
  return gateGroup;
}

export function addHorses(scene: T.Scene, race: Race, pileLayers: number) {
  return race.horses.map((h) => {
    const g = plush(h, {bridle: true, labels: false, pileLayers});
    g.scale.setScalar(HORSE_SCALE);
    g.rotation.order = 'YZX';
    g.userData.number = h.number;
    scene.add(g);
    return g;
  });
}

/** A soft dark patch under each horse's feet; follow it with `place` each frame. */
export function addContactShadows(scene: T.Scene, horses: T.Group[]) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext('2d')!;
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(20, 28, 18, .5)');
  gradient.addColorStop(0.55, 'rgba(20, 28, 18, .3)');
  gradient.addColorStop(1, 'rgba(20, 28, 18, 0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  const map = new T.CanvasTexture(canvas);
  map.colorSpace = T.SRGBColorSpace;
  const material = new T.MeshBasicMaterial({
    map, transparent: true, depthWrite: false, toneMapped: false,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
  });
  const geometry = new T.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
  const box = new T.Box3().setFromObject(horses[0]);
  const size = box.getSize(new T.Vector3());
  const patches = horses.map(() => {
    const patch = new T.Mesh(geometry, material);
    patch.scale.set(size.x * 1.15, 1, size.z * 1.5);
    patch.renderOrder = 3;
    patch.raycast = () => {};
    scene.add(patch);
    return patch;
  });
  return {
    place() {
      patches.forEach((patch, i) => {
        patch.position.copy(horses[i].position);
        patch.position.y += 0.08;
        patch.rotation.y = horses[i].rotation.y;
      });
    },
  };
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
  const geometries = new Set<T.BufferGeometry>();
  const materials = new Set<T.Material>();
  const textures = new Set<T.Texture>();
  scene.traverse((o) => {
    if (o instanceof T.Mesh) {
      if (o instanceof T.InstancedMesh) o.dispose();
      geometries.add(o.geometry);
      // Hair draws its sway into the shadow map with a material of its own.
      if (o.customDepthMaterial) materials.add(o.customDepthMaterial);
    }
    if (o instanceof T.Mesh || o instanceof T.Sprite) {
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
        materials.add(m);
        for (const value of Object.values(m)) {
          if (value instanceof T.Texture) textures.add(value);
        }
      }
    }
  });
  textures.forEach((t) => t.dispose());
  materials.forEach((m) => m.dispose());
  geometries.forEach((g) => g.dispose());
}
