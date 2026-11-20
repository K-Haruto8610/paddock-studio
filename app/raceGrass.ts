import * as T from 'three';
import type {Ground, GroundHit} from './courseGround';
import type {RaceRoute} from './raceRoutes';
import type {TrackModel, TrackPoint} from './trackGeometry';
import {seeded} from './scenery';
import {type BladeField, bladeMaterial, bladeMesh, emptyBladeField, grassUniforms, mowBand, scatterBlade} from './surfaceDetail';
import {NO_WEAR, wearAt, type TurfWear} from './turfWear';

/**
 * Blades on worn turf (app/turfWear.ts): none where the soil is bare, short, flattened and darker where it is trampled.
 * Blade `i` of `field`, standing at world (x, z) `lane` metres off the inside rail.
 */
function wearBlade(field: BladeField, i: number, x: number, z: number, lane: number, wear: TurfWear) {
  if (wear.amount <= 0) return;
  const {trampled, bare} = wearAt(x, z, lane, wear);
  field.blade[i * 4 + 1] *= bare > 0.4 ? 0 : (1 - bare) * (1 - 0.7 * trampled);
  field.blade[i * 4 + 2] += 0.6 * trampled;
  field.blade[i * 4 + 3] -= 0.16 * trampled + 0.2 * bare;
}

// Standing grass on the course a race is run on. A whole course would take tens of millions of
// blades, so they stand only around the field: the course is cut into short stretches along the
// race's route, and the stretches near the horses are grown as the field reaches them.

/** Metres of route covered by one stretch of blades. */
const STRETCH = 10;
/** Blades per square metre. */
const DENSITY = 130;
/** Blades reach this far ahead of the leader and behind the last horse (fading in over the last metres). */
const MARGIN = 32;
/** Longest run of course that carries blades; a field strung out further loses them at the back. */
const MAX_SPAN = 150;
/** The route is sampled on a grid this fine and each blade placed between the samples (for where it stands, not how high). */
const GRID_ALONG = 5;
const GRID_ACROSS = 4;
/** Stretches are grown this many metres before the blades' range reaches them. */
const PREPARE = 40;
/** Blades prepared at each 60 Hz tick, ahead of the visible range. */
const BLADES_PER_TICK = 3200;

export type RaceGrass = ReturnType<typeof addRaceGrass>;

/** Sparse blades continue over the visible lap beyond the dense stand around the horses. */
export function addDistantGrass(scene: T.Scene, track: TrackModel, ground: Ground, uniforms: ReturnType<typeof grassUniforms>,
  wear: TurfWear = NO_WEAR) {
  const material = bladeMaterial({...uniforms, uFrom: {value: -1e6}, uTo: {value: 1e6}});
  const group = new T.Group();
  scene.add(group);
  const bounds = new T.Box3();
  const hit: GroundHit = {y: 0, normal: new T.Vector3(), dirt: false};
  const segmentLength = 30;
  for (let start = 0, segment = 0; start < track.lap; start += segmentLength, segment++) {
    const length = Math.min(segmentLength, track.lap - start);
    const random = seeded(5689 + segment * 7919);
    const field = emptyBladeField(Math.ceil(length * track.width * 6));
    bounds.makeEmpty();
    for (let i = 0; i < field.count; i++) {
      const at = start + length * random();
      const lane = track.width * random();
      const {point, tangent} = track.sample(at, lane);
      const found = ground(point.x, point.z, hit, point.y);
      if (found) point.y = hit.y;
      bounds.expandByPoint(point);
      field.root.set(point.toArray(), i * 3);
      if (found) field.up.set([hit.normal.x, hit.normal.z], i * 2);
      const band = mowBand(lane);
      field.lay.set([tangent.x * band, tangent.z * band, at, 0], i * 4);
      scatterBlade(field, i, random);
      wearBlade(field, i, point.x, point.z, lane, wear);
    }
    const mesh = bladeMesh(field, material);
    mesh.geometry.boundingSphere = bounds.getBoundingSphere(new T.Sphere());
    mesh.geometry.boundingSphere.radius += 1;
    mesh.frustumCulled = true;
    mesh.renderOrder = segment + 500;
    group.add(mesh);
  }
  return () => {
    group.traverse(object => { if (object instanceof T.Mesh) object.geometry.dispose(); });
    group.removeFromParent();
    material.dispose();
  };
}

/**
 * Grass over lanes 0..width of `route`, grown around the field by `update`. Each blade takes its
 * height and lean from `ground`, the surface as drawn, so it stands on the course even on a slope.
 */
export function addRaceGrass(scene: T.Scene, route: RaceRoute, width: number, ground: Ground, wear: TurfWear = NO_WEAR) {
  const uniforms = grassUniforms();
  const material = bladeMaterial(uniforms);
  const group = new T.Group();
  scene.add(group);
  const stretches = new Map<number, T.Mesh>();

  /** A stretch being grown: its blades are placed a part at a time. */
  type Growing = {k: number; field: BladeField; rand: () => number; grid: TrackPoint[][]; done: number};
  const p = new T.Vector3(),
    q = new T.Vector3(),
    t = new T.Vector3(),
    hit: GroundHit = {y: 0, normal: new T.Vector3(0, 1, 0), dirt: false};

  const begin = (k: number): Growing => ({
    k,
    field: emptyBladeField(Math.round(DENSITY * STRETCH * width)),
    // Seeded by the stretch, so seeking back or recording twice grows the same grass.
    rand: seeded(9001 + k * 7919),
    grid: Array.from({length: GRID_ALONG + 1}, (_, i) =>
      Array.from({length: GRID_ACROSS + 1}, (_, j) =>
        route.sample(k * STRETCH + (i / GRID_ALONG) * STRETCH, (j / GRID_ACROSS) * width),
      ),
    ),
    done: 0,
  });

  /** Places the next `count` blades of a stretch. */
  const place = (g: Growing, count: number) => {
    const {field, rand, grid} = g,
      from = g.k * STRETCH,
      end = Math.min(field.count, g.done + count);
    for (let n = g.done; n < end; n++) {
      const a = rand(),
        l = rand(),
        fi = a * GRID_ALONG,
        fj = l * GRID_ACROSS,
        i = Math.min(GRID_ALONG - 1, Math.floor(fi)),
        j = Math.min(GRID_ACROSS - 1, Math.floor(fj));
      p.lerpVectors(grid[i][j].point, grid[i][j + 1].point, fj - j);
      q.lerpVectors(grid[i + 1][j].point, grid[i + 1][j + 1].point, fj - j);
      p.lerp(q, fi - i);
      t.lerpVectors(grid[i][j].tangent, grid[i + 1][j].tangent, fi - i).setY(0).normalize();
      // The same bands as the mowing stripes painted on the turf texture.
      const band = mowBand(l * width);
      // The route is curved and sampled on its own grid, the ribbon is straight between its own
      // samples: the height comes from the ribbon, and the route's only says which one is meant.
      // A blade the ribbon does not reach (none should be) keeps the route's height.
      const found = ground(p.x, p.z, hit, p.y + 0.02);
      field.root[n * 3] = p.x;
      field.root[n * 3 + 1] = found ? hit.y : p.y + 0.02;
      field.root[n * 3 + 2] = p.z;
      if (found) {
        field.up[n * 2] = hit.normal.x;
        field.up[n * 2 + 1] = hit.normal.z;
      }
      field.lay[n * 4] = t.x * band;
      field.lay[n * 4 + 1] = t.z * band;
      field.lay[n * 4 + 2] = from + a * STRETCH;
      scatterBlade(field, n, rand);
      field.blade[n * 4 + 3] += 0.04 * band;
      wearBlade(field, n, p.x, p.z, l * width, wear);
    }
    g.done = end;
  };

  const stand = (g: Growing) => {
    const mesh = bladeMesh(g.field, material);
    mesh.renderOrder = g.k + 1;
    stretches.set(g.k, mesh);
    group.add(mesh);
  };
  /** The stretch being grown ahead of the field, a little each frame. */
  let ahead: Growing | null = null;

  return {
    uniforms,
    /**
     * Keeps blades standing from behind the last horse (`rear`) to ahead of the leader (`front`),
     * both distances along the route. `show` false hides them (the camera is too far to see blades).
     */
    update(rear: number, front: number, show: boolean, ticks: number) {
      group.visible = show;
      if (!show) return;
      const to = front + MARGIN,
        from = Math.max(rear - MARGIN, to - MAX_SPAN);
      uniforms.uFrom.value = from;
      uniforms.uTo.value = to;
      const first = Math.max(0, Math.floor(from / STRETCH)),
        last = Math.floor(to / STRETCH),
        // Placing a stretch's blades takes tens of milliseconds, a visible hitch if done in one
        // frame. The stretches the field is running towards are grown before they are needed,
        // a part of them each frame; blades there have no height yet, so nothing shows.
        prepared = Math.floor((to + PREPARE) / STRETCH);
      for (const [k, mesh] of stretches)
        if (k < first || k > last) {
          group.remove(mesh);
          mesh.geometry.dispose();
          stretches.delete(k);
        }
      if (ahead && (ahead.k < first || ahead.k > prepared)) ahead = null;
      // Anything missing where blades already stand (the start, a seek) cannot wait.
      for (let k = first; k <= last; k++)
        if (!stretches.has(k)) {
          const g = ahead?.k === k ? ahead : begin(k);
          if (g === ahead) ahead = null;
          place(g, g.field.count);
          stand(g);
        }
      if (!ahead && last < prepared && ticks > 0) ahead = begin(last + 1);
      if (ahead) place(ahead, ticks * BLADES_PER_TICK);
    },
    dispose() {
      for (const mesh of stretches.values()) mesh.geometry.dispose();
      stretches.clear();
      group.clear();
      material.dispose();
    },
  };
}

/** Metres per side of one tile of verge grass. */
const VERGE_TILE = 16;
/** Verge blades per square metre: thinner than the course, which is looked at from close by. */
const VERGE_DENSITY = 24;
/** Verge grass stands this far around the camera's focus (the field). */
const VERGE_RADIUS = 64;

/**
 * Grass on all the ground that is not a course surface: the infield, the banks and the land outside the rails.
 * Tiles around `update`'s centre are grown (seeded by the tile, so a seek or a second recording grows the same
 * grass) and dropped once the field has moved on. Course turf stands blades too, except the lanes of the race's own turf (`ownTurf`),
 * which carry denser ones; dirt stays bare.
 */
export function addVergeGrass(scene: T.Scene, heightAt: (x: number, z: number) => number, ground: Ground, ownTurf: boolean) {
  const uniforms = grassUniforms();
  const material = bladeMaterial(uniforms);
  const group = new T.Group();
  scene.add(group);
  const tiles = new Map<string, T.Mesh>();
  const hit: GroundHit = {y: 0, normal: new T.Vector3(0, 1, 0), dirt: false};

  const grow = (i: number, j: number) => {
    const rand = seeded(31337 + i * 7919 + j * 104729),
      field = emptyBladeField(Math.round(VERGE_DENSITY * VERGE_TILE * VERGE_TILE));
    let n = 0;
    for (let k = 0; k < field.count; k++) {
      const x = (i + rand()) * VERGE_TILE,
        z = (j + rand()) * VERGE_TILE;
      scatterBlade(field, n, rand);
      // Turf of the courses the race is not run on (every course, when it is run on dirt) is grass too.
      const found = ground(x, z, hit);
      if (found && (ownTurf || hit.dirt)) continue;
      field.root[n * 3] = x;
      field.root[n * 3 + 1] = found ? hit.y : heightAt(x, z) + 0.03;
      field.root[n * 3 + 2] = z;
      if (found) {
        field.up[n * 2] = hit.normal.x;
        field.up[n * 2 + 1] = hit.normal.z;
      }
      // Unmown: no lean along a course, and a slightly duller shade than the striped turf.
      field.lay[n * 4] = 0;
      field.lay[n * 4 + 1] = 0;
      field.lay[n * 4 + 2] = 0;
      field.blade[n * 4 + 3] -= 0.05;
      n++;
    }
    field.count = n;
    const mesh = bladeMesh(field, material);
    mesh.frustumCulled = true;
    mesh.geometry.boundingSphere = new T.Sphere(
      new T.Vector3((i + 0.5) * VERGE_TILE, heightAt((i + 0.5) * VERGE_TILE, (j + 0.5) * VERGE_TILE), (j + 0.5) * VERGE_TILE),
      VERGE_TILE * 0.75 + 10,
    );
    mesh.renderOrder = 300;
    return mesh;
  };

  return {
    uniforms,
    update(centre: T.Vector3, show: boolean) {
      group.visible = show;
      if (!show) return;
      const r = Math.ceil(VERGE_RADIUS / VERGE_TILE),
        ci = Math.floor(centre.x / VERGE_TILE),
        cj = Math.floor(centre.z / VERGE_TILE);
      const wanted = new Set<string>();
      for (let i = ci - r; i <= ci + r; i++)
        for (let j = cj - r; j <= cj + r; j++) {
          if (Math.hypot((i + 0.5) * VERGE_TILE - centre.x, (j + 0.5) * VERGE_TILE - centre.z) > VERGE_RADIUS) continue;
          const key = `${i},${j}`;
          wanted.add(key);
          if (!tiles.has(key)) {
            const mesh = grow(i, j);
            tiles.set(key, mesh);
            group.add(mesh);
          }
        }
      for (const [key, mesh] of tiles)
        if (!wanted.has(key)) {
          group.remove(mesh);
          mesh.geometry.dispose();
          tiles.delete(key);
        }
    },
    dispose() {
      for (const mesh of tiles.values()) mesh.geometry.dispose();
      tiles.clear();
      group.removeFromParent();
      material.dispose();
    },
  };
}
