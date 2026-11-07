import * as T from 'three';
import type {Race} from './types';
import {courseFor, lapFor, G1_COURSES, type CourseKind} from './courses';
import {COURSE_DATA} from './courseData';

// Official elevation differences are only 2-5m over ~2km: at true scale the
// course looks flat. Heights are drawn exaggerated by this factor (shown in UI).
export const VERTICAL_EXAGGERATION = 8;

// Digitized INNER RAIL from JRA's 568 x 385 Tokyo plan (not an aerial survey).
// Origin is the winning post. Order follows the official left-handed direction.
export const TOKYO_TURF_PIXELS = [
  [399, 270],
  [428, 267],
  [452, 257],
  [471, 239],
  [484, 211],
  [491, 182],
  [489, 154],
  [480, 128],
  [465, 107],
  [445, 90],
  [422, 79],
  [395, 72],
  [366, 69],
  [306, 68],
  [246, 68],
  [186, 68],
  [139, 68],
  [112, 70],
  [89, 77],
  [70, 91],
  [55, 112],
  [44, 139],
  [38, 165],
  [36, 190],
  [39, 215],
  [48, 237],
  [63, 255],
  [84, 266],
  [108, 272],
  [150, 273],
  [220, 272],
  [290, 271],
  [355, 270],
];
const TOKYO_DIRT_PIXELS = [
  [399, 253],
  [425, 249],
  [445, 237],
  [459, 218],
  [470, 192],
  [476, 169],
  [472, 145],
  [461, 123],
  [443, 106],
  [422, 95],
  [398, 88],
  [367, 85],
  [305, 84],
  [245, 84],
  [185, 84],
  [139, 84],
  [114, 86],
  [93, 94],
  [78, 109],
  [65, 132],
  [57, 157],
  [53, 181],
  [53, 202],
  [60, 222],
  [74, 240],
  [94, 251],
  [117, 255],
  [165, 255],
  [230, 254],
  [300, 254],
  [355, 253],
];
// Remaining distance -> relative height (m), digitized from the official section graphs.
export const TURF_HEIGHT = [
  [0, 0],
  [300, 0],
  [460, -2],
  [650, -2.2],
  [880, -2.5],
  [1050, -1.1],
  [1190, -0.3],
  [1260, -0.3],
  [1330, -1.7],
  [1600, -1],
  [1800, -0.5],
  [1950, 0.2],
  [2083.1, 0],
];
const DIRT_HEIGHT = [
  [0, 0],
  [200, 0],
  [500, -2.4],
  [800, -2.5],
  [1000, -1],
  [1090, -0.9],
  [1200, -1.7],
  [1500, -1],
  [1800, -0.3],
  [1899, 0],
];
export function linearProfile(points: number[][], x: number) {
  for (let i = 1; i < points.length; i++) {
    if (x <= points[i][0]) {
      const [a, b] = [points[i - 1], points[i]];
      return T.MathUtils.lerp(a[1], b[1], (x - a[0]) / (b[0] - a[0]));
    }
  }
  return points.at(-1)![1];
}
const mod = (v: number, n: number) => ((v % n) + n) % n;
export type TrackPoint = {point: T.Vector3; tangent: T.Vector3; outward: T.Vector3};
export type TrackModel = {
  kind: CourseKind;
  lap: number;
  width: number;
  scale: number;
  curve: T.CatmullRomCurve3;
  sample: (distance: number, lane?: number) => TrackPoint;
  pixel: (x: number, y: number) => T.Vector3;
  height: (distance: number) => number;
  profile: (remaining: number) => number;
};

export function drawingFor(venueCode: string) {
  if (venueCode === '05')
    return {
      center: [280, 170],
      rails: {turf: TOKYO_TURF_PIXELS, dirt: TOKYO_DIRT_PIXELS} as Record<string, number[][]>,
      profiles: {turf: TURF_HEIGHT, dirt: DIRT_HEIGHT} as Record<string, number[][]>,
    };
  const d = COURSE_DATA[venueCode];
  if (!d?.rails) return null;
  return {
    center: d.center!,
    rails: d.rails as Record<string, number[][]>,
    profiles: d.profiles as Record<string, number[][]>,
  };
}
// Courses drawn in 3D: the venue's GⅠ courses plus the race's own course.
export function availableKinds(race: Race): CourseKind[] {
  const d = drawingFor(race.venueCode),
    own = courseFor(race).kind,
    wanted = G1_COURSES[race.venueCode] ?? ['turf'];
  return (['turf', 'outer', 'dirt'] as CourseKind[]).filter(
    (k) => (k === 'turf' || wanted.includes(k) || k === own) && (!d || d.rails[k] || k === own),
  );
}

// kind: which rail. A boolean is accepted for the original turf/dirt API.
export function makeTrack(
  race: Race,
  kindOrDirt: CourseKind | boolean = courseFor(race).kind,
): TrackModel {
  const kind: CourseKind =
    typeof kindOrDirt === 'boolean' ? (kindOrDirt ? 'dirt' : 'turf') : kindOrDirt;
  const drawing = drawingFor(race.venueCode),
    lap = lapFor(race.venueCode, kind);
  const rail = drawing?.rails[kind] ?? drawing?.rails.turf;
  const [cx, cy] = drawing?.center ?? [280, 170];
  const pixels =
    rail ??
    Array.from({length: 48}, (_, i) => {
      const a = (i / 48) * Math.PI * 2;
      return [280 + 200 * Math.cos(a), 170 + 100 * Math.sin(a)];
    });
  const raw = new T.CatmullRomCurve3(
    pixels.map(([x, z]) => new T.Vector3(x - cx, 0, z - cy)),
    true,
    'centripetal',
  );
  raw.arcLengthDivisions = 4096;
  // Digitized venues share one plan scale (calibrated on the inner turf lap), so
  // rails that coincide on the plan coincide in 3D. Each course is still
  // parameterized by its official lap; the drawn length differs by a few %.
  const shared = race.venueCode !== '05' && kind !== 'turf' && drawing?.rails.turf;
  const scale = shared
    ? lapFor(race.venueCode, 'turf') /
      new T.CatmullRomCurve3(
        drawing!.rails.turf.map(([x, z]) => new T.Vector3(x - cx, 0, z - cy)),
        true,
        'centripetal',
      ).getLength()
    : lap / raw.getLength();
  const curve = new T.CatmullRomCurve3(
    raw.points.map((p) => p.clone().multiplyScalar(scale)),
    true,
    'centripetal',
  );
  curve.arcLengthDivisions = 4096;
  const points = drawing?.profiles[kind] ?? drawing?.profiles.turf;
  let profile = (remaining: number) => (points ? linearProfile(points, remaining) : 0);
  if (kind === 'outer' && drawing?.rails.outer) {
    // Where the outer course runs on the inner course's rail it takes the inner
    // course's heights, blending smoothly to its own section graph as it bulges
    // away. Two graphs read independently never agree exactly; without this the
    // shared stretch showed a ledge between the two ribbons.
    const inner = makeTrack(race, 'turf'),
      innerPts: {x: number; z: number; d: number}[] = [];
    for (let d = 0; d < inner.lap; d += 1) {
      const p = inner.curve.getPointAt(d / inner.lap);
      innerPts.push({x: p.x, z: p.z, d});
    }
    const table: number[][] = [];
    for (let d = 0; d <= lap; d += 2) {
      const p = curve.getPointAt(Math.min(d, lap) / lap);
      let best = innerPts[0],
        e = Infinity;
      for (const q of innerPts) {
        const k = (q.x - p.x) ** 2 + (q.z - p.z) ** 2;
        if (k < e) {
          e = k;
          best = q;
        }
      }
      const w = T.MathUtils.smoothstep(Math.sqrt(e), 1.5, 10),
        own = points ? linearProfile(points, lap - d) : 0;
      table.push([lap - d, T.MathUtils.lerp(inner.profile(inner.lap - best.d), own, w)]);
    }
    table.reverse();
    profile = (remaining: number) => linearProfile(table, remaining);
  }
  const height = (distance: number) => profile(lap - mod(distance, lap)) * VERTICAL_EXAGGERATION;
  const width = race.venueCode === '05' ? (kind === 'dirt' ? 25 : 36) : kind === 'dirt' ? 25 : 30;
  // Positive lanes always point away from the infield: the left normal for a
  // left-handed course, the right normal for a right-handed one.
  const side = courseFor(race).left ? 1 : -1;
  const sample = (distance: number, lane = 0): TrackPoint => {
    const u = mod(distance, lap) / lap;
    const point = curve.getPointAt(u);
    const tangent = curve.getTangentAt(u);
    const outward = new T.Vector3(-tangent.z, 0, tangent.x).normalize().multiplyScalar(side);
    point.addScaledVector(outward, lane);
    point.y = height(distance);
    return {point, tangent, outward};
  };
  return {
    kind,
    lap,
    width,
    scale,
    curve,
    sample,
    height,
    profile,
    pixel: (x, y) => new T.Vector3((x - cx) * scale, 0, (y - cy) * scale),
  };
}
// Ground surface that follows the drawn tracks. Every point stays a little below
// the nearest track surface, so ribbons and horses are never buried by terrain.
export function makeTerrain(tracks: TrackModel[], extras: T.Vector3[] = []) {
  const samples: {x: number; z: number; y: number}[] = [];
  // Extra racing surfaces (chutes) that are not a full lap of a track.
  for (const p of extras) samples.push({x: p.x, z: p.z, y: p.y});
  for (const t of tracks) {
    const step = Math.max(3, t.lap / 600);
    for (let d = 0; d < t.lap; d += step)
      for (const lane of [-4, 0, t.width / 2, t.width, t.width + 4]) {
        const p = t.sample(d, lane).point;
        samples.push({x: p.x, z: p.z, y: p.y});
      }
  }
  // Bucket samples so each lookup only visits nearby track.
  const CELL = 40,
    buckets = new Map<string, typeof samples>();
  for (const s of samples) {
    const k = `${Math.floor(s.x / CELL)},${Math.floor(s.z / CELL)}`;
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k)!.push(s);
  }
  const heightAt = (x: number, z: number) => {
    const cx = Math.floor(x / CELL),
      cz = Math.floor(z / CELL);
    for (let r = 1; r < 40; r++) {
      let num = 0,
        den = 0,
        nearest = Infinity,
        nearestY = 0,
        lowest = Infinity;
      for (let i = -r; i <= r; i++)
        for (let j = -r; j <= r; j++)
          for (const s of buckets.get(`${cx + i},${cz + j}`) ?? []) {
            const d2 = (s.x - x) ** 2 + (s.z - z) ** 2;
            if (d2 < nearest) {
              nearest = d2;
              nearestY = s.y;
            }
            if (d2 < 100) lowest = Math.min(lowest, s.y);
            const w = 1 / (d2 + 400) ** 1.5;
            num += w * s.y;
            den += w;
          }
      // Never above the nearest racing surface or any surface within 10m.
      if (den > 0 && Math.sqrt(nearest) < r * CELL)
        return Math.min(num / den, nearestY, lowest) - 0.4;
    }
    return 0;
  };
  return {heightAt, samples};
}
