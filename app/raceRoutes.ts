import * as T from 'three';
import type {Race} from './types';
import type {Pixel} from './courseData';
import {courseFor, checkpointDistances, type CourseKind} from './courses';
import {makeTrack, drawingFor, type TrackModel, type TrackPoint} from './trackGeometry';

/**
 * Official starting points of the JRA flat GⅠ races, in the pixel space of each
 * venue's plan image (references/jra/<venue>/pic_course_heimenzu.gif; Tokyo: plan.gif).
 *
 * `mark` is the start tick drawn on the plan next to the distance label, e.g.
 * "1,200m(外)". The route is built from it one of four ways:
 *  - main rail (default): the start lies on the course rail, placed by official
 *    remaining distance (rail length - distance). `markError` reports how far that
 *    is from the plan mark.
 *  - `chute` (引込線): a pocket off the main rail. The pixel list runs from the mark to the
 *    rail; its last point is snapped onto the rail and the pocket is stretched so
 *    the whole route is exactly the official distance.
 *  - `pin`: main-rail start taken from the plan mark itself (the drawn rail length
 *    disagrees with the official lap there); the route is rescaled to end at the post.
 *  - `first` (外→内 / 内→外): the race starts on the `first` course, passes the winning
 *    post on it (the two rails coincide there) and then runs the finishing course.
 *    `mark` is projected onto the first rail and the leg is rescaled to the official distance.
 */
export type RouteSpec = {
  plan: string;
  mark: Pixel;
  chute?: Pixel[];
  pin?: boolean;
  first?: CourseKind;
  /** 'polygons': the chute surface is drawn from plan polygons (Tokyo); default draws a ribbon. */
  surface?: 'polygons';
};

// Key: `<JRA venue code>:<finishing course kind>:<distance>`.
export const ROUTE_SPECS: Record<string, RouteSpec> = {
  // ---- 東京 (plan.gif 568x386). Main-rail starts have no chute.
  '05:turf:1600': {plan: '芝 1,600m (2コーナー奥の本線上)', mark: [434, 66]},
  '05:turf:2400': {plan: '芝 2,400m (正面スタンド前の本線上)', mark: [213, 272]},
  // Tokyo chutes, re-read from plan.gif (2026-11-08). Each list is lane 0, the inner
  // rail of the pocket (lanes spread outward = right of the running direction); it
  // starts at the plan's start bar and runs to where the pocket meets the main rail.
  '05:turf:2000': {
    plan: '芝 2,000m (右端の引込線、右上へ延び向正面へ合流)',
    mark: [534, 236],
    surface: 'polygons',
    chute: [[534, 236], [533, 190], [532, 150], [526, 134], [505, 111], [483, 90], [458, 81], [428, 80]], // prettier-ignore
  },
  '05:turf:1800': {
    plan: '芝 1,800m (右側の引込線、3コーナー手前で本線へ合流)',
    mark: [523, 155],
    surface: 'polygons',
    chute: [[523, 155], [505, 140], [480, 120], [458, 101]],
  },
  // The start bar sits on the turf pocket beyond the 2コーナー; the two dashed lines
  // either side of the label (y≈71 and y≈85) bound the starting stalls.
  '05:dirt:1600': {
    plan: 'ダ 1,600m (2コーナー奥の芝引込線から本線へ)',
    mark: [487, 85],
    surface: 'polygons',
    chute: [[487, 85], [462, 86], [437, 87], [414, 88]],
  },
  // ---- 中山 (572x438, 右回り)
  '06:turf:2000': {plan: '芝 2,000m(内) (スタンド前、4コーナー出口側)', mark: [433, 309]},
  // The outer rail's drawn length is 3% above the official lap, so the distance-based
  // start misses the tick by 2.4% of the lap; start from the tick itself.
  '06:outer:1200': {plan: '芝 1,200m(外) (向正面の外回り上、2コーナー出口の先)', mark: [202, 22], pin: true},
  // 2,500m(内) starts on the outer course (JRA: 発走地点は外回りコース上) and finishes on the inner.
  '06:turf:2500': {plan: '芝 2,500m(内) (外回り向正面、3コーナー手前)', mark: [417, 82], first: 'outer'},
  // ---- 京都 (571x306, 右回り)
  '08:turf:2000': {plan: '芝 2,000m(内) (4コーナー出口の先、本線上)', mark: [395, 201]},
  '08:outer:3200': {plan: '芝 3,200m(外) (向正面、2周)', mark: [263, 60]},
  '08:outer:3000': {plan: '芝 3,000m(外) (向正面、2周)', mark: [359, 42]},
  '08:outer:2200': {plan: '芝 2,200m(外) (4コーナー出口の先、本線上)', mark: [440, 206]},
  // 1,600m(外): the pocket on the back straight (向正面左手の引込線); the straight pocket
  // continues the back-straight rail to the left of the 2コーナー and joins it at ~x=216.
  '08:outer:1600': {plan: '芝 1,600m(外) (向正面の引込線)', mark: [120, 88], chute: [[120, 88], [168, 80], [216, 72]]},
  // ---- 阪神 (567x379, 右回り)
  '09:turf:2000': {plan: '芝 2,000m(内) (スタンド前、4コーナー出口の先)', mark: [343, 281]},
  // 2,200m(内): the pocket beyond the 4コーナー, in line with the home straight.
  '09:turf:2200': {plan: '芝 2,200m(内) (ホームストレッチ延長の引込線)', mark: [456, 280], chute: [[456, 280], [423, 279], [392, 277.5]]}, // prettier-ignore
  // 外1,600m and 内1,200m share one start on the back straight (JRA course page).
  '09:outer:1600': {plan: '芝 1,600m(外) (向正面半ば、本線上)', mark: [175, 118]},
  // ---- 中京 (571x325, 左回り)
  '07:turf:1200': {plan: '芝 1,200m (向正面、本線上)', mark: [343, 50]},
  '07:dirt:1800': {plan: 'ダ 1,800m (ホームストレッチ、本線上)', mark: [204.5, 209]},
};

export type RouteSource = 'main-rail' | 'chute' | 'pinned' | 'mixed';
export type RaceRoute = {
  sample: (travel: number, lane?: number) => TrackPoint;
  start: T.Vector3;
  chute: T.Vector3[];
  phaseDistances: number[];
  source: RouteSource;
  spec?: RouteSpec;
  /** Metres between the route's start and the plan's start mark (null without a mark). */
  markError: number | null;
  /** Pocket path for drawing (null when the surface comes from plan polygons or there is no chute). */
  chuteSurface: {length: number; sample: (u: number, lane?: number) => TrackPoint} | null;
};

const mod = (v: number, n: number) => ((v % n) + n) % n;

/** Distance along the track (metres after the winning post) closest to a world point. */
function nearestOnTrack(track: TrackModel, p: {x: number; z: number}) {
  const N = 4096,
    d2 = (d: number) => {
      const q = track.curve.getPointAt(mod(d, track.lap) / track.lap);
      return (q.x - p.x) ** 2 + (q.z - p.z) ** 2;
    };
  let best = 0,
    err = Infinity;
  for (let i = 0; i < N; i++) {
    const e = d2((i / N) * track.lap);
    if (e < err) {
      err = e;
      best = (i / N) * track.lap;
    }
  }
  const step = track.lap / N;
  let refined = best;
  for (let k = -20; k <= 20; k++) {
    const d = best + (k * step) / 20,
      e = d2(d);
    if (e < err) {
      err = e;
      refined = d;
    }
  }
  return {distance: mod(refined, track.lap), gap: Math.sqrt(err)};
}

function blend(a: TrackPoint, b: TrackPoint, w: number): TrackPoint {
  return {
    point: a.point.clone().lerp(b.point, w),
    tangent: a.tangent.clone().lerp(b.tangent, w).normalize(),
    outward: a.outward.clone().lerp(b.outward, w).normalize(),
  };
}

/** Course run first by a 内→外 / 内→外 race (JV-Data track codes 13/19 and 14/20), if drawn. */
function mixedFirst(race: Race, track: TrackModel): CourseKind | null {
  const first: CourseKind | null =
    [13, 19].includes(race.trackCode) && track.kind === 'outer'
      ? 'turf'
      : [14, 20].includes(race.trackCode) && track.kind === 'turf'
        ? 'outer'
        : null;
  return first && drawingFor(race.venueCode)?.rails[first] ? first : null;
}

export function specFor(race: Race, kind: CourseKind = courseFor(race).kind) {
  return ROUTE_SPECS[`${race.venueCode}:${kind}:${race.distance}`];
}

export function makeRoute(race: Race, track: TrackModel): RaceRoute {
  const D = race.distance,
    lap = track.lap,
    spec = specFor(race, track.kind),
    side = courseFor(race).left ? 1 : -1,
    phaseDistances = checkpointDistances(race);
  // Plan pixels share the scale of the turf rail (Tokyo calibrates each rail on its own).
  const turf = track.kind === 'turf' ? track : makeTrack(race, 'turf');
  const out = (
    sample: RaceRoute['sample'],
    source: RouteSource,
    markError: number | null,
    chute: T.Vector3[] = [],
    chuteSurface: RaceRoute['chuteSurface'] = null,
  ): RaceRoute => ({
    sample,
    start: sample(0).point,
    chute,
    phaseDistances,
    source,
    spec,
    markError,
    chuteSurface,
  });

  // 引込線: a pocket from the plan mark onto the main rail.
  if (spec?.chute) {
    const pts = spec.chute.map(([x, y]) => turf.pixel(x, y)),
      join = nearestOnTrack(track, pts.at(-1)!),
      joinPoint = track.sample(join.distance).point;
    pts[pts.length - 1] = joinPoint;
    const branch = new T.CatmullRomCurve3(pts, false, 'centripetal');
    branch.arcLengthDivisions = 1024;
    const length = branch.getLength(),
      // Whole laps run after the join: rail length from the join to the post, plus extra laps.
      laps = Math.max(1, Math.round((D - length + join.distance) / lap)),
      allotted = D - (laps * lap - join.distance),
      height = track.height(join.distance);
    const chutePoint = (u: number, lane = 0): TrackPoint => {
      const point = branch.getPointAt(u),
        tangent = branch.getTangentAt(u),
        outward = new T.Vector3(-tangent.z, 0, tangent.x).normalize().multiplyScalar(side);
      point.addScaledVector(outward, lane);
      point.y = height;
      return {point, tangent, outward};
    };
    const sample = (travel: number, lane = 0) =>
      travel >= allotted
        ? track.sample(join.distance + travel - allotted, lane)
        : chutePoint(T.MathUtils.clamp(travel / allotted, 0, 1), lane);
    return out(
      sample,
      'chute',
      0,
      pts,
      spec.surface === 'polygons' ? null : {length, sample: chutePoint},
    );
  }

  // 外→内 / 内→外: first course until the winning post, then the finishing course.
  const first = spec?.first ?? mixedFirst(race, track);
  if (first && first !== track.kind) {
    const firstTrack = makeTrack(race, first),
      start = spec ? nearestOnTrack(firstTrack, turf.pixel(...spec.mark)).distance : null,
      natural = start === null ? 0 : firstTrack.lap - start,
      laps = Math.max(1, start === null ? Math.ceil(D / lap) - 1 : Math.round((D - natural) / lap)),
      allotted = D - laps * lap,
      s1 = start ?? mod(firstTrack.lap - allotted, firstTrack.lap),
      len1 = start === null ? allotted : natural,
      window = 40;
    const leg1 = (t: number, lane: number) =>
      firstTrack.sample(s1 + (Math.min(t, allotted) / allotted) * len1, lane);
    const sample = (travel: number, lane = 0) => {
      if (travel >= allotted) return track.sample(travel - allotted, lane);
      if (travel <= allotted - window) return leg1(travel, lane);
      const w = T.MathUtils.smoothstep(travel, allotted - window, allotted);
      return blend(leg1(travel, lane), track.sample(travel - allotted, lane), w);
    };
    return out(sample, 'mixed', spec ? 0 : null);
  }

  const startDistance = mod(lap - D, lap),
    pixelMark = spec ? turf.pixel(...spec.mark) : null,
    mark = pixelMark ? nearestOnTrack(track, pixelMark).distance : null;
  // Main rail, start pinned to the plan mark: rescale so the route still ends at the post.
  if (spec?.pin && mark !== null) {
    const total = Math.max(1, Math.round((D + mark) / lap)) * lap - mark;
    return out((travel, lane = 0) => track.sample(mark + (travel / D) * total, lane), 'pinned', 0);
  }
  // Main rail, standard start by official remaining distance.
  let markError: number | null = null;
  if (mark !== null) {
    const gap = Math.abs(mark - startDistance);
    markError = Math.min(gap, lap - gap);
  }
  return out((travel, lane = 0) => track.sample(startDistance + travel, lane), 'main-rail', markError);
}
