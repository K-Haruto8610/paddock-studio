import * as T from 'three';
import type {Race} from './types';
import type {Pixel} from './courseData';
import {courseFor, checkpointDistances, type CourseKind} from './courses';
import {makeTrack, drawingFor, availableKinds, type TrackModel, type TrackPoint} from './trackGeometry';

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
  /** 'polygons': the pocket is also drawn from plan polygons (Tokyo), under the route's own strip. */
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
  /** Pocket path (null without a chute): keeps the ground below it and opens the outer rail. */
  chuteSurface: {
    length: number;
    /** Width of the strip: that of the course the pocket runs on to. */
    width: number;
    sample: (u: number, lane?: number) => TrackPoint;
  } | null;
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

/** Signed turn (radians) from one level direction to another. */
const turn = (a: T.Vector3, b: T.Vector3) => Math.atan2(a.x * b.z - a.z * b.x, a.x * b.x + a.z * b.z);

/**
 * The end of a pocket (引込線) redrawn as one curve that leaves the drawn line (`drawn`, ending
 * on the rail) and arrives running with the course. `onward` holds the course's directions
 * from the join on. Of the curves tried (how far back it starts, how wide it swings), this
 * keeps the one that weaves least: `weave` is how much it turns one way only to turn back,
 * in radians, counted on into the course.
 */
function easeOntoCourse(drawn: T.CatmullRomCurve3, onward: T.Vector3[]) {
  const length = drawn.getLength(),
    end = drawn.getPointAt(1).setY(0),
    level = (v: T.Vector3) => v.setY(0).normalize(),
    into = level(onward[0].clone());
  const hermite = (from: T.Vector3, m0: T.Vector3, m1: T.Vector3, t: number) => {
    const t2 = t * t,
      t3 = t2 * t;
    return from
      .clone()
      .multiplyScalar(2 * t3 - 3 * t2 + 1)
      .addScaledVector(m0, t3 - 2 * t2 + t)
      .addScaledVector(end, -2 * t3 + 3 * t2)
      .addScaledVector(m1, t3 - t2);
  };
  let best: {back: number; pull: number; weave: number} | null = null;
  for (let back = 40; back <= Math.max(40, length * 0.85); back += 10) {
    if (back > length) break;
    const u = 1 - back / length,
      from = drawn.getPointAt(u).setY(0),
      leave = level(drawn.getTangentAt(u)),
      reach = from.distanceTo(end);
    for (const pull of [0.5, 0.75, 1]) {
      const m0 = leave.clone().multiplyScalar(reach * pull),
        m1 = into.clone().multiplyScalar(reach * pull);
      const ways = [leave];
      for (let i = 1; i <= 16; i++)
        ways.push(hermite(from, m0, m1, i / 16).sub(hermite(from, m0, m1, (i - 1) / 16)));
      ways.push(...onward);
      let turned = 0,
        net = 0;
      for (let i = 1; i < ways.length; i++) {
        const step = turn(ways[i - 1], ways[i]);
        turned += Math.abs(step);
        net += step;
      }
      const weave = turned - Math.abs(net);
      if (!best || weave < best.weave - 0.005) best = {back, pull, weave};
    }
  }
  const {back, pull, weave} = best ?? {back: length * 0.6, pull: 1, weave: Infinity};
  const points: T.Vector3[] = [];
  const kept = length - back;
  for (let i = 0, n = Math.max(1, Math.ceil(kept / 8)); i < n; i++)
    points.push(drawn.getPointAt((i / n) * (kept / length)).setY(0));
  const from = drawn.getPointAt(kept / length).setY(0),
    reach = from.distanceTo(end),
    m0 = level(drawn.getTangentAt(kept / length)).multiplyScalar(reach * pull),
    m1 = into.clone().multiplyScalar(reach * pull);
  for (let i = 0, n = Math.ceil(back / 6); i <= n; i++) points.push(hermite(from, m0, m1, i / n));
  const curve = new T.CatmullRomCurve3(points, false, 'centripetal');
  curve.arcLengthDivisions = 1024;
  return {curve, weave};
}

/**
 * Where a pocket joins its course, and the path it takes to get there.
 *
 * The pocket drawn on the plan ends on the rail at an angle (22° for Tokyo's 2,000m): a field
 * following it kinks outward at the join. The end of the pocket is redrawn to arrive running
 * with the course. A pocket in line with a straight (Kyoto's 1,600m, Hanshin's 2,200m) reaches
 * the rail while that is still coming off its bend, so its join moves on down the course, to
 * where it can run in with the least weaving and without cutting inside the rail.
 */
function pocketJoin(track: TrackModel, drawn: T.Vector3[]) {
  const snapped = nearestOnTrack(track, drawn.at(-1)!).distance;
  /** Lanes from the rail (negative: inside it), looking only at the course around the join. */
  const laneOf = (p: T.Vector3, upTo: number) => {
    let lane = 0,
      err = Infinity;
    for (let d = snapped - 60; d <= upTo + 10; d += 2) {
      const s = track.sample(d),
        e = (s.point.x - p.x) ** 2 + (s.point.z - p.z) ** 2;
      if (e < err) {
        err = e;
        lane = (p.x - s.point.x) * s.outward.x + (p.z - s.point.z) * s.outward.z;
      }
    }
    return lane;
  };
  const attempt = (ahead: number) => {
    const at = snapped + ahead,
      to = track.sample(at),
      line = new T.CatmullRomCurve3([...drawn.slice(0, -1), to.point], false, 'centripetal');
    line.arcLengthDivisions = 1024;
    const onward = [0, 15, 30, 45, 60, 80, 100].map((d) => track.sample(at + d).tangent);
    const eased = easeOntoCourse(line, onward);
    // Only the stretch beside the course matters: further back the pocket is well outside it.
    let clear = true;
    for (let i = 0; i <= 24 && clear; i++)
      clear = laneOf(eased.curve.getPointAt(1 - (i / 24) * 0.35), at) > -1.5;
    return {distance: at, branch: eased.curve, weave: clear ? eased.weave : Infinity};
  };
  let best = attempt(0);
  // A join further on has to weave clearly less (2°) to be worth leaving the drawn one.
  for (let ahead = 20; ahead <= 240 && best.weave > 0.035; ahead += 20) {
    const moved = attempt(ahead);
    if (moved.weave < best.weave - 0.035) best = moved;
  }
  return best;
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
  const turf = track.kind === 'turf' ? track : makeTrack(race, 'turf', track.elevationScale);
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
      join = pocketJoin(track, pts),
      branch = join.branch;
    pts[pts.length - 1] = branch.getPointAt(1);
    const length = branch.getLength(),
      // Whole laps run after the join: rail length from the join to the post, plus extra laps.
      laps = Math.max(1, Math.round((D - length + join.distance) / lap)),
      allotted = D - (laps * lap - join.distance);
    // The pocket takes the height of the courses it runs on or beside, weighted by
    // how close each one's centre is, so it meets every surface it overlaps
    // without a step (Hanshin's 2,200m pocket lies under the outer course's 4コーナー).
    const courses = availableKinds(race).map((k) => (k === track.kind ? track : makeTrack(race, k, track.elevationScale)));
    const heights: number[] = [];
    for (let i = 0; i <= 64; i++) {
      // Centre of the pocket (the branch itself is its lane 0).
      const t = branch.getTangentAt(i / 64),
        p = branch
          .getPointAt(i / 64)
          .addScaledVector(new T.Vector3(-t.z, 0, t.x).normalize(), (side * track.width) / 2);
      let num = 0,
        den = 0;
      for (const c of courses) {
        const near = nearestOnTrack(c, p),
          centre = Math.max(0, Math.abs(near.gap - c.width / 2)),
          w = 1 / (centre + 1) ** 4;
        num += w * c.height(near.distance);
        den += w;
      }
      heights.push(num / den);
    }
    // Where the neighbouring courses' graphs disagree with the main rail's, the pocket still has to
    // arrive at the rail's own height at the join: that disagreement is faded in along the pocket.
    // (Pulling the whole pocket towards the join's one height instead sank it below the courses it
    // lies on wherever they slope: Hanshin's 2,200m pocket on the outer course's 4コーナー.)
    const miss = track.height(join.distance) - heights[64];
    for (let i = 0; i <= 64; i++) heights[i] += miss * T.MathUtils.smoothstep(i / 64, 0, 1);
    const height = (u: number) => {
      const f = T.MathUtils.clamp(u, 0, 1) * 64,
        i = Math.min(63, Math.floor(f));
      return T.MathUtils.lerp(heights[i], heights[i + 1], f - i);
    };
    const chutePoint = (u: number, lane = 0): TrackPoint => {
      const point = branch.getPointAt(u),
        tangent = branch.getTangentAt(u),
        outward = new T.Vector3(-tangent.z, 0, tangent.x).normalize().multiplyScalar(side);
      point.addScaledVector(outward, lane);
      point.y = height(u);
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
      {length, width: track.width, sample: chutePoint},
    );
  }

  // 外→内 / 内→外: first course until the winning post, then the finishing course.
  const first = spec?.first ?? mixedFirst(race, track);
  if (first && first !== track.kind) {
    const firstTrack = makeTrack(race, first, track.elevationScale),
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
