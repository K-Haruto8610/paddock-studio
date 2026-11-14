import type {Race} from './types';
export type Placement = {gap: number; lane: number};
// Horses are drawn ~1.7x life size (≈4m long, ≈1.5m wide) so they stay legible
// against a true-scale course. All spacing below is in metres for that size.
export const HORSE_LENGTH = 4.0;
export const HORSE_WIDTH = 1.5;
export const LANE_PITCH = 2.0;
/** Closest a horse's centre gets to the inside rail (metres). */
export const MIN_LANE = 0.9;
export const laneSlots = (count: number) =>
  Array.from({length: count}, (_, j) => 1.8 + j * LANE_PITCH);
export function startLanes(race: Race) {
  const width = race.surface === 'ダート' ? 25 : 30,
    n = race.horses.length;
  const pitch = Math.min(1.6, (width - 4) / Math.max(1, n - 1));
  return race.horses.map((_, i) => 1.2 + i * pitch);
}
export function buildLayouts(race: Race): Placement[][] {
  const starts = startLanes(race).map((lane) => ({gap: 0, lane}));
  const stages = [starts];
  for (let phase = 0; phase < 4; phase++) {
    const order = race.horses
      .map((h, i) => ({h, i, score: h.positions[phase]}))
      .sort((a, b) => a.score - b.score || a.h.number - b.h.number);
    const min = order[0].score,
      result: Placement[] = Array(order.length),
      placed: Placement[] = [];
    for (const [rank, {h, i, score}] of order.entries()) {
      // Distances are visualization assumptions: neither time gaps nor lateral lanes are observed in JV-Data.
      const gap = (score - min) * 110 + rank * 1.8;
      const preferred =
        1.8 +
        Math.min(5, h.gate - 1) * LANE_PITCH * 0.8 +
        (phase >= 2 && h.positions[phase] < h.positions[0] ? LANE_PITCH : 0);
      const candidates = laneSlots(9).sort(
        (a, b) => Math.abs(a - preferred) - Math.abs(b - preferred),
      );
      const lane =
        candidates.find((l) =>
          placed.every(
            (p) =>
              Math.abs(p.gap - gap) >= HORSE_LENGTH * 1.3 ||
              Math.abs(p.lane - l) >= LANE_PITCH * 0.9,
          ),
        ) ?? candidates.at(-1)!;
      result[i] = {gap, lane};
      placed.push(result[i]);
    }
    stages.push(result);
  }
  return stages;
}
/** Plan placements at this travel distance, eased between checkpoints, before any separation. */
export function interpolateLayouts(
  layouts: Placement[][],
  travel: number,
  knots: number[],
): Placement[] {
  let segment = 0;
  while (segment < knots.length - 2 && travel > knots[segment + 1]) segment++;
  const t = Math.max(
    0,
    Math.min(1, (travel - knots[segment]) / (knots[segment + 1] - knots[segment])),
  );
  const smooth = t * t * (3 - 2 * t);
  return layouts[segment].map((a, i) => ({
    gap: a.gap + (layouts[segment + 1][i].gap - a.gap) * smooth,
    lane: a.lane + (layouts[segment + 1][i].lane - a.lane) * smooth,
  }));
}
/** Two horses closer than this along the course are alongside: they need sideways clearance. */
const REACH = HORSE_LENGTH * 1.25;
/** Sideways clearance two horses need at this distance along the course (0 once they are clear). */
const clearanceAt = (dx: number) => (dx >= REACH ? 0 : HORSE_WIDTH * Math.sqrt(1 - (dx / REACH) ** 2));
/** Metres of travel per simulated step of the lane tracks. */
const LANE_STEP = 1;
/** Sideways metres per metre travelled a horse can make up, on top of the plan's own drift. */
const CATCH_UP = 0.12;

/**
 * A pass between two checkpoints: one horse is clearly behind the other at the first (`from`, the
 * difference of their gaps) and clearly ahead at the second (`to`). Two horses running level, or
 * one only drawing level with the other, is not a pass.
 */
export const isPass = (from: number, to: number) =>
  from * to < 0 && Math.min(Math.abs(from), Math.abs(to)) >= HORSE_LENGTH * 0.25;

type LaneTrack = {knots: number[]; dt: number; lanes: Float64Array[]; sides: Int8Array[]};
const laneTracks = new WeakMap<Placement[][], LaneTrack>();

/**
 * The lanes the horses aim at, simulated once from the start to the finish: the plan's lanes,
 * except that a horse passing another never trades sides with it. It stays on the side it came up
 * on until the two are clear, and only then moves across to where the plan has it. A horse running
 * straight up behind the one it passes goes round on the side that has room. Horses with no pass
 * between them are left to the plan and to `separateLayouts`.
 */
function laneTrack(layouts: Placement[][], knots: number[]): LaneTrack {
  const cached = laneTracks.get(layouts);
  if (cached && cached.knots.length === knots.length && cached.knots.every((v, i) => v === knots[i]))
    return cached;
  const total = knots.at(-1)!,
    n = layouts[0].length,
    steps = Math.max(1, Math.ceil(total / LANE_STEP)),
    dt = total / steps;
  // side[i * n + j] for i < j, while one is passing the other: +1 = j goes by outside i, -1 = inside.
  const side = new Int8Array(n * n);
  let lanes = layouts[0].map((p) => p.lane),
    aim = lanes;
  let before = 0;
  const out: Float64Array[] = [],
    sides: Int8Array[] = [];
  for (let s = 0; s <= steps; s++) {
    const travel = Math.min(total, s * dt),
      plan = interpolateLayouts(layouts, travel, knots),
      gaps = plan.map((p) => p.gap);
    let segment = 0;
    while (segment < knots.length - 2 && travel > knots[segment + 1]) segment++;
    const next = layouts[segment + 1];
    // Where the horses are drawn right now: sides are settled from what is on screen.
    const shown = separate(
      gaps.map((gap, i) => ({gap, lane: lanes[i]})),
      side,
    ).map((p) => p.lane);

    // Does the horse coming up behind `p` have a free lane on this side of it?
    const room = (o: number, p: number, dir: number) => {
      if (dir < 0 && shown[p] - MIN_LANE < HORSE_WIDTH) return false;
      for (let k = 0; k < n; k++) {
        if (k === o || k === p) continue;
        const off = (shown[k] - shown[p]) * dir;
        if (
          off > HORSE_WIDTH * 0.5 &&
          off < HORSE_WIDTH * 2 &&
          gaps[k] > gaps[p] - REACH &&
          gaps[k] < gaps[o] + REACH
        )
          return false;
      }
      return true;
    };
    let passing: number[] = [];
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const at = i * n + j;
        if (Math.abs(gaps[i] - gaps[j]) >= REACH) {
          side[at] = 0;
          continue;
        }
        const pass = isPass(
          layouts[segment][i].gap - layouts[segment][j].gap,
          next[i].gap - next[j].gap,
        );
        // Still alongside at a checkpoint with the pass behind them: if they are on the sides the
        // plan has them on there, they go back to following it. If not, the side is kept.
        if (side[at] && !pass && segment !== before) {
          const planned = layouts[segment][j].lane - layouts[segment][i].lane;
          if (planned * side[at] >= 0) side[at] = 0;
        }
        if (!side[at]) {
          if (!pass) continue;
          // The side is settled here and kept until the two are clear of each other again: the
          // side the horse is already on, unless both are in the same lane, nose to tail.
          if (Math.abs(lanes[j] - lanes[i]) >= HORSE_WIDTH * 0.5 || gaps[i] === gaps[j])
            side[at] = shown[j] >= shown[i] ? 1 : -1;
          else {
            // Straight up behind: round the side with room, else the side it is heading for anyway.
            const [o, p] = gaps[i] > gaps[j] ? [i, j] : [j, i];
            const outside = room(o, p, 1),
              inside = room(o, p, -1),
              want = next[o].lane - next[p].lane,
              lean = shown[o] - shown[p];
            const dir =
              outside !== inside
                ? outside
                  ? 1
                  : -1
                : Math.abs(want) > 0.3
                  ? Math.sign(want)
                  : Math.abs(lean) > 0.05
                    ? Math.sign(lean)
                    : 1;
            side[at] = o === j ? dir : -dir;
          }
        }
        passing.push(at);
      }

    // A horse about to pass pulls out into clear air first: the plans keep the field only a few
    // lanes wide, so the lane it is planned in is usually taken by the horses it has to get by.
    const target = plan.map((p, o) => {
      const taken: number[] = [];
      // It cannot pull out across a horse that is alongside it: only as far as its neighbours.
      let blocked = false,
        low = MIN_LANE,
        high = Infinity;
      for (let k = 0; k < n; k++) {
        if (k === o) continue;
        const behind = gaps[o] - gaps[k];
        // Passing `k` between these checkpoints, and close enough behind (or not yet clear ahead).
        const passes =
          behind > -REACH &&
          behind < REACH * 2 &&
          next[o].gap < next[k].gap &&
          isPass(layouts[segment][o].gap - layouts[segment][k].gap, next[o].gap - next[k].gap);
        if (!passes && Math.abs(behind) >= REACH) continue;
        taken.push(shown[k]);
        if (Math.abs(behind) < REACH) {
          if (shown[k] < shown[o]) low = Math.max(low, shown[k] + HORSE_WIDTH);
          else high = Math.min(high, shown[k] - HORSE_WIDTH);
        }
        if (passes && Math.abs(p.lane - shown[k]) < HORSE_WIDTH) blocked = true;
      }
      if (!blocked) return p.lane;
      // The nearest free lane to where it is now; the outside one when both are as near.
      const free = (y: number) =>
        y >= low - 1e-6 &&
        y <= high + 1e-6 &&
        taken.every((l) => Math.abs(y - l) >= HORSE_WIDTH - 1e-6);
      let best = lanes[o];
      if (free(best)) return best;
      let nearest = Infinity;
      for (const l of taken)
        for (const y of [l + HORSE_WIDTH, l - HORSE_WIDTH]) {
          const away = Math.abs(y - lanes[o]);
          if (free(y) && away < nearest - 1e-6) [best, nearest] = [y, away];
        }
      return best;
    });
    // Head for that lane: as fast as the plan itself drifts, plus a little to catch up.
    const tracked = lanes.map((y, i) => {
      const limit = Math.abs(plan[i].lane - aim[i]) + CATCH_UP * dt;
      return y + Math.max(-limit, Math.min(limit, target[i] - y));
    });
    // Keep every passing pair on its side. Sides settled at different moments can contradict each
    // other in a crowd; then the pairs still fighting are let go and follow the plan again.
    let moved = tracked;
    for (let attempt = 0; attempt < 4; attempt++) {
      moved = [...tracked];
      const unsettled = new Set<number>();
      for (let pass = 0, pushed = true; pushed && pass < 400; pass++) {
        pushed = false;
        unsettled.clear();
        for (const at of passing) {
          const i = Math.floor(at / n),
            j = at % n,
            [inner, outer] = side[at] > 0 ? [i, j] : [j, i],
            clearance = clearanceAt(Math.abs(gaps[i] - gaps[j])),
            dy = moved[outer] - moved[inner];
          if (dy < clearance) {
            const need = clearance - dy + 0.001,
              inward = Math.max(0, Math.min(need / 2, moved[inner] - MIN_LANE));
            moved[inner] -= inward;
            moved[outer] += need - inward;
            pushed = true;
            unsettled.add(at);
          }
        }
      }
      if (!unsettled.size) break;
      for (const at of unsettled) side[at] = 0;
      passing = passing.filter((at) => side[at]);
    }
    lanes = moved;
    aim = plan.map((p) => p.lane);
    before = segment;
    out.push(Float64Array.from(lanes));
    sides.push(side.slice());
  }
  const track = {knots: [...knots], dt, lanes: out, sides};
  laneTracks.set(layouts, track);
  return track;
}

/** Placements at this travel distance: the plan's gaps, on lanes where a pass keeps its side. */
export function sampleLayouts(
  layouts: Placement[][],
  travel: number,
  knots: number[],
): Placement[] {
  const result = interpolateLayouts(layouts, travel, knots),
    track = laneTrack(layouts, knots),
    last = track.lanes.length - 1,
    x = Math.max(0, Math.min(last, travel / track.dt)),
    k = Math.min(last - 1, Math.floor(x)),
    f = x - k,
    a = track.lanes[k],
    b = track.lanes[k + 1];
  result.forEach((p, i) => (p.lane = a[i] + (b[i] - a[i]) * f));
  return separate(result, track.sides[Math.round(x)]);
}
/**
 * Pushes overlapping silhouettes apart sideways (mutates and returns the placements). It looks at
 * one moment only: on its own it is for a still field (a hand-edited point); playback goes through
 * `sampleLayouts`, which first puts each passing horse on the side it is passing on.
 */
export function separateLayouts(result: Placement[]): Placement[] {
  return separate(result);
}
/** `side[i * n + j]` (i < j), when given and not 0, fixes which of the two is the outer one. */
function separate(result: Placement[], side?: Int8Array): Placement[] {
  // The elliptical clearance grows smoothly as two horses draw level, rather than snapping the
  // field into fixed columns.
  const reach = REACH;
  // Repeat until no pair needs a push: a crowded field next to the rail can take more than a few passes.
  for (let pass = 0, moved = true; moved && pass < 200; pass++) {
    moved = false;
    for (let i = 0; i < result.length; i++)
      for (let j = i + 1; j < result.length; j++) {
        const a = result[i],
          b = result[j],
          dx = Math.abs(a.gap - b.gap);
        if (dx >= reach) continue;
        // A pair in the middle of a pass keeps its sides; any other pair is split the way it lies.
        const clearance = HORSE_WIDTH * Math.sqrt(1 - (dx / reach) ** 2),
          held = side?.[i * result.length + j],
          [inner, outer] = (held ? held > 0 : b.lane >= a.lane) ? [a, b] : [b, a],
          dy = outer.lane - inner.lane;
        if (dy < clearance) {
          // Split the push, but the inner horse stops at the rail and the outer one takes the rest.
          const need = clearance - dy + 0.001,
            inward = Math.max(0, Math.min(need / 2, inner.lane - MIN_LANE));
          inner.lane -= inward;
          outer.lane += need - inward;
          moved = true;
        }
      }
  }
  return result;
}
