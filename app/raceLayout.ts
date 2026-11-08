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
export function sampleLayouts(
  layouts: Placement[][],
  travel: number,
  knots: number[],
): Placement[] {
  return separateLayouts(interpolateLayouts(layouts, travel, knots));
}
/** Pushes overlapping silhouettes apart sideways (mutates and returns the placements). */
export function separateLayouts(result: Placement[]): Placement[] {
  // Keep silhouettes separate while passing. The elliptical clearance grows smoothly
  // as two horses draw level, rather than snapping the field into fixed columns.
  const reach = HORSE_LENGTH * 1.25;
  // Repeat until no pair needs a push: a crowded field next to the rail can take more than a few passes.
  for (let pass = 0, moved = true; moved && pass < 200; pass++) {
    moved = false;
    for (let i = 0; i < result.length; i++)
      for (let j = i + 1; j < result.length; j++) {
        const a = result[i],
          b = result[j],
          dx = Math.abs(a.gap - b.gap);
        if (dx >= reach) continue;
        const clearance = HORSE_WIDTH * Math.sqrt(1 - (dx / reach) ** 2),
          dy = b.lane - a.lane;
        if (Math.abs(dy) < clearance) {
          // Split the push, but the inner horse stops at the rail and the outer one takes the rest.
          const need = clearance - Math.abs(dy) + 0.001,
            [inner, outer] = dy >= 0 ? [a, b] : [b, a],
            inward = Math.max(0, Math.min(need / 2, inner.lane - MIN_LANE));
          inner.lane -= inward;
          outer.lane += need - inward;
          moved = true;
        }
      }
  }
  return result;
}
