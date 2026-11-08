import type {Race, ReplayRace, PlaybackMode} from './types';
import type {RacePace} from './paceForecast';
import {checkpointDistances, cornerLabels, courseFor} from './courses';
import {
  buildLayouts,
  sampleLayouts,
  laneSlots,
  HORSE_LENGTH,
  LANE_PITCH,
  type Placement,
} from './raceLayout';

export type Checkpoint = {
  label: string;
  distance: number;
  observed: boolean;
  ranks: (number | null)[];
  corner?: number;
};
export type PlaybackPlan = {
  layouts: Placement[][];
  knots: number[];
  checkpoints: Checkpoint[];
  mode: PlaybackMode;
  available: boolean;
  /** Forecast only: 'pace' = imported race-flow model, 'legacy' = per-corner rank model. */
  source?: 'pace' | 'legacy';
};

type PaceSpot = {gap: number; lane: number; rank: number};

/** Imported gaps (lengths) and lanes (slots from the inside) placed on the track in metres.
 * Order follows the imported rank; lanes are the model's; a horse is only pushed back when it
 * would sit on another silhouette. */
function paceLayout(spots: PaceSpot[]): Placement[] {
  const out: Placement[] = Array(spots.length),
    placed: Placement[] = [];
  const order = spots
    .map((v, i) => ({i, v}))
    .sort((a, b) => a.v.rank - b.v.rank || a.v.gap - b.v.gap || a.i - b.i);
  let last = 0;
  for (const {i, v} of order) {
    const lane = 1.8 + v.lane * LANE_PITCH;
    let gap = Math.max(v.gap * HORSE_LENGTH * 1.3, last);
    for (const p of placed)
      if (Math.abs(p.lane - lane) < LANE_PITCH * 0.9 && gap - p.gap < HORSE_LENGTH * 1.05)
        gap = p.gap + HORSE_LENGTH * 1.05;
    // Strictly behind the horse ranked ahead, so the 3D order always matches the imported ranks.
    gap = Math.max(gap, last + 0.05 * (placed.length > 0 ? 1 : 0));
    last = gap;
    out[i] = {gap, lane};
    placed.push(out[i]);
  }
  return out;
}

function paceForecastPlan(
  race: Race,
  pace: RacePace,
  start: Checkpoint,
  startLayout: Placement[],
): PlaybackPlan {
  const distances = checkpointDistances(race),
    checkpoints = [start],
    layouts = [startLayout];
  pace.corners.forEach((corner, k) => {
    layouts.push(paceLayout(pace.horses.map((h) => h.positions[k])));
    checkpoints.push({
      label: `${corner}角`,
      distance: distances[corner],
      observed: false,
      ranks: pace.horses.map((h) => h.positions[k].rank),
      corner,
    });
  });
  // Finish: the model's own estimate when given, else last corner + closing 3F difference (1 s = 5 lengths).
  const best = Math.min(...pace.horses.map((h) => h.last3));
  const finish = pace.horses.map(
    (h) => h.finish ?? {rank: 0, gap: h.positions.at(-1)!.gap + (h.last3 - best) * 5},
  );
  const lead = Math.min(...finish.map((f) => f.gap)),
    ranks = Array<number>(race.horses.length);
  finish
    .map((f, i) => ({f, i}))
    .sort((a, b) => (a.f.rank && b.f.rank ? a.f.rank - b.f.rank : a.f.gap - b.f.gap) || a.i - b.i)
    .forEach((v, k) => (ranks[v.i] = k + 1));
  // A horse shut in at the last corner has to find room before it can kick: it is drawn angling out.
  const finalLane = (h: RacePace['horses'][number]) =>
    h.positions.at(-1)!.lane + ((h.kickDelay ?? 0) > 0.05 ? 1 : 0);
  // 仕掛け遅れ: half-way up the straight, free horses have made half their ground; a delayed horse
  // has made less (none at 0.3 s or more), then finishes late.
  if (pace.horses.some((h) => (h.kickDelay ?? 0) > 0.05)) {
    const last = pace.horses.map((h) => h.positions.at(-1)!.gap);
    const mid = pace.horses.map((h, i) => {
      const share = 0.5 * Math.max(0, 1 - (h.kickDelay ?? 0) / 0.3);
      return last[i] + (finish[i].gap - lead - last[i]) * share;
    });
    const midLead = Math.min(...mid),
      midRanks = Array<number>(race.horses.length);
    mid
      .map((g, i) => ({g, i}))
      .sort((a, b) => a.g - b.g || pace.horses[a.i].positions.at(-1)!.rank - pace.horses[b.i].positions.at(-1)!.rank)
      .forEach((v, k) => (midRanks[v.i] = k + 1));
    const approach = checkpoints.at(-1)!.distance;
    layouts.push(
      paceLayout(
        pace.horses.map((h, i) => ({
          gap: mid[i] - midLead,
          lane: (h.positions.at(-1)!.lane + finalLane(h)) / 2,
          rank: midRanks[i],
        })),
      ),
    );
    checkpoints.push({
      label: '直線',
      distance: approach + (race.distance - approach) * 0.5,
      observed: false,
      ranks: midRanks,
    });
  }
  layouts.push(
    paceLayout(
      pace.horses.map((h, i) => ({
        gap: finish[i].gap - lead,
        lane: finalLane(h),
        rank: ranks[i],
      })),
    ),
  );
  checkpoints.push({label: 'ゴール推定', distance: race.distance, observed: false, ranks});
  return {
    mode: 'forecast',
    source: 'pace',
    available: true,
    layouts,
    knots: checkpoints.map((c) => c.distance),
    checkpoints,
  };
}

function observedLayout(race: Race, ranks: number[], previous?: Placement[]): Placement[] {
  const minimum = Math.min(...ranks),
    out: Placement[] = Array(ranks.length),
    placed: Placement[] = [];
  for (const {rank, i} of ranks
    .map((rank, i) => ({rank, i}))
    .sort((a, b) => a.rank - b.rank || a.i - b.i)) {
    const gap = (rank - minimum) * HORSE_LENGTH * 1.3;
    // Lanes/gaps are illustrative; only ranks come from the recorded results.
    const advancing = previous && gap < previous[i].gap;
    const preferred =
      1.8 +
      Math.min(4, race.horses[i].gate - 1) * LANE_PITCH * 0.8 +
      (advancing ? LANE_PITCH * 1.5 : 0);
    const lanes = laneSlots(10).sort((a, b) => Math.abs(a - preferred) - Math.abs(b - preferred));
    const lane =
      lanes.find((l) =>
        placed.every(
          (p) =>
            Math.abs(p.gap - gap) >= HORSE_LENGTH * 1.3 || Math.abs(p.lane - l) >= LANE_PITCH * 0.9,
        ),
      ) ?? lanes.at(-1)!;
    out[i] = {gap, lane};
    placed.push(out[i]);
  }
  return out;
}

export function buildPlayback(
  race: Race,
  mode: PlaybackMode,
  replay?: ReplayRace,
  pace?: RacePace | null,
): PlaybackPlan {
  const forecast = buildLayouts(race),
    distances = checkpointDistances(race);
  const start: Checkpoint = {
    label: 'スタート',
    distance: 0,
    observed: false,
    ranks: race.horses.map(() => null),
  };
  if (mode === 'replay') {
    if (!replay)
      return {
        mode,
        available: false,
        layouts: [forecast[0], forecast[0]],
        knots: [0, race.distance],
        checkpoints: [start, {...start, label: '実績データなし', distance: race.distance}],
      };
    const checkpoints = [start],
      layouts = [forecast[0]];
    for (const corner of replay.corners) {
      const ranks = race.horses.map(
        (h) => replay.horses[String(h.number)]?.corners[corner - 1] ?? null,
      );
      if (ranks.some((r) => r === null)) continue;
      layouts.push(observedLayout(race, ranks as number[], layouts.at(-1)));
      checkpoints.push({
        label: `${corner}角`,
        distance: distances[corner],
        observed: true,
        ranks,
        corner,
      });
    }
    if (replay.completeFinish) {
      const ranks = race.horses.map((h) => replay.horses[String(h.number)]?.finish ?? null);
      if (ranks.every((r) => r !== null)) {
        layouts.push(observedLayout(race, ranks as number[], layouts.at(-1)));
        checkpoints.push({label: 'ゴール', distance: race.distance, observed: true, ranks});
      }
    }
    // A missing terminal observation is explicitly held, never replaced by prediction results.
    if (checkpoints.at(-1)!.distance < race.distance) {
      layouts.push(layouts.at(-1)!);
      checkpoints.push({
        label: '入線データなし',
        distance: race.distance,
        observed: false,
        ranks: race.horses.map(() => null),
      });
    }
    return {
      mode,
      available: checkpoints.some((c) => c.observed),
      layouts,
      knots: checkpoints.map((c) => c.distance),
      checkpoints,
    };
  }
  if (pace) return paceForecastPlan(race, pace, start, forecast[0]);
  const labels = cornerLabels(race),
    checkpoints = [
      start,
      ...[1, 2, 3, 4].map((c) => ({
        label: labels[c - 1],
        distance: distances[c],
        observed: false,
        ranks: race.horses.map((h) => h.positions[c - 1] * (race.horses.length - 1) + 1),
      })),
    ];
  // Existing last-3F prediction now affects the straight instead of freezing the field.
  // This is a relative-motion estimate, not an observed or separately trained finishing rank.
  const ratio = Math.min(1, courseFor(race).straight / 600),
    best = Math.min(...race.horses.map((h) => h.last3));
  const finalGaps = forecast[4].map((p, i) => p.gap + (race.horses[i].last3 - best) * 16 * ratio);
  const first = Math.min(...finalGaps),
    finish = forecast[4].map((p, i) => ({gap: finalGaps[i] - first, lane: p.lane}));
  const order = finish.map((v, i) => ({gap: v.gap, i})).sort((a, b) => a.gap - b.gap),
    ranks = Array(race.horses.length);
  order.forEach((v, i) => (ranks[v.i] = i + 1));
  checkpoints.push({label: 'ゴール推定', distance: race.distance, observed: false, ranks});
  return {
    mode,
    source: 'legacy',
    available: true,
    layouts: [...forecast, finish],
    knots: [...distances, race.distance],
    checkpoints,
  };
}

export function playbackFrame(race: Race, plan: PlaybackPlan, progress: number) {
  const travel = Math.max(0, Math.min(1, progress)) * race.distance;
  const placements = sampleLayouts(plan.layouts, travel, plan.knots);
  const exact = plan.checkpoints.find((c) => Math.abs(c.distance - travel) < 0.05);
  let previous = 0;
  for (let i = 0; i < plan.checkpoints.length; i++)
    if (plan.checkpoints[i].distance <= travel) previous = i;
  const next = Math.min(previous + 1, plan.checkpoints.length - 1);
  const label = exact
    ? exact.label
    : `${plan.checkpoints[previous].label} → ${plan.checkpoints[next].label}`;
  const order = placements
    .map((p, i) => ({horse: race.horses[i], gap: p.gap, index: i}))
    .sort((a, b) => a.gap - b.gap || a.horse.number - b.horse.number);
  return {travel, placements, exact, label, order, previous: plan.checkpoints[previous]};
}
