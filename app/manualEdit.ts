import type {Race} from './types';
import type {Checkpoint, PlaybackPlan} from './playback';
import {checkpointDistances} from './courses';
import {interpolateLayouts, HORSE_LENGTH, MIN_LANE, type Placement} from './raceLayout';

/** The four moments of the race the user can rearrange by hand. */
export type EditKey = 'early' | 'middle' | 'corner4' | 'stretch';
export type EditPoint = {key: EditKey; label: string; distance: number};
/** Hand-placed horses per edit point, keyed by horse number. Placements are the plan's own units. */
export type RaceEdits = Partial<Record<EditKey, Record<string, Placement>>>;

export const MAX_LANE = 28;
export const clampPlacement = (p: Placement): Placement => ({
  gap: Math.max(-HORSE_LENGTH * 10, Math.min(500, p.gap)),
  lane: Math.max(MIN_LANE, Math.min(MAX_LANE, p.lane)),
});

/** 序盤 = 200 m from the start, 中盤 = the middle of the race, 4角, 最終直線 = half-way up the straight. */
export function editPoints(race: Race): EditPoint[] {
  const corner4 = checkpointDistances(race)[4],
    stretch = corner4 + (race.distance - corner4) * 0.5;
  const early = Math.min(200, corner4 * 0.3);
  // Keep the middle between 序盤 and 4角 even on a short race.
  const middle = Math.max(early + 40, Math.min(race.distance / 2, corner4 - 40));
  return [
    {key: 'early', label: '序盤', distance: early},
    {key: 'middle', label: '中盤', distance: middle},
    {key: 'corner4', label: '4角', distance: corner4},
    {key: 'stretch', label: '最終直線', distance: stretch},
  ];
}

export const editCount = (edits: RaceEdits | undefined) =>
  Object.values(edits ?? {}).reduce((n, e) => n + Object.keys(e ?? {}).length, 0);

const ranksOf = (layout: Placement[]) => {
  const ranks = Array<number>(layout.length);
  layout
    .map((p, i) => ({p, i}))
    .sort((a, b) => a.p.gap - b.p.gap || a.p.lane - b.p.lane || a.i - b.i)
    .forEach((v, k) => (ranks[v.i] = k + 1));
  return ranks;
};

/**
 * The base plan with the four edit points inserted as checkpoints. An unedited horse keeps the
 * base plan's position there; an edited one is placed where the user put it. Base checkpoints
 * within 25 m of an edit point are replaced by it, so the field never jerks between two keys.
 */
export function applyEdits(race: Race, base: PlaybackPlan, edits: RaceEdits): PlaybackPlan {
  const points = editPoints(race);
  const keep = base.checkpoints
    .map((c, k) => ({c, layout: base.layouts[k]}))
    .filter(
      ({c}, k, all) =>
        k === 0 || k === all.length - 1 || points.every((p) => Math.abs(p.distance - c.distance) > 25),
    );
  const added = points.map((p) => {
    const c: Checkpoint = {
      label: p.label,
      distance: p.distance,
      observed: false,
      ranks: [],
      edit: p.key,
      edited: Object.keys(edits[p.key] ?? {}).length > 0,
    };
    return {c, layout: interpolateLayouts(base.layouts, p.distance, base.knots)};
  });
  const all = [...keep, ...added]
    .sort((a, b) => a.c.distance - b.c.distance)
    .map((v) => ({c: {...v.c}, layout: v.layout.map((p) => ({...p}))}));
  // An edit moves the horse from there on: later points keep the same shift from the model
  // (through the finish) until the horse is edited again.
  race.horses.forEach((h, i) => {
    let shift: Placement | null = null;
    for (const {c, layout} of all.slice(1)) {
      const placed = c.edit ? edits[c.edit]?.[String(h.number)] : undefined;
      if (placed) {
        const to = clampPlacement(placed);
        shift = {gap: to.gap - layout[i].gap, lane: to.lane - layout[i].lane};
        layout[i] = to;
      } else if (shift)
        layout[i] = clampPlacement({gap: layout[i].gap + shift.gap, lane: layout[i].lane + shift.lane});
    }
  });
  const changed = editCount(edits) > 0;
  for (const v of all.slice(1))
    if (v.c.edit || changed) v.c.ranks = ranksOf(v.layout);
  return {
    ...base,
    mode: 'forecast',
    edited: true,
    layouts: all.map((v) => v.layout),
    knots: all.map((v) => v.c.distance),
    checkpoints: all.map((v) => v.c),
  };
}

const STORAGE_KEY = 'paddock-studio:manual-edits';
export function loadEdits(): Record<string, RaceEdits> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, RaceEdits>) : {};
  } catch {
    return {};
  }
}
export function saveEdits(all: Record<string, RaceEdits>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {
    // Storage blocked: edits still work for this visit.
  }
}
