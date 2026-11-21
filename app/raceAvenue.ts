import * as T from 'three';
import type {TrackModel} from './trackGeometry';
import type {Species} from './raceSeason';

/**
 * Where the avenue (並木) runs: lines just outside the outermost rail of every course drawn for the race, so the trees stand
 * the same whichever course the race is on. A line is broken wherever something else needs the ground: a pocket (引込線)
 * leaving the course, the start sign, the main stand and its apron.
 */
export type AvenueSite = {
  /** Every course drawn for the race (its own and the venue's other GⅠ courses). */
  tracks: TrackModel[];
  /** The pockets drawn for the race: a strip `width` wide from lane 0, sampled along it by `t` in 0..1. */
  pockets: {sample: (t: number, lane?: number) => {point: T.Vector3}; width: number}[];
  /** Ground the avenue keeps off (the stand and its apron, the start sign). */
  keepOut: ((p: T.Vector3) => boolean)[];
};

/** Metres from the outer rail to the trunks of the first row. */
export const AVENUE_GAP = 2;
/** The lines planted, as offsets from the outermost rail: the trees, the hedge behind them, the second row. */
export const AVENUE_LINES = {front: AVENUE_GAP, hedge: AVENUE_GAP + 3.5, back: AVENUE_GAP + 12} as const;

/** The species of the avenue along each stretch: cherry on the back straight, ginkgo on the far turn, zelkova elsewhere. */
export function avenueSpecies(fraction: number): Species {
  return fraction > 0.3 && fraction < 0.62 ? 'cherry' : fraction >= 0.62 && fraction < 0.85 ? 'ginkgo' : 'zelkova';
}

/** A stretch of an avenue line: points 0.5 m apart on the ground plan (y is the course's), each with its species. */
export type AvenueRun = {points: T.Vector3[]; species: Species[]};

/**
 * Lane across a course of any point near it: metres from its inner rail, outward positive (negative in the infield).
 * Nearest sample of the course's line, 1 m apart, found through a coarse grid.
 */
export function acrossOf(track: TrackModel) {
  const CELL = 40;
  type S = {x: number; z: number; ox: number; oz: number};
  const cells = new Map<string, S[]>();
  for (let d = 0; d < track.lap; d += 1) {
    const {point, outward} = track.sample(d, 0);
    const key = `${Math.floor(point.x / CELL)},${Math.floor(point.z / CELL)}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key)!.push({x: point.x, z: point.z, ox: outward.x, oz: outward.z});
  }
  return (p: T.Vector3) => {
    const cx = Math.floor(p.x / CELL), cz = Math.floor(p.z / CELL);
    for (let r = 1; r < 30; r++) {
      let best: S | undefined, bd = Infinity;
      for (let i = -r; i <= r; i++)
        for (let j = -r; j <= r; j++)
          for (const s of cells.get(`${cx + i},${cz + j}`) ?? []) {
            const d2 = (s.x - p.x) ** 2 + (s.z - p.z) ** 2;
            if (d2 < bd) { bd = d2; best = s; }
          }
      if (best && Math.sqrt(bd) < r * CELL) return (p.x - best.x) * best.ox + (p.z - best.z) * best.oz;
    }
    return Infinity;
  };
}

/**
 * The runs of one avenue line `offset` metres outside the outermost rail. Each course's line is walked (outermost course
 * first); a point is kept when it lies at least `offset` outside every course, clear of the pockets and of `keepOut`, and
 * is not already on a line kept from another course (where two courses share a rail).
 */
export function avenueRuns(site: AvenueSite, offset: number): AvenueRun[] {
  // The trees and the shrubs walk the same lines: work each out once per site.
  const cached = RUNS.get(site)?.get(offset);
  if (cached) return cached;
  const runs = lineRuns(site, offset);
  if (!RUNS.has(site)) RUNS.set(site, new Map());
  RUNS.get(site)!.set(offset, runs);
  return runs;
}
const RUNS = new WeakMap<AvenueSite, Map<number, AvenueRun[]>>();

function lineRuns(site: AvenueSite, offset: number): AvenueRun[] {
  const tracks = [...site.tracks].sort((a, b) => b.lap - a.lap);
  const across = tracks.map(acrossOf);
  const pocketLines = site.pockets.map(pocket => ({
    reach: (pocket.width / 2 + offset + 1.5) ** 2,
    points: Array.from({length: 241}, (_, i) => pocket.sample(i / 240, pocket.width / 2).point),
  }));
  const kept = new Map<string, T.Vector3[]>();
  const keyOf = (p: T.Vector3) => `${Math.floor(p.x / 4)},${Math.floor(p.z / 4)}`;
  const taken = (p: T.Vector3) => {
    const cx = Math.floor(p.x / 4), cz = Math.floor(p.z / 4);
    for (let i = -1; i <= 1; i++)
      for (let j = -1; j <= 1; j++)
        for (const q of kept.get(`${cx + i},${cz + j}`) ?? []) if ((q.x - p.x) ** 2 + (q.z - p.z) ** 2 < 2.25) return true;
    return false;
  };
  const runs: AvenueRun[] = [];
  tracks.forEach((track, k) => {
    const ok = (p: T.Vector3) =>
      across.every((a, j) => j === k || a(p) > tracks[j].width + offset - 0.25) &&
      !pocketLines.some(({reach, points}) => points.some(q => (q.x - p.x) ** 2 + (q.z - p.z) ** 2 < reach)) &&
      !site.keepOut.some(out => out(p)) && !taken(p);
    const steps = Math.floor(track.lap / 0.5);
    const line: {p: T.Vector3; keep: boolean; d: number}[] = [];
    for (let i = 0; i < steps; i++) {
      const d = i * 0.5;
      const p = track.sample(d, track.width + offset).point;
      line.push({p, keep: ok(p), d});
    }
    // Start the walk at a gap (if any), so a run never wraps round the loop's seam.
    const start = Math.max(0, line.findIndex(s => !s.keep));
    const own: AvenueRun[] = [];
    let run: AvenueRun | null = null;
    for (let n = 0; n < steps; n++) {
      const s = line[(start + n) % steps];
      if (s.keep) {
        run ??= {points: [], species: []};
        run.points.push(s.p);
        run.species.push(avenueSpecies(s.d / track.lap));
      } else if (run) {
        own.push(run);
        run = null;
      }
    }
    if (run) own.push(run);
    runs.push(...own);
    // Later courses keep off this one's line (taken), not off their own.
    for (const run of own)
      for (const p of run.points) {
        const key = keyOf(p);
        if (!kept.has(key)) kept.set(key, []);
        kept.get(key)!.push(p);
      }
  });
  return runs.filter(run => run.points.length > 4);
}

/**
 * Places along the runs every `step` metres of ground, the first `first` metres from each run's start: the point, the
 * direction the line runs there (radians about y) and the species. Runs too short for one place are skipped.
 */
export function walkRuns(runs: AvenueRun[], step: number, first: number) {
  const places: {at: T.Vector3; yaw: number; species: Species; index: number}[] = [];
  let index = 0;
  for (const {points, species} of runs) {
    let walked = step - first;
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i];
      walked += Math.hypot(b.x - a.x, b.z - a.z);
      if (walked < step) continue;
      walked = 0;
      const ahead = points[Math.min(points.length - 1, i + 1)];
      places.push({at: b.clone(), yaw: Math.atan2(ahead.x - a.x, ahead.z - a.z) - Math.PI / 2, species: species[i], index: index++});
    }
  }
  return places;
}
