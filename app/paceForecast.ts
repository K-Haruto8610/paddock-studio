// Import format for race-flow (展開予想) models: `paddock.pace-forecast/v1`.
// Any model that writes this shape can be loaded, either bundled at /data/pace_forecasts.json
// or picked from disk in the UI. See README「展開予想モデルの取り込み形式」.
import {z} from 'zod';
import type {Race} from './types';

export const PACE_SCHEMA = 'paddock.pace-forecast/v1';

const position = z.object({
  corner: z.number().int().min(1).max(4),
  rank: z.number().positive(),
  gap: z.number().min(0).max(60), // lengths behind the leader (前後)
  lane: z.number().min(0).max(12), // lateral slot counted from the inside, 0 = rail side (左右)
  // Optional race-interaction output (ポジション争い):
  wantRank: z.number().positive().optional(), // where the horse wanted to be before contention
  held: z.boolean().optional(), // its spot was taken, so it tucked in behind (控えた)
  boxed: z.boolean().optional(), // horse outside and a wall under one length ahead (包まれ)
});
const horse = z.object({
  positions: z.array(position).min(1),
  last3: z.number().min(28).max(50), // closing 3F in seconds (上がり), traffic included
  last3Free: z.number().min(28).max(50).optional(), // closing 3F with a clear run
  kickDelay: z.number().min(0).max(5).optional(), // seconds lost waiting for room (仕掛け遅れ)
  paceCost: z.number().min(0).max(5).optional(), // seconds lost to energy spent early (脚の消耗: pace exposure, 二の足/かかり)
  finish: z.object({rank: z.number().positive(), gap: z.number().min(0).max(80)}).optional(),
});
const race = z.object({
  corners: z.array(z.number().int().min(1).max(4)).min(1),
  horses: z.record(z.string().regex(/^\d{1,2}$/), horse),
  // Optional race pace (前半のペース): early 3F vs the course's usual (s, negative = faster),
  // H/M/S among comparable races (bands), and the horses that usually want the lead.
  pace: z
    .object({
      early: z.number().min(-10).max(10),
      label: z.enum(['H', 'M', 'S']),
      bands: z.array(z.number()).length(2).optional(),
      frontRunners: z.array(z.number().int()).optional(),
    })
    .optional(),
});
export const paceForecastSchema = z.object({
  schema: z.literal(PACE_SCHEMA),
  generatedAt: z.string(),
  model: z
    .object({
      name: z.string(),
      version: z.string().optional(),
      trainingEnd: z.string().optional(),
      validation: z.string().optional(),
      metrics: z
        .record(
          z.string(),
          z.object({
            mae: z.number(),
            baselineMae: z.number(),
            validationRows: z.number(),
            independentMae: z.number().optional(), // same model without the race-interaction layer
          }),
        )
        .optional(),
      notes: z.array(z.string()).optional(),
      // Optional: how well the formation reproduces mid-race moves (first to last corner).
      moves: z
        .object({
          model: z.object({corr: z.number(), upHit: z.number(), downHit: z.number(), falseBig: z.number().optional()}),
          withoutMoves: z
            .object({corr: z.number(), upHit: z.number(), downHit: z.number(), falseBig: z.number().optional()})
            .optional(),
          pastMeanCorr: z.number().optional(),
          rankError: z.number().optional(),
          rankErrorWithoutMoves: z.number().optional(),
        })
        .passthrough()
        .optional(),
      // Optional: validation of the race-pace model.
      pace: z
        .object({corr: z.number(), labelHit: z.number(), mae: z.number().optional(), constantMae: z.number().optional()})
        .passthrough()
        .optional(),
    })
    .passthrough(),
  races: z.record(z.string(), race),
});
export type PaceForecast = z.infer<typeof paceForecastSchema>;
export type PacePosition = z.infer<typeof position>;
export type PaceHorse = z.infer<typeof horse>;
/** One race resolved against the published field, horses in race.horses order. */
export type RacePace = {
  model: string;
  corners: number[];
  horses: PaceHorse[];
  racePace?: z.infer<typeof race>['pace'];
};

export function parsePaceForecast(
  json: unknown,
): {ok: true; data: PaceForecast} | {ok: false; error: string} {
  const r = paceForecastSchema.safeParse(json);
  if (r.success) return {ok: true, data: r.data};
  const first = r.error.issues[0];
  return {ok: false, error: `${first.path.join('.') || '(root)'}: ${first.message}`};
}

/** Match a forecast to the race's field; null with a reason when it cannot be used as-is. */
export function paceForRace(
  file: PaceForecast | null | undefined,
  race: Race,
): {pace: RacePace | null; reason?: string} {
  const entry = file?.races[race.id];
  if (!file || !entry) return {pace: null};
  const corners = [...new Set(entry.corners)].sort((a, b) => a - b);
  const horses: PaceHorse[] = [];
  for (const h of race.horses) {
    const p = entry.horses[String(h.number)];
    if (!p) return {pace: null, reason: `${h.number}番の予想がありません`};
    const have = p.positions.map((v) => v.corner).sort((a, b) => a - b);
    if (have.join() !== corners.join())
      return {pace: null, reason: `${h.number}番の角（${have.join(',')}）が一致しません`};
    horses.push({...p, positions: [...p.positions].sort((a, b) => a.corner - b.corner)});
  }
  const extra = Object.keys(entry.horses).filter(
    (n) => !race.horses.some((h) => String(h.number) === n),
  );
  if (extra.length) return {pace: null, reason: `出走していない馬番 ${extra.join(',')} が含まれます`};
  return {pace: {model: file.model.name, corners, horses, racePace: entry.pace}};
}

export const PACE_LABELS = {H: 'ハイペース', M: '平均ペース', S: 'スローペース'} as const;
export const laneLabel = (lane: number) => (lane < 0.5 ? '内' : lane < 1.5 ? '中' : '外');
