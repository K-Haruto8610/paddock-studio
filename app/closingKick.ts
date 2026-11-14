// Closing-kick (上がり) running form, as pure functions of a horse's own travelled distance.
// Nothing here carries state, so pausing, seeking and MP4 recording always reproduce the same pose.

/** Distance per base stride in race metres, independent of race length and playback rate. */
export const STRIDE_LENGTH = 16;
/** The kick starts this far from the finish and ramps from 0 to full over KICK_RAMP metres. */
export const KICK_WINDOW = 600;
export const KICK_RAMP = 100;
/** Metres a field covers in one second of closing speed: converts 仕掛け遅れ (seconds) into distance. */
export const KICK_SPEED = 17;
/** The slowest closing 3F in the field peaks at this strength, the fastest at 1 (linear in between). */
export const KICK_WEAKEST = 0.4;
/** Strength of a horse with no recorded closing 3F. */
export const KICK_NEUTRAL = 0.7;
/** Extra strides per metre at full kick: the horse switches from long strides to a quick, short pitch gait. */
export const KICK_CADENCE = 0.4;

/** Per horse: where its kick begins (race metres) and how hard it peaks (0..1). */
export type KickPlan = {start: number; strength: number};
/** Kick state at one distance: `level` in 0..1, and the extra strides the faster cadence has added so far. */
export type Kick = {level: number; strides: number};
export const NO_KICK: Kick = {level: 0, strides: 0};

/** Plans for a whole field from its closing 3F times (seconds; null/0 = unknown) and optional kick delays (seconds). */
export function kickPlans(raceDistance: number, last3: (number | null | undefined)[], delays: (number | undefined)[] = []): KickPlan[] {
  const known = last3.filter((v): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0);
  const best = Math.min(...known), worst = Math.max(...known);
  return last3.map((v, i) => {
    const strength = typeof v === 'number' && known.includes(v)
      ? worst - best < 1e-6 ? 1 : 1 - (1 - KICK_WEAKEST) * (v - best) / (worst - best)
      : KICK_NEUTRAL;
    return {start: Math.max(0, raceDistance - KICK_WINDOW + (delays[i] ?? 0) * KICK_SPEED), strength};
  });
}

/**
 * Kick at a distance. The level follows a smoothstep ramp; the cadence is that level integrated over distance
 * in closed form (the antiderivative of smoothstep is u^3 - u^4 / 2), so the stride phase stays continuous
 * and has no jump where the kick begins, however the race is seeked.
 */
export function kickAt(distance: number, plan: KickPlan | undefined): Kick {
  if (!plan) return NO_KICK;
  const u = (distance - plan.start) / KICK_RAMP;
  if (u <= 0) return NO_KICK;
  const level = plan.strength * (u >= 1 ? 1 : u * u * (3 - 2 * u));
  const area = u >= 1 ? u - 0.5 : u * u * u - u * u * u * u / 2;
  return {level, strides: KICK_CADENCE * plan.strength * KICK_RAMP * area / STRIDE_LENGTH};
}
