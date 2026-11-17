/** Estimated speed at 1×. The race data has no total finishing time. */
export const RACE_METRES_PER_SECOND = 17;
export const RACE_FPS = 60;
export const FRAME_SECONDS = 1 / RACE_FPS;

export const raceSeconds = (travel: number) => travel / RACE_METRES_PER_SECOND;
export const outputSeconds = (travel: number, speed: number) => raceSeconds(travel) / speed;
export const progressAtTime = (seconds: number, distance: number, speed: number) =>
  (seconds * speed * RACE_METRES_PER_SECOND) / distance;

/** Use the last complete tick for history. Keep the requested progress for the pose. */
export const completeTick = (seconds: number) => Math.floor(seconds * RACE_FPS + 1e-9);
