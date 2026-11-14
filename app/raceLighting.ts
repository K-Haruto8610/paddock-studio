import * as T from 'three';

/** Stylized seasonal sunlight: low winter sun and nearly overhead summer sun. */
export function raceSunOffset(date: string): T.Vector3 {
  const match = /^(\d{4})-?(\d{2})-?(\d{2})$/.exec(date);
  if (!match) return new T.Vector3(-200, 500, 180);
  const [, year, month, day] = match.map(Number);
  const held = Date.UTC(year, month - 1, day);
  const parsed = new Date(held);
  if (parsed.getUTCFullYear() !== year || parsed.getUTCMonth() !== month - 1 || parsed.getUTCDate() !== day) {
    return new T.Vector3(-200, 500, 180);
  }

  // Fixed solstice anchors are an art direction, not an astronomical simulation.
  const summer = Date.UTC(year, 5, 21);
  const winter = Date.UTC(year, 11, 21);
  const previousWinter = Date.UTC(year - 1, 11, 21);
  const nextSummer = Date.UTC(year + 1, 5, 21);
  const phase = held <= summer
    ? (held - previousWinter) / (summer - previousWinter)
    : held <= winter
      ? 1 + (held - summer) / (winter - summer)
      : (held - winter) / (nextSummer - winter);
  const summerness = (1 - Math.cos(Math.PI * phase)) / 2;
  const elevation = T.MathUtils.degToRad(20 + 65 * summerness);
  const distance = Math.hypot(200, 500, 180);
  const horizontal = distance * Math.cos(elevation);
  const azimuthLength = Math.hypot(200, 180);
  return new T.Vector3(
    -200 / azimuthLength * horizontal,
    distance * Math.sin(elevation),
    180 / azimuthLength * horizontal,
  );
}
