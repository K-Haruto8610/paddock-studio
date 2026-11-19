import * as T from 'three';

/** Latitude and longitude (degrees) of each JRA racecourse, by venue code. */
const VENUES: Record<string, [number, number]> = {
  '01': [43.077, 141.324], // 札幌
  '02': [41.785, 140.775], // 函館
  '03': [37.765, 140.476], // 福島
  '04': [37.95, 139.18], // 新潟
  '05': [35.663, 139.484], // 東京
  '06': [35.727, 139.96], // 中山
  '07': [35.066, 137.016], // 中京
  '08': [34.906, 135.723], // 京都
  '09': [34.779, 135.364], // 阪神
  '10': [33.842, 130.873], // 小倉
};
/** The usual GI post time, for a race whose own is not known. */
export const DEFAULT_POST_TIME = '15:40';

/**
 * Height of the sun above the horizon (degrees) at a racecourse at a local (JST) time on a date: the NOAA solar position
 * approximation (equation of time and declination from the fractional year), good to a fraction of a degree.
 */
export function solarElevation(date: string, venueCode: string, time = DEFAULT_POST_TIME) {
  const match = /^(\d{4})-?(\d{2})-?(\d{2})$/.exec(date);
  const clock = /^(\d{1,2}):?(\d{2})$/.exec(time);
  if (!match || !clock) return null;
  const [, year, month, day] = match.map(Number);
  const [lat, lon] = VENUES[venueCode] ?? VENUES['05'];
  const minutes = Number(clock[1]) * 60 + Number(clock[2]);
  const leap = (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
  const dayOfYear = Math.round((Date.UTC(year, month - 1, day) - Date.UTC(year, 0, 1)) / 86400000);
  const g = (2 * Math.PI / (leap ? 366 : 365)) * (dayOfYear + (minutes / 60 - 9 - 12) / 24);
  const equation = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g) - 0.014615 * Math.cos(2 * g)
    - 0.040849 * Math.sin(2 * g));
  const declination = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g)
    + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  const solarMinutes = minutes + equation + 4 * lon - 60 * 9;
  const hourAngle = T.MathUtils.degToRad(solarMinutes / 4 - 180);
  const phi = T.MathUtils.degToRad(lat);
  const sinElevation = Math.sin(phi) * Math.sin(declination) + Math.cos(phi) * Math.cos(declination) * Math.cos(hourAngle);
  return T.MathUtils.radToDeg(Math.asin(sinElevation));
}

/** Lowest sun drawn: below this the scene would be dusk, which no JRA race is run in. */
const MIN_ELEVATION = 6;

/**
 * Where the sun stands for a race (offset from the field, about 560 m away): as high as the real sun at the venue at post
 * time (`solarElevation`), so a December afternoon has a low sun and long shadows and a June one a high sun. The compass
 * direction stays the art-directed one (the course models carry no north), behind the camera's usual left shoulder.
 */
export function raceSunOffset(date: string, venueCode = '05', postTime = DEFAULT_POST_TIME): T.Vector3 {
  const height = solarElevation(date, venueCode, postTime);
  const elevation = T.MathUtils.degToRad(Math.max(MIN_ELEVATION, height ?? 45));
  const distance = Math.hypot(200, 500, 180);
  const horizontal = distance * Math.cos(elevation);
  const azimuthLength = Math.hypot(200, 180);
  return new T.Vector3(
    -200 / azimuthLength * horizontal,
    distance * Math.sin(elevation),
    180 / azimuthLength * horizontal,
  );
}
