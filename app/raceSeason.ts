import * as T from 'three';

/**
 * Trees around a Japanese racecourse through the year: cherry (桜) blooming on bare branches at the end of March, then
 * leafing; zelkova (ケヤキ), ginkgo (イチョウ) and maple (モミジ) leafing out in April, turning in November and bare in
 * winter; evergreens (クスノキ・マツ) green all year. Dates are the Kanto/Kansai norm, an art direction rather than a
 * phenology model; each tree is a few days early or late (`jitter`) so a row turns unevenly.
 */
export type Species = 'evergreen' | 'zelkova' | 'ginkgo' | 'maple' | 'cherry';

export type Foliage = {
  /** How full the crown is (0 bare branches ... 1 full). */
  leaf: number;
  /** Fine twigs showing where the leaves are gone (0 for evergreens). */
  twig: number;
  /** Main colour of the crown and the colour of its patches (sRGB). */
  main: T.Color;
  patch: T.Color;
};

/** Day of the year (1 Jan = 0) of a race date `YYYYMMDD` or `YYYY-MM-DD`; midsummer if it does not parse. */
export function dayOfYear(date: string) {
  const match = /^(\d{4})-?(\d{2})-?(\d{2})$/.exec(date);
  if (!match) return 180;
  const [, y, m, d] = match.map(Number);
  return Math.round((Date.UTC(y, m - 1, d) - Date.UTC(y, 0, 1)) / 86400000);
}

const ramp = (d: number, from: number, to: number) => T.MathUtils.smoothstep(d, from, to);
const colour = (hex: string) => new T.Color(hex);
const SUMMER = colour('#557a45');
const FRESH = colour('#9fbe5c');

/** New leaves are a light yellow green that deepens to summer green over May and June. */
function leafGreen(d: number, out: number, summer = SUMMER) {
  return FRESH.clone().lerp(summer, ramp(d, out + 15, out + 70));
}

export function foliage(species: Species, day: number, jitter = 0): Foliage {
  const d = day + jitter;
  if (species === 'evergreen') {
    // Dark, with lighter new growth in May.
    const main = colour('#3d5b3a').lerp(colour('#5f7f45'), 0.5 * ramp(d, 120, 140) * (1 - ramp(d, 160, 200)));
    return {leaf: 1, twig: 0, main, patch: colour('#4c6a40')};
  }
  if (species === 'cherry') {
    // Blossom from late March to mid April on bare wood, then leaves; turning red-orange early (October), bare by December.
    const bloom = ramp(d, 78, 88) * (1 - ramp(d, 99, 110));
    const out = ramp(d, 98, 124), turn = ramp(d, 280, 318), fall = ramp(d, 312, 342);
    const leaves = out * (1 - fall);
    const green = leafGreen(d, 100);
    const autumn = green.clone().lerp(colour('#b4553a'), turn);
    const blossom = colour('#f6dde4');
    const share = bloom / Math.max(bloom + leaves, 1e-3);
    return {
      leaf: Math.min(1, Math.max(bloom * 0.92, leaves * 0.95)), twig: 0.45,
      main: autumn.clone().lerp(blossom, share),
      patch: autumn.clone().lerp(colour('#c98a3c'), turn).lerp(colour('#eab9c9'), share),
    };
  }
  const [out, turnFrom, turnTo, fallFrom, fallTo, autumn, patch] = {
    zelkova: [92, 298, 330, 318, 350, '#b26f38', '#c79a45'],
    ginkgo: [100, 305, 330, 326, 352, '#dcb43a', '#e8cd5c'],
    maple: [96, 308, 334, 330, 356, '#b9342b', '#d4692f'],
  }[species] as [number, number, number, number, number, string, string];
  const leaves = ramp(d, out, out + 30) * (1 - ramp(d, fallFrom, fallTo));
  const turn = ramp(d, turnFrom, turnTo);
  const green = leafGreen(d, out, species === 'ginkgo' ? colour('#628346') : SUMMER);
  return {
    leaf: leaves, twig: 0.42,
    main: green.clone().lerp(colour(autumn), turn),
    // Turning starts in patches: part of the crown is still green while the rest has coloured.
    patch: green.clone().lerp(colour(patch), turn * turn),
  };
}
