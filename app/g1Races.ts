import master from '../data/g1_races.json';

// Single source of truth for current JRA flat GI races (data/g1_races.json).
export type G1Race = {
  id: string;
  name: string;
  aliases: string[];
  venueCode: string;
  venue: string;
  surface: '芝' | 'ダート';
  distance: number;
  layout: '' | '内' | '外' | '外→内';
  trackCodes: number[];
  month: number;
  ageSex: string;
  note: string;
};
export type G1VenueOverride = {id: string; year: number; venueCode: string; trackCodes: number[]};
export type G1Master = {
  version: number;
  checkedAt: string;
  sources: string[];
  notes: string[];
  venueOverrides: G1VenueOverride[];
  races: G1Race[];
};

export const G1_MASTER = master as G1Master;

const normalize = (s: string) => s.normalize('NFKC').replace(/\s+/g, '');

export function findG1(name: string): G1Race | undefined {
  const n = normalize(name);
  return G1_MASTER.races.find((r) => [r.name, ...r.aliases].some((v) => normalize(v) === n));
}

// Expected venue / track codes in a given year (renovation relocations applied).
export function g1Expected(race: G1Race, year: number) {
  const o = G1_MASTER.venueOverrides.find((v) => v.id === race.id && v.year === year);
  return {venueCode: o?.venueCode ?? race.venueCode, trackCodes: o?.trackCodes ?? race.trackCodes};
}
