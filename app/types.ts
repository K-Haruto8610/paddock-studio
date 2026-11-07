export type Horse = {
  number: number;
  gate: number;
  name: string;
  coat: string;
  coatCode: string;
  blinker: boolean | null;
  jockey: string;
  age: number;
  weight: number;
  positions: number[];
  last3: number;
  history: {
    date: string;
    name: string;
    distance: number;
    field: number;
    corners: (number | null)[];
    last3: number | null;
  }[];
};
export type Race = {
  id: string;
  date: string;
  name: string;
  venue: string;
  venueCode: string;
  distance: number;
  surface: string;
  trackCode: number;
  course: string;
  horses: Horse[];
  historical: boolean;
};
export type Dataset = {
  generatedAt: string;
  races: Race[];
  model: {
    rows: number;
    races: number;
    excludedRows: number;
    trainingEnd: string;
    validation: string;
    inferenceTrainingEnd: string;
    notes: string[];
    metrics: Record<string, {mae: number; baselineMae: number; validationRows: number}>;
  };
};
export type ReplayHorse = {
  corners: (number | null)[];
  finish: number | null;
  officialFinish: number | null;
  last3: number | null;
  abnormality: string;
};
export type ReplayRace = {
  horses: Record<string, ReplayHorse>;
  corners: number[];
  partialCorners: number[];
  completeFinish: boolean;
  source: string;
  raceDate: string;
};
export type Replays = {generatedAt: string; races: Record<string, ReplayRace>; note: string};
export type PlaybackMode = 'replay' | 'forecast';
