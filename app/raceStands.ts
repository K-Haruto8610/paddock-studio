import * as T from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import type {TrackModel} from './trackGeometry';
import type {WeatherLook} from './raceWeather';
import {seeded} from './scenery';
import {standSeatGeometry} from './standSeat';
import {bevelBox} from './surfaceFinish';
import {standFinish, type StandLook, type StandMaterials} from './standFinish';
import {acrossOf} from './raceAvenue';

// The main stand (正面スタンド) of each GⅠ venue, photoreal, after photographs of the real buildings (Wikimedia Commons,
// 2007-2023): their silhouette, levels and materials, at true size. Built in the stand's own frame: x along the home
// straight in the running direction, z away from the course (0 at the foot of the stand's front), y up from the apron.
// The courses are drawn with exaggerated heights, so the building follows the ground along its length (each x is lifted
// to the apron there) instead of standing level on a slope it would float over.

type Key = keyof StandMaterials;

/** Seats and people on a stand, collected while it is built. */
type Seat = {x: number; y: number; z: number; colour: string};
type Person = {x: number; y: number; z: number; seated: boolean; open: boolean};

/** Collects the stand's pieces by material (merged into one mesh each) and the seats, people and lamps on it. */
class Kit {
  parts = new Map<Key, T.BufferGeometry[]>();
  seats: Seat[] = [];
  people: Person[] = [];
  readonly random: () => number;
  constructor(readonly m: StandMaterials, seed: number, readonly crowd: number) {
    this.random = seeded(seed);
  }

  add(key: Key, geometry: T.BufferGeometry, shade = 1) {
    const g = geometry.index ? geometry.toNonIndexed() : geometry;
    if (!g.attributes.uv) g.setAttribute('uv', new T.Float32BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2));
    g.setAttribute('color', new T.Float32BufferAttribute(new Float32Array(g.attributes.position.count * 3).fill(shade), 3));
    for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv', 'color'].includes(name)) g.deleteAttribute(name);
    if (!this.parts.has(key)) this.parts.set(key, []);
    this.parts.get(key)!.push(g);
  }

  /** A box from corner to corner, edges rounded by `bevel` (0: sharp), uv in metres of its material's tile. */
  box(key: Key, x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, shade = 1, bevel = 0.12) {
    const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0), d = Math.abs(z1 - z0);
    if (w < 1e-3 || h < 1e-3 || d < 1e-3) return;
    const tile = (this.m[key].userData.tile as number | undefined) ?? 4;
    const g = bevel > 0 ? bevelBox(w, h, d, Math.min(bevel, Math.min(w, h, d) / 2.2), tile) : boxUv(w, h, d, tile);
    g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    this.add(key, g, shade);
  }

  /** A flat four-cornered face (corners in order, seen counter-clockwise from the side it faces), uv in metres. */
  quad(key: Key, corners: T.Vector3[], shade = 1, uv?: [number, number][]) {
    const tile = (this.m[key].userData.tile as number | undefined) ?? 4;
    const normal = new T.Vector3().subVectors(corners[1], corners[0]).cross(new T.Vector3().subVectors(corners[2], corners[0])).normalize();
    const ax = Math.abs(normal.x), ay = Math.abs(normal.y), az = Math.abs(normal.z);
    const uvs = uv ?? corners.map(p => (ax >= ay && ax >= az ? [p.z / tile, p.y / tile] : ay >= az ? [p.x / tile, p.z / tile]
      : [p.x / tile, p.y / tile]) as [number, number]);
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.Float32BufferAttribute(corners.flatMap(p => [p.x, p.y, p.z]), 3));
    g.setAttribute('normal', new T.Float32BufferAttribute(corners.flatMap(() => [normal.x, normal.y, normal.z]), 3));
    g.setAttribute('uv', new T.Float32BufferAttribute(uvs.flat(), 2));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    this.add(key, g, shade);
  }

  /** Curtain glass facing the course (-z) at depth z, its uv one map (4 bays by 2 floors) per 4 `bay` x 2 `floor` metres. */
  glass(x0: number, x1: number, y0: number, y1: number, z: number, bay = 3, floor = 4.2, shade = 1, facing: 1 | -1 = -1) {
    const u = (x: number) => x / (4 * bay), v = (y: number) => (y - y0) / (2 * floor);
    const [a, b] = facing < 0 ? [x1, x0] : [x0, x1];
    this.quad(shade < 0.8 ? 'coveredGlass' : 'glass', [new T.Vector3(a, y0, z), new T.Vector3(b, y0, z), new T.Vector3(b, y1, z), new T.Vector3(a, y1, z)], shade,
      [[u(a), v(y0)], [u(b), v(y0)], [u(b), v(y1)], [u(a), v(y1)]]);
  }

  /** Curtain glass on an end face (facing +x or -x) at x, from z0 to z1. */
  glassSide(z0: number, z1: number, y0: number, y1: number, x: number, facing: 1 | -1, bay = 3, floor = 4.2) {
    const u = (z: number) => z / (4 * bay), v = (y: number) => (y - y0) / (2 * floor);
    const [a, b] = facing > 0 ? [z1, z0] : [z0, z1];
    this.quad('glass', [new T.Vector3(x, y0, a), new T.Vector3(x, y0, b), new T.Vector3(x, y1, b), new T.Vector3(x, y1, a)], 1,
      [[u(a), v(y0)], [u(b), v(y0)], [u(b), v(y1)], [u(a), v(y1)]]);
  }

  /** A horizontal face at y, facing down (a soffit, `-1`) or up. */
  flat(key: Key, x0: number, x1: number, z0: number, z1: number, y: number, facing: 1 | -1, shade = 1) {
    const c = facing < 0
      ? [new T.Vector3(x0, y, z0), new T.Vector3(x1, y, z0), new T.Vector3(x1, y, z1), new T.Vector3(x0, y, z1)]
      : [new T.Vector3(x0, y, z1), new T.Vector3(x1, y, z1), new T.Vector3(x1, y, z0), new T.Vector3(x0, y, z0)];
    this.quad(key, c, shade);
  }

  /** A beam w wide and h deep hanging under the line from (z0, y0) to (z1, y1) at x. */
  beam(key: Key, x: number, z0: number, y0: number, z1: number, y1: number, w: number, h: number, shade = 1) {
    const length = Math.hypot(z1 - z0, y1 - y0);
    const g = boxUv(w, h, length, 4);
    g.translate(0, -h / 2, 0);
    g.rotateX(-Math.atan2(y1 - y0, z1 - z0));
    g.translate(x, (y0 + y1) / 2, (z0 + z1) / 2);
    this.add(key, g, shade);
  }

  /** Seats in rows across x0..x1, 0.5 m apart, leaving an aisle every `aisle` metres. */
  seatRow(x0: number, x1: number, y: number, z: number, colour: (x: number) => string, aisle: number, occupancy: number, open: boolean) {
    for (let x = x0 + 0.25; x <= x1 - 0.25; x += 0.5) {
      if (aisle > 0 && Math.abs(x - Math.round(x / aisle) * aisle) < 0.7) continue;
      this.seats.push({x, y, z, colour: colour(x)});
      if (this.random() < occupancy * this.crowd) this.people.push({x, y, z: z + 0.1, seated: true, open});
    }
  }

  /**
   * Concrete terraces rising from (z0, y0) to (z1, y1) in `steps` treads, each with `lines` rows of seats (none: people stand
   * on the steps), an aisle every `aisle` metres; `walls` raises a stepped parapet beside each aisle (Nakayama's white walls).
   */
  terrace(o: {x0: number; x1: number; z0: number; z1: number; y0: number; y1: number; steps: number; lines: number;
    colour: (x: number) => string; aisle: number; occupancy: number; walls?: Key; shade?: number; open?: boolean}) {
    const tread = (o.z1 - o.z0) / o.steps, rise = (o.y1 - o.y0) / o.steps, shade = o.shade ?? 1, open = o.open ?? true;
    this.box('concrete', o.x0, o.x1, -4, o.y0, o.z0 - 0.4, o.z0 + 0.01, shade, 0.05);
    for (let i = 0; i < o.steps; i++) {
      const z = o.z0 + i * tread, y = o.y0 + (i + 1) * rise;
      this.box('concrete', o.x0, o.x1, -4, y, z, o.z1 + (i === o.steps - 1 ? 0 : 0.01), shade, 0.04);
      if (o.lines > 0)
        for (let line = 0; line < o.lines; line++)
          this.seatRow(o.x0, o.x1, y, z + 0.3 + line * (tread / o.lines), o.colour, o.aisle, o.occupancy, open);
      else
        for (let x = o.x0 + 0.3; x < o.x1; x += 0.55)
          if (this.random() < o.occupancy * this.crowd)
            this.people.push({x: x + (this.random() - 0.5) * 0.3, y, z: z + tread * (0.3 + 0.4 * this.random()), seated: false, open});
      if (o.walls)
        for (let x = Math.ceil(o.x0 / o.aisle) * o.aisle; x < o.x1; x += o.aisle)
          for (const side of [-1, 1]) this.box(o.walls, x + side * 0.9 - 0.15, x + side * 0.9 + 0.15, y, y + 1.1, z, z + tread, shade, 0.05);
    }
  }

  /**
   * An open deck: a floor slab whose front carries a fascia, stepped rows of seats rising behind it, glass at the back
   * and a lit ceiling above (`ceiling`: the underside of whatever is above). Under cover: shaded.
   */
  deck(o: {x0: number; x1: number; zf: number; zb: number; y: number; rows: number; rise: number; ceiling: number;
    colour: (x: number) => string; aisle: number; occupancy: number; fascia?: Key; fasciaH?: number; shade?: number}) {
    const shade = o.shade ?? 0.62, fascia = o.fascia ?? 'cladding', fh = o.fasciaH ?? 1.6;
    this.box('concrete', o.x0, o.x1, o.y - 0.9, o.y, o.zf, o.zb, 0.8, 0.05);
    this.box(fascia, o.x0, o.x1, o.y - 1.1, o.y + fh - 0.9, o.zf - 0.35, o.zf + 0.05, 1, 0.08);
    const depth = (o.zb - o.zf - 1.4) / o.rows;
    for (let r = 0; r < o.rows; r++) {
      const z = o.zf + 0.9 + r * depth, y = o.y + r * o.rise;
      this.box('concrete', o.x0, o.x1, o.y - 0.2, y, z, o.zb, shade, 0.03);
      this.seatRow(o.x0, o.x1, y, z + 0.25, o.colour, o.aisle, o.occupancy, false);
    }
    this.glass(o.x0, o.x1, o.y + o.rows * o.rise, o.ceiling, o.zb, 3, 4.2, shade);
    this.flat('ceiling', o.x0, o.x1, o.zf - 0.3, o.zb, o.ceiling, -1, 0.72);
  }

  /**
   * The cantilevered roof from its back (zb, yb) to its front edge (zf, yf): a sheeted top, the underside (panels, timber,
   * or membrane between ribs over the front `membrane` metres), a fascia along the edge, ribs every `ribs` metres under
   * (and over, for a trussed roof), floodlights on the edge every `lamps` metres.
   */
  roof(o: {x0: number; x1: number; zf: number; zb: number; yf: number; yb: number; thick: number; fascia: number;
    ribs?: number; ribDepth?: number; ribsOver?: number; membrane?: number; skylights?: boolean; lamps?: number}) {
    const {x0, x1, zf, zb, yf, yb, thick} = o;
    const at = (z: number) => T.MathUtils.lerp(yf, yb, (z - zf) / (zb - zf));
    this.quad('roofTop', [new T.Vector3(x0, yf, zf), new T.Vector3(x0, yb, zb), new T.Vector3(x1, yb, zb), new T.Vector3(x1, yf, zf)]);
    const split = o.membrane ? zf + o.membrane : zf;
    if (o.membrane) {
      this.quad('membrane', [new T.Vector3(x0, yf - thick, zf), new T.Vector3(x1, yf - thick, zf),
        new T.Vector3(x1, at(split) - thick, split), new T.Vector3(x0, at(split) - thick, split)]);
    }
    this.quad('under', [new T.Vector3(x0, at(split) - thick, split), new T.Vector3(x1, at(split) - thick, split),
      new T.Vector3(x1, yb - thick, zb), new T.Vector3(x0, yb - thick, zb)], 0.85);
    if (o.skylights)
      for (let x = Math.ceil(x0 / (o.ribs ?? 9.6)) * (o.ribs ?? 9.6) + (o.ribs ?? 9.6) / 2; x < x1; x += o.ribs ?? 9.6)
        this.quad('membrane', [new T.Vector3(x - 1.4, yf - thick - 0.02, zf + 2), new T.Vector3(x + 1.4, yf - thick - 0.02, zf + 2),
          new T.Vector3(x + 1.4, at(zb - 8) - thick - 0.02, zb - 8), new T.Vector3(x - 1.4, at(zb - 8) - thick - 0.02, zb - 8)]);
    // Edge fascia and end caps.
    this.box('trim', x0, x1, yf - thick - o.fascia * 0.6, yf + o.fascia * 0.4, zf - 0.4, zf + 0.5, 1, 0.1);
    for (const x of [x0, x1]) {
      const s = x === x0 ? -1 : 1;
      this.quad('trim', s < 0
        ? [new T.Vector3(x, yf - thick, zf), new T.Vector3(x, yb - thick, zb), new T.Vector3(x, yb, zb), new T.Vector3(x, yf, zf)]
        : [new T.Vector3(x, yb - thick, zb), new T.Vector3(x, yf - thick, zf), new T.Vector3(x, yf, zf), new T.Vector3(x, yb, zb)]);
    }
    if (o.ribs)
      for (let x = Math.ceil(x0 / o.ribs) * o.ribs; x <= x1; x += o.ribs) {
        this.beam('ribs', x, zf + 0.6, yf - thick, zb, yb - thick, 0.45, o.ribDepth ?? 1.2, 0.9);
        if (o.ribsOver) this.beam('ribs', x, zf + 4, yf + o.ribsOver, zb - 2, yb + o.ribsOver * 1.4, 0.45, o.ribsOver, 1);
      }
    if (o.lamps)
      for (let x = x0 + o.lamps / 2; x < x1; x += o.lamps) {
        this.box('steel', x - 0.05, x + 0.05, yf + 0.3, yf + 0.9, zf + 0.25, zf + 0.35, 1, 0);
        this.box('steel', x - 0.22, x + 0.22, yf + 0.9, yf + 1.25, zf + 0.05, zf + 0.45, 1, 0);
      }
  }
}

/** A sharp box with uv in metres on each face. */
function boxUv(w: number, h: number, d: number, tile: number) {
  const g = new T.BoxGeometry(w, h, d);
  const p = g.attributes.position, n = g.attributes.normal, uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) {
    const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i)), az = Math.abs(n.getZ(i));
    if (ax >= ay && ax >= az) uv.setXY(i, p.getZ(i) / tile, p.getY(i) / tile);
    else if (ay >= az) uv.setXY(i, p.getX(i) / tile, p.getZ(i) / tile);
    else uv.setXY(i, p.getX(i) / tile, p.getY(i) / tile);
  }
  return g;
}

/** Seats in blocks of one colour `block` metres wide, cycling through `colours`. */
const blocks = (colours: string[], block = 1e6) => (x: number) =>
  colours[((Math.floor(x / block) % colours.length) + colours.length) % colours.length];

type StandSpec = {
  look: StandLook;
  /** Distance along the course of the stand's middle from the winning post (negative: before it). */
  centre: number;
  /** From the outer rail to the foot of the stand's front, and how much of that next to the rail is lawn, not paving. */
  setback: number; lawn: number;
  /** Extent of the building along x (with its end blocks) and back from its front. */
  extent: [number, number]; depth: number;
  build: (k: Kit) => void;
};

/**
 * 東京 フジビュースタンド (2007): open concrete terraces below four long balconies stepping back as they rise, glazed
 * upper floors, and a 45 m cantilevered roof on dark steel ribs that run over and under it with daylight strips between;
 * glass stair towers at both ends. Warm greige concrete, golden seats on the balconies.
 */
function tokyo(k: Kit) {
  const h = 165, gold = blocks(['#c3a04c', '#b48f3f'], 33);
  k.terrace({x0: -h, x1: h, z0: 0, z1: 24, y0: 0.6, y1: 9, steps: 14, lines: 0, colour: gold, aisle: 30, occupancy: 0.85,
    walls: 'concrete'});
  // The concourse behind the terraces, under the first balcony.
  k.glass(-h, h, 9, 12, 24.5, 3.2, 3, 0.55);
  k.flat('ceiling', -h, h, 20, 24.5, 12, -1, 0.6);
  const decks = [{y: 13, zf: 20, zb: 29}, {y: 18.8, zf: 21.5, zb: 30}, {y: 24.6, zf: 23, zb: 31}];
  decks.forEach((d, i) => k.deck({x0: -h, x1: h, zf: d.zf, zb: d.zb, y: d.y, rows: 4, rise: 0.45,
    ceiling: (decks[i + 1]?.y ?? 30.4) - 0.9, colour: gold, aisle: 22, occupancy: 0.75, fascia: 'concrete', fasciaH: 1.8}));
  // Glazed fifth and sixth floors behind a deep band, then the plant floor under the roof.
  k.box('concrete', -h, h, 29.4, 32, 24.5, 33, 1);
  k.glass(-h, h, 32, 40.4, 33, 3.2, 4.2, 0.6);
  k.flat('ceiling', -h, h, 24.5, 33, 40.4, -1, 0.7);
  k.box('cladding', -h, h, 40.4, 43.4, 26, 64, 0.9);
  k.box('cladding', -h, h, -4, 43, 31, 64);
  k.roof({x0: -h - 6, x1: h + 6, zf: -4, zb: 62, yf: 49, yb: 45, thick: 1.4, fascia: 1.6, ribs: 9.6, ribDepth: 1.3,
    ribsOver: 1.6, skylights: true});
  // Glass stair towers at both ends.
  for (const s of [-1, 1]) {
    const xa = s * h, xb = s * (h + 12);
    const [x0, x1] = s < 0 ? [xb, xa] : [xa, xb];
    k.box('trim', x0, x1, -4, 40, 6, 42, 1);
    k.glass(x0 + 0.4, x1 - 0.4, 0, 39, 5.7, 1.6, 4);
    k.glassSide(6.3, 41.7, 0, 39, xb + s * 0.05, s as 1 | -1, 1.6, 4);
    k.box('trim', x0 - 0.5, x1 + 0.5, 40, 41.2, 5.4, 42.6, 1);
  }
}

/**
 * 中山 (1990, renewed 2014): blue seats on white-walled terraces, two floors of dark green glass the whole length under a
 * deep white band, a narrow glazed floor, an upper open deck, and a flat roof with floodlights all along its edge.
 */
function nakayama(k: Kit) {
  const h = 150, blue = blocks(['#2f5fa8', '#2b56a0']);
  k.terrace({x0: -h, x1: h, z0: 0, z1: 16, y0: 0.4, y1: 6.4, steps: 10, lines: 1, colour: blue, aisle: 24, occupancy: 0.7,
    walls: 'cladding'});
  k.glass(-h, h, 6.4, 15.4, 16.2, 3.6, 4.5, 0.75);
  k.flat('ceiling', -h, h, 14.6, 16.2, 15.4, -1, 0.8);
  k.box('cladding', -h, h, 15.4, 18.6, 14.6, 17.5, 1);
  k.glass(-h, h, 18.6, 21, 17.6, 3.6, 2.4, 0.8);
  k.box('cladding', -h, h, 21, 23.4, 16.4, 19, 1);
  k.deck({x0: -h, x1: h, zf: 17, zb: 25, y: 23.4, rows: 4, rise: 0.5, ceiling: 30.4, colour: blue, aisle: 24, occupancy: 0.7,
    fasciaH: 1.3});
  k.box('cladding', -h, h, -4, 31, 25, 50);
  k.roof({x0: -h - 4, x1: h + 4, zf: -6, zb: 50, yf: 33, yb: 31.6, thick: 1.2, fascia: 1.3, lamps: 4.5});
  // End blocks: white stair halls, the upstream one taller.
  k.box('cladding', -h - 16, -h, -4, 27, 4, 46, 1);
  k.glass(-h - 15.6, -h - 0.4, 1, 24, 3.7, 2.6, 4.6);
  k.box('cladding', h, h + 22, -4, 19, 10, 46, 1);
  k.glass(h + 0.4, h + 21.6, 1, 17, 9.7, 3, 4.5);
}

/**
 * 京都 (rebuilt 2023): white seats on low terraces, a continuous ribbon of blue glass between thick white bands, white
 * balcony bands with dark slots above, and a thin roof reaching far out over the apron, its underside lined in timber.
 * The station-side and goal-side halves meet at a recessed glass joint.
 */
function kyoto(k: Kit) {
  const h = 140, white = blocks(['#e9e9e5']);
  k.terrace({x0: -h, x1: h, z0: 0, z1: 15, y0: 0.4, y1: 5.6, steps: 10, lines: 1, colour: white, aisle: 20, occupancy: 0.72,
    walls: 'steel'});
  k.glass(-h, h, 5.6, 7.6, 16, 3, 2, 0.5);
  k.flat('ceiling', -h, h, 12, 16, 7.6, -1, 0.75);
  k.box('cladding', -h, h, 7.6, 9.4, 12, 16.5, 1);
  k.glass(-h, h, 9.4, 14.8, 14.4, 3, 2.7, 0.95);
  k.box('cladding', -h, h, 14.8, 17.6, 11.6, 16, 1);
  for (const [y0, y1, z] of [[17.6, 19.6, 16.6], [22, 24, 17.4], [26.4, 28.6, 18.4]] as const)
    k.glass(-h, h, y0, y1, z, 3, 2.2, 0.7);
  k.box('cladding', -h, h, 19.6, 22, 13.6, 17.4, 1);
  k.box('cladding', -h, h, 24, 26.4, 14.8, 18.4, 1);
  k.box('cladding', -h, h, -4, 30.4, 18.6, 48);
  // The joint between the station-side and goal-side halves.
  k.box('trim', -20.6, -19.4, 5.6, 30.6, 11, 19, 1);
  k.roof({x0: -h - 3, x1: h + 3, zf: -7, zb: 48, yf: 31.8, yb: 30.6, thick: 1, fascia: 0.9});
  // Glazed end bays.
  for (const s of [-1, 1] as const) {
    const [x0, x1] = s < 0 ? [-h - 4, -h] : [h, h + 4];
    k.box('trim', x0, x1, -4, 30.6, 10, 48, 1);
    k.glassSide(10.5, 47.5, 5.6, 29.5, s < 0 ? x0 - 0.05 : x1 + 0.05, s, 3, 4.6);
  }
}

/**
 * 阪神 (1991, renewed 2006): five floors of glass stepping back, each over a projecting silver ledge, a small terrace in
 * front, a thin roof with floodlights along its edge, and silver towers at both ends carrying wings of the roof.
 */
function hanshin(k: Kit) {
  const h = 120, green = blocks(['#3d7a63']);
  k.terrace({x0: -h, x1: h, z0: 0, z1: 9, y0: 0.4, y1: 3.4, steps: 5, lines: 1, colour: green, aisle: 20, occupancy: 0.65});
  for (let i = 0; i < 5; i++) {
    const base = 3.4 + 4.6 * i, z = 9.5 + 0.9 * i;
    k.box('trim', -h, h, base - 0.3, base + 0.35, z - 2, z + 0.6, 1);
    k.glass(-h, h, base + 0.35, base + 4.3, z + 0.5, 3.2, 4, i === 4 ? 0.75 : 0.85);
    k.flat('ceiling', -h, h, z - 2, z + 0.5, base - 0.3, -1, 0.9);
  }
  k.box('trim', -h, h, 26.3, 27.2, 13, 15.5, 1);
  k.box('cladding', -h, h, -4, 29, 15, 46);
  k.roof({x0: -h, x1: h, zf: 2, zb: 46, yf: 30.6, yb: 29.4, thick: 1, fascia: 1.1, lamps: 5});
  for (const s of [-1, 1]) {
    const [x0, x1] = s < 0 ? [-h - 11, -h] : [h, h + 11];
    k.box('cladding', x0, x1, -4, 33, 6, 42, 1);
    k.glass(x0 + 3, x1 - 3, 2, 30, 5.7, 1.2, 4.6);
    k.box('trim', x0 - 3, x1 + 3, 33, 33.9, -2, 44, 1, 0.2);
  }
}

/**
 * 中京 ペガサススタンド (2011): multicoloured seats on low terraces, two floors of glass in the west half and two open
 * decks in the east, under a long cantilevered roof whose front is white membrane between white steel ribs; a grey block
 * with a glass hall and a louvred annex at the east end.
 */
function chukyo(k: Kit) {
  const h = 105, mixed = blocks(['#3f8f4f', '#d98a2e', '#2f6fb6', '#d6c23d'], 12);
  k.box('cladding', -h, h, 0.2, 1.5, -0.45, 0, 1, 0.05);
  k.terrace({x0: -h, x1: h, z0: 0, z1: 12, y0: 0.5, y1: 4.5, steps: 8, lines: 1, colour: mixed, aisle: 18, occupancy: 0.72});
  const w = -12;
  // West: two floors of glass, a band, one more glazed floor, a band.
  k.glass(-h, w, 4.5, 13, 12.6, 3.4, 4.25, 0.9);
  k.box('cladding', -h, w, 8.5, 9.1, 11.8, 13, 1);
  k.box('cladding', -h, w, 13, 15, 11.4, 14.2, 1);
  k.glass(-h, w, 15, 21.6, 13.8, 3.4, 3.3, 0.85);
  k.box('trim', -h, w, 21.6, 24.6, 12.4, 15, 1);
  // East: two open decks.
  k.deck({x0: w, x1: h, zf: 12, zb: 20, y: 5.5, rows: 4, rise: 0.45, ceiling: 11.4, colour: mixed, aisle: 18, occupancy: 0.75,
    fascia: 'trim', fasciaH: 1.4});
  k.deck({x0: w, x1: h, zf: 9.5, zb: 20, y: 12.8, rows: 6, rise: 0.5, ceiling: 21.6, colour: mixed, aisle: 18, occupancy: 0.75,
    fascia: 'trim', fasciaH: 1.4});
  k.box('cladding', -h, h, -4, 28, 20, 42);
  k.roof({x0: -h - 2, x1: h + 2, zf: -10, zb: 42, yf: 31, yb: 28.2, thick: 1.1, fascia: 1.2, ribs: 10, ribDepth: 1.5,
    membrane: 22, lamps: 6});
  // East block with its glass hall, and the louvred annex beyond; a small core at the west end.
  k.box('cladding', h, h + 22, -4, 27, 0, 44, 1);
  k.box('trim', h + 2.5, h + 17.5, 3, 25, -2.4, 0.2, 1, 0.1);
  k.glass(h + 3, h + 17, 3.5, 24.5, -2.45, 1.6, 2.6);
  k.box('trim', h + 22, h + 40, -4, 22, 12, 40, 1);
  k.glass(h + 22.2, h + 39.8, 0, 21.5, 11.9, 1.2, 4.3);
  for (let x = h + 22.6; x < h + 40; x += 1.2) k.box('mullion', x - 0.12, x + 0.12, 0.5, 21.6, 10.8, 11.9, 1, 0);
  k.box('cladding', -h - 10, -h, -4, 24, 8, 42, 1);
}

const STANDS: Record<string, StandSpec> = {
  '05': {centre: -70, setback: 40, lawn: 16, extent: [-177, 177], depth: 64, build: tokyo, look: {
    concrete: '#d4cbba', cladding: '#cdc4b2', panel: [1.8, 1.2], metallic: 0, trim: '#d9d2c3', room: '#2c3034', glass: '#7a868c',
    ceiling: '#cbc4b6', lamps: 2, under: 'panel', underColour: '#bcb4a6', ribs: '#3c3731', paving: '#c6c3bc'}},
  '06': {centre: -40, setback: 30, lawn: 0, extent: [-166, 172], depth: 50, build: nakayama, look: {
    concrete: '#cfcfca', cladding: '#eceae5', panel: [1.5, 1], metallic: 0.1, trim: '#f1f0ec', room: '#1d282c', glass: '#56717a',
    ceiling: '#d7d6d1', lamps: 2, under: 'panel', underColour: '#5a5e61', ribs: '#5a5e61', paving: '#c2c1bc'}},
  '07': {centre: -20, setback: 34, lawn: 0, extent: [-115, 145], depth: 44, build: chukyo, look: {
    concrete: '#d2d2cd', cladding: '#a3a8ac', panel: [1.2, 0.9], metallic: 0.35, trim: '#eeeeec', room: '#1f272c', glass: '#6f8a94',
    ceiling: '#cfcfca', lamps: 2, under: 'membrane', underColour: '#4c5054', ribs: '#dcdedf', paving: '#c4c2bd'}},
  '08': {centre: -30, setback: 32, lawn: 0, extent: [-144, 144], depth: 48, build: kyoto, look: {
    concrete: '#d9d9d4', cladding: '#efefec', panel: [1.8, 1.2], metallic: 0.05, trim: '#f4f4f2', room: '#21303a', glass: '#6293b8',
    ceiling: '#e4e3df', lamps: 2, under: 'timber', underColour: '#9fa3aa', ribs: '#e9e9e6', paving: '#d6d4cf'}},
  '09': {centre: -30, setback: 30, lawn: 0, extent: [-134, 134], depth: 46, build: hanshin, look: {
    concrete: '#cfd1cf', cladding: '#b9bec1', panel: [1.2, 1.2], metallic: 0.6, trim: '#c8cccf', room: '#232a2d', glass: '#64797f',
    ceiling: '#c9cac7', lamps: 2, under: 'panel', underColour: '#8e9295', ribs: '#8e9295', paving: '#cdc3bb'}},
};

/** What people wear in the stands by season: dark coats from November to March, light clothes from June to September. */
const CLOTHES = {
  winter: ['#1f2328', '#2b2f36', '#3a3d42', '#4a4036', '#5b5348', '#26324a', '#6b6f73', '#8a7f6e', '#d8d4cc', '#7a2f2f', '#2e4a3a', '#1a1a1c'],
  summer: ['#f2f2ee', '#e5e1d6', '#9bb5cf', '#3a3f46', '#5f7d99', '#c9c2b0', '#28303a', '#d7c9a8', '#a8b8a0', '#7a8a99', '#ffffff', '#1e2a3a'],
};
const HAIR = ['#17130f', '#1d1915', '#2a211a', '#3a2e24', '#e8e6e0', '#20304a', '#5a5a5a'];

/**
 * A person's outline turned about the vertical (height 0..1, radius in metres), wider at the shoulders than front to back:
 * standing (legs, hips, waist, chest, shoulders, neck) or seated (from the seat). Below `legs` they wear darker trousers.
 */
const STANDING = {profile: [[0.07, 0], [0.1, 0.03], [0.12, 0.45], [0.17, 0.5], [0.15, 0.62], [0.19, 0.8], [0.2, 0.9], [0.12, 0.95],
  [0.06, 0.97], [0.05, 1]], legs: 0.49};
const SEATED = {profile: [[0.05, -0.12], [0.18, -0.05], [0.18, 0.05], [0.15, 0.32], [0.19, 0.66], [0.2, 0.84], [0.12, 0.92],
  [0.06, 0.96], [0.05, 1]], legs: 0.03};
function person({profile, legs}: {profile: number[][]; legs: number}) {
  const g = new T.LatheGeometry(profile.map(([r, y]) => new T.Vector2(r, y)), 7);
  g.scale(1.15, 1, 0.8);
  const y = g.attributes.position;
  const shade = new Float32Array(y.count * 3);
  for (let i = 0; i < y.count; i++) shade.fill(y.getY(i) < legs ? 0.32 : 1, i * 3, i * 3 + 3);
  g.setAttribute('color', new T.BufferAttribute(shade, 3));
  return g;
}

/**
 * Seats (the Blender seat, app/standSeat.ts) and the crowd: low bodies and heads, seated or standing, dressed for the
 * season; in rain or snow the open terraces and the apron mostly empty.
 */
function crowdMeshes(k: Kit, day: number, look: WeatherLook, lift: (x: number) => number) {
  const out: T.Object3D[] = [];
  const seat = standSeatGeometry();
  const matrix = new T.Matrix4(), colour = new T.Color();
  if (seat && k.seats.length) {
    const seats = new T.InstancedMesh(seat, new T.MeshStandardMaterial({color: '#ffffff', vertexColors: true, roughness: 0.6}),
      k.seats.length);
    const random = seeded(5531);
    k.seats.forEach((s, i) => {
      seats.setMatrixAt(i, matrix.makeTranslation(s.x, s.y + lift(s.x), s.z));
      seats.setColorAt(i, colour.set(s.colour).offsetHSL(0, -0.04 * random(), 0.05 * (random() - 0.5)));
    });
    seats.computeBoundingSphere();
    seats.receiveShadow = true;
    seats.name = 'stand-seats';
    out.push(seats);
  }
  const season = day > 150 && day < 275 ? 'summer' : day > 120 && day < 300 ? null : 'winter';
  const palette = season ? CLOTHES[season] : [...CLOTHES.winter, ...CLOTHES.summer];
  const away = look.falling ? 0.25 : 1;
  const random = seeded(8807);
  const people = k.people.filter(p => !p.open || random() < away);
  if (people.length) {
    const standing = person(STANDING), seated = person(SEATED);
    const head = new T.SphereGeometry(0.115, 6, 4);
    const clothes = new T.MeshStandardMaterial({color: '#ffffff', vertexColors: true, roughness: 0.9});
    const sitters = people.filter(p => p.seated), standers = people.filter(p => !p.seated);
    const bodies = [new T.InstancedMesh(seated, clothes, sitters.length), new T.InstancedMesh(standing, clothes, standers.length)];
    const heads = new T.InstancedMesh(head, new T.MeshStandardMaterial({color: '#ffffff', roughness: 0.8}), people.length);
    const scale = new T.Vector3(), position = new T.Vector3(), turn = new T.Quaternion();
    const up = new T.Vector3(0, 1, 0);
    [...sitters, ...standers].forEach((p, i) => {
      const tall = 0.92 + 0.16 * random();
      // Seated: the body from the seat (0.42 m up) to the shoulders; standing: from the floor. Turned a little each way.
      const [foot, length] = p.seated ? [0.42, 0.72 * tall] : [0, 1.5 * tall];
      const y = p.y + lift(p.x);
      turn.setFromAxisAngle(up, (random() - 0.5) * (p.seated ? 0.5 : 1.6));
      const mesh = p.seated ? bodies[0] : bodies[1], j = p.seated ? i : i - sitters.length;
      mesh.setMatrixAt(j, matrix.compose(position.set(p.x, y + foot, p.z), turn, scale.set(tall, length, tall)));
      mesh.setColorAt(j, colour.set(palette[Math.floor(random() * palette.length)]).offsetHSL(0, 0, (random() - 0.5) * 0.06));
      heads.setMatrixAt(i, matrix.compose(position.set(p.x, y + foot + length + 0.1, p.z), turn, scale.set(1, 1.1, 1)));
      heads.setColorAt(i, colour.set(HAIR[Math.floor(random() ** 2 * HAIR.length)]));
    });
    for (const m of [...bodies, heads]) {
      if (!m.count) continue;
      m.computeBoundingSphere();
      m.receiveShadow = true;
      m.name = 'stand-crowd';
      out.push(m);
    }
  }
  return out;
}

export type StandPlacement = {
  /** Ground the trees keep off: the stand, its apron, and room round them. */
  keepOut: (p: T.Vector3) => boolean;
  /** Ground that is paved or built on (no grass grows there). */
  paved: (x: number, z: number) => boolean;
};

/**
 * Builds the venue's main stand outside the home straight (`turf`: the course it is set out from), `setback` metres beyond
 * the outermost rail of all the drawn `courses` along its length, with its paved apron, a spectators' railing along the
 * lawn or rail, and the crowd. Venues without a model get none.
 */
export function addMainStand(scene: T.Scene, venueCode: string, turf: TrackModel, courses: TrackModel[],
  heightAt: (x: number, z: number) => number, day: number, look: WeatherLook): StandPlacement | null {
  const spec = STANDS[venueCode];
  if (!spec) return null;
  // How far outside the turf's outer rail the outermost rail runs along the stand (where two courses' drawn straights
  // do not quite coincide).
  const across = courses.map(acrossOf);
  let beyond = 0;
  for (let d = spec.centre + spec.extent[0] - 10; d <= spec.centre + spec.extent[1] + 10; d += 5)
    for (let lane = turf.width; lane < turf.width + 60; lane += 0.5) {
      const p = turf.sample(d, lane).point;
      if (across.every((a, i) => a(p) > courses[i].width + 0.5)) {
        beyond = Math.max(beyond, lane - turf.width);
        break;
      }
    }
  const front = turf.sample(spec.centre, turf.width + beyond + spec.setback);
  const outward = front.outward.clone().setY(0).normalize();
  const up = new T.Vector3(0, 1, 0);
  const along = up.clone().cross(outward);
  const run = Math.sign(along.dot(front.tangent)) || 1;
  const group = new T.Group();
  group.name = 'main-stand';
  group.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(along, up, outward));
  group.position.set(front.point.x, 0, front.point.z);
  group.scale.x = run;
  group.updateMatrixWorld(true);
  const toWorld = (x: number, z: number) => new T.Vector3(x, 0, z).applyMatrix4(group.matrixWorld);
  const toLocal = new T.Matrix4().copy(group.matrixWorld).invert();

  // The apron's height along the stand's front, smoothed: the building is lifted to it at each x.
  const [ex0, ex1] = spec.extent;
  const xs: number[] = [], ys: number[] = [];
  for (let x = ex0 - 48; x <= ex1 + 48; x += 8) {
    const p = toWorld(x, -1);
    xs.push(x);
    ys.push(heightAt(p.x, p.z));
  }
  const smooth = ys.map((_, i) => {
    let sum = 0, n = 0;
    for (let j = Math.max(0, i - 3); j <= Math.min(ys.length - 1, i + 3); j++) { sum += ys[j]; n++; }
    return sum / n;
  });
  const lift = (x: number) => {
    const t = T.MathUtils.clamp((x - xs[0]) / 8, 0, xs.length - 1.0001);
    const i = Math.floor(t);
    return T.MathUtils.lerp(smooth[i], smooth[i + 1], t - i);
  };

  const materials = standFinish(spec.look, look.snowCover);
  const kit = new Kit(materials, 4049 + Number(venueCode), 1);
  spec.build(kit);

  // The apron: paving draped on the ground from the lawn (or the rail's bank) to the stand, a railing along its edge, and
  // people along it (most near the winning post) and about.
  const zApron = -spec.setback + 4 + spec.lawn;
  const ax0 = ex0 - 6, ax1 = ex1 + 6;
  const ground = (x: number, z: number) => {
    const p = toWorld(x, z);
    return heightAt(p.x, p.z);
  };
  const nx = Math.ceil((ax1 - ax0) / 4), nz = Math.ceil((0.8 - zApron) / 2);
  const positions: number[] = [], uvs: number[] = [], index: number[] = [];
  for (let i = 0; i <= nx; i++)
    for (let j = 0; j <= nz; j++) {
      const x = ax0 + (ax1 - ax0) * i / nx, z = zApron + (0.8 - zApron) * j / nz;
      positions.push(x, ground(x, z) + 0.18, z);
      uvs.push(x / 3.6, z / 3.6);
      if (i < nx && j < nz) {
        const a = i * (nz + 1) + j;
        index.push(a, a + 1, a + nz + 1, a + 1, a + nz + 2, a + nz + 1);
      }
    }
  const apronGeometry = new T.BufferGeometry();
  apronGeometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  apronGeometry.setAttribute('uv', new T.Float32BufferAttribute(uvs, 2));
  apronGeometry.setIndex(index);
  apronGeometry.computeVertexNormals();
  apronGeometry.setAttribute('color', new T.Float32BufferAttribute(new Float32Array(positions.length).fill(1), 3));
  const apron = new T.Mesh(apronGeometry, materials.paving);
  apron.receiveShadow = true;
  apron.name = 'stand-apron';
  group.add(apron);

  const railing = new Kit(materials, 1, 1);
  const finish = -spec.centre;
  for (let x = ax0; x < ax1; x += 2.5) {
    const y0 = ground(x, zApron + 0.4), y1 = ground(x + 2.5, zApron + 0.4);
    railing.box('steel', x - 0.04, x + 0.04, y0, y0 + 1.1, zApron + 0.36, zApron + 0.44, 1, 0);
    railing.quad('steel', [new T.Vector3(x, y0 + 1.05, zApron + 0.4), new T.Vector3(x + 2.5, y1 + 1.05, zApron + 0.4),
      new T.Vector3(x + 2.5, y1 + 1.12, zApron + 0.4), new T.Vector3(x, y0 + 1.12, zApron + 0.4)]);
    railing.quad('steel', [new T.Vector3(x + 2.5, y1 + 1.05, zApron + 0.4), new T.Vector3(x, y0 + 1.05, zApron + 0.4),
      new T.Vector3(x, y0 + 1.12, zApron + 0.4), new T.Vector3(x + 2.5, y1 + 1.12, zApron + 0.4)]);
  }
  const random = seeded(9157);
  const near = (x: number) => 0.3 + 0.7 * Math.exp(-(((x - finish) / 110) ** 2));
  for (let x = ax0 + 1; x < ax1 - 1; x += 0.62)
    for (let row = 0; row < 4; row++)
      if (random() < near(x) * (0.95 - row * 0.2)) {
        const z = zApron + 1 + row * 0.6 + random() * 0.2, px = x + (random() - 0.5) * 0.25;
        kit.people.push({x: px, y: ground(px, z) - lift(px), z, seated: false, open: true});
      }
  const area = (ax1 - ax0) * (-zApron);
  for (let i = 0; i < area * 0.035; i++) {
    const x = ax0 + random() * (ax1 - ax0), z = zApron + 4 + random() * (-zApron - 5);
    if (random() < near(x)) kit.people.push({x, y: ground(x, z) - lift(x), z, seated: false, open: true});
  }

  // Merge each material's pieces into one mesh; lift the building to the apron along its length.
  const merge = (source: Kit, lifted: boolean) => {
    for (const [key, parts] of source.parts) {
      if (!parts.length) continue;
      const geometry = mergeGeometries(parts.filter(p => p.attributes.position.count > 0));
      if (!geometry) continue;
      parts.forEach(p => p.dispose());
      if (lifted) {
        const p = geometry.attributes.position;
        for (let i = 0; i < p.count; i++) p.setY(i, p.getY(i) + lift(p.getX(i)));
      }
      geometry.computeBoundingSphere();
      const mesh = new T.Mesh(geometry, materials[key]);
      // The sun's shadow map only covers the ground round the field: a stand casting into it would show its shadow
      // cut off at the map's edge, so the building keeps its shade baked instead.
      mesh.receiveShadow = true;
      mesh.name = `stand-${key}`;
      group.add(mesh);
    }
  };
  merge(kit, true);
  merge(railing, false);
  for (const m of crowdMeshes(kit, day, look, lift)) group.add(m);
  scene.add(group);

  const local = new T.Vector3();
  return {
    keepOut: p => {
      local.copy(p).setY(0).applyMatrix4(toLocal);
      return local.x > ex0 - 22 && local.x < ex1 + 22 && local.z > -spec.setback - 8 && local.z < spec.depth + 28;
    },
    paved: (x, z) => {
      local.set(x, 0, z).applyMatrix4(toLocal);
      return local.x > ax0 && local.x < ax1 && local.z > zApron - 0.3 && local.z < spec.depth;
    },
  };
}
