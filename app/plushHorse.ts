import * as T from 'three';
import {seededPick} from './seededPick';
import {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
import type {Horse} from './types';
import {STRIDE_LENGTH, NO_KICK, type Kick} from './closingKick';
const coat: Record<string, string> = {
  '01': '#b8773c',
  '02': '#845330',
  '03': '#926039',
  '04': '#45332c',
  '05': '#292b2c',
  '06': '#202126',
  '07': '#c7c9c4',
  '08': '#b69a7f',
  '09': '#958475',
  '10': '#767b81',
  '11': '#f1e8d9',
};
export function label(text: string, width = 512) {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = 128;
  const x = c.getContext('2d')!;
  x.fillStyle = '#fffaf0';
  x.beginPath();
  x.roundRect(4, 12, width - 8, 100, 22);
  x.fill();
  x.fillStyle = '#263b34';
  x.font = width === 128 ? 'bold 82px sans-serif' : 'bold 36px sans-serif';
  x.textAlign = 'center';
  x.textBaseline = 'middle';
  x.fillText(text, width / 2, 64, width - 25);
  const map = new T.CanvasTexture(c);
  map.colorSpace = T.SRGBColorSpace;
  const s = new T.Sprite(new T.SpriteMaterial({map, depthTest: false}));
  s.scale.set((width / 128) * 1.8, 1.8, 1);
  s.renderOrder = 5;
  return s;
}

type Point = [number, number, number];
type LegRig = {mesh: T.Mesh; base: Float32Array; baseNormals: Float32Array; index: Uint32Array; leg: Uint8Array; weight: Float32Array; last: string};
type HairRig = {uniforms: HairUniforms; mesh: T.Mesh};
type HeadRig = {parts: {mesh: T.Mesh; base: Float32Array; baseNormals: Float32Array; index: Uint32Array; weight: Float32Array}[]; last: number};
type PlushRig = {body: T.Group; head: T.Group; tail: T.Group; legs?: LegRig; neck?: HeadRig; hair?: {mane: HairRig; tail: HairRig}};
// Gallop footfall order (right lead): left hind -> right hind -> left fore -> right fore, then a moment of
// suspension. `beat` is when each foot lands, as a fraction of the stride. The horse faces +x; its left is -z.
const LEGS = [
  {x: -0.82, z: -0.43, beat: 0, reach: 0.5, center: -0.06},
  {x: -0.82, z: 0.43, beat: 0.1, reach: 0.5, center: -0.06},
  {x: 0.64, z: -0.43, beat: 0.3, reach: 0.55, center: 0.06},
  {x: 0.64, z: 0.43, beat: 0.4, reach: 0.55, center: 0.06},
] as const;
const LEG_PIVOT_Y = 0.95;
/**
 * Closing-kick running form: how far each part moves from its normal gallop as the kick level goes 0 -> 1.
 * These are the numbers to tune by eye (the plush preview has a kick slider).
 */
export const KICK_HEAD_DOWN = 15 * Math.PI / 180; // neck/head base angle: lowered and stretched forward (nose down)
export const NOD = 0.08; // neck up-down amplitude of the normal gallop, in radians
export const KICK_NOD = 3; // neck up-down amplitude, as a multiple of the normal gallop's
export const KICK_NOD_BEAT = 0.35; // stride phase of the nod's lowest point: the forelegs land at 0.3 and 0.4
export const KICK_LEAN = 4 * Math.PI / 180; // forward lean of the whole body
export const KICK_ROCK = 1.4; // body pitch rocking, as a multiple
// Two gaits: a long, floating stride in the body of the race (ストライド走法), and a short, quick one in the
// closing kick (ピッチ走法). The quicker turnover itself is KICK_CADENCE in closingKick.ts.
export const STRIDE_REACH = 1.15; // leg swing of the normal gallop, as a multiple of the legs' own reach
export const KICK_REACH = 0.8; // leg swing at full kick, on the same scale
export const STRIDE_HOP = 0.09; // height of the suspension between strides in the normal gallop
export const KICK_HOP = 0.035; // and at full kick: the feet barely leave the ground
export const KICK_STREAM = 1.5; // how hard the air streams the mane and tail back, as a multiple

type Leg = (typeof LEGS)[number];

/** Swing of a leg from the vertical (before the gallop's ramp-up), `reach` being the stride's reach multiple. */
function legAngle(l: Leg, phase: number, reach: number) {
  // Foot lands at its beat reaching forward, sweeps back under the body (stance), then swings forward again.
  // The warp makes the stance quicker than the swing, as in a real gallop.
  const t = phase - l.beat;
  const warped = t + 0.5 / (2 * Math.PI) * Math.sin(2 * Math.PI * (t - 0.25));
  return l.center + l.reach * reach * Math.cos(2 * Math.PI * warped);
}

/** How far into the stride (after its beat) each leg is, when the hoof is flat under the hip and when it has swept most of the way back. */
function stanceStage(l: Leg, share: number) {
  // The stance runs from the leg's forward extreme to its rearmost; find the moment its swing is `share` of the way
  // back by bisection (the swing is monotonic through the stance).
  const forward = l.center + l.reach * STRIDE_REACH, back = l.center - l.reach * STRIDE_REACH;
  const target = forward + (back - forward) * share;
  let lo = 0, hi = 0.5;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (legAngle(l, l.beat + mid, STRIDE_REACH) > target) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
const HOOF_STAGES = {
  // Flat under the hip (leg vertical): the hoof is at its lowest and presses the ground.
  down: LEGS.map(l => stanceStage(l, (l.center + l.reach * STRIDE_REACH) / (2 * l.reach * STRIDE_REACH))),
  // Most of the way back: the hoof drives off, throwing the ground behind it.
  push: LEGS.map(l => stanceStage(l, 0.85)),
};

/** Hooves a horse has. */
export const HOOF_COUNT = LEGS.length;
/** The hind hooves push hardest; they are the first two (see LEGS). */
export const isHindHoof = (n: number) => LEGS[n].x < 0;
/** How far the horse's stride has come, in strides: the one number its legs, hair and hooves are all functions of. */
export function stridePhase(distance: number, reducedMotion = false, kick: Kick = NO_KICK) {
  return Math.max(distance, 0) / STRIDE_LENGTH + (reducedMotion ? 0 : kick.strides);
}
/** 0 while the horse stands, 1 once it is galloping (the legs ramp up over the first stride). */
export function gallopAmount(distance: number, reducedMotion = false) {
  return reducedMotion ? 0 : T.MathUtils.smoothstep(Math.max(distance, 0), 0, STRIDE_LENGTH);
}
/**
 * Counts the times leg `n`'s hoof has reached `stage` of its stance by this stride phase: it goes up by one at
 * each footfall ('down') or push-off ('push'). Compare it between two frames to find the footfalls between them;
 * `phase - beat - stage` crosses an integer at the moment, so the fraction of the way is recoverable too.
 */
export function hoofCount(n: number, phase: number, stage: 'down' | 'push') {
  return phase - LEGS[n].beat - HOOF_STAGES[stage][n];
}
/** Where leg `n`'s hoof is at this stride phase, in the horse's frame (x forward, z to its right), at the gallop's full swing. */
export function hoofSpot(n: number, phase: number, kickLevel: number, out: {x: number; z: number}) {
  const l = LEGS[n];
  out.x = l.x + LEG_PIVOT_Y * Math.sin(legAngle(l, phase, T.MathUtils.lerp(STRIDE_REACH, KICK_REACH, kickLevel)));
  out.z = l.z;
}
// STRIDE_LENGTH (race metres per stride) lives in closingKick.ts, next to the cadence that is integrated over it.

export const DEFAULT_LEG_WIDTH = 0.8;
export const HEAD_WIDTH_SCALES = [0.6, 0.7, 0.8, 0.9, 0.95] as const;
export const DEFAULT_HEAD_WIDTH = 0.8;
export type HeadSettings = {
  headWidth: number; noseSize: number; noseTaper: number; foreheadSlope: number;
  faceBlend: number; chestSlope: number; noseColor: number; eyeScale: number; maneHeight: number;
};
export const PREVIOUS_HEAD: HeadSettings = {headWidth: 1, noseSize: 0, noseTaper: 0, foreheadSlope: 0, faceBlend: 0, chestSlope: 0, noseColor: 0, eyeScale: 0, maneHeight: 0};
export const CURRENT_HEAD: HeadSettings = {headWidth: 0.8, noseSize: 1, noseTaper: 1, foreheadSlope: 1, faceBlend: 1, chestSlope: 1, noseColor: 1, eyeScale: 1, maneHeight: 1};
/** The head the user picked with the preview sliders; used everywhere by default. */
export const SAVED_HEAD: HeadSettings = {headWidth: 0.95, noseSize: 0.25, noseTaper: 0.5, foreheadSlope: 0.15, faceBlend: 0.15, chestSlope: 0.25, noseColor: 0.5, eyeScale: 0, maneHeight: 0.3};
type PlushOptions = Partial<HeadSettings> & {bridle?: boolean; fabric?: FabricChoice; labels?: boolean; pileLayers?: number};
export function normalizeHead(options: PlushOptions = {}): HeadSettings {
  const settings = {...SAVED_HEAD};
  for (const key of Object.keys(settings) as (keyof HeadSettings)[]) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) settings[key] = T.MathUtils.clamp(value, key === 'headWidth' ? 0.6 : 0, 1);
  }
  return settings;
}
function smoothUnion(a: number, b: number, k: number) {
  if (k <= 0) return Math.min(a, b);
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

/** The head alone (forehead + muzzle), in body units. */
function faceDistance(px: number, py: number, pz: number, settings: HeadSettings) {
  const {headWidth} = settings;
  const power = 2 / 0.82;
  // A tapered continuous volume replaces the separate forehead and muzzle balls.
  const hx = (px - 0.67) / 1.18;
  const hy = (py - 1.91) / 1.18;
  const hz = pz / (1.18 * headWidth);
  const rounded = (x: number, y: number, z: number, sx: number, sy: number, sz: number) =>
    (Math.pow(Math.pow(Math.abs(x / sx), power) + Math.pow(Math.abs(y / sy), power) + Math.pow(Math.abs(z / sz), power), 1 / power) - 1) * Math.min(sx, sy, sz);
  const oldForehead = rounded(hx - 0.24, hy - 0.15, hz, 0.72, 0.68, 0.61);
  const t = T.MathUtils.clamp((hx + (hy - 0.15) * -0.33) / 1.1089, 0, 1);
  const slopedForehead = Math.hypot(hx - t, hy - (0.15 - 0.33 * t), hz) - (0.59 - 0.32 * t);
  const forehead = T.MathUtils.lerp(oldForehead, slopedForehead, settings.foreheadSlope);
  const nx = hx - 0.79, ny = hy + 0.24;
  const rotatedX = nx * Math.cos(0.13) - ny * Math.sin(0.13);
  const rotatedY = nx * Math.sin(0.13) + ny * Math.cos(0.13);
  const taper = 1 - settings.noseTaper * 0.4 * T.MathUtils.smoothstep(hx, 0.5, 1.35);
  const nose = rounded(rotatedX, rotatedY, hz / taper,
    T.MathUtils.lerp(0.59, 0.44, settings.noseSize), T.MathUtils.lerp(0.47, 0.3, settings.noseSize), T.MathUtils.lerp(0.59, 0.4, settings.noseSize));
  return smoothUnion(forehead, nose, 0.2 * settings.faceBlend) * 1.18 * headWidth;
}

export function bodyDistance(px: number, py: number, pz: number, settings: HeadSettings, includeHead = true) {
  const {headWidth} = settings;
  const blend = (a: number, b: number) => {
    const smoothing = 0.32 * T.MathUtils.smoothstep(py, 0.22, 0.65);
    if (smoothing === 0) return Math.min(a, b);
    const h = Math.max(smoothing - Math.abs(a - b), 0) / smoothing;
    return Math.min(a, b) - h * h * smoothing * 0.25;
  };
  const power = 2 / 0.82;
  const d = (Math.pow(Math.pow(Math.abs((px + 0.15) / 1.22), power) + Math.pow(Math.abs((py - 1.22) / 0.87), power) + Math.pow(Math.abs(pz / 0.8), power), 1 / power) - 1) * 0.8;
  let feetDistance = Infinity;
  for (const lx of [-0.82, 0.64]) for (const lz of [-0.43, 0.43]) {
    // Circular cross-section, with softly rounded ends. The whole upper end
    // merges inside the belly rather than leaving a shoulder outside its outline.
    const bevel = 0.06;
    const radial = Math.hypot(px - lx, pz - lz) - (0.35 * DEFAULT_LEG_WIDTH - bevel);
    const vertical = Math.abs(py - 0.59) - (0.59 - bevel);
    const foot = Math.hypot(Math.max(radial, 0), Math.max(vertical, 0)) + Math.min(Math.max(radial, vertical), 0) - bevel;
    feetDistance = Math.min(feetDistance, foot);
  }
  const body = blend(d, feetDistance);
  if (!includeHead) return body;
  const face = faceDistance(px, py, pz, settings);
  const lean = T.MathUtils.clamp(((px - 0.55) * 0.38 + (py - 1.25) * 0.65) / 0.5669, 0, 1);
  const chest = (Math.hypot(px - (0.55 + 0.38 * lean), py - (1.25 + 0.65 * lean), pz / headWidth) - (0.57 - 0.08 * lean)) * headWidth;
  const base = smoothUnion(body, face, 0.32 * settings.faceBlend);
  const joined = blend(blend(body, chest), face);
  return T.MathUtils.lerp(base, joined, settings.chestSlope);
}

const HOOD_THICKNESS = 0.035;
// Felt colours a menko comes in; each horse wears the one its name picks, in every race.
const MENKO_COLORS = ['#327ca6', '#b8343c', '#2f7d52', '#d9a62e', '#6a4a9c', '#d0688f', '#23396f', '#d9712c', '#2a2b30', '#e9e4d6', '#3aa39a', '#8c2f55'];
const smoothMax = (a: number, b: number, k: number) => -smoothUnion(-a, -b, k);
/**
 * Menko (racing hood) as a solid of felt: the shell grown by the cloth thickness, cut to the hood's outline,
 * with eye holes. The part inside the head is hidden, so only the outer cloth and its rounded rim show.
 */
export function hoodDistance(px: number, py: number, pz: number, settings: HeadSettings) {
  const ux = (px - 0.67) / 1.18, uy = (py - 1.91) / 1.18;
  const k = 0.035;
  // Outline in head coordinates. Front edge: the head/nose border the user traced, from below the eye
  // up to the nose bridge; on the bridge it stops under the noseband so the strap hides the rim.
  const border = ((ux - 0.5) - (uy + 0.39) * 0.76) / Math.hypot(1, 0.76);
  const noseband = ((ux - 0.778) - (uy + 0.15) * 0.34) / Math.hypot(1, 0.34);
  const front = smoothMax(border, noseband, k);
  const below = -0.42 - uy;
  const behind = -0.5 - ux;
  const outline = smoothMax(smoothMax(front, below, k), behind, k) * 1.18;
  const eyeX = T.MathUtils.lerp(0.47, 0.36, settings.eyeScale);
  const eyeY = T.MathUtils.lerp(0.26, 0.25, settings.eyeScale);
  const eyeR = 0.13 * T.MathUtils.lerp(1.24, 1, settings.eyeScale) + 0.035;
  const eyeHole = (Math.hypot((ux - eyeX) / 1.05, uy - eyeY) - eyeR) * 1.18;
  // Follow the real shell (so the cloth never dives into the body), but only where the head is:
  // the rim ends where the shell drifts away from the head volume toward the torso.
  const headOnly = faceDistance(px, py, pz, settings) - 0.07;
  const cloth = bodyDistance(px, py, pz, settings) - HOOD_THICKNESS;
  return smoothMax(smoothMax(smoothMax(cloth, outline, k * 1.18), headOnly, k * 1.18), -eyeHole, k * 1.18);
}

const hoodTemplates = new Map<string, {positions: Float32Array; normals: Float32Array}>();
function hoodGeometry(settings: HeadSettings) {
  const cacheKey = JSON.stringify(settings);
  let template = hoodTemplates.get(cacheKey);
  if (!template) {
    // A finer grid than the body, only around the head, so the cloth rim stays crisp.
    const resolution = 84, size = 2.2, cx = 1.05, cy = 2.1;
    const material = new T.MeshBasicMaterial();
    const field = new MarchingCubes(resolution, material, false, false, 80000);
    field.isolation = 0;
    for (let z = 0; z < resolution; z++) for (let y = 0; y < resolution; y++) for (let x = 0; x < resolution; x++) {
      const px = (x / resolution - 0.5) * size + cx;
      const py = (y / resolution - 0.5) * size + cy;
      const pz = (z / resolution - 0.5) * size;
      field.field[x + resolution * (y + resolution * z)] = -hoodDistance(px, py, pz, settings);
    }
    field.update();
    const count = field.geometry.drawRange.count;
    const positions = new Float32Array(field.geometry.attributes.position.array.slice(0, count * 3));
    for (let i = 0; i < positions.length; i += 3) {
      positions[i] = positions[i] * size / 2 + cx;
      positions[i + 1] = positions[i + 1] * size / 2 + cy;
      positions[i + 2] *= size / 2;
    }
    template = {positions, normals: new Float32Array(field.geometry.attributes.normal.array.slice(0, count * 3))};
    if (hoodTemplates.size >= 8) hoodTemplates.delete(hoodTemplates.keys().next().value!);
    hoodTemplates.set(cacheKey, template);
    field.geometry.dispose();
    material.dispose();
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.BufferAttribute(template.positions.slice(), 3));
  geometry.setAttribute('normal', new T.BufferAttribute(template.normals.slice(), 3));
  const uv = new Float32Array(template.positions.length / 3 * 2);
  for (let i = 0; i < uv.length / 2; i++) {
    uv[i * 2] = template.positions[i * 3] * 0.45 + template.positions[i * 3 + 2] * 0.2;
    uv[i * 2 + 1] = template.positions[i * 3 + 1] * 0.45;
  }
  geometry.setAttribute('uv', new T.BufferAttribute(uv, 2));
  return geometry;
}

// Keep CPU arrays only. Each horse owns a disposable copy of the seamless shell.
const shellTemplates = new Map<string, {positions: Float32Array; normals: Float32Array}>();
function stuffedBodyGeometry(color: T.Color, settings: HeadSettings) {
  const cacheKey = JSON.stringify(settings);
  let shellTemplate = shellTemplates.get(cacheKey);
  if (!shellTemplate) {
    const resolution = 58;
    const temporaryMaterial = new T.MeshBasicMaterial();
    const field = new MarchingCubes(resolution, temporaryMaterial, false, false, 20000);
    field.isolation = 0;
    for (let z = 0; z < resolution; z++) for (let y = 0; y < resolution; y++) for (let x = 0; x < resolution; x++) {
      const px = (x / resolution - 0.5) * 4.2 + 0.35;
      const py = (y / resolution - 0.5) * 4.2 + 1.5;
      const pz = (z / resolution - 0.5) * 4.2;
      field.field[x + resolution * (y + resolution * z)] = -bodyDistance(px, py, pz, settings);
    }
    field.update();
    const count = field.geometry.drawRange.count;
    const positions = new Float32Array(field.geometry.attributes.position.array.slice(0, count * 3));
    for (let i = 0; i < positions.length; i += 3) {
      positions[i] = positions[i] * 2.1 + 0.35;
      positions[i + 1] = Math.max(0, positions[i + 1] * 2.1 + 1.5);
      positions[i + 2] *= 2.1;
    }
    shellTemplate = {positions, normals: new Float32Array(field.geometry.attributes.normal.array.slice(0, count * 3))};
    if (shellTemplates.size >= 8) shellTemplates.delete(shellTemplates.keys().next().value!);
    shellTemplates.set(cacheKey, shellTemplate);
    field.geometry.dispose();
    temporaryMaterial.dispose();
  }
  const geometry = new T.BufferGeometry();
  const positions = shellTemplate.positions.slice();
  const colors = new Float32Array(positions.length);
  const uvs = new Float32Array(positions.length / 3 * 2);
  const sock = new T.Color('#29272a');
  const oldNoseColor = color.clone().lerp(new T.Color('#dac8ae'), 0.28);
  for (let i = 0; i < positions.length / 3; i++) {
    const y = positions[i * 3 + 1];
    // A color change on the same surface: no separate sole or inflated cuff.
    const c = sock.clone().lerp(color, T.MathUtils.smoothstep(y, 0.23, 0.27));
    const noseAmount = T.MathUtils.smoothstep(positions[i * 3], 1.35, 1.65) * T.MathUtils.smoothstep(y, 1.2, 1.5);
    c.lerp(oldNoseColor, noseAmount * (1 - settings.noseColor));
    c.toArray(colors, i * 3);
    uvs[i * 2] = positions[i * 3] * 0.45 + positions[i * 3 + 2] * 0.2;
    uvs[i * 2 + 1] = y * 0.45;
  }
  geometry.setAttribute('position', new T.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new T.BufferAttribute(shellTemplate.normals.slice(), 3));
  geometry.setAttribute('color', new T.BufferAttribute(colors, 3));
  geometry.setAttribute('uv', new T.BufferAttribute(uvs, 2));
  return geometry;
}

// Small repeatable fibers: generated locally, with no downloaded texture assets.
export type FabricKind = 'current' | 'minky' | 'boa' | 'felt' | 'velour' | 'terry' | 'knit' | 'leather';
export const FABRIC_LABELS: Record<FabricKind, string> = {
  current: 'いまの布', minky: 'ミンキー', boa: 'ボア', felt: 'フェルト',
  velour: 'ベロア', terry: 'タオル地', knit: 'ニット', leather: '革',
};
/** Which fabric each part of the plush is made of. Unset parts keep the current cloth. */
export type FabricChoice = {body?: FabricKind; mane?: FabricKind; menko?: FabricKind; bridle?: FabricKind};
/** The fabrics the user picked on the comparison page; used everywhere by default. */
export const SAVED_FABRIC: Required<FabricChoice> = {body: 'velour', mane: 'boa', menko: 'felt', bridle: 'leather'};

type FabricLook = {
  repeat: number; bump: number; roughness: number; sheen: number; sheenRoughness: number; sheenColor: string;
  clearcoat?: number;
  /** Length of the raised pile in body units; fabrics without it are flat cloth. */
  pile?: number;
};
const FABRIC_LOOKS: Record<FabricKind, FabricLook> = {
  current: {repeat: 5, bump: 0.018, roughness: 1, sheen: 0.65, sheenRoughness: 1, sheenColor: '#d9d1c6', pile: 0.028},
  // Dense short pile: almost no visible grain, a soft bright glow at grazing angles.
  minky: {repeat: 4, bump: 0.004, roughness: 0.85, sheen: 1, sheenRoughness: 0.35, sheenColor: '#ffffff', pile: 0.04},
  // Curly long pile: deep bumpy curls.
  boa: {repeat: 2.4, bump: 0.05, roughness: 1, sheen: 0.8, sheenRoughness: 0.55, sheenColor: '#f2ece2', pile: 0.045},
  // Matted fibres, flat and dry.
  felt: {repeat: 2.4, bump: 0.012, roughness: 1, sheen: 0.15, sheenRoughness: 1, sheenColor: '#d9d1c6'},
  // Cut pile laid in one direction: smooth, with a strong shifting sheen.
  velour: {repeat: 3, bump: 0.003, roughness: 0.7, sheen: 1, sheenRoughness: 0.22, sheenColor: '#fff6ea', pile: 0.038},
  // Rows of small raised loops.
  terry: {repeat: 1.5, bump: 0.035, roughness: 1, sheen: 0.35, sheenRoughness: 0.8, sheenColor: '#e8e2d8'},
  // Columns of V stitches.
  knit: {repeat: 1.3, bump: 0.04, roughness: 1, sheen: 0.3, sheenRoughness: 0.8, sheenColor: '#e8e2d8'},
  // Fine pebble grain with a soft polish.
  leather: {repeat: 3, bump: 0.006, roughness: 0.5, sheen: 0, sheenRoughness: 1, sheenColor: '#000000', clearcoat: 0.3},
};

const fabricCanvases = new Map<FabricKind, HTMLCanvasElement>();
function fabricCanvas(kind: FabricKind) {
  const cached = fabricCanvases.get(kind);
  if (cached) return cached;
  const size = kind === 'current' ? 128 : 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  let seed = 831;
  const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const gray = (v: number, a = 1) => `rgba(${v | 0},${v | 0},${v | 0},${a})`;
  const fill = (v: number) => {ctx.fillStyle = gray(v); ctx.fillRect(0, 0, size, size);};
  // Draw a shape at x and its wrapped copies so the tile repeats without seams.
  const wrapped = (x: number, y: number, r: number, draw: (x: number, y: number) => void) => {
    for (const dx of [-size, 0, size]) for (const dy of [-size, 0, size]) {
      if (x + dx > -r && x + dx < size + r && y + dy > -r && y + dy < size + r) draw(x + dx, y + dy);
    }
  };
  if (kind === 'current') {
    ctx.fillStyle = '#eeeae5';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 6500; i++) {
      const x = random() * size, y = random() * size;
      ctx.strokeStyle = gray(190 + random() * 65);
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + random() * 1.6 - 0.8, y + 1 + random() * 2);
      ctx.stroke();
    }
  } else if (kind === 'minky') {
    fill(244);
    for (let i = 0; i < 26000; i++) {
      ctx.fillStyle = gray(222 + random() * 33, 0.7);
      ctx.fillRect(random() * size, random() * size, 1, 1);
    }
  } else if (kind === 'boa') {
    fill(178);
    for (let i = 0; i < 1100; i++) {
      const x = random() * size, y = random() * size, r = 3 + random() * 5, start = random() * Math.PI * 2;
      const light = 215 + random() * 40;
      wrapped(x, y, r + 3, (cx, cy) => {
        ctx.lineCap = 'round';
        ctx.strokeStyle = gray(150, 0.5);
        ctx.lineWidth = 3.4;
        ctx.beginPath();
        ctx.arc(cx + 0.8, cy + 1, r, start, start + 4.2);
        ctx.stroke();
        ctx.strokeStyle = gray(light);
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.arc(cx, cy, r, start, start + 4.2);
        ctx.stroke();
      });
    }
  } else if (kind === 'felt') {
    fill(236);
    for (let i = 0; i < 380; i++) {
      const x = random() * size, y = random() * size, r = 6 + random() * 18, v = 205 + random() * 50;
      wrapped(x, y, r, (cx, cy) => {
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g.addColorStop(0, gray(v, 0.28));
        g.addColorStop(1, gray(v, 0));
        ctx.fillStyle = g;
        ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
      });
    }
    ctx.lineWidth = 0.45;
    for (let i = 0; i < 12000; i++) {
      const x = random() * size, y = random() * size, a = random() * Math.PI * 2, l = 1 + random() * 3;
      ctx.strokeStyle = gray(200 + random() * 55, 0.8);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l);
      ctx.stroke();
    }
  } else if (kind === 'velour') {
    fill(240);
    // Pile streaks in one direction, low contrast; columns wrap so the tile repeats.
    const phase = [random() * 6, random() * 6, random() * 6];
    for (let x = 0; x < size; x++) {
      const t = x / size * Math.PI * 2;
      const v = 238 + 2.5 * Math.sin(t * 3 + phase[0]) + 1.5 * Math.sin(t * 11 + phase[1]) + Math.sin(t * 29 + phase[2]);
      ctx.fillStyle = gray(v);
      ctx.fillRect(x, 0, 1, size);
    }
    for (let i = 0; i < 9000; i++) {
      ctx.fillStyle = gray(225 + random() * 30, 0.5);
      ctx.fillRect(random() * size, random() * size, 1, 2);
    }
  } else if (kind === 'terry') {
    fill(168);
    const n = 24, step = size / n;
    for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
      const x = (col + (row % 2) * 0.5 + (random() - 0.5) * 0.25) * step;
      const y = (row + 0.5 + (random() - 0.5) * 0.25) * step;
      const r = step * (0.36 + random() * 0.08);
      wrapped(x, y, r + 3, (cx, cy) => {
        ctx.strokeStyle = gray(120, 0.6);
        ctx.lineWidth = 3.4;
        ctx.beginPath();
        ctx.arc(cx + 0.7, cy + 0.9, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = gray(232 + random() * 23);
        ctx.lineWidth = 2.8;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      });
    }
  } else if (kind === 'knit') {
    fill(140);
    const columns = 16, rows = 24, w = size / columns, h = size / rows;
    for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
      for (const side of [-1, 1]) {
        const cx = (col + 0.5) * w + side * w * 0.22, cy = (row + 0.5) * h;
        const g = ctx.createRadialGradient(cx - side * 1.5, cy - 1.5, 0, cx, cy, w * 0.42);
        const v = 238 + random() * 17;
        g.addColorStop(0, gray(v));
        g.addColorStop(0.75, gray(v - 40));
        g.addColorStop(1, gray(150));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.ellipse(cx, cy, w * 0.24, h * 0.62, side * -0.55, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  } else {
    fill(232);
    for (let i = 0; i < 1700; i++) {
      const x = random() * size, y = random() * size, r = 1.5 + random() * 3.5, v = 196 + random() * 44;
      wrapped(x, y, r, (cx, cy) => {
        ctx.fillStyle = gray(v, 0.55);
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    for (let i = 0; i < 8000; i++) {
      ctx.fillStyle = gray(210 + random() * 45, 0.5);
      ctx.fillRect(random() * size, random() * size, 1, 1);
    }
  }
  fabricCanvases.set(kind, canvas);
  return canvas;
}

function fabricTexture(kind: FabricKind, repeat: [number, number]) {
  const texture = new T.CanvasTexture(fabricCanvas(kind));
  texture.wrapS = texture.wrapT = T.RepeatWrapping;
  texture.repeat.set(...repeat);
  texture.colorSpace = T.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

/** A cloth material of the given fabric. `stretch` scales the tile along u for long thin parts (straps). */
function fabricMaterial(color: T.ColorRepresentation, kind: FabricKind, stretch: [number, number] = [1, 1]) {
  const look = FABRIC_LOOKS[kind];
  const repeat: [number, number] = kind === 'current'
    ? [5 * stretch[0], 4 * stretch[1]]
    : [look.repeat * stretch[0], look.repeat * stretch[1]];
  const map = fabricTexture(kind, repeat);
  return new T.MeshPhysicalMaterial({
    color, map, bumpMap: map, bumpScale: look.bump,
    roughness: look.roughness, sheen: look.sheen, sheenRoughness: look.sheenRoughness, sheenColor: look.sheenColor,
    clearcoat: look.clearcoat ?? 0, clearcoatRoughness: 0.5,
  });
}

const PILE_LAYERS = 12;
/** Shade at the fibre roots and of the cloth backing under the pile; the fibres brighten from this toward their tips. */
const PILE_ROOT_SHADE = 0.45;
let pileNoise: T.DataTexture | undefined;
/** R: one random height per fibre. G: smooth patches, for the marks left where the pile is brushed. B: tufts, high at their centres. */
function pileNoiseTexture() {
  if (pileNoise) return pileNoise;
  const size = 256, cells = 12;
  let seed = 4271;
  const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const lattice = Float32Array.from({length: cells * cells}, random);
  const tufts = 32;
  const seeds = Array.from({length: tufts * tufts}, () => [random(), random()]);
  // Distance to the nearest tuft centre, wrapping at the tile edge so the pattern repeats seamlessly.
  const tuft = (x: number, y: number) => {
    const fx = x / size * tufts, fy = y / size * tufts, cx = Math.floor(fx), cy = Math.floor(fy);
    let nearest = 4;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const [jx, jy] = seeds[((cx + dx + tufts) % tufts) + tufts * ((cy + dy + tufts) % tufts)];
      nearest = Math.min(nearest, Math.hypot(cx + dx + jx - fx, cy + dy + jy - fy));
    }
    return 1 - T.MathUtils.smoothstep(nearest, 0.1, 0.75);
  };
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const fx = x / size * cells, fy = y / size * cells;
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const tx = T.MathUtils.smoothstep(fx - x0, 0, 1), ty = T.MathUtils.smoothstep(fy - y0, 0, 1);
    const at = (cx: number, cy: number) => lattice[(cx % cells) + cells * (cy % cells)];
    const smooth = T.MathUtils.lerp(T.MathUtils.lerp(at(x0, y0), at(x0 + 1, y0), tx), T.MathUtils.lerp(at(x0, y0 + 1), at(x0 + 1, y0 + 1), tx), ty);
    const i = (x + y * size) * 4;
    data[i] = random() * 256;
    data[i + 1] = smooth * 255;
    data[i + 2] = tuft(x + 0.5, y + 0.5) * 255;
    data[i + 3] = 255;
  }
  pileNoise = new T.DataTexture(data, size, size, T.RGBAFormat);
  pileNoise.wrapS = pileNoise.wrapT = T.RepeatWrapping;
  pileNoise.magFilter = T.LinearFilter;
  pileNoise.minFilter = T.LinearMipmapLinearFilter;
  pileNoise.generateMipmaps = true;
  pileNoise.needsUpdate = true;
  return pileNoise;
}

/**
 * Raised pile: the same surface drawn several times, each copy pushed a little further out and keeping only
 * the fibres tall enough to reach it. This gives the cloth a soft fuzzy outline instead of a hard edge.
 */
function pileMaterial(color: T.ColorRepresentation, kind: FabricKind, layers: number, options: {vertexColors?: boolean; seam?: boolean; length?: number} = {}) {
  const look = FABRIC_LOOKS[kind];
  const material = new T.MeshPhysicalMaterial({
    color, vertexColors: !!options.vertexColors, roughness: 1,
    sheen: look.sheen * 0.3, sheenRoughness: Math.max(look.sheenRoughness, 0.3), sheenColor: look.sheenColor,
    alphaTest: 0.5,
  });
  const length = options.length ?? look.pile ?? 0.03;
  const seam = options.seam ? 1 : 0;
  material.onBeforeCompile = shader => {
    shader.uniforms.pileMap = {value: pileNoiseTexture()};
    shader.uniforms.pileLength = {value: length};
    shader.uniforms.pileLayers = {value: layers};
    shader.uniforms.pileSeam = {value: seam};
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
attribute vec3 rest;
attribute vec3 restNormal;
attribute float pileScale;
uniform float pileLength;
uniform float pileLayers;
varying float vPileScale;
varying vec3 vRest;
varying vec3 vRestNormal;
varying float vPile;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
vPile = (float(gl_InstanceID) + 1.0) / pileLayers;
vRest = rest;
vRestNormal = restNormal;
vPileScale = pileScale;
// Fibres stand along the normal and lean a little down and back, as if stroked.
float pileHeight = pileLength * pileScale * vPile;
// The stroke direction drifts from patch to patch but stays downward and backward overall.
vec3 lean = vec3(-0.2, -0.45, 0.0) + 0.25 * vec3(sin(rest.y * 7.1 + rest.z * 5.3), sin(rest.x * 6.3 + rest.z * 4.1), sin(rest.x * 5.1 + rest.y * 3.7));
transformed += objectNormal * pileHeight + lean * (pileHeight * vPile);`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
uniform sampler2D pileMap;
uniform float pileSeam;
varying float vPileScale;
varying vec3 vRest;
varying vec3 vRestNormal;
varying float vPile;`)
      .replace('#include <color_fragment>', `#include <color_fragment>
{
  // Pile pressed flat under other cloth is not drawn.
  if (vPileScale < 0.08) discard;
  // Project the fibre pattern along the surface's main axis, so it never stretches into streaks.
  vec3 fp = vRest * 1.35;
  vec3 fdx = dFdx(fp) * 0.85, fdy = dFdy(fp) * 0.85;
  vec3 an = abs(vRestNormal);
  vec2 fuv, gx, gy;
  // Each fibre is drawn as a short stroke lying the way the pile is brushed: downward, and backward on top.
  vec2 lie;
  if (an.x > an.y && an.x > an.z) { fuv = fp.yz; gx = fdx.yz; gy = fdy.yz; lie = vec2(0.65, 1.0); }
  else if (an.y > an.z) { fuv = fp.zx; gx = fdx.zx; gy = fdy.zx; lie = vec2(1.0, 0.65); }
  else { fuv = fp.xy; gx = fdx.xy; gy = fdy.xy; lie = vec2(1.0, 0.65); }
  fuv *= lie; gx *= lie; gy *= lie;
  float fibre = textureGrad(pileMap, fuv, gx, gy).r;
  // Tufts: sampled coarser than the fibres, so each one spans several of them.
  float tuft = textureGrad(pileMap, fuv * 3.2, gx * 3.2, gy * 3.2).b;
  float brushed = texture2D(pileMap, vRest.xy * 0.21 + 0.13).g + texture2D(pileMap, vRest.zy * 0.21 + 0.57).g - 1.0;
  // The pile is pinched short along the centre seam where the two halves are sewn together.
  float seam = pileSeam * (1.0 - smoothstep(0.004, 0.034, abs(vRest.z)));
  float reach = (1.0 - 0.7 * seam) * (0.8 + 0.25 * brushed) * mix(0.6, 1.0, tuft);
  diffuseColor.a = fibre - 0.92 * pow(min(vPile / reach, 1.5), 1.2) + 0.5;
  diffuseColor.rgb *= mix(${PILE_ROOT_SHADE.toFixed(2)}, 1.05, pow(vPile, 1.5)) * (1.0 + 0.2 * brushed) * (1.0 - 0.25 * (1.0 - tuft)) * (1.0 - 0.15 * seam);
}`)
      // The soft glow at grazing angles takes the colour of the fibres, so dark cloth stays dark.
      .replace('#include <lights_physical_fragment>', `#include <lights_physical_fragment>
#ifdef USE_SHEEN
material.sheenColor *= mix(diffuseColor.rgb, vec3(1.0), 0.08);
#endif`);
  };
  return material;
}

/** Cover a mesh with pile. The layers share its geometry, so they follow the legs as they swing. */
function addPile(mesh: T.Mesh, material: T.Material, layers: number) {
  const geometry = mesh.geometry;
  if (!geometry.attributes.rest) {
    geometry.setAttribute('rest', new T.BufferAttribute((geometry.attributes.position.array as Float32Array).slice(), 3));
    geometry.setAttribute('restNormal', new T.BufferAttribute((geometry.attributes.normal.array as Float32Array).slice(), 3));
  }
  if (!geometry.attributes.pileScale) {
    geometry.setAttribute('pileScale', new T.BufferAttribute(new Float32Array(geometry.attributes.position.count).fill(1), 1));
  }
  const pile = new T.InstancedMesh(geometry, material, layers);
  pile.name = 'pile';
  pile.receiveShadow = true;
  pile.frustumCulled = false;
  pile.raycast = () => {};
  mesh.add(pile);
  return pile;
}

type Strand = {points: T.Vector3[]; radius: number; shade: number};
type HairTemplate = {
  positions: Float32Array; normals: Float32Array; colors: Float32Array; index: Uint32Array;
  /** For the sway shader: 0 at the root to 1 at the tip, a random seed per hair, the hair's root, and its path per unit of fraction. */
  fractions: Float32Array; seeds: Float32Array; roots: Float32Array; spans: Float32Array;
};
/** Bump when the template layout changes, so stale templates (hot reload) are never reused. */
const HAIR_TEMPLATE_VERSION = 2;
type Solid = (x: number, y: number, z: number) => number;

/** Thin tapering tubes, one per hair, merged into a single mesh. */
function strandTemplate(strands: Strand[]): HairTemplate {
  const sides = 4;
  const rings = strands.reduce((n, s) => n + s.points.length, 0);
  const positions = new Float32Array(rings * sides * 3), normals = new Float32Array(rings * sides * 3), colors = new Float32Array(rings * sides * 3);
  const fractions = new Float32Array(rings * sides), seeds = new Float32Array(rings * sides);
  const roots = new Float32Array(rings * sides * 3), spans = new Float32Array(rings * sides * 3);
  const index: number[] = [];
  const tangent = new T.Vector3(), side = new T.Vector3(), across = new T.Vector3(), offset = new T.Vector3();
  let v = 0;
  for (const [n, strand] of strands.entries()) {
    const last = strand.points.length - 1;
    let length = 0;
    for (let i = 0; i < last; i++) length += strand.points[i].distanceTo(strand.points[i + 1]);
    // Golden-ratio spacing: any run of neighbouring hairs gets well-spread seeds.
    const seed = ((n + 1) * 0.6180339887) % 1;
    for (let i = 0; i <= last; i++) {
      tangent.copy(strand.points[Math.min(i + 1, last)]).sub(strand.points[Math.max(i - 1, 0)]).normalize();
      // Carry the frame along the hair so the tube never twists.
      if (i === 0) side.set(tangent.y, -tangent.x, 0.3).cross(tangent);
      side.addScaledVector(tangent, -side.dot(tangent)).normalize();
      across.crossVectors(tangent, side);
      const t = i / last;
      const radius = strand.radius * (1 - 0.96 * Math.pow(t, 1.35)) * T.MathUtils.lerp(0.6, 1, T.MathUtils.smoothstep(t, 0, 0.15));
      // Darker where the hairs crowd together at the root, lighter toward the loose tips.
      const shade = strand.shade * T.MathUtils.lerp(0.72, 1.04, T.MathUtils.smoothstep(t, 0, 0.7));
      const p = strand.points[i];
      for (let j = 0; j < sides; j++) {
        const a = j / sides * Math.PI * 2;
        offset.copy(side).multiplyScalar(Math.cos(a)).addScaledVector(across, Math.sin(a));
        positions.set([p.x + offset.x * radius, p.y + offset.y * radius, p.z + offset.z * radius], v * 3);
        normals.set([offset.x, offset.y, offset.z], v * 3);
        colors.set([shade, shade, shade], v * 3);
        fractions[v] = t;
        seeds[v] = seed;
        roots.set([strand.points[0].x, strand.points[0].y, strand.points[0].z], v * 3);
        spans.set([tangent.x * length, tangent.y * length, tangent.z * length], v * 3);
        if (i < last) {
          const next = v - j + (j + 1) % sides;
          index.push(v, next, v + sides, next, next + sides, v + sides);
        }
        v++;
      }
    }
  }
  return {positions, normals, colors, index: Uint32Array.from(index), fractions, seeds, roots, spans};
}

// One geometry per template, shared by every horse: the sway is done by the vertex shader, so nothing per-horse is baked in.
const hairGeometries = new WeakMap<HairTemplate, T.BufferGeometry>();
function hairGeometry(template: HairTemplate) {
  const shared = hairGeometries.get(template);
  if (shared) return shared;
  const geometry = new T.BufferGeometry();
  hairGeometries.set(template, geometry);
  geometry.setAttribute('position', new T.BufferAttribute(template.positions, 3));
  geometry.setAttribute('normal', new T.BufferAttribute(template.normals, 3));
  geometry.setAttribute('color', new T.BufferAttribute(template.colors, 3));
  geometry.setAttribute('hairFrac', new T.BufferAttribute(template.fractions, 1));
  geometry.setAttribute('hairSeed', new T.BufferAttribute(template.seeds, 1));
  geometry.setAttribute('hairRoot', new T.BufferAttribute(template.roots, 3));
  geometry.setAttribute('hairSpan', new T.BufferAttribute(template.spans, 3));
  geometry.setIndex(new T.BufferAttribute(template.index, 1));
  return geometry;
}

/**
 * Grow one hair from its root: it leaves in `direction`, bends under gravity, and is pushed back out of
 * `solid` (the stuffed body) so it lies on the surface instead of passing through it.
 */
function growStrand(
  root: T.Vector3, direction: T.Vector3, length: number, segments: number,
  shape: {gravity: T.Vector3; droop: number; solid: Solid; clearance: number; wave: number; random: () => number},
) {
  const {gravity, droop, solid, clearance, wave, random} = shape;
  const points = [root.clone()];
  const p = root.clone(), v = direction.clone().normalize(), previous = new T.Vector3(), normal = new T.Vector3();
  const phase = random() * Math.PI * 2, frequency = 0.7 + random() * 0.9;
  const waveAxis = new T.Vector3(random() - 0.5, random() - 0.5, random() - 0.5).normalize();
  const e = 0.01;
  for (let i = 1; i <= segments; i++) {
    previous.copy(p);
    v.addScaledVector(gravity, droop).addScaledVector(waveAxis, wave * Math.sin(phase + i * frequency)).normalize();
    p.addScaledVector(v, length / segments);
    const gap = clearance * Math.min(1, i / 2);
    let touched = false;
    for (let pass = 0; pass < 2; pass++) {
      const d = solid(p.x, p.y, p.z);
      if (d >= gap) break;
      normal.set(
        solid(p.x + e, p.y, p.z) - solid(p.x - e, p.y, p.z),
        solid(p.x, p.y + e, p.z) - solid(p.x, p.y - e, p.z),
        solid(p.x, p.y, p.z + e) - solid(p.x, p.y, p.z - e),
      );
      if (normal.lengthSq() === 0) break;
      p.addScaledVector(normal.normalize(), gap - d);
      touched = true;
    }
    if (touched) v.copy(p).sub(previous).normalize();
    points.push(p.clone());
  }
  return points;
}

const maneTemplates = new Map<string, HairTemplate>();
/**
 * Outline of the mane as the user drew it, in body units: seen from the side, the range of x at each height
 * (on the centre plane); seen from behind, the range of z. The hair fills the volume inside both outlines.
 */
const MANE_SIDE: [number, number, number][] = [
  [3.1, 0.24, 0.58], [3.04, 0.12, 0.63], [2.95, 0.04, 0.63], [2.9, -0.03, 0.62], [2.8, -0.09, 0.6],
  [2.7, -0.13, 0.62], [2.6, -0.16, 0.6], [2.5, -0.17, 0.52], [2.4, -0.2, 0.46], [2.3, -0.25, 0.41],
  [2.2, -0.34, 0.35], [2.1, -0.5, 0.27], [2.02, -0.52, 0.12],
];
const MANE_BACK: [number, number, number][] = [
  [3.12, -0.08, 0.12], [3.07, -0.22, 0.26], [3.01, -0.28, 0.32], [2.95, -0.33, 0.37], [2.65, -0.33, 0.37],
  [2.45, -0.31, 0.35], [2.35, -0.28, 0.32], [2.25, -0.24, 0.28], [2.19, -0.2, 0.24], [2.13, -0.12, 0.16], [2.03, -0.03, 0.07],
];
/**
 * How far hair may stray past the drawn outline: a little front-to-back, and further sideways where it lies on the
 * side of the head and neck. The side drawing reaches down the neck further than the rear drawing is wide; 0.2 is the
 * measured best balance between the two.
 */
const MANE_SPILL = 0.05, MANE_SIDE_SPILL = 0.2;
function outlineSpan(table: [number, number, number][], y: number): [number, number] | null {
  if (y > table[0][0] || y < table[table.length - 1][0]) return null;
  for (let i = 0; i + 1 < table.length; i++) {
    const [ya, a0, a1] = table[i], [yb, b0, b1] = table[i + 1];
    if (y <= ya && y >= yb) {
      const t = (ya - y) / (ya - yb);
      return [T.MathUtils.lerp(a0, b0, t), T.MathUtils.lerp(a1, b1, t)];
    }
  }
  return null;
}

/** The mane, in body coordinates: locks of hair rising from the crest and swept back, from the poll to the withers. */
function maneGeometry(settings: HeadSettings, menko: boolean) {
  const cacheKey = JSON.stringify(settings) + menko + HAIR_TEMPLATE_VERSION;
  let template = maneTemplates.get(cacheKey);
  if (!template) {
    const {headWidth} = settings;
    // Taller or shorter about the crest, with the drawn outline at the saved setting.
    const stretch = 1 + (SAVED_HEAD.maneHeight - settings.maneHeight) * 0.3;
    const base = 2.0;
    const side = (y: number) => outlineSpan(MANE_SIDE, base + (y - base) / stretch);
    const back = (y: number) => outlineSpan(MANE_BACK, base + (y - base) / stretch);
    const top = base + (MANE_SIDE[0][0] - base) * stretch;
    const inside = (p: T.Vector3, tolerance = 0, sideways = tolerance) => {
      const sx = side(p.y), bz = back(p.y);
      return !!sx && !!bz && p.x >= sx[0] - tolerance && p.x <= sx[1] + tolerance && p.z >= bz[0] - sideways && p.z <= bz[1] + sideways;
    };
    const skin: Solid = (x, y, z) => menko
      ? Math.min(bodyDistance(x, y, z, settings), hoodDistance(x, y, z, settings))
      : bodyDistance(x, y, z, settings);
    // The ears, as thin ellipsoids, so the hair parts around them instead of passing through.
    const ear = (x: number, y: number, z: number) => {
      const hx = (x - 0.67) / 1.18, hy = (y - 1.91) / 1.18, hz = z / 1.18;
      const s = hz < 0 ? -1 : 1, a = -s * 0.2;
      const ly = hy - 0.58, lz = hz - s * 0.3 * headWidth;
      const ry = ly * Math.cos(a) - lz * Math.sin(a), rz = ly * Math.sin(a) + lz * Math.cos(a);
      return (Math.hypot((hx + 0.06) / 0.07, (ry - 0.33) / 0.36, rz / 0.17) - 1) * 0.07 * 1.18;
    };
    const solid: Solid = (x, y, z) => Math.min(skin(x, y, z), ear(x, y, z));
    const gradient = (f: Solid, p: T.Vector3) => {
      const e = 0.01;
      return new T.Vector3(
        f(p.x + e, p.y, p.z) - f(p.x - e, p.y, p.z),
        f(p.x, p.y + e, p.z) - f(p.x, p.y - e, p.z),
        f(p.x, p.y, p.z + e) - f(p.x, p.y, p.z - e),
      ).normalize();
    };
    const normalAt = (p: T.Vector3) => gradient(skin, p);
    // Hair that meets the head or neck lies on it rather than passing through.
    const pushOut = (p: T.Vector3, clearance: number) => {
      for (let pass = 0; pass < 4; pass++) {
        const d = solid(p.x, p.y, p.z);
        if (d >= clearance) return;
        p.addScaledVector(gradient(solid, p), clearance - d);
      }
    };
    const surfaceBelow = (x: number, z: number) => {
      let lo = 1, hi = top + 0.3;
      if (skin(x, hi, z) < 0 || skin(x, lo, z) > 0) return null;
      for (let k = 0; k < 22; k++) {
        const y = (lo + hi) / 2;
        if (skin(x, y, z) < 0) lo = y; else hi = y;
      }
      return new T.Vector3(x, (lo + hi) / 2, z);
    };
    let seed = 9157;
    const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
    const strands: Strand[] = [];
    const steps = 16, perLock = 40;
    const backward = new T.Vector3(-1, 0, 0), sideways = new T.Vector3(0, 0, 1);
    /**
     * One guide hair per lock. It leaves the crest along the skin's normal, sweeps back toward the tail and fans a
     * little to one side, and ends where it meets the drawn outline, so the tips of the locks trace that outline.
     */
    const growGuide = (root: T.Vector3, lean: number, fan: number, curl: number, drape: boolean) => {
      const normal = normalAt(root);
      const v = drape
        ? normal.clone().multiplyScalar(0.35).addScaledVector(backward, 0.4).addScaledVector(sideways, fan < 0 ? -1 : 1).normalize()
        : normal.clone().addScaledVector(backward, lean).addScaledVector(sideways, fan).normalize();
      // Standing locks curl gently backward at the tip; draped ones fall down the side of the neck.
      const bend = drape ? new T.Vector3(-0.25, -1, 0).multiplyScalar(0.22) : new T.Vector3(-0.8, -0.45, 0).multiplyScalar(curl);
      const step = (drape ? 0.45 : 0.8) / steps;
      const points = [root.clone()], p = root.clone();
      for (let i = 1; i <= steps; i++) {
        v.add(bend);
        if (!drape) v.z *= 0.97;
        p.addScaledVector(v.normalize(), step);
        pushOut(p, 0.012);
        if (!drape) {
          // Slide along the side of the outline rather than stopping there: the lock keeps its length.
          const bz = back(p.y);
          if (bz) p.z = T.MathUtils.clamp(p.z, bz[0], bz[1]);
        }
        if (!inside(p, drape ? MANE_SPILL : 0.012, drape ? MANE_SIDE_SPILL : 0.012)) break;
        points.push(p.clone());
      }
      return points;
    };
    // The crest climbs steeply from the withers to the poll: space the roots evenly along it, not along x.
    const crestX: number[] = [], crestLength: number[] = [];
    for (let i = 0, previous: T.Vector3 | null = null, total = 0; i <= 110; i++) {
      const at = surfaceBelow(-0.5 + i * 0.01, 0);
      if (!at) continue;
      if (previous) total += at.distanceTo(previous);
      previous = at;
      crestX.push(at.x); crestLength.push(total);
    }
    const crestAt = (u: number) => {
      const target = u * crestLength[crestLength.length - 1];
      let i = 1;
      while (i < crestLength.length - 1 && crestLength[i] < target) i++;
      return T.MathUtils.mapLinear(target, crestLength[i - 1], crestLength[i], crestX[i - 1], crestX[i]);
    };
    for (let tries = 0; strands.length < 9000 && tries < 6000; tries++) {
      // Roots along the sewn crest line, which widens into a patch between the ears.
      const x = crestAt(random());
      const poll = T.MathUtils.smoothstep(x, 0.15, 0.5);
      const halfWidth = T.MathUtils.lerp(0.15, 0.2, poll);
      const across = random() * 2 - 1;
      const surface = surfaceBelow(x, across * halfWidth);
      if (!surface || !inside(surface, 0.03)) continue;
      const root = surface.addScaledVector(normalAt(surface), -0.01);
      const drape = x > 0.05 && Math.abs(across) > 0.6 && random() < 0.3;
      // Hair fans away from the centre line in proportion to how far out it is rooted, so no two locks cross;
      // further at the poll, where the outline is widest.
      const fan = (across + (random() - 0.5) * 0.3) * T.MathUtils.lerp(0.9, 0.7, poll);
      // Upright at the poll to reach the top of the outline; swept further back down the neck.
      const sweep = T.MathUtils.lerp(0.35 + random() * 0.9, random() * 0.55, poll);
      const curl = T.MathUtils.lerp(0.05, 0.008, poll) * (0.6 + random() * 0.8);
      let guide: T.Vector3[] = [];
      for (let attempt = 0; attempt < 5 && guide.length < 6; attempt++) {
        // Try the full backward sweep first; near the back edge of the outline only upright locks have room.
        guide = growGuide(root, sweep * (1 - attempt / 5), fan, curl, drape);
      }
      if (guide.length < 4) continue;
      const phase = random() * Math.PI * 2, twist = (random() - 0.5) * 2.4;
      const shade = 0.9 + random() * 0.22;
      for (let n = 0; n < perLock; n++) {
        // Each hair starts a little apart from the guide and gathers toward it, so the lock tapers to a tip.
        const angle = random() * Math.PI * 2, r = 0.055 * Math.sqrt(random());
        const at = surfaceBelow(root.x + Math.cos(angle) * r, root.z + Math.sin(angle) * r);
        if (!at) continue;
        const offset = at.sub(root);
        const reach = 0.62 + 0.38 * Math.pow(random(), 0.6);
        const last = Math.max(3, Math.round((guide.length - 1) * reach));
        const frizz = 0.002 + random() * 0.004, wave = random() * Math.PI * 2;
        const points: T.Vector3[] = [];
        for (let i = 0; i <= last; i++) {
          const t = i / (guide.length - 1);
          const gather = T.MathUtils.lerp(1, 0.5, Math.pow(t, 0.8));
          // The lock turns slightly about its own axis as it rises.
          const c = Math.cos(twist * t), s = Math.sin(twist * t);
          const point = guide[i].clone().add(new T.Vector3(
            (offset.x * c - offset.z * s) * gather,
            offset.y * (1 - t),
            (offset.x * s + offset.z * c) * gather,
          ));
          const ripple = Math.sin(t * 9 + phase + wave * 0.3) * frizz * t;
          point.x += ripple;
          point.z += Math.cos(t * 7 + wave) * frizz * t;
          if (i > 0) pushOut(point, 0.008);
          points.push(point);
        }
        strands.push({points, radius: 0.0016 + random() * 0.001, shade: shade * (0.94 + random() * 0.12)});
      }
    }
    template = strandTemplate(strands);
    if (maneTemplates.size >= 8) maneTemplates.delete(maneTemplates.keys().next().value!);
    maneTemplates.set(cacheKey, template);
  }
  return hairGeometry(template);
}

const TAIL_ROOT: Point = [-1.22, 1.38, 0];
const tailTemplates = new Map<string, HairTemplate>();
/** The tail, in the tail group's coordinates: long hairs that leave the dock, arch, and hang clear of the hindquarters. */
function tailGeometry(settings: HeadSettings) {
  const cacheKey = JSON.stringify(settings) + HAIR_TEMPLATE_VERSION;
  let template = tailTemplates.get(cacheKey);
  if (!template) {
    const solid: Solid = (x, y, z) => bodyDistance(x + TAIL_ROOT[0], y + TAIL_ROOT[1], z + TAIL_ROOT[2], settings, false);
    let seed = 2203;
    const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
    const gravity = new T.Vector3(-0.04, -1, 0).normalize();
    const strands: Strand[] = [];
    // One curved centre line keeps the long hairs together at the tip.
    const spine = growStrand(new T.Vector3(-0.15, 0, 0), new T.Vector3(-1, 0.4, 0), 1.5, 36,
      {gravity, droop: 0.085, solid, clearance: 0.12, wave: 0, random});
    for (let n = 0; n < 1500; n++) {
      const angle = random() * Math.PI * 2, spread = Math.sqrt(random());
      // Outer hairs finish earlier; only a few central hairs reach the very end.
      const reach = 0.52 + 0.48 * Math.pow(1 - spread, 0.7);
      const points: T.Vector3[] = [];
      for (let i = 0; i <= 18; i++) {
        const t = i / 18 * reach, sample = t * (spine.length - 1);
        const k = Math.min(Math.floor(sample), spine.length - 2);
        const center = spine[k].clone().lerp(spine[k + 1], sample - k);
        const tangent = spine[k + 1].clone().sub(spine[k]).normalize();
        const across = new T.Vector3(-tangent.y, tangent.x, 0);
        // A narrow dock opens into a full middle, then closes to a tapered tip.
        const envelope = 0.065 * (1 - t) + 0.25 * Math.pow(Math.sin(Math.PI * t), 1.1);
        const radius = spread * envelope;
        center.addScaledVector(across, Math.cos(angle) * radius);
        center.z += Math.sin(angle) * radius;
        points.push(center);
      }
      strands.push({
        points,
        radius: 0.00175 + random() * 0.001,
        shade: 0.86 + random() * 0.3,
      });
    }
    template = strandTemplate(strands);
    if (tailTemplates.size >= 8) tailTemplates.delete(tailTemplates.keys().next().value!);
    tailTemplates.set(cacheKey, template);
  }
  return hairGeometry(template);
}

type HairUniforms = {
  hairPhase: {value: number}; hairAmount: {value: number}; hairKick: {value: number}; hairWind: {value: number}; hairHead: {value: number};
  hairToBody: {value: T.Matrix4}; hairFromBody: {value: T.Matrix4};
};
const hairUniforms = (): HairUniforms => ({
  hairPhase: {value: 0}, hairAmount: {value: 0}, hairKick: {value: 0}, hairWind: {value: 0}, hairHead: {value: 0},
  hairToBody: {value: new T.Matrix4()}, hairFromBody: {value: new T.Matrix4()},
});

/**
 * Sway tuning, one set per kind of hair. Displacements are in units of the hair's own length (times fraction^2), so
 * `wind` 0.35 leans the tip of a mane hair back by about a third of its length.
 * natural/drop: how fast a hair swings by itself (relative to the stride) at the root and how much slower toward the tip;
 * bounce: how much of the body's vertical shake it picks up; wind/lift: streaming back and up; side/sideRate: sideways
 * sway and its rate in cycles per stride; flutter: the fine 2x/3x ripple; travel: radians of phase lag from root to tip.
 */
const HAIR_TUNING = {
  mane: {natural: 3.4, drop: 1.5, bounce: 1.0, wind: 0.35, lift: 0.12, side: 0.05, sideRate: 1, flutter: 0.06, travel: 0.9},
  tail: {natural: 3.0, drop: 1.4, bounce: 1.7, wind: 0.55, lift: 0.06, side: 0.3, sideRate: 0.5, flutter: 0.08, travel: 1.3},
};
// Hair never passes through the body: ellipsoids just inside the stuffed shell (centre, radii), in body coordinates.
// Checked numerically against bodyDistance: every point of each lies at least 0.02 inside the shell.
const HAIR_BARREL = {center: [-0.15, 1.22, 0], radii: [1.19, 0.84, 0.77]};
const HAIR_RUMP = {center: [-0.7, 1.2, 0], radii: [0.6, 0.75, 0.65]};
// The neck: a tapered capsule from `from` to `to`, flattened sideways by `width` like the chest it is built from.
const HAIR_NECK = {from: [0.55, 1.25], to: [0.93, 1.9], radii: [0.54, 0.46], width: 0.92};
const glsl = (v: number[]) => `vec${v.length}(${v.map(n => n.toFixed(3)).join(', ')})`;

/**
 * Vertex-shader sway for mane and tail. Everything is a pure function of the stride phase, so it repeats exactly
 * when paused, seeked or recorded. Per hair: the root is fixed, the offset grows with the square of the root-to-tip
 * fraction, a wave runs toward the tip, and a seed gives it its own phase, stiffness and amplitude.
 * Work is done in body coordinates (the meshes pass matrices to and from them), so wind, the body's shake and the
 * collision ellipsoids are written once, whatever group the hair hangs from.
 */
function hairShader(kind: 'mane' | 'tail', depth: boolean) {
  const k = HAIR_TUNING[kind], f = (n: number) => n.toFixed(3);
  return `
${kind === 'mane' ? '#define HAIR_MANE' : ''}
${depth ? '#define HAIR_DEPTH' : ''}
attribute float hairFrac;
attribute float hairSeed;
attribute vec3 hairRoot;
attribute vec3 hairSpan;
uniform float hairPhase;
uniform float hairAmount;
uniform float hairKick;
uniform float hairWind;
uniform float hairHead;
uniform mat4 hairToBody;
uniform mat4 hairFromBody;

float hairHash(float x) { return fract(sin(x * 91.3458 + 17.17) * 47453.5453); }

// Pull a point out of an ellipsoid along the ray from its centre.
vec3 hairOut(vec3 p, vec3 c, vec3 r) {
  vec3 q = (p - c) / r;
  float k = dot(q, q);
  return k < 1.0 ? c + q * (r * inversesqrt(max(k, 1e-4))) : p;
}
vec3 hairCollide(vec3 p) {
  p = hairOut(p, ${glsl(HAIR_BARREL.center)}, ${glsl(HAIR_BARREL.radii)});
  p = hairOut(p, ${glsl(HAIR_RUMP.center)}, ${glsl(HAIR_RUMP.radii)});
  vec3 a = vec3(${HAIR_NECK.from.map(f).join(', ')}, 0.0), ab = vec3(${HAIR_NECK.to.map(f).join(', ')}, 0.0) - a;
  vec3 q = vec3(p.xy, p.z / ${f(HAIR_NECK.width)});
  float t = clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0);
  vec3 c = a + ab * t, v = q - c;
  float radius = mix(${f(HAIR_NECK.radii[0])}, ${f(HAIR_NECK.radii[1])}, t), l = length(v);
  if (l < radius && l > 1e-4) { q = c + v * (radius / l); p = vec3(q.xy, q.z * ${f(HAIR_NECK.width)}); }
  return p;
}

// Hair bobbing against the body's own vertical shake, as a damped oscillator driven by the body's acceleration
// (x'' + 2 zeta w x' + w^2 x = -y''). Solved in closed form per harmonic n of the stride: a hair that is slow
// to respond (low w) lags the body, so the body dropping lifts it late and the body rising drops it late.
float hairBob(float phi, float w, float n, float weight) {
  float a = n * (phi - 5.5292); // the body is highest at stride phase 0.88
  vec2 den = vec2(w * w - n * n, 0.8 * w * n);
  vec2 inv = vec2(den.x, -den.y) / dot(den, den);
  return n * n * weight * (inv.x * cos(a) - inv.y * sin(a));
}

// Where the hair is pushed, in body coordinates, at fraction f.
vec3 hairSway(float f, float len, vec3 along, vec4 r, float lock) {
  float stiff = mix(0.8, 1.3, r.x), gain = mix(0.7, 1.3, r.y);
  // The wave reaches the tip later than the root; each hair (and each lock of them) is a little out of step.
  float phi = 6.2831853 * hairPhase - ${f(k.travel)} * f - (lock + r.z) * 0.35;
  float w = (${f(k.natural)} - ${f(k.drop)} * f) * stiff;
  float bob = hairBob(phi, w, 1.0, 1.0) + hairBob(phi, w, 2.0, 0.3);
  vec3 d = vec3(0.0);
  d.y = bob * ${f(k.bounce)} * gain * (1.0 + 0.4 * hairKick) * hairAmount;
  d.x = -hairWind * ${f(k.wind)} * (0.8 + 0.4 * r.w);
  d.y += hairWind * ${f(k.lift)};
  d.z = hairAmount * ${f(k.side)} * gain * sin(${f(k.sideRate)} * phi + 6.2831853 * r.y);
  float flutter = 0.6 * sin(2.0 * phi + 6.2831853 * r.z) + 0.4 * sin(3.0 * phi + 6.2831853 * r.w);
  d.yz += ${f(k.flutter)} * hairAmount * vec2(flutter, 0.5 * sin(3.0 * phi + 9.0 * r.x) + 0.5 * cos(2.0 * phi + 3.0 * r.y));
  d *= f * f * len;
  // A hair bends rather than stretches: nearly all of the push along its own length is dropped.
  return d - 0.9 * along * dot(along, d);
}

// A hair keeps its length: the push swings the point about the hair's root instead of dragging it away, so a hard
// wind lays the tail out behind the horse rather than stretching it.
vec3 hairBend(vec3 base, vec3 span, float f, float len, vec3 d) {
  vec3 rel = span * f + d;
  float l = length(rel);
  return base - span * f + (l > 1e-5 ? rel * (f * len / l) : rel);
}

vec3 hairTurn(vec3 p, float a) {
  float c = cos(a), s = sin(a);
  return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

void hairDeform(out vec3 outPosition, out vec3 outNormal) {
  mat3 toBody = mat3(hairToBody);
  vec3 base = (hairToBody * vec4(position, 1.0)).xyz, span = toBody * hairSpan, n0 = toBody * normal;
#ifdef HAIR_MANE
  // The mane rides the neck, not the head: only the hair at the poll turns with the head.
  float turn = hairHead * smoothstep(${f(HEAD_BLEND[0])}, ${f(HEAD_BLEND[1])}, dot(hairRoot.xy - vec2(${HEAD_PIVOT.map(f).join(', ')}), vec2(${HEAD_AXIS.map(f).join(', ')})));
  vec3 pivot = vec3(${HEAD_PIVOT.map(f).join(', ')}, 0.0);
  base = pivot + hairTurn(base - pivot, turn); span = hairTurn(span, turn); n0 = hairTurn(n0, turn);
#endif
  float len = length(span);
  vec3 along = span / max(len, 1e-5);
  vec4 r = vec4(hairHash(hairSeed), hairHash(hairSeed + 0.37), hairHash(hairSeed + 0.71), hairHash(hairSeed + 0.13));
  float lock = dot(hairRoot, vec3(4.0, 2.5, 7.0));
  // Collision is faded in with the motion, so the standing pose is exactly the authored hair.
  float solid = smoothstep(0.0, 0.15, hairAmount);
  vec3 p0 = hairBend(base, span, hairFrac, len, hairSway(hairFrac, len, along, r, lock));
  p0 = mix(p0, hairCollide(p0), solid);
  outPosition = (hairFromBody * vec4(p0, 1.0)).xyz;
#ifdef HAIR_DEPTH
  outNormal = vec3(0.0);
#else
  // Normals follow the bend: turn them by the rotation that carries the hair's rest direction to its bent one.
  float e = 0.06;
  vec3 p1 = hairBend(base + span * e, span, hairFrac + e, len, hairSway(hairFrac + e, len, along, r, lock));
  p1 = mix(p1, hairCollide(p1), solid);
  vec3 b = normalize(p1 - p0), v = cross(along, b);
  float c = dot(along, b);
  vec3 n = n0 * c + cross(v, n0) + v * (dot(v, n0) / max(1.0 + c, 0.1));
  outNormal = normalize(mat3(hairFromBody) * n);
#endif
}`;
}

/** Mane and tail share one look but each gets its own material (so its own uniforms) and a matching shadow material. */
function hairMaterials(kind: 'mane' | 'tail') {
  const uniforms = hairUniforms();
  const material = new T.MeshPhysicalMaterial({color: '#2f2b2d', vertexColors: true, roughness: 0.82, metalness: 0, sheen: 0.85, sheenRoughness: 0.72, sheenColor: '#605650', specularIntensity: 0.35});
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${hairShader(kind, false)}`)
      .replace('#include <beginnormal_vertex>', 'vec3 hairPosition, objectNormal;\nhairDeform(hairPosition, objectNormal);')
      .replace('#include <begin_vertex>', 'vec3 transformed = hairPosition;');
  };
  material.customProgramCacheKey = () => `hair-${kind}`;
  // The sun's shadow map draws the hair with this, so the shadow sways exactly as the hair does.
  const depth = new T.MeshDepthMaterial({depthPacking: T.RGBADepthPacking});
  depth.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${hairShader(kind, true)}`)
      .replace('#include <begin_vertex>', 'vec3 hairNormal, transformed;\nhairDeform(transformed, hairNormal);');
  };
  depth.customProgramCacheKey = () => `hair-${kind}-depth`;
  return {uniforms, material, depth};
}

/** A stuffed, slightly squared sphere avoids the old stack-of-balls silhouette. */
function pillowGeometry() {
  const geometry = new T.SphereGeometry(1, 24, 16);
  const p = geometry.attributes.position;
  const soften = (v: number) => Math.sign(v) * Math.pow(Math.abs(v), 0.82);
  for (let i = 0; i < p.count; i++) {
    p.setXYZ(i, soften(p.getX(i)), soften(p.getY(i)), soften(p.getZ(i)));
  }
  geometry.computeVertexNormals();
  return geometry;
}

/** Two thin cloth faces enclose a little padding; the hollow is geometry, not a patch. */
function earGeometry() {
  const positions: number[] = [], uv: number[] = [], indices: number[] = [];
  const rows = 20, columns = 16;
  for (let side = 0; side < 2; side++) for (let row = 0; row <= rows; row++) {
    const t = row / rows;
    const width = 0.18 * Math.pow(Math.sin(Math.PI * t), 0.8) * (1 - t * 0.35);
    for (let col = 0; col <= columns; col++) {
      const u = col / columns * 2 - 1;
      const bowl = Math.sin(Math.PI * t) * (0.08 * u * u - 0.065 * (1 - u * u));
      const thickness = 0.018 + 0.02 * Math.sin(Math.PI * t) * (1 - u * u);
      positions.push(bowl + (side === 0 ? thickness : -thickness), t * 0.66, u * width);
      uv.push(col / columns, t);
    }
  }
  const stride = columns + 1, layer = (rows + 1) * stride;
  for (let side = 0; side < 2; side++) for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
    const a = side * layer + row * stride + col, b = a + 1, c = a + stride, d = c + 1;
    if (side === 0) indices.push(a, c, b, b, c, d);
    else indices.push(a, b, c, b, d, c);
  }
  for (let row = 0; row < rows; row++) for (const col of [0, columns]) {
    const a = row * stride + col, b = a + stride;
    indices.push(a, a + layer, b, b, a + layer, b + layer);
  }
  for (const row of [0, rows]) for (let col = 0; col < columns; col++) {
    const a = row * stride + col;
    indices.push(a, a + 1, a + layer, a + 1, a + layer + 1, a + layer);
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new T.Float32BufferAttribute(uv, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

export function plush(h: Horse, options: PlushOptions = {}) {
  const layers = options.pileLayers ?? PILE_LAYERS;
  const settings = normalizeHead(options);
  const {headWidth} = settings;
  const g = new T.Group();
  const body = new T.Group();
  body.name = 'plush-body';
  g.add(body);
  const fabrics = options.fabric ?? SAVED_FABRIC;
  const cloth = (color: T.ColorRepresentation) => fabricMaterial(color, 'current');
  const brown = fabricMaterial(coat[h.coatCode] || '#999999', fabrics.body ?? 'current');
  const dark = cloth('#29272a');
  const hair = fabricMaterial('#29272a', fabrics.mane ?? 'current');
  const bodyFabric = fabrics.body ?? 'current';
  const piled = !!FABRIC_LOOKS[bodyFabric].pile;
  const blue = cloth('#234b98');
  const hood = h.blinker ? fabricMaterial(seededPick(h.name, MENKO_COLORS), fabrics.menko ?? 'current') : brown;
  const eye = new T.MeshStandardMaterial({color: '#151416', roughness: 0.48});
  const shine = new T.MeshStandardMaterial({color: '#e1d9c9', roughness: 0.6});
  const sphere = new T.SphereGeometry(1, 16, 12);
  const pillow = pillowGeometry();
  const ball = (parent: T.Object3D, mat: T.Material, p: Point, s: Point, stuffed = true) => {
    const m = new T.Mesh(stuffed ? pillow : sphere, mat);
    m.position.set(...p);
    m.scale.set(...s);
    m.castShadow = m.receiveShadow = true;
    parent.add(m);
    return m;
  };

  const shellMaterial = fabricMaterial('#ffffff', fabrics.body ?? 'current');
  shellMaterial.vertexColors = true;
  const shell = new T.Mesh(stuffedBodyGeometry(brown.color, settings), shellMaterial);
  shell.name = 'seamless-body-and-feet';
  shell.castShadow = shell.receiveShadow = true;
  body.add(shell);
  const pileLength = FABRIC_LOOKS[bodyFabric].pile ?? 0;
  if (piled) {
    shellMaterial.color.setScalar(PILE_ROOT_SHADE);
    if (h.blinker) {
      // The hood presses the pile flat wherever it covers the head.
      const at = shell.geometry.attributes.position.array as Float32Array;
      const scale = new Float32Array(at.length / 3);
      for (let i = 0; i < scale.length; i++) {
        scale[i] = at[i * 3 + 1] < 1.2 ? 1 : T.MathUtils.smoothstep(hoodDistance(at[i * 3], at[i * 3 + 1], at[i * 3 + 2], settings), -0.012, 0.03);
      }
      shell.geometry.setAttribute('pileScale', new T.BufferAttribute(scale, 1));
    }
    addPile(shell, pileMaterial('#ffffff', bodyFabric, layers, {vertexColors: true, seam: true}), layers);
  }
  // Eyes, nostrils and the saddlecloth sit on top of the pile rather than sinking into it.
  const raise = Math.max(0, pileLength - 0.02);
  const legs = legRig(shell);

  const nodding = [shell];
  if (h.blinker) {
    const menko = new T.Mesh(hoodGeometry(settings), hood);
    nodding.push(menko);
    menko.name = 'felt-menko';
    menko.castShadow = menko.receiveShadow = true;
    body.add(menko);
  }

  const head = new T.Group();
  head.position.set(0.67, 1.91, 0);
  head.scale.setScalar(1.18);
  body.add(head);
  head.name = h.blinker ? 'menko-head' : 'bare-head';
  const surfaceZ = (x: number, y: number) => {
    let lo = 0, hi = 1;
    for (let i = 0; i < 20; i++) {
      const z = (lo + hi) / 2;
      if (bodyDistance(0.67 + x * 1.18, 1.91 + y * 1.18, z * 1.18, settings) < 0) lo = z;
      else hi = z;
    }
    return (lo + hi) / 2;
  };
  for (const side of [-1, 1]) {
    const ear = new T.Mesh(earGeometry(), hood);
    ear.position.set(-0.06, 0.58, side * 0.3 * headWidth);
    ear.rotation.x = side * 0.2;
    ear.name = 'cupped-cloth-ear';
    ear.castShadow = ear.receiveShadow = true;
    head.add(ear);
    if (piled && !h.blinker) addPile(ear, pileMaterial(brown.color, bodyFabric, layers, {length: 0.016}), layers);
    const eyeX = T.MathUtils.lerp(0.47, 0.36, settings.eyeScale);
    const eyeY = T.MathUtils.lerp(0.26, 0.25, settings.eyeScale);
    const eyeZ = surfaceZ(eyeX, eyeY);
    const eyeSize = T.MathUtils.lerp(1.24, 1, settings.eyeScale);
    ball(head, eye, [eyeX, eyeY, side * (eyeZ + 0.012 + raise / 1.18)], [0.085 * eyeSize, 0.105 * eyeSize, 0.035], false);
    ball(head, shine, [eyeX + 0.02, eyeY + 0.03, side * (eyeZ + 0.044 + raise / 1.18)], [0.019, 0.022, 0.006], false);
    const noseZ = surfaceZ(1.1, -0.13);
    ball(head, dark, [1.1, -0.13, side * (noseZ + 0.006 + raise / 1.18)], [0.027, 0.039, 0.02], false);
  }
  // Tube UVs run 0..1 along each strap, so the grain is tiled many times along its length.
  if (options.bridle) addBridle(head, fabricMaterial('#5b3826', fabrics.bridle ?? 'current', [12, 0.3]), settings, !!h.blinker, pileLength);

  // Mane and tail are real hairs that sway in the vertex shader. The mane is grown in body coordinates and hangs from
  // the body: it rides the neck, and the shader turns only the hair at the poll with the head, which now nods far more.
  const maneHair = hairMaterials('mane'), tailHair = hairMaterials('tail');
  const mane = new T.Mesh(maneGeometry(settings, !!h.blinker), maneHair.material);
  mane.name = 'hair-mane';
  mane.castShadow = mane.receiveShadow = true;
  mane.customDepthMaterial = maneHair.depth;
  // The sway reaches outside the resting bounds, so the box must not cull it.
  mane.frustumCulled = false;
  body.add(mane);
  const tail = new T.Group();
  tail.position.set(...TAIL_ROOT);
  body.add(tail);
  // A small stuffed dock closes the root, where the hairs are sewn in.
  ball(tail, hair, [-0.12, 0, 0], [0.12, 0.1, 0.1]).name = 'tail-dock';
  const tailMesh = new T.Mesh(tailGeometry(settings), tailHair.material);
  tailMesh.name = 'hair-tail';
  tailMesh.castShadow = tailMesh.receiveShadow = true;
  tailMesh.customDepthMaterial = tailHair.depth;
  tailMesh.frustumCulled = false;
  tail.add(tailMesh);

  // Saddlecloth drapes over the back. Numbers are printed on both physical sides.
  const clothPoint = (x: number, t: number, lift = 0): Point => {
    // Follow the actual blended shell, including where the feet meet the belly.
    let low = 0;
    let high = 1.6;
    for (let i = 0; i < 16; i++) {
      const r = (low + high) / 2;
      if (bodyDistance(x, 1.22 + r * Math.cos(t), r * Math.sin(t), settings, false) < 0) low = r;
      else high = r;
    }
    const r = (low + high) / 2 + Math.max(0.025, pileLength + 0.012) + lift;
    return [x, 1.22 + r * Math.cos(t), r * Math.sin(t)];
  };
  const clothGeometry = new T.PlaneGeometry(1.08, 2.8, 10, 24);
  const pos = clothGeometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const t = pos.getY(i) / 1.4 * 1.72;
    pos.setXYZ(i, ...clothPoint(pos.getX(i) - 0.65, t));
  }
  clothGeometry.computeVertexNormals();
  blue.side = T.DoubleSide;
  const saddle = new T.Mesh(clothGeometry, blue);
  saddle.castShadow = saddle.receiveShadow = true;
  body.add(saddle);
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 192;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, 256, 192);
  ctx.fillStyle = '#fff8e9';
  ctx.textAlign = 'center';
  ctx.font = 'bold 146px sans-serif';
  ctx.fillText(String(h.number), 128, 132);
  ctx.font = 'bold 22px sans-serif';
  ctx.fillText(h.name, 128, 176, 230);
  const printMap = new T.CanvasTexture(canvas);
  printMap.colorSpace = T.SRGBColorSpace;
  const printMaterial = new T.MeshStandardMaterial({map: printMap, roughness: 1, transparent: true, depthWrite: false});
  for (const side of [-1, 1]) {
    const geometry = new T.PlaneGeometry(1, 1, 12, 12);
    const vertices = geometry.attributes.position;
    for (let i = 0; i < vertices.count; i++) {
      const x = -0.65 + vertices.getX(i) * 0.9 * side;
      const t = side * (1.25 - vertices.getY(i) * 0.85);
      vertices.setXYZ(i, ...clothPoint(x, t, 0.008));
    }
    geometry.computeVertexNormals();
    const print = new T.Mesh(geometry, printMaterial);
    body.add(print);
  }
  if (options.labels !== false) {
    const plate = label(`${h.number}  ${h.name}`);
    plate.position.set(0, 8, 0);
    plate.scale.set(18, 4.5, 1);
    plate.name = 'nameplate';
    g.add(plate);
    const num = label(String(h.number), 128);
    num.name = 'numberplate';
    num.position.set(0, 4.4, 0);
    num.scale.set(2.2, 2.2, 1);
    g.add(num);
  }
  g.userData.plushRig = {body, head, tail, legs, neck: headRig(nodding), hair: {mane: {uniforms: maneHair.uniforms, mesh: mane}, tail: {uniforms: tailHair.uniforms, mesh: tailMesh}}} satisfies PlushRig;
  return g;
}

/** Weights each low shell vertex to the nearest leg, so a leg can swing without splitting the seamless body. */
/**
 * The head is part of the one seamless shell, so nodding has to bend the cloth itself: every vertex past the neck
 * turns about HEAD_PIVOT, fading out across the neck. Eyes, ears and bridle ride the head group, which poseHead
 * carries through the same turn, so they stay on the surface instead of sliding over it and sinking in.
 */
const HEAD_ORIGIN = [0.67, 1.91]; // the head group's place at rest
const HEAD_PIVOT = [0.55, 1.6]; // inside the neck
const HEAD_AXIS = [Math.SQRT1_2, Math.SQRT1_2]; // from the pivot toward the head
const HEAD_BLEND = [0, 0.25]; // distance along the axis over which the turn fades in
const headWeight = (x: number, y: number) =>
  T.MathUtils.smoothstep((x - HEAD_PIVOT[0]) * HEAD_AXIS[0] + (y - HEAD_PIVOT[1]) * HEAD_AXIS[1], HEAD_BLEND[0], HEAD_BLEND[1])
  // The legs are posed separately: keep the two sets of vertices apart.
  * T.MathUtils.smoothstep(y, LEG_PIVOT_Y, LEG_PIVOT_Y + 0.2);
function headRig(meshes: T.Mesh[]): HeadRig {
  return {last: 0, parts: meshes.map(mesh => {
    const position = mesh.geometry.attributes.position as T.BufferAttribute, normal = mesh.geometry.attributes.normal as T.BufferAttribute;
    const index: number[] = [], weight: number[] = [];
    for (let i = 0; i < position.count; i++) {
      const w = headWeight(position.getX(i), position.getY(i));
      if (w > 0.001) {index.push(i); weight.push(w);}
    }
    return {mesh, base: (position.array as Float32Array).slice(), baseNormals: (normal.array as Float32Array).slice(), index: Uint32Array.from(index), weight: Float32Array.from(weight)};
  })};
}
function poseHead(rig: HeadRig, head: T.Group, angle: number) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const ox = HEAD_ORIGIN[0] - HEAD_PIVOT[0], oy = HEAD_ORIGIN[1] - HEAD_PIVOT[1];
  head.position.set(HEAD_PIVOT[0] + ox * c - oy * s, HEAD_PIVOT[1] + ox * s + oy * c, 0);
  head.rotation.z = angle;
  if (Math.abs(angle - rig.last) < 1e-4) return;
  rig.last = angle;
  for (const part of rig.parts) {
    const position = part.mesh.geometry.attributes.position as T.BufferAttribute, normal = part.mesh.geometry.attributes.normal as T.BufferAttribute;
    const p = position.array as Float32Array, n = normal.array as Float32Array;
    for (let k = 0; k < part.index.length; k++) {
      const i = part.index[k] * 3, a = angle * part.weight[k];
      const ca = Math.cos(a), sa = Math.sin(a);
      const dx = part.base[i] - HEAD_PIVOT[0], dy = part.base[i + 1] - HEAD_PIVOT[1];
      p[i] = HEAD_PIVOT[0] + dx * ca - dy * sa;
      p[i + 1] = HEAD_PIVOT[1] + dx * sa + dy * ca;
      const nx = part.baseNormals[i], ny = part.baseNormals[i + 1];
      n[i] = nx * ca - ny * sa;
      n[i + 1] = nx * sa + ny * ca;
    }
    position.needsUpdate = normal.needsUpdate = true;
  }
}

function legRig(mesh: T.Mesh): LegRig {
  const position = mesh.geometry.attributes.position as T.BufferAttribute;
  const index: number[] = [], leg: number[] = [], weight: number[] = [];
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i), y = position.getY(i), z = position.getZ(i);
    const vertical = 1 - T.MathUtils.smoothstep(y, 0.3, LEG_PIVOT_Y);
    if (vertical <= 0) continue;
    let best = 0, bestLeg = 0;
    LEGS.forEach((l, n) => {
      const w = 1 - T.MathUtils.smoothstep(Math.hypot(x - l.x, z - l.z), 0.3, 0.42);
      if (w > best) {best = w; bestLeg = n;}
    });
    if (best * vertical < 0.001) continue;
    index.push(i); leg.push(bestLeg); weight.push(best * vertical);
  }
  const normal = mesh.geometry.attributes.normal as T.BufferAttribute;
  return {
    mesh, base: (position.array as Float32Array).slice(), baseNormals: (normal.array as Float32Array).slice(),
    index: Uint32Array.from(index), leg: Uint8Array.from(leg), weight: Float32Array.from(weight), last: '',
  };
}

/** Swing each leg about its hip (rotation around z), bending softly where it meets the belly. */
function poseLegs(rig: LegRig, angles: number[]) {
  const key = angles.map(a => a.toFixed(4)).join();
  if (key === rig.last) return;
  rig.last = key;
  const position = rig.mesh.geometry.attributes.position as T.BufferAttribute;
  const normal = rig.mesh.geometry.attributes.normal as T.BufferAttribute;
  const p = position.array as Float32Array, n = normal.array as Float32Array;
  for (let k = 0; k < rig.index.length; k++) {
    const i = rig.index[k] * 3;
    const a = angles[rig.leg[k]] * rig.weight[k];
    const c = Math.cos(a), s = Math.sin(a);
    const l = LEGS[rig.leg[k]];
    const dx = rig.base[i] - l.x, dy = rig.base[i + 1] - LEG_PIVOT_Y;
    p[i] = l.x + dx * c - dy * s;
    p[i + 1] = LEG_PIVOT_Y + dx * s + dy * c;
    const nx = rig.baseNormals[i], ny = rig.baseNormals[i + 1];
    n[i] = nx * c - ny * s;
    n[i + 1] = nx * s + ny * c;
  }
  position.needsUpdate = normal.needsUpdate = true;
}

const STRAP_WIDTH = 0.062;
const STRAP_THICKNESS = 0.009;
const STRAP_LIFT = 0.022; // Clearance for the leather backing over flat cloth; over pile the strap rides on the fibre tips.

/**
 * Leather head straps built only from plane sections ("rings") and straight runs, so they read as clean
 * loops and lines. Every open end finishes exactly on another strap's centre line under a flat leather joint,
 * so there are no cut ends or gaps. All coordinates are in the head group's frame.
 */
function addBridle(head: T.Group, leather: T.Material, settings: HeadSettings, menko: boolean, pileLength = 0) {
  const distance = (v: T.Vector3) => {
    const px = 0.67 + v.x * 1.18, py = 1.91 + v.y * 1.18, pz = v.z * 1.18;
    const shell = bodyDistance(px, py, pz, settings);
    return (menko ? Math.min(shell, hoodDistance(px, py, pz, settings)) : shell) / 1.18;
  };
  // The straps sit on top of the coat: with their underside at the fibre tips, the pile cannot swallow them.
  // The hood presses the pile flat, so over it the plain clearance is enough.
  const pileLift = Math.max(STRAP_LIFT, pileLength / 1.18 + STRAP_THICKNESS / 2 + 0.003);
  const liftAt = (v: T.Vector3) => {
    if (!menko) return pileLift;
    const px = 0.67 + v.x * 1.18, py = 1.91 + v.y * 1.18, pz = v.z * 1.18;
    const over = hoodDistance(px, py, pz, settings) - bodyDistance(px, py, pz, settings);
    return T.MathUtils.lerp(STRAP_LIFT, pileLift, T.MathUtils.smoothstep(over, 0, 0.04));
  };
  const normalAt = (v: T.Vector3) => {
    const e = 0.002, n = new T.Vector3(
      distance(new T.Vector3(v.x + e, v.y, v.z)) - distance(new T.Vector3(v.x - e, v.y, v.z)),
      distance(new T.Vector3(v.x, v.y + e, v.z)) - distance(new T.Vector3(v.x, v.y - e, v.z)),
      distance(new T.Vector3(v.x, v.y, v.z + e)) - distance(new T.Vector3(v.x, v.y, v.z - e)),
    );
    return n.lengthSq() > 0 ? n.normalize() : new T.Vector3(0, 1, 0);
  };
  // Surface point on the line origin + s * dir (origin inside or near the shell).
  const surfaceAlong = (origin: T.Vector3, dir: T.Vector3, from: number, to: number) => {
    let lo = from, hi = to;
    for (let i = 0; i < 24; i++) {
      const s = (lo + hi) / 2;
      if (distance(origin.clone().addScaledVector(dir, s)) < 0) lo = s; else hi = s;
    }
    return origin.clone().addScaledVector(dir, (lo + hi) / 2);
  };
  const lifted = (surface: T.Vector3) => surface.clone().addScaledVector(normalAt(surface), liftAt(surface));
  // Plane section around a centre inside the head: the plane holds `up` and the z axis.
  const ring = (center: T.Vector3, up: T.Vector3, from: number, to: number, steps: number) => {
    const side = new T.Vector3(0, 0, 1);
    const points: T.Vector3[] = [];
    for (let i = 0; i <= steps; i++) {
      const phi = T.MathUtils.lerp(from, to, i / steps);
      const dir = up.clone().multiplyScalar(Math.cos(phi)).addScaledVector(side, Math.sin(phi));
      points.push(lifted(surfaceAlong(center, dir, 0, 1)));
    }
    return points;
  };
  // Straight run: points on the chord a-b pushed onto the shell along one shared normal,
  // so the strap lies in a single plane and reads as a straight line.
  const straight = (a: T.Vector3, b: T.Vector3, steps: number) => {
    const n = normalAt(a).add(normalAt(b)).normalize();
    const points = [a.clone()];
    for (let i = 1; i < steps; i++) {
      const chord = a.clone().lerp(b, i / steps);
      points.push(lifted(surfaceAlong(chord, n, -0.5, 0.3)));
    }
    points.push(b.clone());
    return points;
  };
  // Remove small wobble, then make sure no point sinks below the strap's clearance.
  const fair = (points: T.Vector3[], closed: boolean, pinned: number[] = []) => {
    for (let pass = 0; pass < 6; pass++) {
      const copy = points.map(p => p.clone());
      for (let i = 0; i < points.length; i++) {
        if (!closed && (i === 0 || i === points.length - 1)) continue;
        if (pinned.includes(i)) continue;
        const prev = copy[(i - 1 + points.length) % points.length], next = copy[(i + 1) % points.length];
        points[i].lerp(prev.clone().add(next).multiplyScalar(0.5), 0.5);
      }
      for (const p of points) {
        const gap = distance(p) - liftAt(p);
        if (gap < 0) p.addScaledVector(normalAt(p), -gap);
      }
    }
    return points;
  };
  const strap = (name: string, points: T.Vector3[], closed: boolean) => {
    const curve = new T.CatmullRomCurve3(points, closed, 'centripetal');
    const steps = points.length * 3;
    const w = STRAP_WIDTH / 2, h = STRAP_THICKNESS / 2, bevel = 0.002;
    // A broad flat face, a thin edge, and tiny bevels that catch the light.
    // Frame the section against the head, so the leather lies flat around the nose and poll.
    const section = [
      [-w + bevel, h], [w - bevel, h], [w, h - bevel], [w, -h + bevel],
      [w - bevel, -h], [-w + bevel, -h], [-w, -h + bevel], [-w, h - bevel],
    ];
    const positions: number[] = [], uvs: number[] = [], indices: number[] = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps, center = curve.getPointAt(t), tangent = curve.getTangentAt(t).normalize();
      const outward = normalAt(center);
      outward.addScaledVector(tangent, -outward.dot(tangent)).normalize();
      const across = new T.Vector3().crossVectors(tangent, outward).normalize();
      // Duplicate section corners per face to preserve the flat leather faces.
      for (let edge = 0; edge < section.length; edge++) {
        for (const corner of [edge, (edge + 1) % section.length]) {
          const [x, y] = section[corner];
          const vertex = center.clone().addScaledVector(across, x).addScaledVector(outward, y);
          positions.push(vertex.x, vertex.y, vertex.z);
          uvs.push(t, (x / STRAP_WIDTH + 0.5));
        }
        if (i < steps) {
          const a = i * 16 + edge * 2, b = a + 1, c = a + 16, d = c + 1;
          indices.push(a, b, c, b, d, c);
        }
      }
    }
    // Close the cut ends; the closed noseband already shares its first and last section.
    if (!closed) for (const end of [0, steps]) {
      const start = positions.length / 3;
      for (let corner = 0; corner < section.length; corner++) {
        const source = (end * 16 + corner * 2) * 3;
        positions.push(...positions.slice(source, source + 3));
        uvs.push(section[corner][0] / STRAP_WIDTH + 0.5, section[corner][1] / STRAP_THICKNESS + 0.5);
      }
      for (let j = 1; j < section.length - 1; j++) {
        if (end === 0) indices.push(start, start + j + 1, start + j);
        else indices.push(start, start + j, start + j + 1);
      }
    }
    const geometry = new T.BufferGeometry();
    geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new T.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const mesh = new T.Mesh(geometry, leather);
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.name = name;
    head.add(mesh);
  };
  // Flatten the joining patches too, keeping the old spherical knots out of the silhouette.
  const joint = (at: T.Vector3) => {
    const patch = new T.Mesh(new T.SphereGeometry(1, 16, 8), leather);
    const normal = normalAt(at);
    patch.scale.set(STRAP_WIDTH * 0.54, STRAP_WIDTH * 0.54, STRAP_THICKNESS * 0.65);
    patch.quaternion.setFromUnitVectors(new T.Vector3(0, 0, 1), normal);
    patch.position.copy(at).addScaledVector(normal, STRAP_THICKNESS * 0.2);
    patch.castShadow = patch.receiveShadow = true;
    patch.name = 'bridle-joint';
    head.add(patch);
  };

  // Noseband: one ring around the muzzle, tilted to follow the user's trace of the head/nose seam.
  const noseUp = new T.Vector3(0.34, 1, 0).normalize();
  const nose = fair(ring(new T.Vector3(0.778, -0.15, 0), noseUp, 0, Math.PI * 2, 96).slice(0, -1), true);
  strap('bridle-noseband', nose, true);
  const nearest = (points: T.Vector3[], target: T.Vector3) => points.reduce((best, p) => p.distanceTo(target) < best.distanceTo(target) ? p : best);
  const noseRight = nearest(nose, new T.Vector3(0.74, -0.2, 0.6)).clone();
  const noseLeft = nearest(nose, new T.Vector3(0.74, -0.2, -0.6)).clone();

  // Headstall: straight cheekpiece -> ring over the poll behind the ears -> straight cheekpiece.
  const crown = ring(new T.Vector3(-0.2, 0.05, 0), new T.Vector3(-0.25, 1, 0).normalize(), -1.75, 1.75, 48);
  const pollLeft = crown[0], pollRight = crown[crown.length - 1];
  const cheekLeft = straight(noseLeft, pollLeft, 24);
  const cheekRight = straight(pollRight, noseRight, 24);
  const headstall = [...cheekLeft, ...crown.slice(1, -1), ...cheekRight];
  const corners = [cheekLeft.length - 1, cheekLeft.length + crown.length - 2];
  fair(headstall, false, corners);
  strap('bridle-headstall', headstall, false);

  // Browband: a ring from one poll corner, up in front of the ears, over the top, to the other corner.
  const corner = headstall[corners[1]];
  const top = lifted(surfaceAlong(new T.Vector3(0.08, 0.3, 0), new T.Vector3(0, 1, 0), 0, 1));
  const center = new T.Vector3((corner.x + top.x) / 2, (corner.y + top.y) / 2, 0);
  const up = top.clone().sub(center).normalize();
  const offset = corner.clone().sub(center);
  const reach = Math.atan2(offset.z, offset.dot(up));
  const brow = ring(center, up, -reach, reach, 56);
  brow[0] = headstall[corners[0]].clone();
  brow[brow.length - 1] = corner.clone();
  strap('bridle-browband', fair(brow, false), false);

  for (const p of [noseLeft, noseRight, headstall[corners[0]], headstall[corners[1]]]) joint(p);
}

/**
 * Distance-based posing: pausing and seeking the replay always reproduce the same pose. `kick` is the closing-kick
 * state at this distance (see closingKick.ts): its level blends in the sprinting form, and its stride count keeps
 * the faster cadence continuous in distance.
 */
export function posePlush(g: T.Group, distance: number, reducedMotion = false, kick: Kick = NO_KICK) {
  const rig = g.userData.plushRig as PlushRig | undefined;
  if (!rig) return;
  const amount = gallopAmount(distance, reducedMotion);
  const k = reducedMotion ? 0 : T.MathUtils.clamp(kick.level, 0, 1);
  // Every motion below is a smooth periodic function of the stride phase, so strides join with no seam.
  const phase = stridePhase(distance, reducedMotion, kick);
  const wave = (offset: number) => Math.cos(2 * Math.PI * (phase - offset));
  const reach = T.MathUtils.lerp(STRIDE_REACH, KICK_REACH, k);
  const angles = LEGS.map(l => legAngle(l, phase, reach) * amount);
  if (rig.legs) poseLegs(rig.legs, angles);
  // Rocking: nose up while the hinds carry the body, nose down while the fores do.
  const pitch = 0.06 * (1 + (KICK_ROCK - 1) * k) * wave(0.2);
  // Keep the lowest foot on the ground (a soft minimum, so the body never jerks between feet),
  // then add the hop of the suspension phase after the leading fore pushes off. The kick's forward lean is part of
  // the body angle here, so the hooves neither float nor sink when it changes.
  const angle = (pitch - KICK_LEAN * k) * amount;
  const sinP = Math.sin(angle), cosP = Math.cos(angle);
  const softness = 30;
  let sum = 0;
  LEGS.forEach((l, n) => {
    const footX = l.x + LEG_PIVOT_Y * Math.sin(angles[n]);
    const footY = LEG_PIVOT_Y * (1 - Math.cos(angles[n]));
    sum += Math.exp(-softness * (footX * sinP + footY * cosP));
  });
  const lowest = -Math.log(sum / LEGS.length) / softness;
  const hop = T.MathUtils.lerp(STRIDE_HOP, KICK_HOP, k) * Math.max(0, wave(0.88)) ** 2;
  rig.body.position.y = -lowest + hop * amount;
  rig.body.rotation.z = angle;
  // The head counter-rotates the rocking; in the kick it nods harder, bottoming out as the forelegs land, on a lower base.
  const nod = -NOD * (1 + (KICK_NOD - 1) * k) * Math.cos(2 * Math.PI * (phase - T.MathUtils.lerp(0.2, KICK_NOD_BEAT, k)));
  const headAngle = (nod - KICK_HEAD_DOWN * k) * amount;
  if (rig.neck) poseHead(rig.neck, rig.head, headAngle); else rig.head.rotation.z = headAngle;
  rig.tail.rotation.z = -amount * (0.3 + 0.08 * wave(0.3));
  rig.tail.rotation.x = amount * 0.05 * Math.sin(2 * Math.PI * phase);
  if (rig.hair) poseHair(rig, phase, amount, k);
}

/** Per-frame inputs of the hair shaders: a handful of numbers, and the tail's matrix to and from body coordinates. */
function poseHair(rig: PlushRig, phase: number, amount: number, kick: number) {
  const {mane, tail} = rig.hair!;
  // The air streams the hair back in proportion to how hard the horse is running.
  const wind = amount * (1 + (KICK_STREAM - 1) * kick);
  for (const u of [mane.uniforms, tail.uniforms]) {
    u.hairPhase.value = phase;
    u.hairAmount.value = amount;
    u.hairKick.value = kick * amount;
    u.hairWind.value = wind;
    u.hairHead.value = rig.head.rotation.z;
  }
  // The mane hangs straight from the body, so its matrices stay identity; the tail turns about its dock.
  rig.tail.updateMatrix();
  tail.uniforms.hairToBody.value.copy(rig.tail.matrix).multiply(tail.mesh.matrix);
  tail.uniforms.hairFromBody.value.copy(tail.uniforms.hairToBody.value).invert();
}
