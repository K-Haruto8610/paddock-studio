import * as T from 'three';
import {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
import type {Horse} from './types';
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
type PlushRig = {body: T.Group; head: T.Group; tail: T.Group; legs?: LegRig};
// Gallop footfall order (right lead): left hind -> right hind -> left fore -> right fore, then a moment of
// suspension. `beat` is when each foot lands, as a fraction of the stride. The horse faces +x; its left is -z.
const LEGS = [
  {x: -0.82, z: -0.43, beat: 0, reach: 0.5, center: -0.06},
  {x: -0.82, z: 0.43, beat: 0.1, reach: 0.5, center: -0.06},
  {x: 0.64, z: -0.43, beat: 0.3, reach: 0.55, center: 0.06},
  {x: 0.64, z: 0.43, beat: 0.4, reach: 0.55, center: 0.06},
] as const;
const LEG_PIVOT_Y = 0.95;
// In race metres: at normal playback speed a 2400 m race shows about two strides a second.
const STRIDE_LENGTH = 16;

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
type PlushOptions = Partial<HeadSettings> & {bridle?: boolean; fabric?: FabricChoice};
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
};
const FABRIC_LOOKS: Record<FabricKind, FabricLook> = {
  current: {repeat: 5, bump: 0.018, roughness: 1, sheen: 0.65, sheenRoughness: 1, sheenColor: '#d9d1c6'},
  // Dense short pile: almost no visible grain, a soft bright glow at grazing angles.
  minky: {repeat: 4, bump: 0.004, roughness: 0.85, sheen: 1, sheenRoughness: 0.35, sheenColor: '#ffffff'},
  // Curly long pile: deep bumpy curls.
  boa: {repeat: 2.4, bump: 0.05, roughness: 1, sheen: 0.8, sheenRoughness: 0.55, sheenColor: '#f2ece2'},
  // Matted fibres, flat and dry.
  felt: {repeat: 2.4, bump: 0.012, roughness: 1, sheen: 0.15, sheenRoughness: 1, sheenColor: '#d9d1c6'},
  // Cut pile laid in one direction: smooth, with a strong shifting sheen.
  velour: {repeat: 3, bump: 0.003, roughness: 0.7, sheen: 1, sheenRoughness: 0.22, sheenColor: '#fff6ea'},
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

let tailTemplate: {positions: Float32Array; normals: Float32Array} | undefined;
function tailGeometry() {
  if (!tailTemplate) {
    const paths: {points: Point[]; start: number; end: number}[] = [{points: [[0, 0, 0], [-0.2, -0.12, 0], [-0.28, -0.4, 0], [-0.32, -0.6, 0]], start: 0.16, end: 0.09}];
    for (let i = -1; i <= 1; i++) paths.push({points: [[-0.28, -0.44, i * 0.025], [-0.34, -0.68, i * 0.075], [-0.38 + Math.abs(i) * 0.06, -0.92 + Math.abs(i) * 0.08, i * 0.14]], start: 0.1, end: 0.006});
    const segments: {a: T.Vector3; delta: T.Vector3; length2: number; r0: number; r1: number}[] = [];
    for (const path of paths) {
      const curve = new T.CatmullRomCurve3(path.points.map(p => new T.Vector3(...p)));
      for (let i = 0; i < 14; i++) {
        const a = curve.getPoint(i / 14), b = curve.getPoint((i + 1) / 14);
        const delta = b.sub(a);
        segments.push({a, delta, length2: delta.lengthSq(), r0: T.MathUtils.lerp(path.start, path.end, i / 14), r1: T.MathUtils.lerp(path.start, path.end, (i + 1) / 14)});
      }
    }
    const resolution = 44;
    const material = new T.MeshBasicMaterial();
    const field = new MarchingCubes(resolution, material, false, false, 10000);
    field.isolation = 0;
    for (let z = 0; z < resolution; z++) for (let y = 0; y < resolution; y++) for (let x = 0; x < resolution; x++) {
      const px = (x / resolution - 0.5) * 1.5 - 0.2;
      const py = (y / resolution - 0.5) * 1.5 - 0.45;
      const pz = (z / resolution - 0.5) * 1.5;
      let d = Infinity;
      // Min within each strand, then soften branch unions without exposed tube caps.
      for (let path = 0; path < paths.length; path++) {
        let strand = Infinity;
        for (let i = path * 14; i < (path + 1) * 14; i++) {
          const seg = segments[i];
          const dx = px - seg.a.x, dy = py - seg.a.y, dz = pz - seg.a.z;
          const t = T.MathUtils.clamp((dx * seg.delta.x + dy * seg.delta.y + dz * seg.delta.z) / seg.length2, 0, 1);
          strand = Math.min(strand, Math.hypot(dx - seg.delta.x * t, dy - seg.delta.y * t, dz - seg.delta.z * t) - T.MathUtils.lerp(seg.r0, seg.r1, t));
        }
        d = smoothUnion(d, strand, 0.035);
      }
      field.field[x + resolution * (y + resolution * z)] = -d;
    }
    field.update();
    const count = field.geometry.drawRange.count;
    const positions = new Float32Array(field.geometry.attributes.position.array.slice(0, count * 3));
    for (let i = 0; i < positions.length; i += 3) {
      positions[i] = positions[i] * 0.75 - 0.2;
      positions[i + 1] = positions[i + 1] * 0.75 - 0.45;
      positions[i + 2] *= 0.75;
    }
    tailTemplate = {positions, normals: new Float32Array(field.geometry.attributes.normal.array.slice(0, count * 3))};
    field.geometry.dispose();
    material.dispose();
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.BufferAttribute(tailTemplate.positions.slice(), 3));
  geometry.setAttribute('normal', new T.BufferAttribute(tailTemplate.normals.slice(), 3));
  const uv = new Float32Array(tailTemplate.positions.length / 3 * 2);
  for (let i = 0; i < uv.length / 2; i++) {uv[i * 2] = tailTemplate.positions[i * 3]; uv[i * 2 + 1] = tailTemplate.positions[i * 3 + 1];}
  geometry.setAttribute('uv', new T.BufferAttribute(uv, 2));
  return geometry;
}

export function plush(h: Horse, options: PlushOptions = {}) {
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
  const blue = cloth('#234b98');
  const hood = h.blinker ? fabricMaterial('#327ca6', fabrics.menko ?? 'current') : brown;
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
  const legs = legRig(shell);

  if (h.blinker) {
    const menko = new T.Mesh(hoodGeometry(settings), hood);
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
    const eyeX = T.MathUtils.lerp(0.47, 0.36, settings.eyeScale);
    const eyeY = T.MathUtils.lerp(0.26, 0.25, settings.eyeScale);
    const eyeZ = surfaceZ(eyeX, eyeY);
    const eyeSize = T.MathUtils.lerp(1.24, 1, settings.eyeScale);
    ball(head, eye, [eyeX, eyeY, side * (eyeZ + 0.012)], [0.085 * eyeSize, 0.105 * eyeSize, 0.035], false);
    ball(head, shine, [eyeX + 0.02, eyeY + 0.03, side * (eyeZ + 0.044)], [0.019, 0.022, 0.006], false);
    const noseZ = surfaceZ(1.1, -0.13);
    ball(head, dark, [1.1, -0.13, side * (noseZ + 0.006)], [0.027, 0.039, 0.018], false);
  }
  // Tube UVs run 0..1 along each strap, so the grain is tiled many times along its length.
  if (options.bridle) addBridle(head, fabricMaterial('#5b3826', fabrics.bridle ?? 'current', [12, 0.3]), settings, !!h.blinker);

  // One thin piece of zigzag-cut felt, running along the middle of the head.
  const maneOutline = new T.Shape();
  maneOutline.moveTo(-0.3, -0.1);
  maneOutline.lineTo(-0.32, 0.4);
  maneOutline.lineTo(-0.22, 0.62);
  maneOutline.lineTo(-0.12, 0.84);
  maneOutline.lineTo(0.57, 0.72);
  maneOutline.lineTo(0.76, 0.6);
  maneOutline.lineTo(0.68, 0.94);
  maneOutline.lineTo(0.49, 0.83);
  maneOutline.lineTo(0.39, 1.05);
  maneOutline.lineTo(0.2, 0.89);
  maneOutline.lineTo(0.08, 1.1);
  maneOutline.lineTo(-0.1, 0.91);
  maneOutline.lineTo(-0.28, 1.04);
  maneOutline.lineTo(-0.36, 0.81);
  maneOutline.lineTo(-0.57, 0.86);
  maneOutline.lineTo(-0.54, 0.61);
  maneOutline.lineTo(-0.73, 0.6);
  maneOutline.lineTo(-0.59, 0.36);
  maneOutline.lineTo(-0.77, 0.3);
  maneOutline.lineTo(-0.6, 0.1);
  maneOutline.closePath();
  const mane = new T.Mesh(new T.ExtrudeGeometry(maneOutline, {depth: 0.075, bevelEnabled: true, bevelSize: 0.012, bevelThickness: 0.012, bevelSegments: 1, steps: 1}), hair);
  mane.position.z = -0.0375;
  mane.scale.y = T.MathUtils.lerp(1, 0.78, settings.maneHeight);
  mane.name = 'felt-zigzag-mane';
  mane.castShadow = true;
  head.add(mane);
  const tail = new T.Group();
  tail.position.set(-1.22, 1.38, 0);
  body.add(tail);
  const tailMesh = new T.Mesh(tailGeometry(), hair);
  tailMesh.name = 'seamless-forked-tail';
  tailMesh.castShadow = tailMesh.receiveShadow = true;
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
    const r = (low + high) / 2 + 0.025 + lift;
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
  ctx.font = 'bold 114px sans-serif';
  ctx.fillText(String(h.number), 128, 119);
  ctx.font = 'bold 23px sans-serif';
  ctx.fillText(h.name, 128, 164, 230);
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
  g.userData.plushRig = {body, head, tail, legs} satisfies PlushRig;
  return g;
}

/** Weights each low shell vertex to the nearest leg, so a leg can swing without splitting the seamless body. */
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

const STRAP_RADIUS = 0.026;
const STRAP_LIFT = 0.036; // centre-line height above the shell: leaves ~0.01 of air under the strap

/**
 * Leather head straps built only from plane sections ("rings") and straight runs, so they read as clean
 * loops and lines. Every open end finishes exactly on another strap's centre line under a round joint,
 * so there are no cut ends or gaps. All coordinates are in the head group's frame.
 */
function addBridle(head: T.Group, leather: T.Material, settings: HeadSettings, menko: boolean) {
  const distance = (v: T.Vector3) => {
    const px = 0.67 + v.x * 1.18, py = 1.91 + v.y * 1.18, pz = v.z * 1.18;
    const shell = bodyDistance(px, py, pz, settings);
    return (menko ? Math.min(shell, hoodDistance(px, py, pz, settings)) : shell) / 1.18;
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
  const lifted = (surface: T.Vector3) => surface.clone().addScaledVector(normalAt(surface), STRAP_LIFT);
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
        const gap = distance(p) - STRAP_LIFT;
        if (gap < 0) p.addScaledVector(normalAt(p), -gap);
      }
    }
    return points;
  };
  const strap = (name: string, points: T.Vector3[], closed: boolean) => {
    const curve = new T.CatmullRomCurve3(points, closed, 'centripetal');
    const mesh = new T.Mesh(new T.TubeGeometry(curve, points.length * 3, STRAP_RADIUS, 12, closed), leather);
    mesh.castShadow = true;
    mesh.name = name;
    head.add(mesh);
  };
  // A ball the size of the strap section fills each joint and hides tube ends inside the other strap.
  const joint = (at: T.Vector3) => {
    const ball = new T.Mesh(new T.SphereGeometry(STRAP_RADIUS * 1.08, 16, 12), leather);
    ball.position.copy(at);
    ball.castShadow = true;
    ball.name = 'bridle-joint';
    head.add(ball);
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

/** Distance-based posing: pausing and seeking the replay always reproduce the same pose. */
export function posePlush(g: T.Group, distance: number, reducedMotion = false) {
  const rig = g.userData.plushRig as PlushRig | undefined;
  if (!rig) return;
  const amount = reducedMotion ? 0 : T.MathUtils.smoothstep(Math.max(distance, 0), 0, STRIDE_LENGTH);
  // Every motion below is a smooth periodic function of the stride phase, so strides join with no seam.
  const phase = Math.max(distance, 0) / STRIDE_LENGTH;
  const wave = (offset: number) => Math.cos(2 * Math.PI * (phase - offset));
  const angles = LEGS.map(l => {
    // Foot lands at its beat reaching forward, sweeps back under the body (stance), then swings forward again.
    // The warp makes the stance quicker than the swing, as in a real gallop.
    const t = phase - l.beat;
    const warped = t + 0.5 / (2 * Math.PI) * Math.sin(2 * Math.PI * (t - 0.25));
    return (l.center + l.reach * Math.cos(2 * Math.PI * warped)) * amount;
  });
  if (rig.legs) poseLegs(rig.legs, angles);
  // Rocking: nose up while the hinds carry the body, nose down while the fores do.
  const pitch = 0.06 * wave(0.2);
  // Keep the lowest foot on the ground (a soft minimum, so the body never jerks between feet),
  // then add the hop of the suspension phase after the leading fore pushes off.
  const sinP = Math.sin(pitch * amount), cosP = Math.cos(pitch * amount);
  const softness = 30;
  let sum = 0;
  LEGS.forEach((l, n) => {
    const footX = l.x + LEG_PIVOT_Y * Math.sin(angles[n]);
    const footY = LEG_PIVOT_Y * (1 - Math.cos(angles[n]));
    sum += Math.exp(-softness * (footX * sinP + footY * cosP));
  });
  const lowest = -Math.log(sum / LEGS.length) / softness;
  const hop = 0.07 * Math.max(0, wave(0.88)) ** 2;
  rig.body.position.y = -lowest + hop * amount;
  rig.body.rotation.z = pitch * amount;
  rig.head.rotation.z = -0.7 * pitch * amount;
  rig.tail.rotation.z = -amount * (0.3 + 0.08 * wave(0.3));
  rig.tail.rotation.x = amount * 0.05 * Math.sin(2 * Math.PI * phase);
}
