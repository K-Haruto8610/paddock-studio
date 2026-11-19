import * as T from 'three';
import {seeded, SURFACE_TILE, tileNoise} from './scenery';
import type {WeatherLook} from './raceWeather';

/**
 * Turf cut up along the inside rail on a wet day (内の芝が荒れる): on soft ground the field runs close to the rail and
 * the hooves tear the turf there, so the first few metres off the inside rail are trampled flat and yellowed, with bare,
 * muddy patches and divots where the turf has been kicked out. The worn ground and the blades that stand on it read one
 * shared pattern (a small tiling texture, sampled the same way in the shader and in `wearAt`), so a bare patch on the
 * ground has no blades on it and a trampled one carries short, flattened grass.
 */

/** Metres covered by one repeat of the wear pattern, and its resolution. */
const TILE = 48;
const SIZE = 256;

/**
 * How worn the turf is along the inside after `courseDay` race days on the present rail position (開催後半の内の荒れ):
 * none for the first two days after the rail is moved, then the field running close to the rail tramples and tears it,
 * levelling off at 0.55 by the twelfth day. On good going it is mostly trampled, flattened and yellowed grass with some
 * bare, dry patches.
 */
export function meetingWear(courseDay: number | null) {
  return courseDay ? 0.55 * T.MathUtils.smoothstep(courseDay, 2, 12) : 0;
}

/**
 * How worn the inside is (0 none ... 1 heavily cut up): the weather and the official going (1 良 ... 4 不良) on the day,
 * and the days raced on the present rail. Rain on turf already worn by the meeting cuts it up further.
 */
export function turfWearAmount(look: WeatherLook, going: number | null, courseDay: number | null = null) {
  const byGoing = going ? [0, 0, 0.4, 0.7, 0.95][going] ?? 0 : 0;
  const wet = Math.max(0.8 * look.wet, byGoing);
  return 1 - (1 - wet) * (1 - meetingWear(courseDay));
}

export type TurfWear = {amount: number; wet: number};
export const NO_WEAR: TurfWear = {amount: 0, wet: 0};

let pattern: {data: Uint8Array; texture: T.DataTexture} | null = null;

/** R: worn patches (clumps 1-4 m, 0..1); G: divots (kicked-out sods 15-35 cm). Tiles over TILE metres. */
function wearPattern() {
  if (pattern) return pattern;
  const rand = seeded(6151);
  const n = [12, 24, 48, 96].map(k => tileNoise(rand, k));
  const raw = new Float32Array(SIZE * SIZE);
  let lo = Infinity, hi = -Infinity;
  for (let y = 0; y < SIZE; y++)
    for (let x = 0; x < SIZE; x++) {
      const u = x / SIZE, v = y / SIZE;
      const value = 0.45 * n[0](u, v) + 0.3 * n[1](u, v) + 0.17 * n[2](u, v) + 0.08 * n[3](u, v);
      raw[y * SIZE + x] = value;
      lo = Math.min(lo, value);
      hi = Math.max(hi, value);
    }
  const divots = new Float32Array(SIZE * SIZE);
  const metre = SIZE / TILE;
  for (let k = 0; k < 1400; k++) {
    // An oval sod, longer along a random hoof direction, with a soft edge.
    const cx = rand() * SIZE, cy = rand() * SIZE, r = (0.15 + 0.2 * rand()) * metre, angle = rand() * Math.PI;
    const ca = Math.cos(angle), sa = Math.sin(angle), reach = Math.ceil(r * 1.6) + 1;
    for (let dy = -reach; dy <= reach; dy++)
      for (let dx = -reach; dx <= reach; dx++) {
        const a = (dx * ca + dy * sa) / (r * 1.5), b = (-dx * sa + dy * ca) / r;
        const d = Math.hypot(a, b);
        if (d >= 1) continue;
        const i = (((Math.round(cy) + dy) % SIZE + SIZE) % SIZE) * SIZE + (((Math.round(cx) + dx) % SIZE + SIZE) % SIZE);
        divots[i] = Math.max(divots[i], T.MathUtils.smoothstep(1 - d, 0, 0.4));
      }
  }
  const data = new Uint8Array(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i++) {
    data[i * 4] = Math.round(255 * (raw[i] - lo) / (hi - lo));
    data[i * 4 + 1] = Math.round(255 * divots[i]);
    data[i * 4 + 3] = 255;
  }
  const texture = new T.DataTexture(data, SIZE, SIZE, T.RGBAFormat);
  texture.wrapS = texture.wrapT = T.RepeatWrapping;
  texture.magFilter = T.LinearFilter;
  texture.minFilter = T.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  pattern = {data, texture};
  return pattern;
}

const smooth = (x: number, a: number, b: number) => T.MathUtils.smoothstep(x, a, b);

/**
 * The classification, written twice with the same numbers (here and in `WEAR_GLSL`): how far off the inside rail the wear
 * reaches, which patches are trampled, which are bare.
 */
function classify(patch: number, divot: number, lane: number, amount: number) {
  const w = (1 - smooth(lane, 2.0, 7.5)) * amount;
  const trampled = smooth(patch, 0.78 - 0.62 * w, 0.93 - 0.62 * w) * smooth(w, 0, 0.15);
  const bare = Math.max(smooth(patch, 0.9 - 0.42 * w, 0.98 - 0.42 * w) * smooth(w, 0.25, 0.55), divot * smooth(w, 0.08, 0.35));
  return {trampled, bare};
}

const WEAR_GLSL = /* glsl */ `
uniform sampler2D uWearMap;
uniform float uWear;
uniform float uWearWet;
varying vec2 vWearXZ;
varying float vWearLane;
vec2 turfWear() {
  vec2 s = texture2D(uWearMap, vWearXZ / ${TILE.toFixed(1)}).rg;
  float w = (1.0 - smoothstep(2.0, 7.5, vWearLane)) * uWear;
  float trampled = smoothstep(0.78 - 0.62 * w, 0.93 - 0.62 * w, s.r) * smoothstep(0.0, 0.15, w);
  float bare = max(smoothstep(0.9 - 0.42 * w, 0.98 - 0.42 * w, s.r) * smoothstep(0.25, 0.55, w), s.g * smoothstep(0.08, 0.35, w));
  return vec2(trampled, bare);
}`;

/** Worn state at world (x, z), `lane` metres off the inside rail: what the blades growing there read. */
export function wearAt(x: number, z: number, lane: number, wear: TurfWear) {
  if (wear.amount <= 0) return {trampled: 0, bare: 0};
  const {data} = wearPattern();
  // Bilinear, texel centres at half-texels, wrapping: as the GPU samples the base level.
  const fx = (x / TILE) * SIZE - 0.5, fy = (z / TILE) * SIZE - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0;
  const at = (i: number, j: number, c: number) =>
    data[((((j % SIZE) + SIZE) % SIZE) * SIZE + (((i % SIZE) + SIZE) % SIZE)) * 4 + c] / 255;
  const sample = (c: number) => T.MathUtils.lerp(
    T.MathUtils.lerp(at(x0, y0, c), at(x0 + 1, y0, c), tx),
    T.MathUtils.lerp(at(x0, y0 + 1, c), at(x0 + 1, y0 + 1, c), tx), ty);
  return classify(sample(0), sample(1), lane, wear.amount);
}

/**
 * Wears a turf ribbon's material (uv: v = lane / SURFACE_TILE). Trampled turf is flattened and yellowed; bare patches are
 * dark wet soil, smoother where wet and without the felt's sheen (it is mud, not turf). Chains the material's own
 * onBeforeCompile (the world colour variation).
 */
export function wearTurf<M extends T.MeshStandardMaterial>(material: M, wear: TurfWear): M {
  if (wear.amount <= 0) return material;
  const previous = material.onBeforeCompile.bind(material);
  const key = material.customProgramCacheKey.bind(material);
  material.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer);
    Object.assign(shader.uniforms, {
      uWearMap: {value: wearPattern().texture}, uWear: {value: wear.amount}, uWearWet: {value: wear.wet},
    });
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vWearXZ;\nvarying float vWearLane;')
      .replace('#include <project_vertex>', `#include <project_vertex>
        vWearXZ = (modelMatrix * vec4(transformed, 1.0)).xz;
        vWearLane = uv.y * ${SURFACE_TILE.toFixed(1)};`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${WEAR_GLSL}`)
      .replace('#include <map_fragment>', `#include <map_fragment>
        vec2 worn = turfWear();
        diffuseColor.rgb *= mix(vec3(1.0), vec3(0.8, 0.74, 0.46), 0.85 * worn.x);
        diffuseColor.rgb = mix(diffuseColor.rgb, mix(vec3(0.2, 0.15, 0.095), vec3(0.13, 0.095, 0.06), uWearWet), worn.y);`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, 0.55, worn.y * uWearWet);`)
      .replace('#include <lights_physical_fragment>', `#include <lights_physical_fragment>
        #ifdef USE_SHEEN
          material.sheenColor *= 1.0 - worn.y;
        #endif`);
  };
  material.customProgramCacheKey = () => `${key()}-turf-wear`;
  return material;
}
