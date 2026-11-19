import * as T from 'three';
import type {TrackModel} from './trackGeometry';

// Procedural surface textures and course furniture that is not read from a plan.

/** Metres of course covered by one repeat of the turf and dirt textures. */
export const SURFACE_TILE = 24;
/** Metres of ground covered by one repeat of the ground texture. */
export const GROUND_TILE = 48;
const GROUND_COLOR = '#8c9f74';
/** The flat land beyond the modelled ground lies this far below the lowest racing surface. */
export const FAR_GROUND_DROP = 0.6;

export function seeded(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Smooth noise in -1..1 that repeats over the unit square (n cells a side). */
export function tileNoise(rand: () => number, n: number) {
  const grid = Float32Array.from({length: n * n}, () => rand() * 2 - 1);
  const ease = (t: number) => t * t * (3 - 2 * t);
  return (u: number, v: number) => {
    const x = u * n,
      y = v * n,
      x0 = Math.floor(x),
      y0 = Math.floor(y),
      fx = ease(x - x0),
      fy = ease(y - y0),
      at = (i: number, j: number) => grid[(((j % n) + n) % n) * n + (((i % n) + n) % n)];
    return T.MathUtils.lerp(
      T.MathUtils.lerp(at(x0, y0), at(x0 + 1, y0), fx),
      T.MathUtils.lerp(at(x0, y0 + 1), at(x0 + 1, y0 + 1), fx),
      fy,
    );
  };
}

/** Seamless tile of `base`, each pixel lightened or darkened by the fraction `tone(u, v)`. */
function tileTexture(base: string, tone: (u: number, v: number) => number, size = 512) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  const img = g.createImageData(size, size);
  const hex = parseInt(base.slice(1), 16);
  const rgb = [hex >> 16, (hex >> 8) & 255, hex & 255];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const k = 1 + tone(x / size, y / size);
      const i = (y * size + x) * 4;
      for (let ch = 0; ch < 3; ch++) img.data[i + ch] = Math.max(0, Math.min(255, rgb[ch] * k));
      img.data[i + 3] = 255;
    }
  g.putImageData(img, 0, 0);
  const tex = new T.CanvasTexture(c);
  tex.colorSpace = T.SRGBColorSpace;
  tex.wrapS = tex.wrapT = T.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}

/** Turf: mottled grass; with `stripes`, mowing bands 3m wide running along the course (v = across). */
export function turfTexture(stripes = true) {
  const rand = seeded(11);
  const n = [8, 16, 32, 64].map((k) => tileNoise(rand, k));
  return tileTexture('#7faf66', (u, v) => {
    const mown = stripes ? 0.05 * Math.tanh(3 * Math.sin(Math.PI * 8 * v)) : 0;
    return mown + 0.04 * n[0](u, v) + 0.035 * n[1](u, v) + 0.03 * n[2](u, v) + 0.035 * n[3](u, v);
  });
}

/** Dirt: sand with faint harrow lines along the course. */
export function dirtTexture() {
  const rand = seeded(23);
  const n = [8, 16, 32, 64].map((k) => tileNoise(rand, k));
  return tileTexture(
    '#ceb28a',
    (u, v) =>
      0.022 * Math.sin(Math.PI * 64 * v) +
      0.03 * n[0](u, v) +
      0.03 * n[1](u, v) +
      0.03 * n[2](u, v) +
      0.04 * n[3](u, v),
  );
}

/** Ground around and inside the courses: pale lawn with broad, soft patches. */
export function groundTexture() {
  const rand = seeded(37);
  const n = [4, 8, 16, 64].map((k) => tileNoise(rand, k));
  return tileTexture(
    GROUND_COLOR,
    (u, v) => 0.012 * n[0](u, v) + 0.018 * n[1](u, v) + 0.03 * n[2](u, v) + 0.035 * n[3](u, v),
  );
}

/** Paved apron: pale concrete with soft stains. */
export function apronTexture() {
  const rand = seeded(53);
  const n = [4, 12, 40, 96].map((k) => tileNoise(rand, k));
  return tileTexture(
    '#c0c6c0',
    (u, v) => 0.02 * n[0](u, v) + 0.025 * n[1](u, v) + 0.025 * n[2](u, v) + 0.03 * n[3](u, v) +
      0.012 * Math.tanh(6 * Math.sin(Math.PI * 24 * u)) * Math.tanh(6 * Math.sin(Math.PI * 24 * v)),
  );
}

/**
 * Tints a ground material by slow noise in world coordinates, so no texture tile shows as a
 * repeat. The periods (about 240, 95 and 40m) are all longer than the 24m and 48m tiles.
 */
export function worldVariation<M extends T.Material>(material: M, strength = 0.1): M {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vVaryXZ;')
      .replace('#include <project_vertex>',
        '#include <project_vertex>\nvVaryXZ = (modelMatrix * vec4(transformed, 1.0)).xz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec2 vVaryXZ;
        float vHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float vNoise(vec2 p) {
          vec2 i = floor(p), f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(mix(vHash(i), vHash(i + vec2(1.0, 0.0)), f.x),
                     mix(vHash(i + vec2(0.0, 1.0)), vHash(i + vec2(1.0, 1.0)), f.x), f.y) * 2.0 - 1.0;
        }`)
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          float tone = 0.5 * vNoise(vVaryXZ / 240.0) + 0.3 * vNoise(vVaryXZ / 95.0 + 7.3) + 0.2 * vNoise(vVaryXZ / 40.0 + 19.1);
          diffuseColor.rgb *= vec3(1.0 + ${strength.toFixed(3)} * tone * 1.2, 1.0 + ${strength.toFixed(3)} * tone, 1.0 + ${strength.toFixed(3)} * tone * 0.7);
        }`);
  };
  material.customProgramCacheKey = () => `world-variation-${strength}`;
  return material;
}

/** Beyond the modelled ground the same lawn carries on flat, so its edge never shows. */
/** The fields beyond the course; `snow` (0..1) whitens them under lying snow. */
export function addFarGround(scene: T.Scene, floor: number, snow = 0) {
  const lawn = groundTexture();
  lawn.repeat.setScalar(10400 / GROUND_TILE);
  const far = new T.Mesh(
    new T.CircleGeometry(5200, 48),
    new T.MeshStandardMaterial({map: lawn, roughness: 0.96, color: new T.Color('#ffffff').lerp(new T.Color(2.4, 2.4, 2.5), snow)}),
  );
  far.rotation.x = -Math.PI / 2;
  far.position.y = floor - FAR_GROUND_DROP;
  far.receiveShadow = true;
  scene.add(far);
}

/** Red and white bands for a furlong pole. */
function bandTexture() {
  const c = document.createElement('canvas');
  c.width = 8;
  c.height = 64;
  const g = c.getContext('2d')!;
  for (let i = 0; i < 8; i++) {
    g.fillStyle = i % 2 ? '#fffdf4' : '#c8202a';
    g.fillRect(0, i * 8, 8, 8);
  }
  const tex = new T.CanvasTexture(c);
  tex.colorSpace = T.SRGBColorSpace;
  return tex;
}

function numberBoardTexture(text: string) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d')!;
  g.fillStyle = '#c8202a';
  g.beginPath();
  g.roundRect(0, 0, 128, 128, 20);
  g.fill();
  g.fillStyle = '#fffdf4';
  g.beginPath();
  g.roundRect(10, 10, 108, 108, 12);
  g.fill();
  g.fillStyle = '#263b34';
  g.font = 'bold 84px sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, 64, 70, 100);
  const tex = new T.CanvasTexture(c);
  tex.colorSpace = T.SRGBColorSpace;
  return tex;
}

/**
 * ハロン棒: a banded pole just inside the inner rail every 200m, its board showing the
 * remaining distance in hundreds of metres and readable from both sides of the course.
 */
export function addFurlongPoles(scene: T.Scene, track: TrackModel) {
  const bands = bandTexture();
  const poleGeo = new T.CylinderGeometry(0.1, 0.1, 3.6, 10);
  const poleMat = new T.MeshStandardMaterial({map: bands, roughness: 0.7});
  const boardGeo = new T.PlaneGeometry(1.5, 1.5);
  for (let remaining = 200; remaining < track.lap; remaining += 200) {
    const at = track.sample(track.lap - remaining, -1.6);
    const group = new T.Group();
    group.position.copy(at.point);
    group.rotation.y = -Math.atan2(at.tangent.z, at.tangent.x);
    group.scale.setScalar(1.6); // drawn at the same exaggeration as the horses and the goal post
    const pole = new T.Mesh(poleGeo, poleMat);
    pole.position.y = 1.8;
    pole.castShadow = true;
    group.add(pole);
    const face = new T.MeshStandardMaterial({
      map: numberBoardTexture(String(remaining / 100)),
      roughness: 0.7,
      toneMapped: false,
    });
    for (const turn of [0, Math.PI]) {
      const board = new T.Mesh(boardGeo, face);
      board.position.y = 4.3;
      board.rotation.y = turn;
      board.castShadow = true;
      group.add(board);
    }
    scene.add(group);
  }
}
