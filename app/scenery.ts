import * as T from 'three';
import type {TrackModel} from './trackGeometry';

// Procedural surface textures and course furniture that is not read from a plan.

/** Metres of course covered by one repeat of the turf and dirt textures. */
export const SURFACE_TILE = 24;
/** Metres of ground covered by one repeat of the ground texture. */
export const GROUND_TILE = 48;
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
  const hex = parseInt(base.slice(1), 16);
  const rgb: [number, number, number] = [hex >> 16, (hex >> 8) & 255, hex & 255];
  return colourTexture((u, v) => {
    const k = 1 + tone(u, v);
    return [rgb[0] * k, rgb[1] * k, rgb[2] * k];
  }, size);
}

/** Seamless tile whose pixels are `colour(u, v)` (sRGB 0..255). */
function colourTexture(colour: (u: number, v: number) => [number, number, number], size = 512) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  const img = g.createImageData(size, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const rgb = colour(x / size, y / size);
      const i = (y * size + x) * 4;
      for (let ch = 0; ch < 3; ch++) img.data[i + ch] = Math.max(0, Math.min(255, rgb[ch]));
      img.data[i + 3] = 255;
    }
  g.putImageData(img, 0, 0);
  const tex = new T.CanvasTexture(c);
  tex.colorSpace = T.SRGBColorSpace;
  tex.wrapS = tex.wrapT = T.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}

const mix3 = (a: number[], b: number[], t: number): [number, number, number] =>
  [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** Healthy race-day turf (sRGB), the straw of dry or thin patches, and the darker blue-green of lusher ones. */
const TURF_GREEN = [84, 124, 52], TURF_STRAW = [138, 136, 78], TURF_LUSH = [60, 104, 50];

/**
 * Turf as a photograph of it from a few metres up: deep green, patches drifting toward straw or a lusher blue-green, a
 * per-pixel grain of light and dark blades, and with `stripes` the mowing bands 3 m wide along the course (v = across).
 */
export function turfTexture(stripes = true) {
  const rand = seeded(11);
  const n = [3, 16, 40, 96].map((k) => tileNoise(rand, k));
  return colourTexture((u, v) => {
    const dry = T.MathUtils.smoothstep(0.7 * n[0](u, v) + 0.3 * n[1](u, v), 0.15, 0.75);
    const lush = T.MathUtils.smoothstep(n[2](u + 0.3, v), 0.2, 0.8);
    let rgb = mix3(mix3(TURF_GREEN, TURF_LUSH, 0.2 * lush), TURF_STRAW, 0.12 * dry);
    const mown = stripes ? 0.09 * Math.tanh(3 * Math.sin(Math.PI * 8 * v)) : 0;
    const grain = 0.1 * (rand() - 0.5) + 0.05 * n[3](u, v);
    rgb = rgb.map(c => c * (1 + mown + grain)) as [number, number, number];
    return rgb;
  });
}

/**
 * Dirt as the sand of a JRA dirt course: pale grey-beige with a speckle of darker and lighter grains, soft damp-dark
 * patches, and faint harrow lines that wander along the course (v = across).
 */
export function dirtTexture() {
  const rand = seeded(23);
  const n = [6, 16, 40, 96].map((k) => tileNoise(rand, k));
  const sand = [200, 178, 146], damp = [168, 146, 116];
  return colourTexture((u, v) => {
    const wet = T.MathUtils.smoothstep(0.6 * n[0](u, v) + 0.4 * n[1](u, v), 0.1, 0.8);
    const harrow = 0.025 * Math.sin(Math.PI * 64 * v + 1.5 * n[1](u, v));
    const r = rand();
    // Single grains: most the sand's own tone, a few dark (stone, clay) and a few bright (quartz).
    const grain = r < 0.06 ? -0.28 * rand() : r > 0.95 ? 0.18 * rand() : 0.08 * (rand() - 0.5);
    const k = 1 + harrow + grain + 0.04 * n[2](u, v) + 0.03 * n[3](u, v);
    return mix3(sand, damp, 0.55 * wet).map(c => c * k) as [number, number, number];
  });
}

/** Ground around and inside the courses: unmown lawn, greener in hollows and straw-coloured where it dries out. */
export function groundTexture() {
  const rand = seeded(37);
  const n = [4, 8, 16, 64].map((k) => tileNoise(rand, k));
  const lawn = [96, 128, 64], straw = [146, 140, 88], lush = [70, 108, 54];
  return colourTexture((u, v) => {
    const dry = T.MathUtils.smoothstep(0.5 * n[0](u, v) + 0.5 * n[1](u, v), 0.1, 0.8);
    const wetter = T.MathUtils.smoothstep(n[2](u, v), 0.3, 0.9);
    const k = 1 + 0.1 * (rand() - 0.5) + 0.06 * n[3](u, v);
    return mix3(mix3(lawn, lush, 0.4 * wetter), straw, 0.3 * dry).map(c => c * k) as [number, number, number];
  });
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
  // Kept after any shader changes the material already has (a scan's detail on the turf and dirt).
  const previous = material.onBeforeCompile.bind(material);
  const previousKey = material.customProgramCacheKey.bind(material);
  material.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer);
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
  material.customProgramCacheKey = () => `${previousKey()}|world-variation-${strength}`;
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
