import * as T from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';
import {apronTexture, seeded, tileNoise} from './scenery';
import {photoTextures, type PhotoSet} from './photoTextures';

// Surface finishes from docs/STYLE_GUIDE.md: the turf is matte felt (a soft green face, faint mottling and mowing stripes,
// a fine nap that only shows in raking light), the dirt is real sand (grain, and roughness that varies from patch to patch).

/** Metres of felt covered by one repeat of the nap map. */
export const FELT_TILE = 1.6;

export function normalFromHeight(height: Float32Array, size: number, strength: number) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  const img = g.createImageData(size, size);
  const at = (x: number, y: number) => height[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength, dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const k = 1 / Math.hypot(dx, dy, 1);
      const i = (y * size + x) * 4;
      img.data[i] = (-dx * k * 0.5 + 0.5) * 255;
      // Canvas rows run downwards while v runs upwards.
      img.data[i + 1] = (dy * k * 0.5 + 0.5) * 255;
      img.data[i + 2] = (k * 0.5 + 0.5) * 255;
      img.data[i + 3] = 255;
    }
  g.putImageData(img, 0, 0);
  const tex = new T.CanvasTexture(c);
  tex.wrapS = tex.wrapT = T.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}

let feltBase: T.CanvasTexture | null = null;

/**
 * Normal map of felt: a fine, even nap of short fibres, with no lines or blades in it. A copy per use (they share the
 * image), so each surface sets its own repeat; `size` metres is what one repeat should cover.
 */
export function feltNormalMap(repeatX: number, repeatY: number) {
  if (!feltBase) {
    const size = 256, rand = seeded(71);
    const n = [40, 96, 160].map(k => tileNoise(rand, k));
    const height = new Float32Array(size * size);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const u = x / size, v = y / size;
        height[y * size + x] = 1.0 * n[0](u, v) + 0.9 * n[1](u, v) + 0.7 * n[2](u, v) + 0.9 * (rand() - 0.5);
      }
    feltBase = normalFromHeight(height, size, 0.5);
  }
  const tex = feltBase.clone();
  tex.needsUpdate = true;
  tex.repeat.set(repeatX, repeatY);
  return tex;
}

/**
 * The material of the turf: matte (roughness 1) with the soft glow of fibres at grazing angles (sheen), the felt nap as
 * a normal map, and the colour and mowing stripes carried by `map`. Never shiny.
 */
export function feltMaterial(map: T.Texture | null, repeatX: number, repeatY: number, lift = 0, color = '#ffffff') {
  return new T.MeshPhysicalMaterial({
    color, map, roughness: 1, metalness: 0,
    sheen: 0.22, sheenRoughness: 0.9, sheenColor: new T.Color('#b9d69b'),
    normalMap: feltNormalMap(repeatX, repeatY), normalScale: new T.Vector2(0.45, 0.45),
    side: T.DoubleSide,
    polygonOffset: lift !== 0, polygonOffsetFactor: -lift, polygonOffsetUnits: -lift,
  });
}

/**
 * Roughness of sand: dry, loose patches rough (1), damp or packed ones a little smoother (about 0.7), at the scale of a hoof
 * print down to a grain. Three reads roughness from the green channel.
 */
export function sandRoughnessMap(size = 512) {
  const rand = seeded(97);
  const n = [6, 18, 60, 160].map(k => tileNoise(rand, k));
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  const img = g.createImageData(size, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const u = x / size, v = y / size;
      const t = 0.5 * n[0](u, v) + 0.3 * n[1](u, v) + 0.15 * n[2](u, v) + 0.1 * n[3](u, v) + 0.12 * (rand() - 0.5);
      const rough = T.MathUtils.clamp(0.86 + 0.16 * t, 0.62, 1);
      const i = (y * size + x) * 4;
      img.data[i] = img.data[i + 2] = 255;
      img.data[i + 1] = rough * 255;
      img.data[i + 3] = 255;
    }
  g.putImageData(img, 0, 0);
  const tex = new T.CanvasTexture(c);
  tex.wrapS = tex.wrapT = T.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}

let railBase: T.CanvasTexture | null = null;

/**
 * Painted rail: white paint with a little wear — faint grime streaks, darker toward the underside, a few chipped specks. A copy
 * per use; `repeatAlong` is how many times the pattern repeats along the rail (u), once around it (v).
 */
export function railPaintMap(repeatAlong: number) {
  if (!railBase) {
    const size = 256, rand = seeded(113);
    const n = [6, 24, 90].map(k => tileNoise(rand, k));
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const g = c.getContext('2d')!;
    const img = g.createImageData(size, size);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const u = x / size, v = y / size;
        // v runs round the tube: the underside (v near 0.5 in a tube's own frame) collects more dirt than the top.
        const under = 0.5 + 0.5 * Math.cos(2 * Math.PI * (v - 0.5));
        // Streaks run along the rail (u): stretch the noise by sampling it at a wide aspect.
        const streak = n[0](u, v * 0.5) * 0.5 + n[1](u * 0.5, v) * 0.35;
        let k = 0.95 - 0.07 * under + 0.05 * streak + 0.03 * n[2](u, v);
        if (rand() < 0.0016) k -= 0.18;                       // a chipped speck
        const i = (y * size + x) * 4;
        img.data[i] = Math.max(0, 247 * k);
        img.data[i + 1] = Math.max(0, 246 * k);
        img.data[i + 2] = Math.max(0, 234 * k);
        img.data[i + 3] = 255;
      }
    g.putImageData(img, 0, 0);
    railBase = new T.CanvasTexture(c);
    railBase.colorSpace = T.SRGBColorSpace;
    railBase.wrapS = railBase.wrapT = T.RepeatWrapping;
    railBase.anisotropy = 8;
  }
  const tex = railBase.clone();
  tex.needsUpdate = true;
  tex.repeat.set(repeatAlong, 1);
  return tex;
}

/** Painted steel: some sheen from the paint, never plastic. */
export function paintedMetal(map: T.Texture) {
  return new T.MeshPhysicalMaterial({color: '#ffffff', map, roughness: 0.52, metalness: 0.2, clearcoat: 0.12, clearcoatRoughness: 0.5});
}

/**
 * A rail post with a small bevel at both ends and smooth sides (a lathe, not a faceted cylinder): 1 m tall, centred,
 * 75 mm radius at the foot, 60 mm at the top.
 */
export function postGeometry() {
  const profile = [[0, -0.5], [0.062, -0.5], [0.0745, -0.488], [0.0605, 0.488], [0.049, 0.5], [0, 0.5]]
    .map(([r, y]) => new T.Vector2(r, y));
  return new T.LatheGeometry(profile, 18);
}

/** Metres covered by one repeat of the concrete maps. */
export const CONCRETE_TILE = 12;

let concreteMaps: {roughness: T.CanvasTexture; normal: T.CanvasTexture} | null = null;

/**
 * Cast concrete: pores and a little grain in the normal map, roughness that wanders between 0.75 and 0.95 so wet-looking
 * and dry patches differ. Procedural, since no photo set ships with the project: swap in photo PBR maps here when there are some.
 */
function concreteDetail() {
  if (!concreteMaps) {
    const size = 256, rand = seeded(131);
    const n = [8, 28, 90, 200].map(k => tileNoise(rand, k));
    const height = new Float32Array(size * size);
    const rough = document.createElement('canvas');
    rough.width = rough.height = size;
    const g = rough.getContext('2d')!;
    const img = g.createImageData(size, size);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const u = x / size, v = y / size;
        height[y * size + x] = 0.4 * n[1](u, v) + 0.8 * n[2](u, v) + 0.8 * n[3](u, v) + 0.7 * (rand() - 0.5);
        const r = T.MathUtils.clamp(0.86 + 0.1 * n[0](u, v) + 0.06 * n[1](u, v), 0.7, 1);
        const i = (y * size + x) * 4;
        img.data[i] = img.data[i + 2] = 255;
        img.data[i + 1] = r * 255;
        img.data[i + 3] = 255;
      }
    g.putImageData(img, 0, 0);
    const roughness = new T.CanvasTexture(rough);
    roughness.wrapS = roughness.wrapT = T.RepeatWrapping;
    concreteMaps = {roughness, normal: normalFromHeight(height, size, 0.35)};
  }
  return concreteMaps;
}

/** Concrete for the stands and buildings; the colour map comes from the caller (it carries the stains). */
export function concreteMaterial(map: T.Texture, color = '#ffffff') {
  const {roughness, normal} = concreteDetail();
  return new T.MeshStandardMaterial({color, map, roughness: 1, roughnessMap: roughness, normalMap: normal, normalScale: new T.Vector2(0.5, 0.5), metalness: 0});
}

/**
 * A box with its edges rounded by `bevel` (nothing in the world has a perfect right angle) and uv in metres, so a tiling
 * map keeps its scale on a 100 m wall as on a 1 m step.
 */
export function bevelBox(w: number, h: number, d: number, bevel = 0.18, tile = CONCRETE_TILE) {
  const geometry = new RoundedBoxGeometry(w, h, d, 2, Math.min(bevel, Math.min(w, h, d) / 2.2));
  const p = geometry.attributes.position, n = geometry.attributes.normal, uv = geometry.attributes.uv;
  for (let i = 0; i < uv.count; i++) {
    const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i)), az = Math.abs(n.getZ(i));
    if (ax >= ay && ax >= az) uv.setXY(i, p.getZ(i) / tile, p.getY(i) / tile);
    else if (ay >= az) uv.setXY(i, p.getX(i) / tile, p.getZ(i) / tile);
    else uv.setXY(i, p.getX(i) / tile, p.getY(i) / tile);
  }
  return geometry;
}

/** Metres covered by one repeat of the scanned wall and roof photos (a precast panel, a few corrugations). */
export const PHOTO_TILE = {concrete: 9, roof: 3};

function photoMaterial(set: PhotoSet, repeatTile: number, tint: T.ColorRepresentation) {
  // The maps are shared; a copy of each per material would only repeat them differently, and uv are already in metres / tile.
  const material = new T.MeshStandardMaterial({
    color: tint, map: set.map, roughnessMap: set.roughnessMap, roughness: 1, normalMap: set.normalMap,
    normalScale: new T.Vector2(0.8, 0.8), metalness: 0,
  });
  material.userData.tile = repeatTile;
  return material;
}

/**
 * Materials of the stands and buildings: the scanned photo PBR sets (concrete panels, corrugated roofing) when they have
 * loaded, the procedural concrete otherwise. `userData.tile` is the metres one uv repeat should cover (see `bevelBox`).
 */
export function standMaterials() {
  const photos = photoTextures();
  if (photos) {
    return {
      // The scans are dark: lift them a little (colour values above 1 simply brighten the map).
      concrete: photoMaterial(photos.concrete, PHOTO_TILE.concrete, new T.Color(1.55, 1.5, 1.4)),
      roof: photoMaterial(photos.roof, PHOTO_TILE.roof, new T.Color(0.6, 0.85, 0.92)),
    };
  }
  const concrete = concreteMaterial(apronTexture(), '#f2e8d2');
  concrete.userData.tile = CONCRETE_TILE;
  return {concrete, roof: new T.MeshStandardMaterial({color: '#405d62', roughness: 0.85})};
}
