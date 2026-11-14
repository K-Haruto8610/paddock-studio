import * as T from 'three';
import {seeded, tileNoise} from './scenery';

// Surfaces with depth: grass that stands up from the turf, and relief on the dirt.

/** Shared by the blades and the shells so both sway, lean and stand to the same settings. */
export type GrassUniforms = {
  uTime: {value: number};
  /** 0 = still air, 1 = a stiff breeze. */
  uWind: {value: number};
  /** Blade length in metres. */
  uHeight: {value: number};
  /** Height of the shell layer in metres. */
  uShellHeight: {value: number};
  /** 1 leans alternate 3m bands up and down the course, as mowing does; 0 leaves the grass unmown. */
  uStripe: {value: number};
  /** Blades stand between these two distances along the course, growing in over `BLADE_FADE` metres at each end. */
  uFrom: {value: number};
  uTo: {value: number};
};

/** Blade length in metres: turf at its race-day cut. */
export const DEFAULT_BLADE_HEIGHT = 0.25;

export function grassUniforms(): GrassUniforms {
  return {
    uTime: {value: 0}, uWind: {value: 0.5}, uHeight: {value: DEFAULT_BLADE_HEIGHT}, uShellHeight: {value: DEFAULT_BLADE_HEIGHT * 0.55}, uStripe: {value: 1},
    uFrom: {value: -1e6}, uTo: {value: 1e6},
  };
}

/** Metres over which blades grow from nothing at each end of the stretch they stand on. */
export const BLADE_FADE = 15;
/** How far mown grass lies over, as a fraction of its length. */
const MOW_LEAN = 0.35;

/** How far the wind pushes a blade at world (x, z) over, as a fraction of its length. */
const WIND = /* glsl */ `
uniform float uTime;
uniform float uWind;
uniform float uStripe;
vec2 wind(vec2 at, float phase) {
  float gust = 0.6 * sin(uTime * 1.9 + at.x * 0.45 + at.y * 0.31 + phase) + sin(uTime * 0.7 + at.x * 0.09 - at.y * 0.05);
  return vec2(1.0, 0.35) * uWind * (0.22 + 0.16 * gust);
}
`;

/** Light and dark of mowing: 3m bands across the course, +1 mown one way and -1 the other. */
export const mowBand = (lane: number) => Math.tanh(3 * Math.sin((Math.PI * lane) / 3));

const BLADE_WIDTH = 0.034;
/** Half-widths of a blade at each quarter of its length; the tip is a single point. */
const BLADE_TAPER = [1, 0.9, 0.7, 0.42];

/** Where each blade of a stretch of grass stands, as the per-blade attributes the shader reads. */
export type BladeField = {
  count: number;
  /** World x, y, z of the root. */
  root: Float32Array;
  /** Yaw, length factor, own lean, shade (fraction lighter or darker). */
  blade: Float32Array;
  /** Mowing lean (x, z: the course direction times the band), distance along the course, gust phase. */
  lay: Float32Array;
  /** x, z of the unit normal of the ground under the blade (y follows); zeros leave the blade upright. */
  up: Float32Array;
};

export function emptyBladeField(count: number): BladeField {
  return {count, root: new Float32Array(count * 3), blade: new Float32Array(count * 4), lay: new Float32Array(count * 4), up: new Float32Array(count * 2)};
}

/** Fills blade `i` of a field: everything but where it stands and how it was mown. */
export function scatterBlade(field: BladeField, i: number, rand: () => number) {
  field.blade[i * 4] = rand() * Math.PI * 2;
  field.blade[i * 4 + 1] = 0.6 + 0.55 * rand();
  field.blade[i * 4 + 2] = 0.1 + 0.35 * rand();
  field.blade[i * 4 + 3] = (rand() - 0.5) * 0.26;
  field.lay[i * 4 + 3] = rand() * Math.PI * 2;
}

/**
 * Blades over a flat width x depth patch centred on the origin, the course running along x.
 * They are laid out in random order, so drawing only the first n thins the patch evenly.
 */
export function flatBladeField(width: number, depth: number, count: number) {
  const rand = seeded(71);
  const field = emptyBladeField(count);
  for (let i = 0; i < count; i++) {
    const z = (rand() - 0.5) * depth;
    field.root[i * 3] = (rand() - 0.5) * width;
    field.root[i * 3 + 2] = z;
    field.lay[i * 4] = mowBand(z);
    scatterBlade(field, i, rand);
    field.blade[i * 4 + 3] += 0.05 * mowBand(z);
  }
  return field;
}

/** One material for every mesh of blades that shares these uniforms. */
export function bladeMaterial(uniforms: GrassUniforms) {
  const material = new T.MeshStandardMaterial({color: '#7faf66', roughness: 0.85, side: T.DoubleSide});
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        attribute vec3 aRoot;
        attribute vec4 aBlade;
        attribute vec4 aLay;
        attribute vec2 aUp;
        uniform float uHeight;
        uniform float uFrom;
        uniform float uTo;
        varying vec3 vGrass;
        ${WIND}
        // The blade's own axes: up along the ground's normal, and x, z kept as near the world's as the slope allows.
        mat3 standing() {
          vec3 n = vec3(aUp.x, sqrt(max(1.0 - dot(aUp, aUp), 0.0)), aUp.y);
          vec3 ex = normalize(vec3(1.0, 0.0, 0.0) - n.x * n);
          return mat3(ex, n, cross(ex, n));
        }
        vec2 bladeBend() {
          vec2 face = vec2(-sin(aBlade.x), cos(aBlade.x));
          return face * aBlade.z + aLay.xy * (uStripe * ${MOW_LEAN.toFixed(2)}) + wind(aRoot.xz, aLay.w);
        }`,
      )
      // Lit as if every blade faced the sky, tilted by its bend: gusts then show as moving sheen.
      .replace(
        '#include <beginnormal_vertex>',
        `vec2 bendN = bladeBend();
        mat3 stand = standing();
        vec3 objectNormal = normalize(stand * vec3(bendN.x * 0.7, 1.0, bendN.y * 0.7));`,
      )
      .replace(
        '#include <begin_vertex>',
        `vec2 bend = bladeBend();
        float reach = min(length(bend), 0.92);
        vec2 dir = reach > 1e-4 ? normalize(bend) : vec2(0.0);
        float t = position.y;
        float grown = smoothstep(uFrom, uFrom + ${BLADE_FADE.toFixed(1)}, aLay.z)
          * (1.0 - smoothstep(uTo - ${BLADE_FADE.toFixed(1)}, uTo, aLay.z));
        float len = uHeight * aBlade.y * grown;
        float side = position.x * ${BLADE_WIDTH.toFixed(4)} * min(grown * 4.0, 1.0);
        float over = reach * t * t * len;
        vec3 transformed = aRoot + stand * vec3(
          cos(aBlade.x) * side + dir.x * over,
          t * sqrt(1.0 - reach * reach * t * t) * len,
          sin(aBlade.x) * side + dir.y * over);
        float patchTone = 1.0 + 0.07 * sin(aRoot.x * 0.21 + 1.3) * sin(aRoot.z * 0.17);
        vGrass = mix(vec3(0.5, 0.56, 0.44), vec3(1.25, 1.27, 1.05), t) * (1.0 + aBlade.w) * patchTone;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vGrass;')
      .replace('#include <color_fragment>', 'diffuseColor.rgb *= vGrass;')
      // Both faces keep the sky-facing normal; flipping it would black out every back face.
      .replace('#include <normal_fragment_begin>', T.ShaderChunk.normal_fragment_begin.replace('normal *= faceDirection;', ''));
  };
  return material;
}

/** Single blades of grass standing where `field` says, drawn as one instanced mesh. */
export function bladeMesh(field: BladeField, material: T.Material) {
  const positions: number[] = [];
  const index: number[] = [];
  BLADE_TAPER.forEach((w, i) => {
    const t = i / BLADE_TAPER.length;
    positions.push(-0.5 * w, t, 0, 0.5 * w, t, 0);
  });
  positions.push(0, 1, 0);
  for (let i = 0; i < BLADE_TAPER.length - 1; i++) {
    const a = i * 2;
    index.push(a, a + 1, a + 3, a, a + 3, a + 2);
  }
  const top = (BLADE_TAPER.length - 1) * 2;
  index.push(top, top + 1, top + 2);
  const geometry = new T.InstancedBufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geometry.setIndex(index);
  geometry.setAttribute('aRoot', new T.InstancedBufferAttribute(field.root, 3));
  geometry.setAttribute('aBlade', new T.InstancedBufferAttribute(field.blade, 4));
  geometry.setAttribute('aLay', new T.InstancedBufferAttribute(field.lay, 4));
  geometry.setAttribute('aUp', new T.InstancedBufferAttribute(field.up, 2));
  geometry.instanceCount = field.count;
  const mesh = new T.Mesh(geometry, material);
  // Blades are placed in the vertex shader, so the geometry's own bounds say nothing.
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  return mesh;
}

/** Blades per metre along each axis of the shell layer. */
const SHELL_DENSITY = 46;

/**
 * Grass as a stack of see-through sheets: each layer keeps only the tufts tall enough to reach it.
 * Cheap enough to cover a whole course; from the side the sheets show.
 */
export function shellGrass(width: number, depth: number, layers: number, map: T.Texture, uniforms: GrassUniforms) {
  const geometry = new T.PlaneGeometry(width, depth);
  geometry.rotateX(-Math.PI / 2);
  const material = new T.MeshStandardMaterial({map, roughness: 0.92});
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uShellHeight;\nvarying vec3 vShell;')
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        float layer = float(gl_InstanceID) / ${(layers - 1).toFixed(1)};
        transformed.y += layer * uShellHeight;
        vShell = vec3(position.x, position.z, layer);`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform float uShellHeight;
        varying vec3 vShell;
        ${WIND}
        float mowBand(float z) { return tanh(3.0 * sin(3.14159265 * z / 3.0)); }
        float tuft(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }`,
      )
      .replace(
        '#include <map_fragment>',
        `#include <map_fragment>
        {
          float layer = vShell.z;
          vec2 lean = vec2(mowBand(vShell.y) * uStripe * ${MOW_LEAN.toFixed(2)}, 0.0) + wind(vShell.xy, 0.0);
          vec2 p = vShell.xy - lean * layer * layer * uShellHeight;
          vec2 cell = p * ${SHELL_DENSITY.toFixed(1)};
          vec2 id = floor(cell);
          vec2 f = fract(cell) - 0.5 + (vec2(tuft(id + 17.0), tuft(id + 43.0)) - 0.5) * 0.5;
          float reach = mix(0.4, 1.0, tuft(id));
          if (layer > 0.0 && (layer > reach || length(f) > 0.5 * (1.0 - layer / reach) + 0.06)) discard;
          diffuseColor.rgb *= mix(0.42, 1.15, layer) * (0.92 + 0.16 * tuft(id + 7.0));
        }`,
      );
  };
  const mesh = new T.InstancedMesh(geometry, material, layers);
  const m = new T.Matrix4();
  for (let i = 0; i < layers; i++) mesh.setMatrixAt(i, m);
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  return mesh;
}

/** Metres of dirt covered by one repeat of the relief map. */
export const RELIEF_TILE = 6;

/** Normal map for harrowed sand: furrows along the course (u), clods and grain. v runs across. */
export function dirtReliefMap(size = 1024) {
  const rand = seeded(53);
  const n = [12, 40, 110].map((k) => tileNoise(rand, k));
  const height = new Float32Array(size * size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const u = x / size,
        v = y / size;
      // Furrows wander a little so they do not read as ruled lines.
      const furrow = Math.sin(2 * Math.PI * (8 * v + 0.08 * n[0](u, v)));
      height[y * size + x] = 5 * furrow + 4 * n[0](u, v) + 3 * n[1](u, v) + 1.6 * n[2](u, v) + 0.9 * (rand() - 0.5);
    }
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  const img = g.createImageData(size, size);
  const at = (x: number, y: number) => height[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const dx = at(x + 1, y) - at(x - 1, y),
        dy = at(x, y + 1) - at(x, y - 1);
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
