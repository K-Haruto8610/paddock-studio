import * as T from 'three';
import {seeded} from './scenery';

/** The race-day weather the scene draws (JV-Data TENKO_CODE 1-6). */
export type Weather = 'clear' | 'cloudy' | 'drizzle' | 'rain' | 'lightsnow' | 'snow';
export const WEATHERS: Weather[] = ['clear', 'cloudy', 'drizzle', 'rain', 'lightsnow', 'snow'];
export const WEATHER_LABEL: Record<Weather, string> = {
  clear: '晴', cloudy: '曇', drizzle: '小雨', rain: '雨', lightsnow: '小雪', snow: '雪',
};
export const GOING_LABEL: Record<number, string> = {1: '良', 2: '稍重', 3: '重', 4: '不良'};

/** Rail position and how many turf race days it has carried (scripts/build_race_conditions.py). */
export type RaceMeeting = {kai: number; day: number; course?: string; courseDay?: number; estimated?: boolean};
/** public/data/race_conditions.json (scripts/build_race_conditions.py). */
export type RaceConditionsFile = {races: Record<string, {
  weather?: number; going?: number; meeting?: RaceMeeting; postTime?: string; postTimeEstimated?: boolean;
}>};

const BY_CODE: Record<number, Weather> = {1: 'clear', 2: 'cloudy', 3: 'rain', 4: 'drizzle', 5: 'snow', 6: 'lightsnow'};
export function weatherFromCode(code: number | undefined): Weather | null {
  return code === undefined ? null : BY_CODE[code] ?? null;
}

/**
 * How a weather looks: cloud over the sky, how much of the sun's disc and light get through, the haze, what falls and how
 * wet or white the ground is. One table, read by the sky, the lights, the fog, the trees and the precipitation.
 */
export type WeatherLook = {
  /** Cloud cover of the sky (0.4 fair-weather cumulus ... 1.05 a closed grey deck). */
  cover: number;
  /** How grey and heavy the clouds are underneath (0 white ... 1 rain cloud). */
  grey: number;
  /** How much of the sun's disc shows (0 hidden behind cloud). */
  sun: number;
  /** Multipliers of the sunlight, of the light from the sky around, and of the shadow's darkness. */
  sunLight: number;
  skyLight: number;
  shadow: number;
  /** Haze: the fog colour mixed toward this grey by `murk`, and the fog's near and far distances (m). */
  murk: number;
  murkColour: string;
  fog: [number, number];
  /** Rain or snow falling, and its density (0..1). */
  falling: 'rain' | 'snow' | null;
  density: number;
  /** Wet surfaces (darker, less rough) and snow lying on trees and the far fields (0..1). */
  wet: number;
  snowCover: number;
};

export const WEATHER_LOOKS: Record<Weather, WeatherLook> = {
  clear: {cover: -1, grey: 0, sun: 1, sunLight: 1, skyLight: 1, shadow: 1, murk: 0, murkColour: '#c9cdcc', fog: [720, 2450],
    falling: null, density: 0, wet: 0, snowCover: 0},
  cloudy: {cover: 0.93, grey: 0.35, sun: 0.12, sunLight: 0.34, skyLight: 1.22, shadow: 0.35, murk: 0.55, murkColour: '#c4c8c8',
    fog: [600, 2200], falling: null, density: 0, wet: 0, snowCover: 0},
  drizzle: {cover: 1.02, grey: 0.55, sun: 0.03, sunLight: 0.24, skyLight: 1.18, shadow: 0.2, murk: 0.8, murkColour: '#b4babc',
    fog: [380, 1750], falling: 'rain', density: 0.45, wet: 0.65, snowCover: 0},
  rain: {cover: 1.08, grey: 0.75, sun: 0, sunLight: 0.17, skyLight: 1.08, shadow: 0.1, murk: 1, murkColour: '#a6aeb1',
    fog: [230, 1350], falling: 'rain', density: 1, wet: 1, snowCover: 0},
  lightsnow: {cover: 1.02, grey: 0.4, sun: 0.03, sunLight: 0.28, skyLight: 1.25, shadow: 0.2, murk: 0.85, murkColour: '#cfd3d7',
    fog: [300, 1500], falling: 'snow', density: 0.4, wet: 0.35, snowCover: 0.45},
  snow: {cover: 1.08, grey: 0.5, sun: 0, sunLight: 0.2, skyLight: 1.3, shadow: 0.12, murk: 1, murkColour: '#d3d7db',
    fog: [170, 950], falling: 'snow', density: 1, wet: 0.5, snowCover: 0.85},
};

/**
 * Wet surfaces: rain darkens dirt, concrete, paint and grass and takes the roughness off the hard ones. The felt turf
 * stays matte (STYLE_GUIDE: never shiny) and only darkens. Applied once to the scene's own materials, before the horses
 * are added (a plush horse in the rain stays a dry toy).
 */
export function wetSurfaces(scene: T.Scene, wet: number) {
  if (wet <= 0) return;
  const done = new Set<T.Material>();
  scene.traverse(o => {
    if (!(o instanceof T.Mesh)) return;
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      if (done.has(m) || !(m instanceof T.MeshStandardMaterial) || m.emissiveMap) continue;
      done.add(m);
      const felt = m instanceof T.MeshPhysicalMaterial && m.sheen > 0;
      m.color.multiplyScalar(1 - (felt ? 0.14 : 0.3) * wet);
      if (!felt) m.roughness *= 1 - 0.45 * wet;
    }
  });
}

/**
 * Rain or snow around the camera: drops (thin streaks) or flakes (soft round points) in a box that travels with the
 * camera, so the near air is always full and nothing is drawn far away (the fog stands in for distant rain). Positions
 * come from the race clock, so a frame is the same every time it is rendered (video export, verification).
 */
export function addPrecipitation(scene: T.Scene, look: WeatherLook) {
  if (!look.falling || look.density <= 0) return null;
  const snow = look.falling === 'snow';
  const box = snow ? new T.Vector3(100, 44, 100) : new T.Vector3(90, 44, 90);
  const count = Math.round((snow ? 42000 : 26000) * look.density);
  const random = seeded(snow ? 811 : 677);
  const vertices = snow ? 1 : 2;
  const base = new Float32Array(count * vertices * 3);
  const end = new Float32Array(count * vertices);
  const seed = new Float32Array(count * vertices);
  for (let i = 0; i < count; i++) {
    const x = random() * box.x, y = random() * box.y, z = random() * box.z, s = random();
    for (let v = 0; v < vertices; v++) {
      const k = i * vertices + v;
      base.set([x, y, z], k * 3);
      end[k] = v;
      seed[k] = s;
    }
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.BufferAttribute(base, 3));
  geometry.setAttribute('aEnd', new T.BufferAttribute(end, 1));
  geometry.setAttribute('aSeed', new T.BufferAttribute(seed, 1));
  const uniforms = {
    uTime: {value: 0},
    uCamera: {value: new T.Vector3()},
    uBox: {value: box},
    // Rain falls fast with a little wind; snow drifts down and sideways.
    uFall: {value: snow ? new T.Vector3(0.35, -1.15, 0.2) : new T.Vector3(1.4, -9, 0.7)},
    uLength: {value: 0.75},
    uOpacity: {value: snow ? 0.9 : 0.16 + 0.12 * look.density},
    uScale: {value: 500},
  };
  const wrap = `
    uniform float uTime;
    uniform vec3 uCamera;
    uniform vec3 uBox;
    uniform vec3 uFall;
    attribute float aSeed;
    varying float vFade;
    vec3 place(vec3 p) {
      vec3 travelled = p + uFall * uTime * (0.8 + 0.4 * aSeed);
      return uCamera + mod(travelled - uCamera, uBox) - 0.5 * uBox;
    }
    float fade(vec3 world) {
      // Thin out toward the box's walls (no visible edge) and right at the lens.
      vec3 d = abs(world - uCamera) / (0.5 * uBox);
      float edge = 1.0 - smoothstep(0.6, 1.0, max(max(d.x, d.y), d.z));
      return edge * smoothstep(0.4, 2.0, distance(world, uCamera));
    }`;
  const material = new T.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, fog: false,
    vertexShader: snow ? `${wrap}
      uniform float uScale;
      void main() {
        vec3 world = place(position);
        float t = uTime * (0.6 + 0.5 * aSeed) + aSeed * 40.0;
        world.xz += 0.35 * vec2(sin(t), cos(t * 0.8));
        vFade = fade(world);
        vec4 view = viewMatrix * vec4(world, 1.0);
        // Flakes as the camera sees them: soft clumps a few centimetres across, blurred by their fall.
        gl_PointSize = clamp((0.07 + 0.11 * aSeed) * uScale / -view.z, 1.5, 28.0);
        gl_Position = projectionMatrix * view;
      }` : `${wrap}
      uniform float uLength;
      attribute float aEnd;
      void main() {
        vec3 world = place(position) - normalize(uFall) * uLength * (0.6 + 0.6 * aSeed) * aEnd;
        vFade = fade(world) * (1.0 - 0.6 * aEnd);
        gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
      }`,
    fragmentShader: snow ? `
      uniform float uOpacity;
      varying float vFade;
      void main() {
        float r = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.15, r) * uOpacity * vFade;
        if (a < 0.01) discard;
        gl_FragColor = vec4(0.97, 0.98, 1.0, a);
      }` : `
      uniform float uOpacity;
      varying float vFade;
      void main() {
        gl_FragColor = vec4(0.86, 0.9, 0.94, uOpacity * vFade);
      }`,
  });
  const mesh = snow ? new T.Points(geometry, material) : new T.LineSegments(geometry, material);
  mesh.name = snow ? 'snowfall' : 'rainfall';
  mesh.frustumCulled = false;
  mesh.renderOrder = 50;
  scene.add(mesh);
  return {
    /** Moves the falling box with the camera to the race clock `time` (s); `pixelsHigh` sizes the flakes. */
    update(camera: T.PerspectiveCamera, time: number, pixelsHigh: number) {
      uniforms.uTime.value = time;
      uniforms.uCamera.value.copy(camera.position);
      uniforms.uScale.value = pixelsHigh / (2 * Math.tan(T.MathUtils.degToRad(camera.fov) / 2));
    },
    /** disposeScene only frees meshes and sprites. */
    dispose() {
      scene.remove(mesh);
      geometry.dispose();
      material.dispose();
    },
  };
}
