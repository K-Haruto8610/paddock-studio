import * as T from 'three';
import type {TrackModel} from './trackGeometry';
import type {Terrain} from './sceneBuilders';
import {seeded} from './scenery';
import {standSeatGeometry} from './standSeat';
import {bevelBox, CONCRETE_TILE, standMaterials} from './surfaceFinish';
import {addTrees} from './raceTrees';
import {addMainStand} from './raceStands';
import type {AvenueSite} from './raceAvenue';
import {addShrubs} from './raceShrubs';
import {WEATHER_LOOKS, type WeatherLook} from './raceWeather';

const FAIR = WEATHER_LOOKS.clear;

/**
 * Horizon and zenith colours of the day's sky: a pale blue-grey haze (aerial perspective: distant land fades toward it),
 * clearer in summer; under cloud the haze turns to the weather's grey.
 */
function skyColours(sunOffset: T.Vector3, look: WeatherLook = FAIR) {
  // The sun at post time stands about 8 degrees up in late December and 41 in June (sin 0.14 ... 0.66).
  const summer = T.MathUtils.smoothstep(sunOffset.clone().normalize().y, 0.18, 0.62);
  return {
    horizon: new T.Color('#c9d5de').lerp(new T.Color('#c4d6e6'), summer).lerp(new T.Color(look.murkColour), look.murk),
    zenith: new T.Color('#709bb9').lerp(new T.Color('#4999cf'), summer),
    summer,
  };
}

/**
 * The sky dome: a hazy horizon rising to blue, the sun (its disc, a bright corona and a wide warm glow), and soft clouds
 * drifting at one height (fractal noise on a flat layer, so they shrink toward the horizon, brighter on the sun's side and
 * grey underneath). The weather sets how much cloud, how grey, and how much of the sun shows through.
 */
function skyDome(sunOffset: T.Vector3, look: WeatherLook = FAIR) {
  const {horizon, zenith, summer} = skyColours(sunOffset, look);
  const sky = new T.Mesh(new T.SphereGeometry(4400, 40, 20), new T.ShaderMaterial({
    side: T.BackSide,
    depthWrite: false,
    toneMapped: false,
    uniforms: {
      horizon: {value: horizon}, zenith: {value: zenith},
      sun: {value: sunOffset.clone().normalize()},
      cover: {value: look.cover < 0 ? 0.5 - 0.12 * summer : look.cover},
      grey: {value: look.grey}, sunShow: {value: look.sun},
      turbidity: {value: 2}, rayleigh: {value: 2.4}, mieCoefficient: {value: 0.003},
    },
    // Atmospheric scattering after Preetham et al., as in three's Sky (MIT): the blue of the sky and the pale haze toward the
    // horizon come from Rayleigh and Mie scattering of the sun's light, not from a painted gradient.
    vertexShader: `
      uniform vec3 sun;
      uniform float rayleigh;
      uniform float turbidity;
      uniform float mieCoefficient;
      varying vec3 direction;
      varying float vSunfade;
      varying vec3 vBetaR;
      varying vec3 vBetaM;
      varying float vSunE;
      const float e = 2.718281828459045;
      const vec3 totalRayleigh = vec3(5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5);
      const vec3 MieConst = vec3(1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14);
      void main() {
        direction = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        float zenithCos = clamp(sun.y, -1.0, 1.0);
        vSunE = 1000.0 * max(0.0, 1.0 - pow(e, -((1.6110731556870734 - acos(zenithCos)) / 1.5)));
        vSunfade = 1.0 - clamp(1.0 - exp(sun.y * 400000.0 / 450000.0), 0.0, 1.0);
        vBetaR = totalRayleigh * (rayleigh - (1.0 - vSunfade));
        vBetaM = 0.434 * (0.2 * turbidity * 10E-18) * MieConst * mieCoefficient;
      }`,
    fragmentShader: `
      varying vec3 direction;
      uniform vec3 horizon;
      uniform vec3 zenith;
      uniform vec3 sun;
      uniform float cover;
      uniform float grey;
      uniform float sunShow;
      varying float vSunfade;
      varying vec3 vBetaR;
      varying vec3 vBetaM;
      varying float vSunE;
      float rayleighPhase(float c) { return 0.05968310365946075 * (1.0 + c * c); }
      float hgPhase(float c, float g) { float g2 = g * g; return 0.07957747154594767 * ((1.0 - g2) / pow(1.0 - 2.0 * g * c + g2, 1.5)); }
      // The sky's colour looking along the ray: sunlight scattered into the line of sight.
      vec3 scattering(vec3 ray) {
        float zenith = acos(max(0.0, ray.y));
        float inv = 1.0 / (cos(zenith) + 0.15 * pow(93.885 - degrees(zenith), -1.253));
        vec3 Fex = exp(-(vBetaR * 8.4E3 * inv + vBetaM * 1.25E3 * inv));
        float c = dot(ray, sun);
        vec3 betaRTheta = vBetaR * rayleighPhase(c * 0.5 + 0.5);
        // The forward (Mie) glare stops growing 15 deg from the sun: closer in, the haze would be as white as the disc, which
        // would then vanish. The sun's own halo is drawn in main().
        vec3 betaMTheta = vBetaM * hgPhase(min(c, 0.966), 0.8);
        vec3 inscatter = vSunE * ((betaRTheta + betaMTheta) / (vBetaR + vBetaM));
        vec3 Lin = pow(inscatter * (1.0 - Fex), vec3(1.5));
        Lin *= mix(vec3(1.0), pow(inscatter * Fex, vec3(0.5)), clamp(pow(1.0 - sun.y, 5.0), 0.0, 1.0));
        vec3 colour = (Lin + vec3(0.1) * Fex) * 0.04 + vec3(0.0, 0.0003, 0.00075);
        return pow(colour, vec3(1.0 / (1.2 + 1.2 * vSunfade)));
      }
      float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float noise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
      }
      float fbm(vec2 p) {
        float v = 0.0, a = 0.5;
        for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; }
        return v;
      }
      void main() {
        vec3 ray = normalize(direction);
        // Scattered sky above, the haze colour (the fog's) at and below the horizon, so the two meet without a seam.
        // Bright sky rolls off softly toward 0.84 instead of clipping, so the sun's disc (1.0) stays brighter than its haze.
        vec3 bright = scattering(ray) * 0.62;
        vec3 over = max(bright - 0.7, 0.0);
        bright = min(bright, vec3(0.7)) + 0.14 * (1.0 - exp(-over / 0.14));
        vec3 colour = mix(horizon, bright, smoothstep(0.0, 0.16, ray.y));
        float facing = max(dot(ray, sun), 0.0);
        float glow = pow(facing, 18.0);
        colour = mix(colour, vec3(1.0, 0.82, 0.55), 0.2 * glow * (0.4 + 0.6 * sunShow));
        // The sun: a disc about twice its true size (0.5 deg; true size reads as a dot on a phone), a tight bright corona
        // round it, both behind the clouds.
        float disc = smoothstep(0.999955, 0.999975, facing);
        float corona = pow(facing, 2500.0) * 0.55 + pow(facing, 300.0) * 0.22;
        float density = 0.0;
        // Clouds on a flat layer: the further toward the horizon, the closer together and the fainter.
        if (ray.y > 0.0) {
          vec2 p = ray.xz / (ray.y + 0.12) * 0.55;
          float body = fbm(p + vec2(3.1, 1.7));
          density = smoothstep(1.0 - cover - 0.1, 1.0 - cover + 0.18, body + 0.18 * fbm(p * 3.0 + 5.0));
          // Lit from the sun's side, shaded and a little grey underneath; rain cloud is heavier and darker.
          float lit = clamp(0.62 + 0.9 * (fbm(p + sun.xz * 0.16 + vec2(3.1, 1.7)) - body), 0.0, 1.0);
          vec3 cloud = mix(vec3(0.66, 0.7, 0.76), vec3(1.0, 0.99, 0.96), lit);
          cloud = mix(cloud, mix(vec3(0.5, 0.53, 0.57), vec3(0.74, 0.76, 0.78), lit), grey);
          density *= smoothstep(0.02, 0.3, ray.y);
          // On a fine day the sun is out: the clouds thin out gradually toward it (over about 30 deg, so no hole shows) and
          // none sits on the disc.
          density *= 1.0 - smoothstep(0.5, 1.0, sunShow) * smoothstep(0.866, 0.9976, facing);
          colour = mix(colour, cloud, density * mix(0.85, 1.0, grey));
        }
        // Through thin cloud the sun is a bright smudge; through a closed deck, nothing.
        float through = sunShow * (1.0 - 0.85 * density);
        colour = mix(colour, vec3(1.0, 0.98, 0.92), clamp(disc * through + corona * through, 0.0, 1.0));
        colour += vec3(0.35, 0.33, 0.3) * pow(facing, 40.0) * sunShow * density * 0.5;
        gl_FragColor = vec4(colour, 1.0);
        #include <colorspace_fragment>
      }`,
  }));
  sky.renderOrder = -100;
  sky.frustumCulled = false;
  return sky;
}

export function addRaceSky(scene: T.Scene, sunOffset: T.Vector3, look: WeatherLook = FAIR) {
  const sky = skyDome(sunOffset, look);
  scene.add(sky);
  scene.fog = new T.Fog(skyColours(sunOffset, look).horizon, ...look.fog);
  return sky;
}

/**
 * Ambient light taken from the sky itself, so the light the surfaces and horses receive from around them is the colour of
 * the sky above them and the haze at the horizon (STYLE_GUIDE: one sky, one light for everything).
 * The texture is the caller's to dispose.
 */
export function skyEnvironment(renderer: T.WebGLRenderer, sunOffset: T.Vector3, look: WeatherLook = FAIR) {
  const dome = new T.Scene();
  const sky = skyDome(sunOffset, look);
  dome.add(sky);
  const pmrem = new T.PMREMGenerator(renderer);
  const target = pmrem.fromScene(dome, 0.04, 1, 10000);
  pmrem.dispose();
  sky.geometry.dispose();
  (sky.material as T.Material).dispose();
  return target.texture;
}

type Terrace = {floor: number; front: number; lines: number};

/**
 * Rows of the Blender stadium seat (app/standSeat.ts) on a stand's terraces: `lines` rows on each tread from its front edge,
 * 1.1 m apart, seats 0.5 m apart across `half` metres either side of the centre, with an aisle every `aisle` metres (where
 * the roof's columns stand). One InstancedMesh, in the stand's frame (track at -z); each seat's colour drifts a little, as
 * moulded seats fade unevenly in the sun.
 */
function seatRows(geometry: T.BufferGeometry, terraces: Terrace[], half: number, aisle: number, tint: string, seed: number) {
  const places: [number, number, number][] = [];
  for (const {floor, front, lines} of terraces) {
    for (let line = 0; line < lines; line++) {
      for (let x = -half; x <= half; x += 0.5) {
        if (Math.abs(x - Math.round(x / aisle) * aisle) < 0.8) continue;   // aisles
        places.push([x, floor, front + 0.35 + line * 1.1]);
      }
    }
  }
  const material = new T.MeshStandardMaterial({color: '#ffffff', vertexColors: true, roughness: 0.72, side: T.DoubleSide});
  const seats = new T.InstancedMesh(geometry, material, places.length);
  seats.name = 'stand-seats';
  const random = seeded(seed);
  const base = new T.Color(tint), color = new T.Color();
  const matrix = new T.Matrix4();
  places.forEach(([x, y, z], i) => {
    seats.setMatrixAt(i, matrix.makeTranslation(x, y, z));
    color.copy(base).offsetHSL(0, -0.04 * random(), 0.05 * (random() - 0.5));
    seats.setColorAt(i, color);
  });
  seats.receiveShadow = true;
  return seats;
}

/**
 * The infield board's LED face: a dark panel of round lamps, 220 x 52, lit amber for the title and white for the lines
 * (race, course, weather and going, as the boards show them), dim where unlit. Emissive, so it reads in any light.
 */
function ledBoard(title: string, lines: string[]) {
  const cols = 220, rows = 52, pitch = 6;
  const text = document.createElement('canvas');
  text.width = cols;
  text.height = rows;
  const g = text.getContext('2d', {willReadFrequently: true})!;
  g.fillStyle = '#000';
  g.fillRect(0, 0, cols, rows);
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.font = 'bold 15px sans-serif';
  g.fillStyle = '#ffb030';
  g.fillText(title, cols / 2, 10, cols - 8);
  g.font = 'bold 12px sans-serif';
  g.fillStyle = '#f2f2ea';
  lines.slice(0, 2).forEach((line, k) => g.fillText(line, cols / 2, 27 + k * 15, cols - 8));
  const lit = g.getImageData(0, 0, cols, rows).data;
  const face = document.createElement('canvas');
  face.width = cols * pitch;
  face.height = rows * pitch;
  const out = face.getContext('2d')!;
  const image = out.createImageData(face.width, face.height);
  for (let y = 0; y < face.height; y++)
    for (let x = 0; x < face.width; x++) {
      const c = Math.floor(x / pitch), r = Math.floor(y / pitch);
      const lamp = Math.hypot(x % pitch - pitch / 2 + 0.5, y % pitch - pitch / 2 + 0.5) < pitch * 0.38;
      const k = (r * cols + c) * 4, o = (y * face.width + x) * 4;
      // Lamps take the lit colour (thresholded, as an LED is on or off); unlit lamps are a faint grey dot.
      const on = Math.max(lit[k], lit[k + 1], lit[k + 2]) > 110;
      for (let ch = 0; ch < 3; ch++) image.data[o + ch] = lamp ? (on ? Math.min(255, lit[k + ch] * 1.15) : 16) : 0;
      image.data[o + 3] = 255;
    }
  out.putImageData(image, 0, 0);
  const texture = new T.CanvasTexture(face);
  texture.colorSpace = T.SRGBColorSpace;
  texture.anisotropy = 8;
  return new T.MeshStandardMaterial({color: '#060606', roughness: 0.3, emissive: '#ffffff', emissiveMap: texture, emissiveIntensity: 1.3});
}

export type LandscapeOptions = {
  venueCode: string;
  /** Every course drawn for the race, the one outermost along the home straight, and the pockets drawn. */
  courses: TrackModel[];
  turf: TrackModel;
  pockets: AvenueSite['pockets'];
  /** Points the avenue keeps clear of (the start sign beside the rail). */
  clear: T.Vector3[];
  /** Day of the year of the race (app/raceSeason.ts dresses the trees for it). */
  day: number;
  look: WeatherLook;
  /** What the infield board shows: a title and up to two lines. */
  board: {title: string; lines: string[]};
};

export function addRaceLandscape(scene: T.Scene, track: TrackModel, terrain: Terrain, floor: number, options: LandscapeOptions) {
  const {look} = options;
  const tree = new T.IcosahedronGeometry(1, 1);
  const box = new T.BoxGeometry(1, 1, 1);
  const soil = new T.MeshStandardMaterial({color: '#3b2f26', roughness: 1});
  // Scanned concrete panels and corrugated roofing (procedural concrete if the photos did not load).
  const {concrete, roof} = standMaterials();
  const seats = new T.MeshStandardMaterial({color: '#809491', roughness: 0.95});

  const at = (distance: number, lane: number) => {
    const frame = track.sample(distance, lane);
    frame.point.y = Math.max(floor, terrain.heightAt(frame.point.x, frame.point.z) - 0.3);
    return frame;
  };
  const piece = (group: T.Group, geometry: T.BufferGeometry, material: T.Material,
    x: number, y: number, z: number, w: number, h: number, d: number) => {
    // Boxes get small bevelled edges (and uv in metres); anything else keeps the unit-geometry scaling.
    const bevelled = geometry === box ? bevelBox(w, h, d, 0.2, (material.userData.tile as number | undefined) ?? CONCRETE_TILE) : null;
    const mesh = new T.Mesh(bevelled ?? geometry, material);
    mesh.position.set(x, y, z);
    if (!bevelled) mesh.scale.set(w, h, d);
    group.add(mesh);
    return mesh;
  };
  // A building's frame: x along the course, the track at -z. Taken from the direction to the course's middle lane, not from
  // the tangent, which turns round between left- and right-handed courses.
  const groupAt = (distance: number, lane: number) => {
    const {point} = at(distance, lane);
    const toward = track.sample(distance, track.width / 2).point.sub(point).setY(0).normalize();
    const group = new T.Group();
    group.position.copy(point);
    group.rotation.y = Math.atan2(-toward.x, -toward.z);
    scene.add(group);
    return group;
  };

  // The venue's main stand (app/raceStands.ts) outside the home straight, with its apron and crowd.
  const main = addMainStand(scene, options.venueCode, options.turf, options.courses, terrain.heightAt, options.day, look);
  const seat = standSeatGeometry();

  // The infield stand faces out across the course: a front walk, three concrete steps of seats, a concourse behind, and a
  // roof on three columns.
  const infield = groupAt(track.lap * 0.68, -115);
  piece(infield, box, concrete, 0, 2.4, 0, 47, 5, 16);
  for (let row = 0; row < 3; row++) {
    const height = 0.5 + row * 0.8;
    piece(infield, box, seat ? concrete : seats, 0, 4.9 + height / 2, -3.6 + row * 2.4, 44, height, 2.4);
  }
  piece(infield, box, concrete, 0, 5.95, 5.2, 47, 2.1, 5.6);
  if (seat) {
    const terraces = [0, 1, 2].map(r => ({floor: 5.4 + r * 0.8, front: -4.8 + r * 2.4, lines: 2}));
    infield.add(seatRows(seat, terraces, 20.5, 10, '#7d8f72', 6203));
  }
  for (const x of [-21, 0, 21]) piece(infield, box, roof, x, 9.2, 4.6, 0.5, 4.6, 0.5);
  piece(infield, box, roof, 0, 11.7, 0.6, 50, 0.6, 13);

  const onGround = (p: T.Vector3) => p.clone().setY(Math.max(floor, terrain.heightAt(p.x, p.z) - 0.3));
  const site: AvenueSite = {
    tracks: options.courses, pockets: options.pockets,
    keepOut: [
      ...(main ? [main.keepOut] : []),
      (p: T.Vector3) => options.clear.some(c => (c.x - p.x) ** 2 + (c.z - p.z) ** 2 < 49),
    ],
  };
  const {sight} = addTrees(scene, site, track, onGround, options.day, look.snowCover);

  const hills = new T.MeshBasicMaterial({color: new T.Color('#97aba7').lerp(new T.Color('#d9dde0'), 0.7 * look.snowCover), toneMapped: false});
  for (let i = 0; i < 28; i++) {
    const random = seeded(3271 + i * 23);
    const position = track.sample(track.lap * i / 28, 710 + random() * 280).point;
    const hill = new T.Mesh(tree, hills);
    hill.position.set(position.x, floor - 10, position.z);
    hill.rotation.y = i;
    hill.scale.set(110 + random() * 140, 27 + random() * 39, 110 + random() * 135);
    scene.add(hill);
  }

  // The infield board: a framed LED face toward the course (-z), on two legs.
  const board = groupAt(-track.lap * 0.45, -50);
  piece(board, box, roof, 0, 7, 0, 22, 7, 1.8);
  const face = new T.Mesh(new T.PlaneGeometry(19, 4.8), ledBoard(options.board.title, options.board.lines));
  face.position.set(0, 7, -0.92);
  face.rotation.y = Math.PI;
  board.add(face);
  for (const x of [-8, 8]) piece(board, box, concrete, x, 2, 0, 1, 5, 1);
  // Infield beds: a concrete kerb round dark soil; the shrubs in them come from app/raceShrubs.ts.
  const beds: T.Matrix4[] = [];
  for (const fraction of [0.04, 0.16, 0.39, 0.57, 0.76, 0.92]) {
    const bed = groupAt(track.lap * fraction, -38);
    piece(bed, box, concrete, 0, 0.2, 0, 24, 0.6, 7);
    piece(bed, box, soil, 0, 0.5, 0, 23, 0.1, 6.2);
    bed.updateMatrixWorld(true);
    beds.push(bed.matrixWorld.clone().multiply(new T.Matrix4().makeTranslation(0, 0.52, 0)));
  }
  addShrubs(scene, site, onGround, beds, options.day, look.snowCover);
  return {sight, paved: main?.paved ?? (() => false)};
}
