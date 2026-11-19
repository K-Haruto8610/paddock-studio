import * as T from 'three';
import type {TrackModel} from './trackGeometry';
import type {Terrain} from './sceneBuilders';
import {seeded} from './scenery';
import {standSeatGeometry} from './standSeat';
import {bevelBox, CONCRETE_TILE, foliageGeometry, leafNormalMap, standMaterials, trunkGeometry} from './surfaceFinish';

/** Horizon and zenith colours of the day's sky: pale haze in winter, a deeper blue near the summer sun. */
function skyColours(sunOffset: T.Vector3) {
  const summer = T.MathUtils.smoothstep(sunOffset.clone().normalize().y, 0.3, 1);
  return {
    horizon: new T.Color('#e1e5d5').lerp(new T.Color('#e1edf3'), summer),
    zenith: new T.Color('#709bb9').lerp(new T.Color('#4999cf'), summer),
    summer,
  };
}

/**
 * The sky dome: a hazy horizon rising to blue, the sun's warm glow, and soft clouds drifting at one height (fractal noise
 * on a flat layer, so they shrink toward the horizon, brighter on the sun's side and grey underneath).
 */
function skyDome(sunOffset: T.Vector3) {
  const {horizon, zenith, summer} = skyColours(sunOffset);
  const sky = new T.Mesh(new T.SphereGeometry(4400, 40, 20), new T.ShaderMaterial({
    side: T.BackSide,
    depthWrite: false,
    toneMapped: false,
    uniforms: {
      horizon: {value: horizon}, zenith: {value: zenith},
      sun: {value: sunOffset.clone().normalize()},
      cover: {value: 0.5 - 0.12 * summer},
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
        vec3 betaMTheta = vBetaM * hgPhase(c, 0.8);
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
        vec3 colour = mix(horizon, min(scattering(ray) * 0.62, vec3(0.95)), smoothstep(0.0, 0.3, ray.y));
        float glow = pow(max(dot(ray, sun), 0.0), 18.0);
        colour = mix(colour, vec3(1.0, 0.82, 0.55), 0.28 * glow);
        // Clouds on a flat layer: the further toward the horizon, the closer together and the fainter.
        if (ray.y > 0.0) {
          vec2 p = ray.xz / (ray.y + 0.12) * 0.55;
          float body = fbm(p + vec2(3.1, 1.7));
          float density = smoothstep(1.0 - cover - 0.1, 1.0 - cover + 0.18, body + 0.18 * fbm(p * 3.0 + 5.0));
          // Lit from the sun's side, shaded and a little grey underneath.
          float lit = clamp(0.62 + 0.9 * (fbm(p + sun.xz * 0.16 + vec2(3.1, 1.7)) - body), 0.0, 1.0);
          vec3 cloud = mix(vec3(0.66, 0.7, 0.76), vec3(1.0, 0.99, 0.96), lit);
          colour = mix(colour, cloud, density * 0.85 * smoothstep(0.02, 0.3, ray.y));
        }
        gl_FragColor = vec4(colour, 1.0);
        #include <colorspace_fragment>
      }`,
  }));
  sky.renderOrder = -100;
  sky.frustumCulled = false;
  return sky;
}

export function addRaceSky(scene: T.Scene, sunOffset: T.Vector3) {
  const sky = skyDome(sunOffset);
  scene.add(sky);
  scene.fog = new T.Fog(skyColours(sunOffset).horizon, 720, 2450);
  return sky;
}

/**
 * Ambient light taken from the sky itself, so the light the surfaces and horses receive from around them is the colour of
 * the sky above them and the haze at the horizon (STYLE_GUIDE: one sky, one light for everything).
 * The texture is the caller's to dispose.
 */
export function skyEnvironment(renderer: T.WebGLRenderer, sunOffset: T.Vector3) {
  const dome = new T.Scene();
  const sky = skyDome(sunOffset);
  dome.add(sky);
  const pmrem = new T.PMREMGenerator(renderer);
  const target = pmrem.fromScene(dome, 0.04, 1, 10000);
  pmrem.dispose();
  sky.geometry.dispose();
  (sky.material as T.Material).dispose();
  return target.texture;
}

/**
 * Rows of the Blender stadium seat (app/standSeat.ts) on the main stand's five terraces: two rows on each 2.2 m tread, four on
 * the deep top one, 0.5 m apart, with an aisle every 11 m (where the roof's columns stand). One InstancedMesh, in the stand's
 * frame; each seat's colour drifts a little, as moulded seats fade unevenly in the sun.
 */
function seatRows(geometry: T.BufferGeometry) {
  const places: [number, number, number][] = [];
  for (let terrace = 0; terrace < 5; terrace++) {
    const front = (terrace - 2) * 2.2 - 2.5, floor = 6.6 + terrace * 0.9;
    for (let line = 0; line < (terrace === 4 ? 4 : 2); line++) {
      for (let x = -46; x <= 46; x += 0.5) {
        if (Math.abs(x - Math.round(x / 11) * 11) < 0.8) continue;   // aisles
        places.push([x, floor, front + 0.35 + line * 1.1]);
      }
    }
  }
  const material = new T.MeshStandardMaterial({color: '#ffffff', vertexColors: true, roughness: 0.72, side: T.DoubleSide});
  const seats = new T.InstancedMesh(geometry, material, places.length);
  seats.name = 'stand-seats';
  const random = seeded(5531);
  const base = new T.Color('#6f8a88'), color = new T.Color();
  const matrix = new T.Matrix4();
  places.forEach(([x, y, z], i) => {
    seats.setMatrixAt(i, matrix.makeTranslation(x, y, z));
    color.copy(base).offsetHSL(0, -0.04 * random(), 0.05 * (random() - 0.5));
    seats.setColorAt(i, color);
  });
  seats.receiveShadow = true;
  return seats;
}

export function addRaceLandscape(scene: T.Scene, track: TrackModel, terrain: Terrain, floor: number) {
  const tree = new T.IcosahedronGeometry(1, 1);
  // Broadleaf crowns: lumpy lobes with leaf-clump relief (colour per tree from the instance, shading from the vertices).
  const leaf = new T.MeshStandardMaterial({color: '#fff', roughness: 1, vertexColors: true, normalMap: leafNormalMap(), normalScale: new T.Vector2(0.7, 0.7)});
  const trunk = new T.MeshStandardMaterial({color: '#665845', roughness: 1});
  const box = new T.BoxGeometry(1, 1, 1);
  const shrub = new T.MeshStandardMaterial({color: '#55764b', roughness: 1});
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
  const groupAt = (distance: number, lane: number) => {
    const {point, tangent} = at(distance, lane);
    const group = new T.Group();
    group.position.copy(point);
    group.rotation.y = -Math.atan2(tangent.z, tangent.x);
    scene.add(group);
    return group;
  };

  const stand = groupAt(0, track.width + 92);
  piece(stand, box, concrete, 0, 2.5, 0, 96, 6, 20);
  // Terraces: five steps rising away from the track (-z). With the Blender seat, the steps are concrete and carry rows of
  // seats; without it, they are flat seat blocks as before.
  const seat = standSeatGeometry();
  for (let row = 0; row < 5; row++) {
    piece(stand, box, seat ? concrete : seats, 0, 6 + row * 0.9, (row - 2) * 2.2, 93, 1.2, 5);
  }
  if (seat) stand.add(seatRows(seat));
  for (const x of [-44, -22, 0, 22, 44]) piece(stand, box, roof, x, 10, 6, 0.8, 12, 0.8);
  piece(stand, box, roof, 0, 16, 2, 98, 1.1, 23);
  piece(stand, box, seats, 0, 15, -9, 98, 1.2, 0.5);
  const infield = groupAt(track.lap * 0.68, -115);
  piece(infield, box, concrete, 0, 2.4, 0, 47, 5, 16);
  piece(infield, box, seats, 0, 4.6, 0, 38, 2.4, 15);
  piece(infield, box, roof, 0, 6.2, 1, 50, 1.3, 19);

  const trees: {position: T.Vector3; height: number; width: number; color: T.Color}[] = [];
  for (let segment = 0; segment < 48; segment++) {
    const random = seeded(7283 + segment * 457);
    const distance = track.lap * segment / 48;
    const nearStand = Math.min(distance, track.lap - distance) < 90;
    for (const lane of [track.width + (nearStand ? 153 : 106), track.width + 176]) {
      for (let i = 0; i < 3; i++) {
        const position = at(distance + (random() - 0.5) * 29, lane + random() * 24).point;
        trees.push({position, height: 8 + random() * 10, width: 3.2 + random() * 4.6,
          color: new T.Color().setHSL(0.24 + random() * 0.08, 0.19 + random() * 0.15, 0.26 + random() * 0.15)});
      }
    }
  }
  const crowns = new T.InstancedMesh(foliageGeometry(), leaf, trees.length);
  const trunks = new T.InstancedMesh(trunkGeometry(), trunk, trees.length);
  const instance = new T.Object3D();
  trees.forEach(({position, height, width, color}, i) => {
    instance.position.copy(position).y += height * 0.72;
    instance.scale.set(width * 0.9, height * 0.5, width * 0.9);
    instance.rotation.y = i;
    instance.updateMatrix();
    crowns.setMatrixAt(i, instance.matrix);
    crowns.setColorAt(i, color);
    instance.position.copy(position).y += height * 0.2;
    instance.scale.set(width * 0.16, height * 0.6, width * 0.16);
    instance.updateMatrix();
    trunks.setMatrixAt(i, instance.matrix);
  });
  scene.add(trunks, crowns);

  const hills = new T.MeshBasicMaterial({color: '#97aba7', toneMapped: false});
  for (let i = 0; i < 28; i++) {
    const random = seeded(3271 + i * 23);
    const position = track.sample(track.lap * i / 28, 710 + random() * 280).point;
    const hill = new T.Mesh(tree, hills);
    hill.position.set(position.x, floor - 10, position.z);
    hill.rotation.y = i;
    hill.scale.set(110 + random() * 140, 27 + random() * 39, 110 + random() * 135);
    scene.add(hill);
  }

  const board = groupAt(-track.lap * 0.45, -50);
  piece(board, box, roof, 0, 7, 0, 22, 7, 1.8);
  piece(board, box, seats, 0, 7, 0.94, 19, 4.8, 0.08);
  for (const x of [-8, 8]) piece(board, box, concrete, x, 2, 0, 1, 5, 1);
  for (const fraction of [0.04, 0.16, 0.39, 0.57, 0.76, 0.92]) {
    const bed = groupAt(track.lap * fraction, -38);
    piece(bed, box, concrete, 0, 0.2, 0, 24, 0.6, 7);
    for (const x of [-9, -6, -3, 0, 3, 6, 9]) piece(bed, tree, shrub, x, 1.35, 0, 2, 1.8, 2.6);
  }
}
