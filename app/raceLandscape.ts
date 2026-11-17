import * as T from 'three';
import type {TrackModel} from './trackGeometry';
import type {Terrain} from './sceneBuilders';
import {seeded} from './scenery';

export function addRaceSky(scene: T.Scene, sunOffset: T.Vector3) {
  const summer = T.MathUtils.smoothstep(sunOffset.clone().normalize().y, 0.3, 1);
  const horizon = new T.Color('#e1e5d5').lerp(new T.Color('#e1edf3'), summer);
  const zenith = new T.Color('#709bb9').lerp(new T.Color('#4999cf'), summer);
  const sky = new T.Mesh(new T.SphereGeometry(4400, 40, 20), new T.ShaderMaterial({
    side: T.BackSide,
    depthWrite: false,
    toneMapped: false,
    uniforms: {
      horizon: {value: horizon}, zenith: {value: zenith},
      sun: {value: sunOffset.clone().normalize()},
    },
    vertexShader: `
      varying vec3 direction;
      void main() {
        direction = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      varying vec3 direction;
      uniform vec3 horizon;
      uniform vec3 zenith;
      uniform vec3 sun;
      void main() {
        vec3 ray = normalize(direction);
        vec3 colour = mix(horizon, zenith, smoothstep(-0.02, 0.66, ray.y));
        float glow = pow(max(dot(ray, sun), 0.0), 18.0);
        colour = mix(colour, vec3(1.0, 0.82, 0.55), 0.28 * glow);
        gl_FragColor = vec4(colour, 1.0);
        #include <colorspace_fragment>
      }`,
  }));
  sky.renderOrder = -100;
  sky.frustumCulled = false;
  scene.add(sky);
  scene.fog = new T.Fog(horizon, 720, 2450);
  return sky;
}

export function addRaceLandscape(scene: T.Scene, track: TrackModel, terrain: Terrain, floor: number) {
  const tree = new T.IcosahedronGeometry(1, 1);
  const leaf = new T.MeshStandardMaterial({color: '#fff', roughness: 1});
  const trunk = new T.MeshStandardMaterial({color: '#665845', roughness: 1});
  const box = new T.BoxGeometry(1, 1, 1);
  const shrub = new T.MeshStandardMaterial({color: '#55764b', roughness: 1});
  const concrete = new T.MeshStandardMaterial({color: '#c4bca8', roughness: 0.86});
  const seats = new T.MeshStandardMaterial({color: '#809491', roughness: 0.95});
  const roof = new T.MeshStandardMaterial({color: '#405d62', roughness: 0.85});

  const at = (distance: number, lane: number) => {
    const frame = track.sample(distance, lane);
    frame.point.y = Math.max(floor, terrain.heightAt(frame.point.x, frame.point.z) - 0.3);
    return frame;
  };
  const piece = (group: T.Group, geometry: T.BufferGeometry, material: T.Material,
    x: number, y: number, z: number, w: number, h: number, d: number) => {
    const mesh = new T.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.scale.set(w, h, d);
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
  for (let row = 0; row < 5; row++) {
    piece(stand, box, seats, 0, 6 + row * 0.9, (row - 2) * 2.2, 93, 1.2, 5);
  }
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
  const crowns = new T.InstancedMesh(tree, leaf, trees.length);
  const trunks = new T.InstancedMesh(box, trunk, trees.length);
  const instance = new T.Object3D();
  trees.forEach(({position, height, width, color}, i) => {
    instance.position.copy(position).y += height * 0.72;
    instance.scale.set(width, height * 0.55, width);
    instance.rotation.y = i;
    instance.updateMatrix();
    crowns.setMatrixAt(i, instance.matrix);
    crowns.setColorAt(i, color);
    instance.position.copy(position).y += height * 0.2;
    instance.scale.set(width * 0.15, height * 0.6, width * 0.15);
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
