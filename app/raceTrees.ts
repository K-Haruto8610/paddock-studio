import * as T from 'three';
import type {TrackModel} from './trackGeometry';
import {seeded} from './scenery';
import {photoTextures} from './photoTextures';
import {foliage, type Foliage, type Species} from './raceSeason';
import {chainShader} from './surfaceFinish';
import {AVENUE_LINES, avenueRuns, walkRuns, type AvenueSite} from './raceAvenue';
import {barkTextures, foliageMaterial, leafAtlas, SHAPES, treeModel, type LeafKind, type TreeShape} from './treeModels';

type Tree = {position: T.Vector3; height: number; width: number; species: Species; shape: TreeShape; variant: number;
  jitter: number; tilt: number};

/** Layouts per shape: enough that neighbours in a row never look like copies. */
const VARIANTS = 3;

const WOOD_MIX: [Species, number][] = [['evergreen', 0.45], ['zelkova', 0.22], ['maple', 0.12], ['cherry', 0.13], ['ginkgo', 0.08]];

/** Bark colour of each shape (multiplies the procedural bark texture): grey zelkova, dark cherry, red-brown pine. */
const BARK: Record<TreeShape, string> = {
  zelkova: '#9a8f80', ginkgo: '#7d7264', cherry: '#5a4639', maple: '#8a8272', camphor: '#6e6253', pine: '#8c6650',
};
/**
 * Linear tints of the scanned barks: the scans were shot in their own light, most of them pale; these bring each one's
 * mean to the bark of the living tree (grey zelkova, dark red-brown cherry, grey-brown camphor and ginkgo, red-brown pine),
 * shifting the hue only halfway so the photograph's own colour survives. The ginkgo shares the camphor scan.
 */
const PHOTO_BARK: Record<TreeShape, [number, number, number]> = {
  zelkova: [0.55, 0.6, 0.66], cherry: [0.42, 0.37, 0.39], camphor: [0.38, 0.4, 0.42], ginkgo: [0.36, 0.4, 0.44],
  pine: [1, 0.92, 0.94], maple: [1.4, 1.5, 1.6],
};

function pick(r: number): Species {
  for (const [species, share] of WOOD_MIX) if ((r -= share) < 0) return species;
  return 'evergreen';
}

/** Evergreens are camphor (クスノキ) or, one in three, pine (マツ); the others are their own shape. */
const shapeOf = (species: Species, r: number): TreeShape =>
  species === 'evergreen' ? (r < 0.67 ? 'camphor' : 'pine') : species;

/**
 * Where the trees stand. Two kinds of planting, as round a real course:
 * - avenues (並木): two even rows along a promenade just outside the outermost rail (app/raceAvenue.ts), the first row
 *   2 m from it, one species to a stretch, every 8.5 m along the ground (so the rows stay even round the turns). One tree
 *   in sixteen is a younger replacement, smaller than its neighbours;
 * - woods: mixed groves further out, mostly evergreen, three deep, leaning a little more.
 * Neither stands on ground the site keeps clear (the main stand and its apron).
 */
function plantTrees(site: AvenueSite, track: TrackModel, ground: (p: T.Vector3) => T.Vector3) {
  const trees: Tree[] = [];
  for (const [row, offset] of [AVENUE_LINES.front, AVENUE_LINES.back].entries()) {
    const random = seeded(9011 + row * 31);
    for (const {at, species} of walkRuns(avenueRuns(site, offset), 8.5, row ? 8.5 : 4.25)) {
      const shape = shapeOf(species, 0);
      const {height, width} = SHAPES[shape];
      const young = random() < 1 / 16 ? 0.62 + 0.15 * random() : 1;
      trees.push({position: ground(at), species, shape, variant: Math.floor(random() * VARIANTS),
        jitter: (random() - 0.5) * 8, tilt: 0.035 * random(),
        height: height * young * (0.92 + 0.16 * random()), width: width * young * (0.92 + 0.16 * random())});
    }
  }
  const clear = (p: T.Vector3) => !site.keepOut.some(out => out(p));
  for (let segment = 0; segment < 96; segment++) {
    const random = seeded(7283 + segment * 457);
    const distance = track.lap * segment / 96;
    for (const lane of [track.width + 168, track.width + 186, track.width + 204]) {
      for (let i = 0; i < 2; i++) {
        const species = pick(random());
        const shape = shapeOf(species, random());
        const {height, width} = SHAPES[shape];
        const size = 0.75 + 0.5 * random();
        const at = track.sample(distance + (random() - 0.5) * 22, lane + random() * 16).point;
        const tree = {position: ground(at), species, shape,
          variant: Math.floor(random() * VARIANTS), jitter: (random() - 0.5) * 14, tilt: 0.07 * random(),
          height: height * size, width: width * size * (0.85 + 0.3 * random())};
        if (clear(at)) trees.push(tree);
      }
    }
  }
  return trees;
}

/**
 * What the camera looks at, for the trees to step aside: the avenue stands 2 m from the rail, between the course and a
 * camera outside it. Each frame the scene sets `from` (the camera) and `to` (the field's centre); `reach` is how far
 * the view of the field widens toward it (its half spread).
 */
export type TreeSight = {from: T.Vector3; to: T.Vector3; reach: number};

/**
 * Fades out (by dithering, so it stays in the opaque pass) a tree that stands between the camera and the field and is tall
 * enough to cut the view of it, or that the camera is inside: as broadcast pictures leave out what stands in front of
 * the lens. Its shadow stays. `size` is the shape's nominal height and width; each instance's own comes from its scale.
 */
function sightFade<M extends T.Material>(material: M, sight: TreeSight, size: {height: number; width: number}) {
  return chainShader(material, 'tree-sight', shader => {
    shader.uniforms.uSightFrom = {value: sight.from};
    shader.uniforms.uSightTo = {value: sight.to};
    shader.uniforms.uSightReach = {get value() { return sight.reach; }};
    shader.uniforms.uTreeSize = {value: new T.Vector2(size.height, size.width)};
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
        uniform vec3 uSightFrom;
        uniform vec3 uSightTo;
        uniform float uSightReach;
        uniform vec2 uTreeSize;
        varying float vSightHide;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vSightHide = 0.0;
        #ifdef USE_INSTANCING
          vec3 treeBase = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
          float treeTop = treeBase.y + uTreeSize.x * length(instanceMatrix[1].xyz);
          float crown = 0.5 * uTreeSize.y * length(instanceMatrix[0].xyz);
          vec2 view = uSightTo.xz - uSightFrom.xz;
          float along = dot(treeBase.xz - uSightFrom.xz, view) / max(dot(view, view), 1.0);
          float off = length(uSightFrom.xz + view * clamp(along, 0.0, 1.0) - treeBase.xz);
          float reach = crown + 1.5 + uSightReach * clamp(along, 0.0, 1.0);
          // The lowest line of sight, to the horses' feet: hidden only if the tree reaches up into it.
          float lineY = mix(uSightFrom.y, uSightTo.y - 2.0, clamp(along, 0.0, 1.0));
          float between = step(0.0, along) * (1.0 - smoothstep(0.97, 1.03, along));
          float across = 1.0 - smoothstep(reach, reach + 2.0, off);
          float under = 1.0 - smoothstep(treeTop, treeTop + 1.5, lineY);
          float inside = 1.0 - smoothstep(crown + 2.0, crown + 4.0, length(treeBase.xz - uSightFrom.xz));
          vSightHide = max(between * across * under, inside * step(uSightFrom.y, treeTop + 2.0));
        #endif`)
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying float vSightHide;`)
      .replace('void main() {', `void main() {
        if (vSightHide > 0.001 && fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) < vSightHide) discard;`);
  });
}

/**
 * Leaf cards for a set of instances sharing one model: the cards cloned for the instance attributes (season per instance).
 */
export function leafInstances(cards: T.BufferGeometry, kind: LeafKind, snow: number, looks: Foliage[], matrices: T.Matrix4[],
  seed: number) {
  const geometry = cards.clone();
  const n = looks.length;
  const state = new Float32Array(n * 4);
  const main = new Float32Array(n * 3), patch = new Float32Array(n * 3), blossom = new Float32Array(n * 3);
  const random = seeded(seed);
  looks.forEach((look, i) => {
    // A little more or less light and saturation from plant to plant.
    const vary = 0.9 + 0.2 * random();
    state.set([look.leaf, look.twig, look.bloom, random()], i * 4);
    main.set([look.main.r * vary, look.main.g * vary, look.main.b * vary], i * 3);
    patch.set([look.patch.r * vary, look.patch.g * vary, look.patch.b * vary], i * 3);
    blossom.set([look.blossom.r, look.blossom.g, look.blossom.b], i * 3);
  });
  geometry.setAttribute('aState', new T.InstancedBufferAttribute(state, 4));
  geometry.setAttribute('aMain', new T.InstancedBufferAttribute(main, 3));
  geometry.setAttribute('aPatch', new T.InstancedBufferAttribute(patch, 3));
  geometry.setAttribute('aBlossom', new T.InstancedBufferAttribute(blossom, 3));
  const mesh = new T.InstancedMesh(geometry, foliageMaterial(leafAtlas(kind), snow), n);
  matrices.forEach((m, i) => mesh.setMatrixAt(i, m));
  mesh.computeBoundingSphere();
  return mesh;
}

/**
 * The bark material of a shape, tinted per instance: the species' scanned bark (Poly Haven; colour, roughness, relief) at
 * its true size on the tubes' uv (0.6 m round, 1.2 m up per repeat), or the procedural bark when the scans are not loaded.
 */
export function barkMaterial(shape: TreeShape) {
  const scan = photoTextures()?.bark[shape];
  if (scan) {
    const copy = (texture: T.Texture) => {
      const c = texture.clone();
      c.repeat.set(0.6 / scan.size, 1.2 / scan.size);
      return c;
    };
    return new T.MeshStandardMaterial({color: '#fff', map: copy(scan.map), normalMap: copy(scan.normalMap),
      roughnessMap: copy(scan.roughnessMap), roughness: 1, normalScale: new T.Vector2(1.3, 1.3)});
  }
  const {map, normalMap} = barkTextures();
  return new T.MeshStandardMaterial({color: '#fff', map, normalMap, normalScale: new T.Vector2(1.2, 1.2), roughness: 0.95});
}

/**
 * The trees round the course, dressed for the race day (`day` of the year) and the weather's snow cover: for each shape and
 * layout, one instanced skeleton of bark and one of leaf cards (seasonal leaves, blossom, or bare twigs).
 */
export function addTrees(scene: T.Scene, site: AvenueSite, track: TrackModel, ground: (p: T.Vector3) => T.Vector3,
  day: number, snow: number) {
  const trees = plantTrees(site, track, ground);
  const sight: TreeSight = {from: new T.Vector3(0, 1e4, 0), to: new T.Vector3(), reach: 14};
  const groups = new Map<string, number[]>();
  trees.forEach((tree, i) => {
    const key = `${tree.shape}:${tree.variant}`;
    groups.set(key, [...(groups.get(key) ?? []), i]);
  });
  const random = seeded(2203);
  const instance = new T.Object3D();
  const matrices = trees.map(({position, height, width, shape, tilt}, i) => {
    const nominal = SHAPES[shape];
    instance.position.copy(position);
    instance.rotation.set(0, 0, 0);
    // Turned any way round, leaning a little toward a random side.
    instance.rotateY(i * 2.39996);
    instance.rotateX(tilt * Math.cos(i * 1.3));
    instance.rotateZ(tilt * Math.sin(i * 1.3));
    instance.scale.set(width / nominal.width, height / nominal.height, width / nominal.width);
    instance.updateMatrix();
    return instance.matrix.clone();
  });
  const woods = new Map<TreeShape, T.MeshStandardMaterial>();
  const photo = !!photoTextures();
  for (const [key, members] of groups) {
    const [shape, variant] = key.split(':') as [TreeShape, string];
    const model = treeModel(shape, Number(variant));
    if (!woods.has(shape)) woods.set(shape, sightFade(barkMaterial(shape), sight, SHAPES[shape]));
    const bark = new T.InstancedMesh(model.wood, woods.get(shape)!, members.length);
    const tint = new T.Color();
    members.forEach((tree, k) => {
      bark.setMatrixAt(k, matrices[tree]);
      (photo ? tint.setRGB(...PHOTO_BARK[shape]) : tint.set(BARK[shape])).multiplyScalar(0.88 + 0.24 * random());
      bark.setColorAt(k, tint);
    });
    bark.computeBoundingSphere();
    bark.name = `tree-bark-${key}`;
    const looks = members.map(tree => foliage(trees[tree].species, day, trees[tree].jitter));
    const leaves = leafInstances(model.leaves, shape, snow, looks, members.map(tree => matrices[tree]), 2203 + members[0]);
    leaves.name = `tree-leaves-${key}`;
    sightFade(leaves.material as T.Material, sight, SHAPES[shape]);
    scene.add(bark, leaves);
  }
  return {count: trees.length, sight};
}
