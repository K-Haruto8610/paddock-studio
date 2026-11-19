import * as T from 'three';
import type {TrackModel} from './trackGeometry';
import {seeded} from './scenery';
import {foliage, type Species} from './raceSeason';
import {branchGeometry, foliageGeometry, leafNormalMap, trunkGeometry} from './surfaceFinish';

type Tree = {position: T.Vector3; height: number; width: number; species: Species; jitter: number};

/** Height and crown width (m) of each species, before the small per-tree variation. */
const SIZE: Record<Species, [number, number]> = {
  evergreen: [11, 6.8], zelkova: [12, 6.6], ginkgo: [13, 4.8], maple: [8, 5.6], cherry: [7.5, 7.2],
};
const WOOD_MIX: [Species, number][] = [['evergreen', 0.45], ['zelkova', 0.22], ['maple', 0.12], ['cherry', 0.13], ['ginkgo', 0.08]];

function pick(r: number): Species {
  for (const [species, share] of WOOD_MIX) if ((r -= share) < 0) return species;
  return 'evergreen';
}

/**
 * Where the trees stand. Two kinds of planting, as round a real course:
 * - avenues (並木): two even rows flanking a promenade outside the course, one species to a stretch — a cherry avenue on
 *   the back straight, ginkgo on the far turn, zelkova elsewhere — every 8.5 m along the ground (not along the course,
 *   so the rows stay even round the turns); behind the main stand they step back to pass behind it;
 * - woods: mixed groves further out, mostly evergreen, three deep.
 */
function plantTrees(track: TrackModel, ground: (distance: number, lane: number) => T.Vector3) {
  const trees: Tree[] = [];
  const stand = 100;
  const nearStand = (d: number) => Math.min(d, track.lap - d) < stand;
  for (const row of [0, 1]) {
    const random = seeded(9011 + row * 31);
    let walked = 0, previous: T.Vector3 | null = null;
    for (let d = 0; d < track.lap; d += 1) {
      const lane = track.width + (nearStand(d) ? 150 : 118) + row * 12;
      const here = track.sample(d, lane).point;
      if (previous) walked += Math.hypot(here.x - previous.x, here.z - previous.z);
      previous = here;
      if (walked < 8.5) continue;
      walked = 0;
      const f = d / track.lap;
      const species: Species = f > 0.3 && f < 0.62 ? 'cherry' : f >= 0.62 && f < 0.85 ? 'ginkgo' : 'zelkova';
      const [height, width] = SIZE[species];
      trees.push({position: ground(d, lane), species, jitter: (random() - 0.5) * 8,
        height: height * (0.92 + 0.16 * random()), width: width * (0.92 + 0.16 * random())});
    }
  }
  for (let segment = 0; segment < 96; segment++) {
    const random = seeded(7283 + segment * 457);
    const distance = track.lap * segment / 96;
    for (const lane of [track.width + 168, track.width + 186, track.width + 204]) {
      for (let i = 0; i < 2; i++) {
        const species = pick(random());
        const [height, width] = SIZE[species];
        trees.push({position: ground(distance + (random() - 0.5) * 22, lane + random() * 16), species,
          jitter: (random() - 0.5) * 14, height: height * (0.75 + 0.5 * random()), width: width * (0.75 + 0.45 * random())});
      }
    }
  }
  return trees;
}

/**
 * The crown's material: the leaf-clump relief of before, plus the season. Each tree carries how full its crown is
 * (`aLeaf`), how much bare twig shows (`aTwig`) and its two colours (`aMain`, `aPatch`). Where the leaves have gone, the
 * crown is cut away in leaf-sized clumps and, among them, a fine haze of twigs (grey-brown) remains, as a bare tree looks
 * from a distance; blossom and autumn colour come in patches. Snow lies on the upward faces.
 */
function crownMaterial(snow: number) {
  const material = new T.MeshStandardMaterial({
    color: '#fff', roughness: 1, vertexColors: true, side: T.DoubleSide,
    normalMap: leafNormalMap(), normalScale: new T.Vector2(0.7, 0.7),
  });
  material.onBeforeCompile = shader => {
    shader.uniforms.uSnow = {value: snow};
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
        attribute float aLeaf;
        attribute float aTwig;
        attribute float aSeed;
        attribute vec3 aMain;
        attribute vec3 aPatch;
        varying float vLeaf;
        varying float vTwig;
        varying vec3 vMain;
        varying vec3 vPatch;
        varying vec3 vLocal;
        varying float vUp;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vLeaf = aLeaf; vTwig = aTwig; vMain = aMain; vPatch = aPatch;
        vLocal = position + aSeed * 17.0;
        vUp = normal.y;`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform float uSnow;
        varying float vLeaf;
        varying float vTwig;
        varying vec3 vMain;
        varying vec3 vPatch;
        varying vec3 vLocal;
        varying float vUp;
        float treeHash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
        float treeNoise(vec3 p) {
          vec3 i = floor(p), f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(mix(treeHash(i), treeHash(i + vec3(1, 0, 0)), f.x), mix(treeHash(i + vec3(0, 1, 0)), treeHash(i + vec3(1, 1, 0)), f.x), f.y),
            mix(mix(treeHash(i + vec3(0, 0, 1)), treeHash(i + vec3(1, 0, 1)), f.x), mix(treeHash(i + vec3(0, 1, 1)), treeHash(i + vec3(1, 1, 1)), f.x), f.y),
            f.z);
        }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        float clumps = 0.65 * treeNoise(vLocal * 6.0) + 0.35 * treeNoise(vLocal * 14.0);
        float edge = vLeaf > 0.98 ? 2.0 : vLeaf < 0.02 ? -1.0 : mix(0.18, 0.82, vLeaf);
        bool leafy = clumps < edge;
        bool twig = !leafy && treeNoise(vLocal * 40.0) < vTwig * 0.62;
        if (!leafy && !twig) discard;
        vec3 tint = leafy ? mix(vMain, vPatch, smoothstep(0.35, 0.65, treeNoise(vLocal * 3.0 + 5.0))) : vec3(0.075, 0.06, 0.05);
        diffuseColor.rgb *= tint;
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.86, 0.88, 0.9), uSnow * smoothstep(0.2, 0.8, vUp) * (leafy ? 1.0 : 0.6));`);
  };
  material.customProgramCacheKey = () => 'seasonal-crown';
  return material;
}

/**
 * The trees round the course, dressed for the race day (`day` of the year) and the weather's snow cover: crowns (seasonal
 * leaves, blossom or bare twigs), trunks, and the bare limbs of the broadleaf trees when their leaves are thin.
 */
export function addTrees(scene: T.Scene, track: TrackModel, ground: (distance: number, lane: number) => T.Vector3,
  day: number, snow: number) {
  const trees = plantTrees(track, ground);
  const geometry = foliageGeometry(2).clone();
  const leaf = new Float32Array(trees.length), twig = new Float32Array(trees.length), seed = new Float32Array(trees.length);
  const main = new Float32Array(trees.length * 3), patch = new Float32Array(trees.length * 3);
  const looks = trees.map(t => foliage(t.species, day, t.jitter));
  const random = seeded(2203);
  looks.forEach((look, i) => {
    // A little more or less light and saturation from tree to tree.
    const vary = 0.9 + 0.2 * random();
    leaf[i] = look.leaf;
    twig[i] = look.twig;
    seed[i] = random();
    main.set([look.main.r * vary, look.main.g * vary, look.main.b * vary], i * 3);
    patch.set([look.patch.r * vary, look.patch.g * vary, look.patch.b * vary], i * 3);
  });
  geometry.setAttribute('aLeaf', new T.InstancedBufferAttribute(leaf, 1));
  geometry.setAttribute('aTwig', new T.InstancedBufferAttribute(twig, 1));
  geometry.setAttribute('aSeed', new T.InstancedBufferAttribute(seed, 1));
  geometry.setAttribute('aMain', new T.InstancedBufferAttribute(main, 3));
  geometry.setAttribute('aPatch', new T.InstancedBufferAttribute(patch, 3));
  const crowns = new T.InstancedMesh(geometry, crownMaterial(snow), trees.length);
  crowns.name = 'tree-crowns';
  const wood = new T.MeshStandardMaterial({color: '#665845', roughness: 1});
  const trunks = new T.InstancedMesh(trunkGeometry(), wood, trees.length);
  // Limbs only where some show: broadleaf trees whose crowns are not full.
  const bare = trees.map((t, i) => i).filter(i => trees[i].species !== 'evergreen' && looks[i].leaf < 0.95);
  const limbs = bare.length ? new T.InstancedMesh(branchGeometry(), wood, bare.length) : null;
  const instance = new T.Object3D();
  trees.forEach(({position, height, width}, i) => {
    instance.rotation.y = i;
    instance.position.copy(position).y += height * 0.72;
    instance.scale.set(width * 0.9, height * 0.5, width * 0.9);
    instance.updateMatrix();
    crowns.setMatrixAt(i, instance.matrix);
    instance.position.copy(position).y += height * 0.2;
    instance.scale.set(width * 0.16, height * 0.6, width * 0.16);
    instance.updateMatrix();
    trunks.setMatrixAt(i, instance.matrix);
  });
  bare.forEach((tree, k) => {
    const {position, height, width} = trees[tree];
    instance.rotation.y = tree;
    instance.position.copy(position);
    instance.scale.set(width, height, width);
    instance.updateMatrix();
    limbs!.setMatrixAt(k, instance.matrix);
  });
  scene.add(trunks, crowns);
  if (limbs) scene.add(limbs);
  return trees.length;
}
