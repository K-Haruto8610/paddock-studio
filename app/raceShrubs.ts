import * as T from 'three';
import {seeded} from './scenery';
import {shrubFoliage, type Foliage, type Shrub} from './raceSeason';
import {shrubModel, type ShrubForm} from './treeModels';
import {leafInstances} from './raceTrees';
import {AVENUE_LINES, avenueRuns, walkRuns, type AvenueSite} from './raceAvenue';

type Plant = {matrix: T.Matrix4; form: ShrubForm; kind: Shrub; variant: number; look: Foliage};

/** The hedge under each stretch of avenue: azalea under the cherries, box under the ginkgo, enkianthus under the zelkova. */
const HEDGE: Record<string, Shrub> = {cherry: 'azalea', ginkgo: 'box', zelkova: 'enkianthus'};

/**
 * The clipped planting (植え込み) round the course, dressed for the race day and the snow:
 * - a clipped hedge along the promenade behind the first row of the avenue, broken every 60 m or so where a path comes
 *   through (app/raceAvenue.ts: the lines run just outside the outermost rail, broken at pockets and the stand);
 * - azalea balls (ツツジの玉作り) between the trees of the second row;
 * - in each infield bed (`beds`: its frame, standing on the bed's soil, x along it), a row of azalea balls between two low
 *   enkianthus hedges.
 */
export function addShrubs(scene: T.Scene, site: AvenueSite, ground: (p: T.Vector3) => T.Vector3,
  beds: T.Matrix4[], day: number, snow: number) {
  const plants: Plant[] = [];
  const random = seeded(4421);
  const place = (at: T.Vector3, yaw: number, scale: T.Vector3) =>
    new T.Matrix4().compose(at, new T.Quaternion().setFromEuler(new T.Euler(0, yaw, 0)), scale);
  const look = (kind: Shrub, hue = random()) => shrubFoliage(kind, day, (random() - 0.5) * 8, hue);

  for (const {at, yaw, species, index} of walkRuns(avenueRuns(site, AVENUE_LINES.hedge), 3.4, 3.4)) {
    if (index % 18 === 9) continue;
    const kind = HEDGE[species];
    // A hedge of azalea flowers in runs of one colour, 5 pieces (17 m) long.
    plants.push({matrix: place(ground(at), yaw, new T.Vector3(1, 0.95 + 0.1 * random(), 1)), form: 'hedge', kind,
      variant: index % 3, look: look(kind, (Math.floor(index / 5) * 0.618) % 1)});
  }
  for (const {at, yaw, index} of walkRuns(avenueRuns(site, AVENUE_LINES.back), 8.5, 4.25)) {
    const s = 0.85 + 0.3 * random();
    plants.push({matrix: place(ground(at), yaw + random() * 6.3, new T.Vector3(s, s, s)), form: 'ball', kind: 'azalea',
      variant: index % 3, look: look('azalea')});
  }

  // The infield beds.
  beds.forEach((bed, b) => {
    for (let i = 0; i < 7; i++) {
      const s = i % 2 ? 0.8 : 1.05;
      const local = place(new T.Vector3(-8.4 + i * 2.8, 0, 0), random() * 6.3, new T.Vector3(s, s, s));
      plants.push({matrix: bed.clone().multiply(local), form: 'ball', kind: 'azalea', variant: (i + b) % 3, look: look('azalea')});
    }
    for (const z of [-2.35, 2.35])
      for (let i = 0; i < 6; i++) {
        const local = place(new T.Vector3(-8.75 + i * 3.5, 0, z), 0, new T.Vector3(1, 0.75, 0.62));
        plants.push({matrix: bed.clone().multiply(local), form: 'hedge', kind: 'enkianthus', variant: (i + b) % 3,
          look: look('enkianthus')});
      }
  });

  const groups = new Map<string, Plant[]>();
  for (const plant of plants) {
    const key = `${plant.form}:${plant.kind}:${plant.variant}`;
    groups.set(key, [...(groups.get(key) ?? []), plant]);
  }
  const coreMaterial = new T.MeshStandardMaterial({color: '#fff', roughness: 1});
  const hidden = new T.Matrix4().makeScale(0, 0, 0);
  for (const [key, members] of groups) {
    const {form, kind, variant} = members[0];
    const model = shrubModel(form, variant);
    // The dark inside of the bush; gone when a deciduous hedge is bare (its twigs show through).
    const core = new T.InstancedMesh(model.core, coreMaterial, members.length);
    const tint = new T.Color();
    members.forEach((plant, k) => {
      core.setMatrixAt(k, plant.look.leaf < 0.35 ? hidden : plant.matrix);
      core.setColorAt(k, tint.copy(plant.look.main).multiplyScalar(0.32));
    });
    core.computeBoundingSphere();
    core.name = `shrub-core-${key}`;
    const leaves = leafInstances(model.leaves, kind, snow, members.map(p => p.look), members.map(p => p.matrix), 5501 + plants.indexOf(members[0]));
    leaves.name = `shrub-leaves-${key}`;
    scene.add(core, leaves);
  }
  return plants.length;
}
