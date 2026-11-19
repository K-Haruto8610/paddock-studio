import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

// The plush horse's shapes as modelled in Blender (blender/horse.py, exported by blender/export.py): the body
// (torso, neck, head and legs as one seamless shell), the menko, the head's small parts, the saddle cloth, and the guide hairs
// the mane and tail are grown around. Every geometry comes out in
// the horse's own frame (metres, x forward, y up, z to its right, ground at y = 0), the frame app/plushHorse.ts builds in,
// so the rig that bends the shell for the gallop and the nod works on it unchanged. Colours, fabrics and the fur
// are the app's own, by part name: the model carries no materials.

export const HORSE_MODEL_URL = '/models/horse.glb';

export type HorseModel = {
  /** Torso, neck, head and legs, one closed shell. Positions and normals only. */
  body: T.BufferGeometry;
  /** The racing hood, cut to its outline, with a closed rim. */
  menko: T.BufferGeometry;
  /** Ears, eyes and nostrils, keyed by their Blender names (`horse_ear_L`, ...). */
  head: Record<string, T.BufferGeometry>;
  /** Bridle straps (`horse_bridle_noseband`, ...) and their flat joints (`horse_bridle_joint_0`, ...). They carry `uv` and `_cover` (1 where lying on the hood). */
  bridle: Record<string, T.BufferGeometry>;
  /** Saddle cloth and the two number panels, 25 mm off the trunk (the app lifts them over the fur). */
  saddle: {cloth: T.BufferGeometry; printL: T.BufferGeometry; printR: T.BufferGeometry};
  /** Guide hairs the app grows the mane and tail around, in the horse frame, root to tip. */
  hair: {maneGuides: T.Vector3[][]; maneGuidesMenko: T.Vector3[][]; tailSpine: T.Vector3[]};
};

let loaded: HorseModel | null = null;
let loading: Promise<HorseModel> | null = null;

/** The model once `loadHorseModel` has finished; null before that. */
export function horseModel() {
  return loaded;
}

export function loadHorseModel(url = HORSE_MODEL_URL): Promise<HorseModel> {
  loading ??= new GLTFLoader().loadAsync(url).then(gltf => {
    gltf.scene.updateMatrixWorld(true);
    const parts: Record<string, T.BufferGeometry> = {};
    gltf.scene.traverse(object => {
      if (!(object as T.Mesh).isMesh && !(object as T.LineSegments).isLineSegments) return;
      const mesh = object as T.Mesh;
      // Bake the node transforms (the head group sits at the neck pivot) into the geometry: everything in the horse frame.
      const geometry = mesh.geometry.clone();
      geometry.applyMatrix4(mesh.matrixWorld);
      parts[mesh.name] = geometry;
    });
    const need = (name: string) => {
      const found = parts[name];
      if (!found) throw new Error(`${url} has no part named ${name}`);
      return found;
    };
    const pick = (prefix: string) => Object.fromEntries(Object.entries(parts).filter(([name]) => name.startsWith(prefix)));
    // Guide hairs are loose edges with a per-point `_strand` index: group the points by it, in order.
    const polylines = (name: string) => {
      const geometry = need(name);
      const position = geometry.attributes.position, strand = geometry.attributes._strand;
      const runs = new Map<number, T.Vector3[]>();
      for (let i = 0; i < position.count; i++) {
        const key = Math.round(strand.getX(i));
        if (!runs.has(key)) runs.set(key, []);
        runs.get(key)!.push(new T.Vector3(position.getX(i), position.getY(i), position.getZ(i)));
      }
      return [...runs.entries()].sort((a, b) => a[0] - b[0]).map(([, points]) => points);
    };
    const model: HorseModel = {
      body: need('horse_body'),
      menko: need('horse_menko'),
      head: {...pick('horse_ear_'), ...pick('horse_eye_'), ...pick('horse_nostril_')},
      bridle: pick('horse_bridle_'),
      saddle: {cloth: need('horse_saddle'), printL: need('horse_saddle_print_L'), printR: need('horse_saddle_print_R')},
      hair: {maneGuides: polylines('horse_mane_guides'), maneGuidesMenko: polylines('horse_mane_guides_menko'), tailSpine: polylines('horse_tail_spine')[0]},
    };
    loaded = model;
    return model;
  });
  return loading;
}
