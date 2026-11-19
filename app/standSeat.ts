import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

// The stadium seat modelled in Blender (blender/seats.py, public/models/stand.glb): a moulded bucket shell on a steel bracket,
// 72 triangles. The grandstand sets it in rows on its terraces (app/raceLandscape.ts). Frame: metres, origin on the floor
// under the middle of the front edge, facing -z, y up. Loaded once; scenes wait for it with the horse (ensureHorseModel).

export const STAND_MODEL_URL = '/models/stand.glb';

let loaded: T.BufferGeometry | null = null;
let loading: Promise<T.BufferGeometry> | null = null;

/**
 * A copy of the seat for one scene (the scene disposes its own), or null before `loadStandSeat` has finished or if it failed:
 * the stand then keeps its flat seat blocks. `color` holds the shade: 1 on the shell, darker on the bracket.
 */
export function standSeatGeometry() {
  return loaded ? loaded.clone() : null;
}

export function loadStandSeat(url = STAND_MODEL_URL): Promise<T.BufferGeometry> {
  loading ??= new GLTFLoader().loadAsync(url).then(gltf => {
    let found: T.BufferGeometry | null = null;
    gltf.scene.traverse(object => {
      // The mesh only: its node offset is where it stood in the Blender scene.
      if ((object as T.Mesh).isMesh && object.name === 'stand_seat') found = (object as T.Mesh).geometry;
    });
    if (!found) throw new Error(`${url} has no stand_seat`);
    const geometry = found as T.BufferGeometry;
    const shade = geometry.attributes._shade;
    const colors = new Float32Array(shade.count * 3);
    for (let i = 0; i < shade.count; i++) colors.fill(shade.getX(i), i * 3, i * 3 + 3);
    geometry.setAttribute('color', new T.BufferAttribute(colors, 3));
    geometry.deleteAttribute('_shade');
    loaded = geometry;
    return geometry;
  });
  return loading;
}
