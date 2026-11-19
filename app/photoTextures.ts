import * as T from 'three';

// Scanned PBR textures (Poly Haven, CC0; see public/textures/README.md) for the stands and buildings: colour, roughness and
// normal maps, each loaded once and shared. Scenes wait for them (ensureHorseModel) so a frame never shows a bare surface.

export type PhotoSet = {map: T.Texture; roughnessMap: T.Texture; normalMap: T.Texture};
export type PhotoTextures = {concrete: PhotoSet; roof: PhotoSet};

let loaded: PhotoTextures | null = null;
let loading: Promise<PhotoTextures> | null = null;

/** The textures once `loadPhotoTextures` has finished; null before that, or if they could not be read. */
export function photoTextures() {
  return loaded;
}

export function loadPhotoTextures(): Promise<PhotoTextures> {
  loading ??= (async () => {
    const loader = new T.TextureLoader();
    const set = async (asset: string): Promise<PhotoSet> => {
      const base = `/textures/${asset}/${asset}_`;
      const [map, roughnessMap, normalMap] = await Promise.all(['diff', 'rough', 'nor_gl'].map(kind => loader.loadAsync(`${base}${kind}_1k.jpg`)));
      map.colorSpace = T.SRGBColorSpace;
      for (const texture of [map, roughnessMap, normalMap]) {
        texture.wrapS = texture.wrapT = T.RepeatWrapping;
        texture.anisotropy = 8;
      }
      return {map, roughnessMap, normalMap};
    };
    const [concrete, roof] = await Promise.all([set('concrete_slab_wall_02'), set('corrugated_iron')]);
    loaded = {concrete, roof};
    return loaded;
  })();
  return loading;
}
