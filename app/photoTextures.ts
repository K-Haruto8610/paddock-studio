import * as T from 'three';
import type {TreeShape} from './treeModels';

// Scanned PBR textures (Poly Haven, CC0; see public/textures/README.md): colour, roughness and normal maps for the stands and
// buildings, the dirt and the turf, and the bark of each kind of tree, each loaded once and shared. Scenes wait for them
// (ensureHorseModel) so a frame never shows a bare surface; if they cannot be read, the procedural finishes stand in.

export type PhotoSet = {
  map: T.Texture; roughnessMap: T.Texture; normalMap: T.Texture;
  /** Mean colour of `map` (linear), to use the scan as a detail that keeps another map's colour on average. */
  average: T.Color;
  /** Metres one repeat of the scan covers. */
  size: number;
};
export type PhotoTextures = {
  concrete: PhotoSet; roof: PhotoSet;
  dirt: PhotoSet; grass: PhotoSet;
  bark: Record<TreeShape, PhotoSet>;
};

let loaded: PhotoTextures | null = null;
let loading: Promise<PhotoTextures> | null = null;

/** The textures once `loadPhotoTextures` has finished; null before that, or if they could not be read. */
export function photoTextures() {
  return loaded;
}

/** Mean linear colour of an image, from a 32 x 32 reduction. */
function meanColour(image: CanvasImageSource) {
  const c = document.createElement('canvas');
  c.width = c.height = 32;
  const g = c.getContext('2d', {willReadFrequently: true})!;
  g.imageSmoothingQuality = 'high';
  g.drawImage(image, 0, 0, 32, 32);
  const data = g.getImageData(0, 0, 32, 32).data;
  const linear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const sum = [0, 0, 0];
  for (let i = 0; i < data.length; i += 4)
    for (let ch = 0; ch < 3; ch++) sum[ch] += linear(data[i + ch] / 255);
  return new T.Color(sum[0] / 1024, sum[1] / 1024, sum[2] / 1024);
}

export function loadPhotoTextures(): Promise<PhotoTextures> {
  loading ??= (async () => {
    const loader = new T.TextureLoader();
    const set = async (asset: string, size: number): Promise<PhotoSet> => {
      const base = `/textures/${asset}/${asset}_`;
      const [map, roughnessMap, normalMap] = await Promise.all(['diff', 'rough', 'nor_gl'].map(kind => loader.loadAsync(`${base}${kind}_1k.jpg`)));
      map.colorSpace = T.SRGBColorSpace;
      for (const texture of [map, roughnessMap, normalMap]) {
        texture.wrapS = texture.wrapT = T.RepeatWrapping;
        texture.anisotropy = 8;
      }
      return {map, roughnessMap, normalMap, average: meanColour(map.image as CanvasImageSource), size};
    };
    const [concrete, roof, dirt, grass, zelkova, cherry, camphor, pine, maple] = await Promise.all([
      set('concrete_slab_wall_02', 9), set('corrugated_iron', 3), set('raked_dirt', 2), set('sparse_grass', 2),
      set('japanese_zelkova_bark', 1.8), set('sakura_bark', 1.6), set('japanese_camphor_bark', 1.8), set('pine_bark', 2),
      set('trident_maple_bark', 1.8),
    ]);
    // No ginkgo scan: its grey-brown, vertically furrowed bark is closest to the camphor's.
    loaded = {concrete, roof, dirt, grass, bark: {zelkova, cherry, camphor, pine, maple, ginkgo: camphor}};
    return loaded;
  })();
  return loading;
}
