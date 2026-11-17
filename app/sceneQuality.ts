import * as T from 'three';

export type SceneQuality = {
  antialias: boolean;
  renderScale: number;
  pileLayers: number;
  fullAnisotropy: boolean;
  extendedGrass: boolean;
  maxPixelRatio: number;
  shadowSize: number;
  toneMapping: T.ToneMapping;
  exposure: number;
  colorSpace: T.ColorSpace;
  clearColor: string;
};

export const PREVIEW_QUALITY: SceneQuality = {
  antialias: true,
  renderScale: 1,
  pileLayers: 12,
  fullAnisotropy: false,
  extendedGrass: false,
  maxPixelRatio: 2,
  shadowSize: 2048,
  toneMapping: T.ACESFilmicToneMapping,
  exposure: 1.03,
  colorSpace: T.SRGBColorSpace,
  clearColor: '#e8eddf',
};

export const EXPORT_QUALITY: SceneQuality = {
  ...PREVIEW_QUALITY, renderScale: 2, shadowSize: 4096, pileLayers: 16, fullAnisotropy: true, extendedGrass: true,
};

export function createRaceRenderer(quality: SceneQuality, preserveDrawingBuffer = false) {
  const renderer = new T.WebGLRenderer({antialias: quality.antialias, preserveDrawingBuffer});
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFSoftShadowMap;
  renderer.toneMapping = quality.toneMapping;
  renderer.toneMappingExposure = quality.exposure;
  renderer.outputColorSpace = quality.colorSpace;
  renderer.setClearColor(quality.clearColor);
  return renderer;
}

