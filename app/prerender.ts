/**
 * Pre-rendered playback: the 3D view rendered frame by frame into an MP4 once, kept in the
 * browser's private file storage (OPFS) and played back as video. Nothing leaves the browser.
 *
 * A video is found again by a key over everything that changes the picture. The scene code is
 * not part of the key: raise PRERENDER_VERSION whenever the look of the race view changes, so
 * old videos are no longer used.
 */
import type {CameraMode} from './raceCamera';
import type {PlaybackPlan} from './playback';
import type {Race} from './types';
import type {Weather} from './raceWeather';
import type {VideoStore} from './videoExport';

export const PRERENDER_VERSION = 5;
const FOLDER = 'prerendered';
/** The widest video, in pixels. */
const MAX_WIDTH = 2560;

export type PrerenderInput = {
  race: Race;
  plan: PlaybackPlan;
  cameraMode: CameraMode;
  speed: number;
  weather: Weather;
  going: number | null;
  courseDay: number | null;
  postTime: string | undefined;
  reduced: boolean;
  /** Stage width ÷ height. */
  aspect: number;
};

/** Stages whose shapes round to the same value share a video (shown cropped to fill). */
export const aspectBucket = (aspect: number) => Math.round(aspect * 20) / 20;

export async function prerenderKey(input: PrerenderInput) {
  const text = JSON.stringify({
    version: PRERENDER_VERSION,
    ...input,
    aspect: aspectBucket(input.aspect),
  });
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)));
  return Array.from(digest.slice(0, 16), (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Video size for a stage of cssWidth × cssHeight CSS pixels: as sharp as the screen (at most 2×),
 * even dimensions for H.264. `width`/`height` are in the scene's label units (CSS pixels).
 */
export function prerenderSize(cssWidth: number, cssHeight: number, devicePixelRatio: number) {
  const ratio = Math.min(devicePixelRatio || 1, 2, MAX_WIDTH / cssWidth);
  const pixelWidth = 2 * Math.max(1, Math.round((cssWidth * ratio) / 2));
  const pixelHeight = 2 * Math.max(1, Math.round((cssHeight * ratio) / 2));
  return {width: pixelWidth / ratio, height: pixelHeight / ratio, pixelRatio: ratio, pixelWidth, pixelHeight};
}

async function folder() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(FOLDER, {create: true});
}

const fileName = (key: string) => `${key}.mp4`;

/** The kept video for this key, or null (also when the browser has no file storage). */
export async function loadPrerender(key: string): Promise<File | null> {
  try {
    const file = await (await folder()).getFileHandle(fileName(key));
    return await file.getFile();
  } catch {
    return null;
  }
}

/** Where to keep a new video; null when the browser has no file storage (the video lives in memory). */
export async function prerenderStore(key: string): Promise<VideoStore | null> {
  try {
    // Ask the browser not to evict the videos under storage pressure. It may decline silently.
    await navigator.storage.persist?.().catch(() => false);
    return {folder: await folder(), name: fileName(key)};
  } catch {
    return null;
  }
}

type Entries = FileSystemDirectoryHandle & {
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
};

/** Kept videos: how many and how many bytes. Unfinished files are counted too, so they can be removed. */
export async function prerenderUsage() {
  let count = 0, bytes = 0;
  try {
    for await (const [name, handle] of (await folder() as Entries).entries()) {
      if (handle.kind !== 'file') continue;
      if (name.endsWith('.mp4')) count++;
      bytes += (await (handle as FileSystemFileHandle).getFile()).size;
    }
  } catch {}
  return {count, bytes};
}

export async function clearPrerenders() {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(FOLDER, {recursive: true}).catch(() => {});
}

export const formatBytes = (bytes: number) =>
  bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)}GB` : `${Math.max(bytes / 1e6, bytes ? 0.1 : 0).toFixed(1)}MB`;
