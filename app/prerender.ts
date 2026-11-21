/**
 * Pre-rendered playback: the 3D view rendered frame by frame into an MP4 once, kept in a folder
 * the viewer picks on their computer (in browsers without a folder picker, the browser's private
 * file storage, OPFS) and played back as video. Nothing is sent to a server.
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

type Entries = FileSystemDirectoryHandle & {
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
};
/** A folder on the viewer's computer, picked once (Chrome, Edge) and kept across visits in IndexedDB. */
type PickedFolder = FileSystemDirectoryHandle & {
  queryPermission?(descriptor: {mode: 'readwrite'}): Promise<PermissionState>;
  requestPermission?(descriptor: {mode: 'readwrite'}): Promise<PermissionState>;
};
type DirectoryPicker = (options: {
  id?: string;
  mode?: 'readwrite';
  startIn?: string;
}) => Promise<FileSystemDirectoryHandle>;

const DB_NAME = 'paddock-studio';
const DB_STORE = 'handles';
const FOLDER_KEY = 'prerender-folder';

function handles<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open(DB_NAME, 1);
    open.onupgradeneeded = () => open.result.createObjectStore(DB_STORE);
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const db = open.result;
      const request = run(db.transaction(DB_STORE, mode).objectStore(DB_STORE));
      request.onsuccess = () => {
        resolve(request.result);
        db.close();
      };
      request.onerror = () => {
        reject(request.error);
        db.close();
      };
    };
  });
}

/** Whether this browser can keep the videos in a folder the viewer picks. */
export const canPickFolder = () =>
  typeof window !== 'undefined' &&
  typeof (window as unknown as {showDirectoryPicker?: unknown}).showDirectoryPicker === 'function';

async function pickedFolder(): Promise<PickedFolder | null> {
  if (!canPickFolder()) return null;
  try {
    return (await handles<PickedFolder | undefined>('readonly', (store) => store.get(FOLDER_KEY))) ?? null;
  } catch {
    return null;
  }
}
const access = async (picked: PickedFolder): Promise<PermissionState> =>
  (await picked.queryPermission?.({mode: 'readwrite'}).catch(() => 'denied' as const)) ?? 'granted';

async function browserFolder() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(FOLDER, {create: true});
}

/**
 * Where the videos are kept: the picked folder, or the browser's private storage (OPFS) when none
 * was picked. A picked folder the browser may not use on this visit (access not granted) throws.
 */
async function folder(): Promise<FileSystemDirectoryHandle> {
  const picked = await pickedFolder();
  if (!picked) return browserFolder();
  if ((await access(picked)) !== 'granted') throw Error(`フォルダー「${picked.name}」へのアクセスが許可されていません`);
  return picked;
}

export type VideoFolder = {kind: 'local'; name: string; ready: boolean} | {kind: 'browser'; canPick: boolean};
export async function videoFolder(): Promise<VideoFolder> {
  const picked = await pickedFolder();
  if (picked) return {kind: 'local', name: picked.name, ready: (await access(picked)) === 'granted'};
  return {kind: 'browser', canPick: canPickFolder()};
}

/**
 * Opens the folder picker; call it from a click. Videos kept in the browser so far move into the
 * picked folder. Null when the viewer cancels.
 */
export async function pickVideoFolder(): Promise<{name: string; moved: number} | null> {
  let picked: FileSystemDirectoryHandle;
  try {
    picked = await (window as unknown as {showDirectoryPicker: DirectoryPicker}).showDirectoryPicker({
      id: 'paddock-videos',
      mode: 'readwrite',
      startIn: 'videos',
    });
  } catch (error) {
    if ((error as DOMException)?.name === 'AbortError') return null;
    throw error;
  }
  await handles('readwrite', (store) => store.put(picked, FOLDER_KEY));
  return {name: picked.name, moved: await moveBrowserVideos(picked)};
}

/** Asks for the picked folder again (browsers may forget the grant between visits); call it from a click. */
export async function allowVideoFolder() {
  const picked = await pickedFolder();
  if (!picked) return false;
  return ((await picked.requestPermission?.({mode: 'readwrite'})) ?? 'granted') === 'granted';
}

/**
 * Before a video is made (from the click): pick the folder if none is picked yet, or ask for it
 * again. False when the viewer cancels or refuses. Browsers without the picker keep using OPFS.
 */
export async function readyVideoFolder() {
  if (!canPickFolder()) return true;
  const picked = await pickedFolder();
  if (!picked) return !!(await pickVideoFolder());
  return (await access(picked)) === 'granted' || allowVideoFolder();
}

/**
 * Files are named `<label>_<first 12 hex of the key>.mp4`, so the folder reads as a list of races.
 * Videos made before the label are `<key>.mp4`. Only names of these shapes are counted or deleted.
 */
const shortKey = (key: string) => key.slice(0, 12);
const OWN = /(?:_[0-9a-f]{12}|^[0-9a-f]{32})\.mp4(?:\.part)?$/;
const isVideo = (name: string) => OWN.test(name) && name.endsWith('.mp4');
const fileName = (key: string, label: string) =>
  `${label.replace(/[\\/:*?"<>|\s]+/g, '_').slice(0, 120)}_${shortKey(key)}.mp4`;

async function ownFiles(dir: FileSystemDirectoryHandle) {
  const files: [string, FileSystemFileHandle][] = [];
  for await (const [name, handle] of (dir as Entries).entries())
    if (handle.kind === 'file' && OWN.test(name)) files.push([name, handle as FileSystemFileHandle]);
  return files;
}

async function moveBrowserVideos(target: FileSystemDirectoryHandle) {
  let moved = 0;
  try {
    const from = await browserFolder();
    for (const [name, handle] of await ownFiles(from)) {
      if (isVideo(name)) {
        const to = /^[0-9a-f]{32}\.mp4$/.test(name) ? `滑らか再生_${shortKey(name)}.mp4` : name;
        const out = await (await target.getFileHandle(to, {create: true})).createWritable();
        await (await handle.getFile()).stream().pipeTo(out);
        moved++;
      }
      await from.removeEntry(name);
    }
  } catch {
    // Whatever could not be moved stays in the browser's storage, no longer shown.
  }
  return moved;
}

/** The kept video for this key, or null (also when there is no file storage or no access). */
export async function loadPrerender(key: string): Promise<File | null> {
  try {
    const found = (await ownFiles(await folder())).find(
      ([name]) => name.endsWith(`_${shortKey(key)}.mp4`) || name === `${key}.mp4`,
    );
    return found ? await found[1].getFile() : null;
  } catch {
    return null;
  }
}

/**
 * Where to keep a new video, named after `label` (race, playback, camera, speed, weather); null
 * when there is no file storage (the video lives in memory).
 */
export async function prerenderStore(key: string, label: string): Promise<VideoStore | null> {
  try {
    const dir = await folder();
    // Ask the browser not to evict videos kept in its own storage. It may decline silently.
    if (!(await pickedFolder())) await navigator.storage.persist?.().catch(() => false);
    return {folder: dir, name: fileName(key, label)};
  } catch {
    return null;
  }
}

/** Kept videos: how many and how many bytes. Unfinished files are counted too, so they can be removed. */
export async function prerenderUsage() {
  let count = 0, bytes = 0;
  try {
    for (const [name, handle] of await ownFiles(await folder())) {
      if (isVideo(name)) count++;
      bytes += (await handle.getFile()).size;
    }
  } catch {}
  return {count, bytes};
}

/** Deletes this app's videos (and unfinished ones) from the folder; other files there are left alone. */
export async function clearPrerenders() {
  const dir = await folder();
  for (const [name] of await ownFiles(dir)) await dir.removeEntry(name).catch(() => {});
}

export const formatBytes = (bytes: number) =>
  bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)}GB` : `${Math.max(bytes / 1e6, bytes ? 0.1 : 0).toFixed(1)}MB`;
