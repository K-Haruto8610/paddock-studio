import {BufferTarget, EncodedPacket, EncodedVideoPacketSource, Mp4OutputFormat, Output, StreamTarget} from 'mediabunny';
import {RACE_FPS, outputSeconds, progressAtTime} from './raceClock';
import {drawRecordingFrame, RECORDING_SCENE_SIZE, VIDEO_HEIGHT, type RecordingOverlay} from './recorder';

const BITRATE = 24_000_000;
const WIDTH = RECORDING_SCENE_SIZE.width;

/** H.264 at the given size: the recording layout by default. Offline encodes ask for 'quality'. */
export async function h264Config(width = WIDTH, height = VIDEO_HEIGHT, {bitrate = BITRATE, latencyMode = 'realtime'}: {
  bitrate?: number; latencyMode?: LatencyMode;
} = {}) {
  if (typeof VideoEncoder === 'undefined') return null;
  for (const codec of ['avc1.640033', 'avc1.4d0033', 'avc1.640032', 'avc1.640034']) {
    const config: VideoEncoderConfig = {
      codec, width, height, bitrate, framerate: RACE_FPS,
      latencyMode, avc: {format: 'avc'},
    };
    try {
      if ((await VideoEncoder.isConfigSupported(config)).supported) return config;
    } catch {}
  }
  return null;
}

export function videoTimeline(distance: number, speed: number, end: number) {
  const finish = Math.ceil(outputSeconds(end * distance, speed) * RACE_FPS);
  return {
    count: finish + 1 + RACE_FPS,
    at: (k: number) => ({
      seconds: k / RACE_FPS,
      progress: Math.min(end, progressAtTime(k / RACE_FPS, distance, speed)),
      timestamp: Math.round(k * 1_000_000 / RACE_FPS),
      duration: Math.round((k + 1) * 1_000_000 / RACE_FPS) - Math.round(k * 1_000_000 / RACE_FPS),
    }),
  };
}

type Frame = ReturnType<ReturnType<typeof videoTimeline>['at']>;

/** Where a finished video is kept: `name` in `folder` (written as `name.part`, renamed at the end). */
export type VideoStore = {folder: FileSystemDirectoryHandle; name: string};

export async function encodeRaceVideo({
  config, timeline, signal, draw, overlay, progress, hash, failAt, store,
}: {
  config: VideoEncoderConfig;
  timeline: ReturnType<typeof videoTimeline>;
  signal: AbortSignal;
  draw: (frame: Frame) => Promise<HTMLCanvasElement>;
  /** The page's lower panels under the 3D view. Without it the video is the 3D view alone, at the config's size. */
  overlay?: (progress: number) => RecordingOverlay;
  progress: (done: number, count: number) => void;
  hash?: (hash: string, frame: Frame) => void;
  failAt?: number;
  /** Keep the video as a file there and return it; otherwise the temporary file is removed. */
  store?: VideoStore;
}) {
  const width = overlay ? WIDTH : config.width;
  const height = overlay ? VIDEO_HEIGHT : config.height;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  // Keep compositing on one backend even when developer readbacks are enabled.
  const context = canvas.getContext('2d', {alpha: false, willReadFrequently: true})!;
  const name = store ? store.name + '.part' : 'paddock-' + crypto.randomUUID() + '.mp4';
  let folder: FileSystemDirectoryHandle | undefined;
  let file: FileSystemFileHandle | undefined;
  let stream: FileSystemWritableFileStream | undefined;
  try {
    folder = store?.folder ?? await navigator.storage.getDirectory();
    file = await folder.getFileHandle(name, {create: true});
    stream = await file.createWritable();
  } catch {
    if (file) await folder?.removeEntry(name).catch(() => {});
    file = undefined;
  }

  const target = stream ? new StreamTarget(stream) : new BufferTarget();
  const output = new Output({format: new Mp4OutputFormat({fastStart: false}), target});
  const source = new EncodedVideoPacketSource('avc');
  output.addVideoTrack(source, {frameRate: RACE_FPS});
  let pending = Promise.resolve();
  let failure: unknown;
  let encoder: VideoEncoder | undefined;
  let unblock: (() => void) | undefined;
  const cancel = () => { if (encoder && encoder.state !== 'closed') encoder.close(); };
  signal.addEventListener('abort', cancel, {once: true});
  try {
    signal.throwIfAborted();
    encoder = new VideoEncoder({
      output(chunk, meta) {
        pending = pending.then(() => source.add(EncodedPacket.fromEncodedChunk(chunk), meta))
          .catch(error => { failure ??= error; });
      },
      error(error) { failure ??= error; unblock?.(); },
    });
    encoder.configure(config);
    await output.start();
    for (let k = 0; k < timeline.count; k++) {
      signal.throwIfAborted();
      if (failure) throw failure;
      const frame = timeline.at(k);
      const scene = await draw(frame);
      if (overlay) drawRecordingFrame(context, scene, overlay(frame.progress));
      else context.drawImage(scene, 0, 0, width, height);
      signal.throwIfAborted();
      if (hash) {
        const pixels = context.getImageData(0, 0, width, height).data;
        const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', pixels));
        hash(Array.from(bytes, b => b.toString(16).padStart(2, '0')).join(''), frame);
      }
      const video = new VideoFrame(canvas, {timestamp: frame.timestamp, duration: frame.duration});
      try {
        if (k === failAt) encoder.close();
        encoder.encode(video, {keyFrame: k % (RACE_FPS * 2) === 0});
      } finally {
        video.close();
      }
      if (encoder.encodeQueueSize >= 3) {
        await new Promise<void>((resolve) => {
          const done = () => {
            encoder!.removeEventListener('dequeue', done);
            signal.removeEventListener('abort', done);
            unblock = undefined;
            resolve();
          };
          unblock = done;
          encoder!.addEventListener('dequeue', done, {once: true});
          signal.addEventListener('abort', done, {once: true});
          if (encoder!.encodeQueueSize < 3 || signal.aborted) done();
        });
      }
      await pending;
      // Give UI events a turn even if the encoder drains the queue immediately.
      if (k % 6 === 0 || k + 1 === timeline.count) {
        progress(k + 1, timeline.count);
        const scheduler = (globalThis as typeof globalThis & {scheduler?: {yield(): Promise<void>}}).scheduler;
        if (scheduler) await scheduler.yield();
        else await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    await encoder.flush();
    await pending;
    signal.throwIfAborted();
    if (failure) throw failure;
    source.close();
    await output.finalize();
    signal.throwIfAborted();
    if (file && store) {
      // A partly written file never carries the final name, so a cached video is always complete.
      const handle = file as FileSystemFileHandle & {move?: (name: string) => Promise<void>};
      await store.folder.removeEntry(store.name).catch(() => {});
      if (handle.move) await handle.move(store.name);
      else {
        const copy = await (await store.folder.getFileHandle(store.name, {create: true})).createWritable();
        await copy.write(await file.getFile());
        await copy.close();
        await store.folder.removeEntry(name);
      }
      signal.throwIfAborted();
      return (await store.folder.getFileHandle(store.name)).getFile();
    }
    if (file) {
      const data = await file.getFile();
      // Decouple the download from the temporary OPFS file, which is deleted immediately.
      const blob = new Blob([await data.arrayBuffer()], {type: 'video/mp4'});
      await folder!.removeEntry(name);
      signal.throwIfAborted();
      return blob;
    }
    return new Blob([(target as BufferTarget).buffer!], {type: 'video/mp4'});
  } catch (error) {
    await output.cancel().catch(() => {});
    if (file) await folder?.removeEntry(name).catch(() => {});
    throw error;
  } finally {
    signal.removeEventListener('abort', cancel);
    cancel();
  }
}
