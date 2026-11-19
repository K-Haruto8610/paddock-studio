'use client';
import {useEffect, useRef, useState} from 'react';
import TrackScene, {type VerifyFrame} from '../TrackScene';
import {ensureHorseModel} from '../blenderHorse';
import {buildPlayback, playbackEndProgress} from '../playback';
import {parsePaceForecast, paceForRace} from '../paceForecast';
import type {Dataset, PlaybackMode, Replays} from '../types';
import {applyEdits, type RaceEdits} from '../manualEdit';
import {EXPORT_QUALITY, PREVIEW_QUALITY} from '../sceneQuality';
import {drawRecordingFrame, prepareRecordingOverlay, VIDEO_HEIGHT} from '../recorder';
import {raceOverlay} from '../raceOverlay';
import {encodeRaceVideo, h264Config, videoTimeline} from '../videoExport';
import entries from '../../docs/visual-baseline/forecasts.json';
import forecasts from '../../docs/visual-baseline/pace_forecasts.json';
import type {Weather} from '../raceWeather';

type VerifyRequest = Parameters<VerifyFrame>[0] & {
  raceId: string;
  mode?: PlaybackMode;
  forecast?: unknown;
  edits?: RaceEdits;
  quality?: 'preview' | 'export';
  composite?: boolean;
  elevationScale?: 2 | 4 | 8;
  /** Race-day weather (default fine, as the baselines were taken) and the going shown on the infield board. */
  weather?: Weather;
  going?: number;
  /** Turf race days on the present rail (wear along the inside); default none. */
  courseDay?: number;
  /** Post time (JST, HH:MM) for the sun's height; default 15:40. */
  postTime?: string;
};

type SceneInput = {
  race: Dataset['races'][number];
  plan: ReturnType<typeof buildPlayback>;
  quality: 'preview' | 'export';
  elevationScale?: 2 | 4 | 8;
  weather: Weather;
  going: number | null;
  courseDay: number | null;
  postTime?: string;
  resolve: () => void;
};

const fixedEntries: Dataset = entries;
const bundled = parsePaceForecast(forecasts);

export default function VisualCheck() {
  const [input, setInput] = useState<SceneInput>();
  const verification = useRef<VerifyFrame | null>(null);
  const loaded = useRef<{raceId: string; mode: PlaybackMode; forecast: unknown; edits: string | undefined;
    quality: string; elevationScale?: number; weather: string; going?: number; courseDay?: number; postTime?: string} | null>(null);

  useEffect(() => {
    input?.resolve();
  }, [input]);

  useEffect(() => {
    if (process.env.NODE_ENV === 'production') return;
    let busy = false;
    const capture = async (request: VerifyRequest) => {
      if (busy) throw Error('A capture is already running');
      busy = true;
      try {
        const race = fixedEntries.races.find(r => r.id === request.raceId);
        if (!race) throw Error('Unknown race ID');
        const mode = request.mode ?? 'forecast';
        const quality = request.quality ?? 'preview';
        const parsed = request.forecast === undefined ? bundled : parsePaceForecast(request.forecast);
        if (!parsed.ok) throw Error(parsed.error);
        const pace = paceForRace(parsed.data, race).pace;
        const replay = mode === 'replay'
          ? (await (await fetch('/data/replays.json')).json() as Replays).races[race.id]
          : undefined;
        const base = buildPlayback(race, mode, replay, pace);
        const plan = request.edits ? applyEdits(race, base, request.edits) : base;
        if (!plan.available) throw Error('Playback data is unavailable for this race');
        const end = playbackEndProgress(race, plan);
        if ([request.progress, ...(request.history ?? [])].some(p => !Number.isFinite(p) || p < 0 || p > end) ||
          !Number.isFinite(request.speed) || request.speed <= 0 ||
          ![request.width, request.height].every(v => Number.isInteger(v) && v > 0 && v <= 4096)) {
          throw Error('Invalid frame dimensions or timeline input');
        }
        await document.fonts.ready;
        await ensureHorseModel();     // every horse in the scene is made from the same model
        const edits = JSON.stringify(request.edits);
        const prior = loaded.current;
        if (!prior || prior.raceId !== race.id || prior.mode !== mode ||
          prior.forecast !== request.forecast || prior.edits !== edits || prior.quality !== quality ||
          prior.elevationScale !== request.elevationScale || prior.weather !== (request.weather ?? 'clear') ||
          prior.going !== request.going || prior.courseDay !== request.courseDay || prior.postTime !== request.postTime) {
          const {elevationScale} = request;
          const weather = request.weather ?? 'clear';
          await new Promise<void>(resolve => setInput({race, plan, quality, elevationScale, weather,
            going: request.going ?? null, courseDay: request.courseDay ?? null, postTime: request.postTime, resolve}));
          loaded.current = {raceId: race.id, mode, forecast: request.forecast, edits, quality, elevationScale, weather,
            going: request.going, courseDay: request.courseDay, postTime: request.postTime};
        }
        if (!verification.current) throw Error('WebGL renderer is unavailable');
        const overlay = raceOverlay(race, plan, request.progress, !!request.edits, pace?.model);
        if (request.composite) await prepareRecordingOverlay(overlay.seek.profileImage);
        const frame = await verification.current(request);
        if (!request.composite) {
          const copy = document.createElement('canvas');
          copy.width = frame.width;
          copy.height = frame.height;
          copy.getContext('2d')!.drawImage(frame, 0, 0);
          return copy;
        }
        const composite = document.createElement('canvas');
        composite.width = 1920;
        composite.height = VIDEO_HEIGHT;
        drawRecordingFrame(composite.getContext('2d', {alpha: false, willReadFrequently: true})!, frame, overlay);
        return composite;
      } finally {
        busy = false;
      }
    };
    const smokeVideo = async () => {
      const config = await h264Config();
      if (!config) throw Error('H.264 WebCodecs is unavailable');
      const race = fixedEntries.races[0];
      const pace = paceForRace(bundled.ok ? bundled.data : null, race).pace;
      const plan = buildPlayback(race, 'forecast', undefined, pace);
      const overlay = (position: number) => raceOverlay(race, plan, position, false, pace?.model);
      await prepareRecordingOverlay(overlay(0).seek.profileImage);
      const request: VerifyRequest = {raceId: race.id, progress: 0, selected: null, cameraMode: 'pack',
        speed: 2, width: 1920, height: 1020, quality: 'export', reduced: false};
      await capture(request);
      const timeline = videoTimeline(race.distance, 2, 0.001);
      const hashes: string[] = [];
      const blob = await encodeRaceVideo({
        config, timeline, signal: new AbortController().signal,
        draw: frame => verification.current!({...request, progress: frame.progress, outputTime: frame.seconds, sequential: true}),
        overlay, progress: () => {}, hash: hash => hashes.push(hash),
      });
      const direct = await capture({...request, composite: true});
      const rgba = direct.getContext('2d')!.getImageData(0, 0, direct.width, direct.height).data;
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', rgba));
      const directHash = Array.from(digest, b => b.toString(16).padStart(2, '0')).join('');
      const {Input, BlobSource, MP4} = await import('mediabunny');
      const media = new Input({source: new BlobSource(blob), formats: [MP4]});
      try {
        const track = await media.getPrimaryVideoTrack();
        if (!track) throw Error('No MP4 video track');
        const stats = await track.computePacketStats();
        return {hashes, directHash, count: timeline.count, duration: await media.computeDuration(),
          width: track.displayWidth, height: track.displayHeight, stats};
      } finally {
        media.dispose();
      }
    };
    const debug = window as Window & {captureRaceFrame?: typeof capture; checkRaceVideo?: typeof smokeVideo};
    debug.captureRaceFrame = capture;
    debug.checkRaceVideo = smokeVideo;
    return () => { delete debug.captureRaceFrame; delete debug.checkRaceVideo; };
  }, []);

  if (process.env.NODE_ENV === 'production') return null;
  return (
    <main>
      <p>開発用の定点ショット。ブラウザーから captureRaceFrame を呼び出してください。</p>
      {input && <TrackScene race={input.race} plan={input.plan} progress={0} selected={null}
        followKey={0} onSelect={() => {}} quality={input.quality === 'export' ? EXPORT_QUALITY : PREVIEW_QUALITY}
        verification={verification} elevationScale={input.elevationScale} weather={input.weather} going={input.going}
        courseDay={input.courseDay} postTime={input.postTime} />}
    </main>
  );
}
