'use client';
import {useEffect, useMemo, useRef, useState} from 'react';
import Link from 'next/link';
import TrackScene, {type VerifyFrame} from './TrackScene';
import type {CameraMode} from './raceCamera';
import PlaybackNotes from './PlaybackNotes';
import HorseTable, {dateLabel} from './HorseTable';
import {framePalette} from './frameColors';
import ModelNotes from './ModelNotes';
import {courseFor} from './courses';
import {availableKinds, raceElevationScale} from './trackGeometry';
import {buildPlayback, playbackFrame, playbackEndProgress} from './playback';
import type {Dataset, Replays, PlaybackMode} from './types';
import {parsePaceForecast, paceForRace, type PaceForecast} from './paceForecast';
import EditPanel from './EditPanel';
import {
  applyEdits,
  loadEdits,
  saveEdits,
  clampPlacement,
  editCount,
  editsFile,
  parseEditsFile,
  type RaceEdits,
} from './manualEdit';
import type {Placement} from './raceLayout';
import {RaceRecorder, mp4MimeType, RECORDING_SCENE_SIZE, prepareRecordingOverlay} from './recorder';
import {completeTick, outputSeconds, progressAtTime, RACE_FPS} from './raceClock';
import {EXPORT_QUALITY} from './sceneQuality';
import {raceOverlay} from './raceOverlay';

async function readJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw Error('データの読み込みに失敗しました');
  return r.json() as Promise<T>;
}

export default function Home() {
  const [data, setData] = useState<Dataset | null>(null),
    [replays, setReplays] = useState<Replays | null>(null),
    [err, setErr] = useState(''),
    [replayError, setReplayError] = useState('');
  const [raceId, setRaceId] = useState(''),
    [selected, setSelected] = useState<number | null>(null),
    [progress, setProgress] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(2),
    [followKey, setFollowKey] = useState(0),
    [cameraMode, setCameraMode] = useState<CameraMode>('pack'),
    [tab, setTab] = useState('horses'),
    [modeChoice, setMode] = useState<PlaybackMode>('replay'),
    // Race-flow model output: bundled file, optionally replaced by a file the user imports.
    [bundledPace, setBundledPace] = useState<PaceForecast | null>(null),
    [importedPace, setImportedPace] = useState<{file: PaceForecast; name: string} | null>(null),
    [paceMessage, setPaceMessage] = useState(''),
    [useLegacy, setUseLegacy] = useState(false),
    // Manual edit mode: the forecast with hand-placed horses at four moments of the race.
    [editing, setEditing] = useState(false),
    // Read on the client only; nothing edited is rendered until edit mode is switched on.
    [allEdits, setAllEdits] = useState<Record<string, RaceEdits>>(() =>
      typeof window === 'undefined' ? {} : loadEdits(),
    ),
    [editMessage, setEditMessage] = useState(''),
    // Video recording of one forecast playthrough (race view + running order) as MP4.
    [recording, setRecording] = useState(false),
    [recordMessage, setRecordMessage] = useState(''),
    [recordSupport, setRecordSupport] = useState<'checking' | 'export' | 'simple' | 'unavailable'>('checking'),
    [exportStatus, setExportStatus] = useState<{done: number; count: number} | null>(null);
  const recorder = useRef<RaceRecorder | null>(null);
  const exportFrame = useRef<VerifyFrame | null>(null);
  const exportAbort = useRef<AbortController | null>(null);
  const codec = useRef<VideoEncoderConfig | null>(null);
  useEffect(() => {
    let mounted = true;
    import('./videoExport').then(module => module.h264Config()).catch(() => null).then(config => {
      if (!mounted) return;
      codec.current = config;
      setRecordSupport(config ? 'export' : mp4MimeType() ? 'simple' : 'unavailable');
    });
    return () => {
      mounted = false;
      exportAbort.current?.abort();
    };
  }, []);
  const progressRef = useRef(0);
  const updateProgress = (value: number) => {
    progressRef.current = value;
    setProgress(value);
  };
  useEffect(() => {
    Promise.allSettled([
      readJson<Dataset>('/data/forecasts.json'),
      readJson<Replays>('/data/replays.json'),
      readJson<unknown>('/data/pace_forecasts.json'),
    ]).then(([f, r, p]) => {
      if (p.status === 'fulfilled') {
        const parsed = parsePaceForecast(p.value);
        if (parsed.ok) setBundledPace(parsed.data);
        else setPaceMessage(`同梱の展開予想を読み込めません（${parsed.error}）`);
      }
      if (f.status === 'fulfilled') {
        setData(f.value);
        setRaceId(f.value.races[0]?.id || '');
      } else setErr(String(f.reason));
      if (r.status === 'fulfilled') setReplays(r.value);
      else
        setReplayError(
          '実績データを読み込めませんでした。再読み込みするか、モデル予想に切り替えてください。',
        );
    });
  }, []);
  const race = data?.races.find((r) => r.id === raceId);
  const distance = race?.distance ?? 0;
  const horse = race?.horses.find((h) => h.number === selected),
    course = race ? courseFor(race) : null,
    replay = replays?.races[raceId],
    // A race forecast from its entry card has no replay until a result is added (data/results).
    mode: PlaybackMode =
      editing || (race && !replay && (race.historical === false || replays !== null)) ? 'forecast' : modeChoice;
  const paceFile = importedPace?.file ?? bundledPace;
  const paceMatch = useMemo(
    () => (race ? paceForRace(paceFile, race) : {pace: null}),
    [race, paceFile],
  );
  const pace = useLegacy ? null : paceMatch.pace;
  const raceEdits = allEdits[raceId];
  const plan = useMemo(() => {
    if (!race) return null;
    const base = buildPlayback(race, mode, replay, pace);
    return editing ? applyEdits(race, base, raceEdits ?? {}) : base;
  }, [race, mode, replay, pace, editing, raceEdits]);
  const endProgress = useMemo(
    () => (race && plan ? playbackEndProgress(race, plan) : 1),
    [race, plan],
  );
  useEffect(() => {
    if (!playing || distance <= 0) return;
    const started = performance.now();
    const from = outputSeconds(progressRef.current * distance, speed);
    const endTime = outputSeconds(endProgress * distance, speed);
    let previousTick = completeTick(from);
    let id = 0;
    const tick = (now: number) => {
      const seconds = from + Math.max(0, now - started) / 1000;
      if (seconds >= endTime) {
        updateProgress(endProgress);
        setPlaying(false);
        return;
      }
      const targetTick = completeTick(seconds);
      if (targetTick > previousTick) {
        previousTick = targetTick;
        updateProgress(progressAtTime(targetTick / RACE_FPS, distance, speed));
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, speed, distance, endProgress]);
  const updateEdits = (change: (e: RaceEdits) => RaceEdits) =>
    setAllEdits((all) => {
      const next = {...all, [raceId]: change(all[raceId] ?? {})};
      saveEdits(next);
      return next;
    });
  const exportEdits = () => {
    if (!race || !raceEdits) return;
    const blob = new Blob([JSON.stringify(editsFile({[raceId]: raceEdits}), null, 1)], {
      type: 'application/json',
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `manual-edits_${raceId}_${race.name}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    setEditMessage(`${a.download} を保存しました`);
  };
  const importEdits = async (file: File) => {
    let parsed: ReturnType<typeof parseEditsFile>;
    try {
      parsed = parseEditsFile(JSON.parse(await file.text()));
    } catch (e) {
      return setEditMessage(`${file.name} はJSONとして読めません（${String(e)}）`);
    }
    if (!parsed.ok) return setEditMessage(`${file.name} は読み込めません：${parsed.error}`);
    const ids = Object.keys(parsed.races);
    const overwritten = ids.filter((id) => editCount(allEdits[id]) > 0);
    if (
      overwritten.length &&
      !confirm(`${overwritten.length}レースの今の手動エディットをファイルの内容で置き換えますか？`)
    )
      return;
    const races = parsed.races;
    setAllEdits((all) => {
      const next = {...all, ...races};
      saveEdits(next);
      return next;
    });
    const names = ids.map((id) => data?.races.find((r) => r.id === id)?.name ?? id);
    setEditMessage(
      `${names.join('・')}の編集を読み込みました` +
        (ids.includes(raceId) ? '' : '（このレースの編集は入っていません）'),
    );
  };
  const importPace = async (file: File) => {
    try {
      const parsed = parsePaceForecast(JSON.parse(await file.text()));
      if (!parsed.ok) return setPaceMessage(`${file.name} は取り込めません：${parsed.error}`);
      setImportedPace({file: parsed.data, name: file.name});
      setUseLegacy(false);
      setMode('forecast');
      setPaceMessage(
        `${file.name}（${parsed.data.model.name}）を取り込みました：${Object.keys(parsed.data.races).length}レース`,
      );
    } catch (e) {
      setPaceMessage(`${file.name} はJSONとして読めません（${String(e)}）`);
    }
  };
  const frame = useMemo(
    () => (race && plan ? playbackFrame(race, plan, progress) : null),
    [race, plan, progress],
  );
  // Seek-bar background: JRA's official section graph, one lap per tile, travel left->right
  // with the winning post on the right edge. Longer races repeat the lap leftwards.
  // The bar runs the way the field passes the stands: left to right on a left-handed course,
  // and mirrored, right to left, on a right-handed one.
  const reversed = !!course && !course.left;
  const profileStyle =
    race && course
      ? {
          backgroundImage: `url(${course.profileImage})`,
          backgroundSize: `${(course.lap / race.distance) * 100}% 100%`,
        }
      : undefined;
  const actual = mode === 'replay',
    seek = (p: number) => {
      setPlaying(false);
      updateProgress(p);
    };
  // The edit point the race is paused on, if any: horses can be dragged only there.
  const editPoint = editing && !playing ? frame?.exact?.edit : undefined;
  const editKnot = editPoint ? plan!.knots.indexOf(frame!.exact!.distance) : -1;
  const moveHorse = (index: number, placement: Placement) => {
    if (!race || !editPoint) return;
    const number = String(race.horses[index].number);
    updateEdits((e) => ({...e, [editPoint]: {...e[editPoint], [number]: clampPlacement(placement)}}));
  };
  const saveVideo = (blob: Blob, name: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };

  const startExport = async (config: VideoEncoderConfig) => {
    if (!race || !plan || !course || exportAbort.current) return;
    const abort = new AbortController();
    exportAbort.current = abort;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const wasPlaying = playing;
    setPlaying(false);
    setExportStatus({done: 0, count: 0});
    setRecordMessage('');
    try {
      const {encodeRaceVideo, videoTimeline} = await import('./videoExport');
      const timeline = videoTimeline(race.distance, speed, endProgress);
      setExportStatus({done: 0, count: timeline.count});
      await prepareRecordingOverlay(course.profileImage);
      abort.signal.throwIfAborted();
      if (!exportFrame.current) throw Error('3D書き出しを開始できません');
      const debug = process.env.NODE_ENV === 'development'
        ? window as Window & {paddockExportHashes?: string[]; paddockFailAt?: number}
        : null;
      if (debug?.paddockExportHashes) debug.paddockExportHashes.length = 0;
      const blob = await encodeRaceVideo({
        config, timeline, signal: abort.signal,
        draw: frame => exportFrame.current!({
          ...RECORDING_SCENE_SIZE, progress: frame.progress, cameraMode, speed, selected, reduced,
          outputTime: frame.seconds, sequential: true,
        }),
        overlay: position => raceOverlay(race, plan, position, editing, pace?.model),
        progress: (done, count) => setExportStatus({done, count}),
        hash: debug?.paddockExportHashes ? hash => debug.paddockExportHashes!.push(hash) : undefined,
        failAt: debug?.paddockFailAt,
      });
      const name = `${editing ? '手動エディット' : 'モデル予想'}_${race.date}_${race.name}.mp4`;
      saveVideo(blob, name);
      setRecordMessage(`${name} を保存しました`);
    } catch (error) {
      setRecordMessage(abort.signal.aborted ? '書き出しを中止しました。' : `MP4を書き出せませんでした（${String(error)}）`);
    } finally {
      exportAbort.current = null;
      setExportStatus(null);
      setPlaying(wasPlaying);
    }
  };
  const overlay = useRef<() => ReturnType<typeof raceOverlay>>(() => { throw Error('No race'); });
  if (race && plan) overlay.current = () => raceOverlay(race, plan, progressRef.current, editing, pace?.model);
  const startRecording = () => {
    if (!race || recording || exportStatus || exportAbort.current) return;
    if (codec.current) return void startExport(codec.current);
    const type = mp4MimeType();
    if (!type)
      return setRecordMessage(
        'このブラウザーはMP4の録画に対応していません。Chrome・Edge（最新版）またはSafariでお試しください。',
      );
    try {
      recorder.current = new RaceRecorder(type, () => overlay.current());
    } catch (e) {
      return setRecordMessage(`録画を開始できません（${String(e)}）`);
    }
    const name = `${editing ? '手動エディット' : 'モデル予想'}_${race.date}_${race.name}.mp4`;
    recorder.current.name = name;
    setRecording(true);
    setRecordMessage('簡易録画中。上位6頭（6頭未満なら全頭）がゴールするまで再生します。一時停止すると録画を終了します。');
    updateProgress(0);
    setPlaying(true);
  };
  // A recording is one playthrough: it ends when playback stops (the finish, a pause, a seek
  // or a race/mode change), holding the last frame briefly at the finish.
  useEffect(() => {
    if (!recording || playing) return;
    const rec = recorder.current;
    const timer = setTimeout(
      () => {
        recorder.current = null;
        setRecording(false);
        rec
          ?.stop()
          .then((blob) => {
            saveVideo(blob, rec.name);
            setRecordMessage(`${rec.name} を保存しました`);
          })
          .catch((e) => setRecordMessage(`録画を保存できませんでした（${String(e)}）`));
      },
      progress >= endProgress ? 1000 : 0,
    );
    return () => clearTimeout(timer);
  }, [recording, playing, progress, endProgress]);
  const startEditing = () => {
    if (!race) return;
    setEditing(true);
    setPlaying(false);
    updateProgress(0);
  };
  return (
    <main>
      {exportStatus && (
        <div className="export-status" role="status">
          <span>{exportStatus.count ? `MP4書き出し中　${exportStatus.done.toLocaleString()} / ${exportStatus.count.toLocaleString()} コマ` : 'MP4書き出しを準備中'}</span>
          <button onClick={() => exportAbort.current?.abort()}>中止</button>
        </div>
      )}
      <header>
        <Link className="brand" href="/">
          <span className="brand-icon">♞</span>
          <span>
            PADDOCK<small>STUDIO</small>
          </span>
        </Link>
        <nav>
          <span>展開シアター</span>
          <button onClick={() => setTab('model')}>モデルについて</button>
        </nav>
        <span className="header-tag">JRA GⅠ · RACE FLOW</span>
      </header>
      <div className="page-head">
        <div>
          <p className="eyebrow">READ THE RACE. WATCH IT UNFOLD.</p>
          <h1>レースの流れを、目で読む。</h1>
          <p className="lead">実際の通過順位とモデル予想を、小さな競馬場で見比べる。</p>
        </div>
        <div className="version">
          ● {actual ? '実績再生' : pace ? `展開予想 ${pace.model}` : '展開モデル v0.1'}
          <small>
            {actual
              ? '通過順位・入線順位を反映'
              : pace
                ? '途中順位・前後・内外・上がりを取り込み'
                : '結果を使わないモデル予想'}
          </small>
        </div>
      </div>
      {err && (
        <p role="alert" className="notice">
          {err}。再読み込みしてください。
        </p>
      )}
      {!data && !err && <p className="notice">馬柱と展開データを読み込んでいます…</p>}
      {race && course && data && plan && frame && (
        <div inert={!!exportStatus}>
          <section className="race-bar">
            <div>
              <span className="pill">GⅠ</span>
              <h2>{race.name}</h2>
              <span className="muted">
                {dateLabel(race.date)}　{race.venue} {race.surface} {race.distance.toLocaleString()}
                m　{race.horses.length}頭
              </span>
            </div>
            <label>
              レースを選択
              <select
                aria-label="レースを選択"
                value={raceId}
                onChange={(e) => {
                  setRaceId(e.target.value);
                  setSelected(null);
                  seek(0);
                }}
              >
                {data.races.map((r) => (
                  <option value={r.id} key={r.id}>
                    {dateLabel(r.date)} {r.name}
                  </option>
                ))}
              </select>
            </label>
          </section>
          {race.entries && (
            <p className="notice" role="status">
              出馬表：{race.entries.source}（{race.entries.fetchedAt.replace('T', ' ').slice(0, 16)}
              取得）
              {race.entries.scratched.length > 0 &&
                `　取消：${race.entries.scratched.map((h) => `${h.number ?? ''}${h.name}`).join('、')}`}
              {replay
                ? `・予想は出馬表と発走前のデータのみで作成。実績：${replay.source}`
                : '・発走前のため実績再生はありません。'}
            </p>
          )}
          <section className="mode-bar" aria-label="再生データの選択">
            <div className="mode-switch">
              <button
                aria-pressed={actual}
                disabled={!replay}
                onClick={() => {
                  setMode('replay');
                  setEditing(false);
                  seek(0);
                }}
              >
                実績再生
              </button>
              <button
                aria-pressed={!actual && !editing}
                onClick={() => {
                  setMode('forecast');
                  setEditing(false);
                  seek(0);
                }}
              >
                モデル予想
              </button>
              <button aria-pressed={editing} onClick={startEditing}>
                手動エディット
              </button>
            </div>
            <p>
              {editing
                ? '予想をもとに、序盤・中盤・4角・最終直線の4地点で馬を手で動かせます。編集は自動保存されます。'
                : actual
                ? '各コーナーと入線順位は実データ。その間の動きは補間。'
                : pace
                  ? `展開予想モデル（${pace.model}）の途中順位・前後の馬身差・内外・上がりで再生。実績は使用しません。`
                  : '馬柱を学習した予想。実際の通過順位・入線順位は使用しません。'}
            </p>
            {!actual && (
              <div className="pace-source" data-testid="pace-source">
                <label>
                  予想データ
                  <select
                    aria-label="予想データ"
                    value={useLegacy || !paceMatch.pace ? 'legacy' : 'pace'}
                    onChange={(e) => setUseLegacy(e.target.value === 'legacy')}
                  >
                    <option value="pace" disabled={!paceMatch.pace}>
                      展開予想モデル{importedPace ? '（取り込み）' : ''}
                    </option>
                    <option value="legacy">通過順位モデル（旧）</option>
                  </select>
                </label>
                <label className="file-button">
                  展開予想JSONを取り込む
                  <input
                    type="file"
                    accept="application/json,.json"
                    aria-label="展開予想JSONを取り込む"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) importPace(f);
                      e.target.value = '';
                    }}
                  />
                </label>
                {importedPace && (
                  <button
                    className="text-button"
                    onClick={() => {
                      setImportedPace(null);
                      setPaceMessage('取り込んだ展開予想を外し、同梱データに戻しました');
                    }}
                  >
                    取り込みを解除
                  </button>
                )}
              </div>
            )}
          </section>
          {!actual && (paceMessage || paceMatch.reason || (paceFile && !paceMatch.pace)) && (
            <p className="notice" role="status">
              {paceMessage}
              {paceMatch.reason
                ? `${paceMessage ? '　' : ''}このレースの展開予想は使えません：${paceMatch.reason}。旧モデルで表示しています。`
                : paceFile && !paceMatch.pace
                  ? `${paceMessage ? '　' : ''}このレースは展開予想データに含まれていないため旧モデルで表示しています。`
                  : ''}
            </p>
          )}
          {actual && replayError && (
            <p role="alert" className="notice">
              {replayError}
            </p>
          )}
          <div className="workspace">
            <section className="theater">
              {exportStatus && (
                <div className="export-renderer" aria-hidden="true">
                  <TrackScene race={race} plan={plan} progress={0} selected={selected} followKey={0}
                    onSelect={() => {}} quality={EXPORT_QUALITY} verification={exportFrame} />
                </div>
              )}
              <div className="stage">
                <TrackScene
                  race={race}
                  plan={plan}
                  progress={progress}
                  playing={playing}
                  speed={speed}
                  selected={selected}
                  followKey={followKey}
                  cameraMode={cameraMode}
                  onSelect={setSelected}
                  editAt={editPoint ? frame.exact!.distance : null}
                  onEdit={moveHorse}
                  onRendered={recording ? (c) => recorder.current?.drawFrame(c) : undefined}
                  recordingSize={recording ? RECORDING_SCENE_SIZE : undefined}
                  suspended={!!exportStatus}
                />
                <div className="scene-top">
                  <span
                    className={`scene-status ${actual ? 'recorded-status' : ''}`}
                    data-testid="playback-status"
                  >
                    ● {editing ? 'エディット' : actual ? '実績' : '予想'} · {frame.label}
                    {actual && frame.exact?.observed
                      ? '（記録地点）'
                      : editPoint
                        ? '（ドラッグで移動）'
                        : progress > 0
                        ? '（補間）'
                        : ''}
                  </span>
                  <div className="view-buttons">
                    <button aria-pressed={cameraMode === 'pack'} onClick={() => {
                      setCameraMode('pack');
                      setFollowKey((k) => k + 1);
                    }}>馬群を追尾</button>
                    <button aria-pressed={cameraMode === 'broadcast'} title="スタートは俯瞰・コーナーは先頭馬のアオリ・直線は全頭・最終直線は上位6頭" onClick={() => {
                      setCameraMode('broadcast');
                      setFollowKey((k) => k + 1);
                    }}>自動演出</button>
                  </div>
                </div>
                <div className="scene-bottom">
                  <span>
                    {editPoint
                      ? '馬をドラッグ：前後＝進行方向 · 左右＝内外 · 背景ドラッグで回転'
                      : 'ドラッグで回転 · スクロールで拡大 · 馬を選択'}
                  </span>
                  <span>
                    JRA公式図面ベース · 馬は約1.7倍 · 高低差は{raceElevationScale(race.venueCode)}倍で表示
                  </span>
                </div>
              </div>
              <div className="playback">
                <button
                  className="play"
                  disabled={!plan.available}
                  aria-label={playing ? '一時停止' : '展開を再生'}
                  onClick={() => {
                    if (progress >= endProgress) updateProgress(0);
                    setPlaying(!playing);
                  }}
                >
                  {playing ? 'Ⅱ' : '▶'}
                </button>
                <button className="reset" aria-label="スタートに戻す" onClick={() => seek(0)}>
                  ↺
                </button>
                <div className={reversed ? 'timeline reversed' : 'timeline'}>
                  <div className="profile-track" aria-hidden="true" style={profileStyle}>
                    <span className="profile-done" style={{width: `${Math.min(1, progress) * 100}%`}} />
                    {plan.checkpoints.map((c) => (
                      <span
                        key={`${c.label}-${c.distance}`}
                        className="profile-tick"
                        style={{left: `${(c.distance / race.distance) * 100}%`}}
                      />
                    ))}
                    <span className="profile-now" style={{left: `${Math.min(1, progress) * 100}%`}} />
                  </div>
                  <input
                    aria-label="展開の進行"
                    type="range"
                    min="0"
                    max="1"
                    step="any"
                    value={Math.min(1, progress)}
                    onChange={(e) => seek(Number(e.target.value))}
                  />
                  <div className="phases">
                    {plan.checkpoints.map((c) => (
                      <button
                        key={`${c.label}-${c.distance}`}
                        aria-label={`${c.label}へ移動`}
                        onClick={() => seek(c.distance / race.distance)}
                        style={{
                          left: `${(reversed ? 1 - c.distance / race.distance : c.distance / race.distance) * 100}%`,
                        }}
                        className={[
                          frame.exact === c ? 'checkpoint-active' : '',
                          c.edit ? 'checkpoint-edit' : '',
                          c.edited ? 'checkpoint-edited' : '',
                        ].join(' ')}
                      >
                        {c.label}
                      </button>
                    ))}
                  </div>
                </div>
                <select
                  aria-label="再生速度"
                  value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                >
                  <option value=".5">0.5×</option>
                  <option value="1">1×</option>
                  <option value="2">2×</option>
                </select>
                {!actual && (
                  <button
                    className={`record ${recording ? 'recording' : ''}`}
                    disabled={!plan.available || recording || recordSupport === 'checking' || recordSupport === 'unavailable'}
                    aria-label={recording ? '録画中' : '予想を録画（MP4）'}
                    title="レース画面と隊列をMP4で録画"
                    onClick={startRecording}
                  >
                    <i aria-hidden="true" />
                    {recording ? '録画中' : recordSupport === 'simple' ? '簡易録画' : 'MP4書き出し'}
                  </button>
                )}
              </div>
              {!actual && recordSupport === 'unavailable' && (
                <p className="notice record-message">このブラウザーはMP4の録画に対応していません。Chrome・Edge（最新版）またはSafariでお試しください。</p>
              )}
              {!actual && recordMessage && (
                <p className="notice record-message" role="status">
                  {recordMessage}
                </p>
              )}
              {editing && (
                <EditPanel
                  race={race}
                  plan={plan}
                  edits={raceEdits ?? {}}
                  active={editPoint}
                  layout={editKnot >= 0 ? plan.layouts[editKnot] : null}
                  selected={selected}
                  onSeek={(d) => seek(d / race.distance)}
                  onMove={moveHorse}
                  onReset={(key, number) =>
                    updateEdits((e) => {
                      if (!key) return {};
                      if (number === undefined) return {...e, [key]: undefined};
                      const rest = {...e[key]};
                      delete rest[String(number)];
                      return {...e, [key]: rest};
                    })
                  }
                  onExit={() => setEditing(false)}
                  onExport={exportEdits}
                  onImport={importEdits}
                  message={editMessage}
                />
              )}
              <div className="live-order-panel">
                <div className="live-order-heading">
                  <b>
                    {progress === 0
                      ? 'ゲート配置'
                      : actual && frame.exact?.observed
                        ? `${frame.exact.label}の実順位`
                        : '現在の隊列'}
                  </b>
                  <small>
                    {progress === 0
                      ? '左から馬番順'
                      : actual && frame.exact?.observed
                        ? '記録された順位順・数字は馬番'
                        : '先頭 → 後方（補間表示）'}
                  </small>
                </div>
                <div className="live-order" data-testid="live-order">
                  {frame.order.map(({horse: h, index}) => (
                    <button
                      key={h.number}
                      data-horse={h.number}
                      aria-label={`${h.number} ${h.name}を選択`}
                      title={`${h.name}${actual && frame.exact?.observed ? '：' + frame.exact.ranks[index] + '番手' : ''}`}
                      onClick={() => setSelected(h.number)}
                      style={{
                        ...framePalette(h.gate),
                      }}
                    >
                      {h.number}
                    </button>
                  ))}
                </div>
              </div>
              <div className="track-key">
                <span>
                  <i className="key-turf" />
                  芝コース
                </span>
                {availableKinds(race).includes('dirt') && (
                  <span>
                    <i className="key-dirt" />
                    ダート
                  </span>
                )}
                <span>馬を選択すると馬名を表示</span>
              </div>
              <div className="course-strip">
                <div>
                  <small>コース</small>
                  <strong>
                    {race.venue} · {course.left ? '左回り' : '右回り'}
                  </strong>
                </div>
                <div>
                  <small>直線距離</small>
                  <strong>
                    {course.straight}
                    <em>m</em>
                  </strong>
                </div>
                <div>
                  <small>高低差</small>
                  <strong>
                    {course.rise}
                    <em>m</em>
                  </strong>
                </div>
                <a href={course.source} target="_blank" rel="noreferrer">
                  JRA公式コース情報 ↗
                </a>
              </div>
              <p className="caption">
                {race.venue}：JRA公式平面図・周長・高低断面図を基に立体化（{course.label}
                ）。図面読取誤差を含みます。高低差は{raceElevationScale(race.venueCode)}
                倍、馬は見やすさのため約1.7倍で表示しています。
                観測地点間の追い抜き位置・時刻、馬間距離と内外の進路は補間です。地点ボタンで通過順位を確認できます。
              </p>
            </section>
            <PlaybackNotes
              race={race}
              horse={horse}
              replay={replay}
              mode={mode}
              plan={plan}
              pace={pace}
              onClear={() => setSelected(null)}
            />
          </div>
          <section className="detail">
            <div className="tabs">
              <button className={tab === 'horses' ? 'active' : ''} onClick={() => setTab('horses')}>
                {actual ? '馬柱と実通過順位' : '馬柱と予想隊列'} <span>{race.horses.length}</span>
              </button>
              <button className={tab === 'model' ? 'active' : ''} onClick={() => setTab('model')}>
                モデル・データの根拠
              </button>
            </div>
            {tab === 'horses' ? (
              <HorseTable
                race={race}
                horse={horse}
                replay={replay}
                mode={mode}
                pace={pace}
                onSelect={setSelected}
              />
            ) : (
              <ModelNotes data={data} pace={paceFile} />
            )}
          </section>
        </div>
      )}
      <footer>
        <span>PADDOCK STUDIO</span>
        <p>馬データ：指定フォルダのmykeibadb / JV-Data　·　コース情報：JRA</p>
        <span>小さな馬たちと、展開を読む。</span>
      </footer>
    </main>
  );
}
