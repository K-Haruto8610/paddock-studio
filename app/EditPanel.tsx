'use client';
import type {Race} from './types';
import type {PlaybackPlan} from './playback';
import {editPoints, editCount, type EditKey, type RaceEdits} from './manualEdit';
import {HORSE_LENGTH, LANE_PITCH, type Placement} from './raceLayout';
import {frameColors} from './HorseTable';

/** Controls for manual edit mode: jump to one of the four edit points, nudge the selected horse, undo. */
export default function EditPanel({
  race,
  plan,
  edits,
  active,
  layout,
  selected,
  onSeek,
  onMove,
  onReset,
  onExit,
}: {
  race: Race;
  plan: PlaybackPlan;
  edits: RaceEdits;
  /** The edit point the race is paused on. */
  active: EditKey | undefined;
  /** The plan's placements at that point. */
  layout: Placement[] | null;
  selected: number | null;
  onSeek: (distance: number) => void;
  onMove: (index: number, placement: Placement) => void;
  /** No key: clear every edit of the race. No number: clear the whole point. */
  onReset: (key?: EditKey, number?: number) => void;
  onExit: () => void;
}) {
  const points = editPoints(race),
    index = race.horses.findIndex((h) => h.number === selected),
    horse = index >= 0 ? race.horses[index] : null,
    here = active ? plan.checkpoints.find((c) => c.edit === active) : undefined;
  const nudge = (gap: number, lane: number) => {
    if (!layout || index < 0) return;
    const p = layout[index];
    onMove(index, {gap: p.gap + gap, lane: p.lane + lane});
  };
  const total = editCount(edits);
  return (
    <section className="edit-panel" aria-label="手動エディット" data-testid="edit-panel">
      <div className="edit-head">
        <b>手動エディット</b>
        <small>
          地点を選んで馬をドラッグ。動かした差は次に編集するまで以降の地点にも引き継がれます。
          {total > 0 ? `編集 ${total}件` : '未編集'}
        </small>
        <button className="text-button" onClick={onExit}>
          エディットを終了
        </button>
      </div>
      <div className="edit-points" role="group" aria-label="編集する地点">
        {points.map((p) => {
          const n = Object.keys(edits[p.key] ?? {}).length;
          return (
            <button
              key={p.key}
              aria-pressed={active === p.key}
              onClick={() => onSeek(p.distance)}
            >
              <span>{p.label}</span>
              <small>{Math.round(p.distance).toLocaleString()}m</small>
              {n > 0 && <em aria-label={`${n}頭を編集済み`}>{n}</em>}
            </button>
          );
        })}
      </div>
      {!active ? (
        <p className="edit-hint">上の地点ボタンで移動すると、その地点の馬を動かせます（再生中は編集できません）。</p>
      ) : horse && layout ? (
        <div className="edit-horse">
          <span
            className="edit-badge"
            style={{background: frameColors[horse.gate - 1], color: horse.gate === 1 ? '#333' : '#fff'}}
          >
            {horse.number}
          </span>
          <span className="edit-name">
            {horse.name}
            <small>
              {here?.ranks[index]}番手 · 先頭から{(layout[index].gap / HORSE_LENGTH).toFixed(1)}馬身 ·
              ラチから{layout[index].lane.toFixed(1)}m
            </small>
          </span>
          <div className="edit-nudge" role="group" aria-label="位置の微調整">
            <button onClick={() => nudge(-HORSE_LENGTH, 0)} aria-label="1馬身前へ">
              ▲ 前へ
            </button>
            <button onClick={() => nudge(HORSE_LENGTH, 0)} aria-label="1馬身後ろへ">
              ▼ 後ろへ
            </button>
            <button onClick={() => nudge(0, -LANE_PITCH)} aria-label="1頭分内へ">
              ◀ 内へ
            </button>
            <button onClick={() => nudge(0, LANE_PITCH)} aria-label="1頭分外へ">
              外へ ▶
            </button>
          </div>
          {edits[active]?.[String(horse.number)] && (
            <button className="text-button" onClick={() => onReset(active, horse.number)}>
              この馬を戻す
            </button>
          )}
        </div>
      ) : (
        <p className="edit-hint">馬をクリックまたはドラッグして選択してください。</p>
      )}
      <div className="edit-reset">
        {active && edits[active] && Object.keys(edits[active]!).length > 0 && (
          <button className="text-button" onClick={() => onReset(active)}>
            この地点の編集を戻す
          </button>
        )}
        {total > 0 && (
          <button
            className="text-button"
            onClick={() => {
              if (confirm('このレースの手動エディットをすべて消去しますか？')) onReset();
            }}
          >
            すべて戻す
          </button>
        )}
      </div>
    </section>
  );
}
