import {courseFor} from './courses';
import {framePalette} from './frameColors';
import {playbackFrame, type PlaybackPlan} from './playback';
import type {RecordingOverlay} from './recorder';
import type {Race} from './types';

export function raceOverlay(race: Race, plan: PlaybackPlan, progress: number, editing: boolean,
  paceModel?: string): RecordingOverlay {
  const frame = playbackFrame(race, plan, progress);
  const course = courseFor(race);
  const date = race.date.replace(/^(\d{4})(\d{2})(\d{2})$/, '$1.$2.$3');
  const mode = editing ? '手動エディット' : `モデル予想${paceModel ? ' ' + paceModel : ''}`;
  return {
    title: `${race.name}　${date} ${race.venue}${race.surface}${race.distance}m`,
    status: `● ${mode} · ${frame.label}`,
    orderTitle: progress === 0 ? 'ゲート配置' : '現在の隊列',
    orderNote: progress === 0 ? '左から馬番順' : '先頭 → 後方',
    order: frame.order.map(({horse}) => ({number: horse.number, ...framePalette(horse.gate)})),
    seek: {
      progress,
      reversed: !course.left,
      profileImage: course.profileImage,
      lapFraction: course.lap / race.distance,
      checkpoints: plan.checkpoints.map(c => ({
        label: c.label, at: c.distance / race.distance, active: frame.exact === c,
        edit: !!c.edit, edited: !!c.edited,
      })),
    },
  };
}
