import type {Race, Horse, ReplayRace, PlaybackMode} from './types';
import type {PlaybackPlan} from './playback';
export default function PlaybackNotes({
  race,
  horse,
  replay,
  mode,
  plan,
  onClear,
}: {
  race: Race;
  horse?: Horse;
  replay?: ReplayRace;
  mode: PlaybackMode;
  plan: PlaybackPlan;
  onClear: () => void;
}) {
  const actual = mode === 'replay',
    record = horse ? replay?.horses[String(horse.number)] : undefined;
  const progressing =
    actual && replay
      ? race.horses
          .map((h) => {
            const r = replay.horses[String(h.number)],
              first = r.corners.find((v) => v !== null);
            return {
              h,
              gain:
                first === undefined || first === null || r.corners[3] === null
                  ? 0
                  : first - r.corners[3],
            };
          })
          .sort((a, b) => b.gain - a.gain)
          .slice(0, 2)
      : [];
  return (
    <aside>
      <div className="aside-heading">
        <p className="eyebrow">{actual ? 'RECORDED RACE' : 'MODEL FORECAST'}</p>
        <h3>
          {horse
            ? `${horse.number} ${horse.name}`
            : actual
              ? '実際の順位の変化'
              : '予想隊列の見どころ'}
        </h3>
        {horse ? (
          <button className="text-button" onClick={onClear}>
            全体に戻る ×
          </button>
        ) : (
          <p>
            {actual
              ? '記録された通過順位を使って再生します。'
              : '過去走を学習したモデルの推定です。'}
          </p>
        )}
      </div>
      {horse ? (
        <>
          <div className="horse-meta">
            <b>{horse.coat}</b>
            <span>{horse.age}歳</span>
            <span>{horse.weight}kg</span>
            <span>{horse.jockey}</span>
          </div>
          <div className="equipment">
            <span>馬具</span>
            <b>
              {horse.blinker === null
                ? '情報なし'
                : horse.blinker
                  ? 'ブリンカー着用'
                  : 'ブリンカーなし'}
            </b>
            {horse.blinker && <small>3Dでは青いメンコで代用</small>}
          </div>
          <h4>{actual ? '通過順位 → 入線順位' : '予想位置の変化'}</h4>
          <div className="mini-flow result-flow">
            {plan.checkpoints.slice(1).map((c, i) => {
              const rank = c.ranks[race.horses.indexOf(horse)];
              return (
                <div key={i}>
                  <span>{c.label}</span>
                  <strong>
                    {rank === null ? '—' : actual ? rank : Math.round(rank)}
                    <small>{actual ? '番手' : '番手相当'}</small>
                  </strong>
                </div>
              );
            })}
          </div>
          <p className="aside-note">
            {actual ? '実際の上がり3F' : '予想上がり3F'}{' '}
            <b>{(actual ? record?.last3 : horse.last3)?.toFixed(1) ?? '—'}秒</b>
          </p>
          {actual && record?.finish !== record?.officialFinish && (
            <p className="aside-note">
              入線順位と確定着順が異なります。確定着順：{record?.officialFinish ?? '—'}位
            </p>
          )}
        </>
      ) : actual ? (
        <>
          <div className="note-card">
            <span>OBSERVED MOVEMENT</span>
            <h4>道中で位置を上げた馬</h4>
            {progressing.map(({h}) => (
              <p key={h.number}>
                <b>
                  {h.number} {h.name}
                </b>
                <br />
                {replay!.horses[String(h.number)].corners.filter((v) => v !== null).join(' → ')} →
                入線{replay!.horses[String(h.number)].finish ?? '—'}位
              </p>
            ))}
          </div>
          <p className="aside-note">
            下の地点ボタンで各コーナーとゴールに移動できます。馬番の並びも再生に合わせて更新します。
          </p>
        </>
      ) : (
        <>
          <div className="note-card">
            <span>EARLY POSITION</span>
            <h4>前方をうかがう馬</h4>
            <p>
              {[...race.horses]
                .sort((a, b) => a.positions[0] - b.positions[0])
                .slice(0, 3)
                .map((h) => `${h.number} ${h.name}`)
                .join('、')}
            </p>
          </div>
          <p className="aside-note">
            直線は4角予想と予想上がり3Fから相対的な前進量を補間します。入線順位の実績は使用しません。
          </p>
        </>
      )}
      <div className="data-note">
        <span>● {actual ? '実績に基づく再生' : 'モデルによる予想'}</span>
        <p>
          {actual
            ? '各コーナーとゴールの順位はDBの記録です。その間の追い抜き位置・時刻、馬間距離、内外の進路は補間です。'
            : '学習は2025年末まで。当日の結果を使わず生成した予想です。最新出馬表の自動更新は未接続です。'}
        </p>
      </div>
    </aside>
  );
}
