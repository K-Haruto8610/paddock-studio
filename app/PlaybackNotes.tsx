import type {Race, Horse, ReplayRace, PlaybackMode} from './types';
import type {PlaybackPlan} from './playback';
import {laneLabel, PACE_LABELS, type RacePace} from './paceForecast';
export default function PlaybackNotes({
  race,
  horse,
  replay,
  mode,
  plan,
  pace,
  onClear,
}: {
  race: Race;
  horse?: Horse;
  replay?: ReplayRace;
  mode: PlaybackMode;
  plan: PlaybackPlan;
  pace?: RacePace | null;
  onClear: () => void;
}) {
  const paced = mode !== 'replay' && pace ? pace : null,
    paceHorse = horse && paced ? paced.horses[race.horses.indexOf(horse)] : undefined;
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
              const rank = c.ranks[race.horses.indexOf(horse)],
                at = paceHorse?.positions.find((p) => p.corner === c.corner);
              return (
                <div key={i}>
                  <span>{c.label}</span>
                  <strong>
                    {rank === null ? '—' : actual || paced ? rank : Math.round(rank)}
                    <small>{actual || paced ? '番手' : '番手相当'}</small>
                  </strong>
                  {at && (
                    <small className="pace-detail">
                      {at.gap.toFixed(1)}馬身 · {laneLabel(at.lane)}
                      {at.held ? ' · 控え' : ''}
                      {at.boxed ? ' · 包まれ' : ''}
                    </small>
                  )}
                </div>
              );
            })}
          </div>
          <p className="aside-note">
            {actual ? '実際の上がり3F' : '予想上がり3F'}{' '}
            <b>
              {(actual ? record?.last3 : (paceHorse?.last3 ?? horse.last3))?.toFixed(1) ?? '—'}秒
            </b>
          </p>
          {paceHorse && (paceHorse.kickDelay ?? 0) > 0.05 && (
            <p className="aside-note">
              最終角で外に馬・前が壁になり仕掛けが遅れる予想：+{paceHorse.kickDelay!.toFixed(2)}秒
              （前が開けば{paceHorse.last3Free?.toFixed(1) ?? '—'}秒）
            </p>
          )}
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
          {paced?.racePace && (
            <div className="note-card" data-testid="pace-card">
              <span>RACE PACE</span>
              <h4>予想ペース：{PACE_LABELS[paced.racePace.label]}（GⅠ比）</h4>
              <p>
                前半3Fはコースの標準より{Math.abs(paced.racePace.early).toFixed(1)}秒
                {paced.racePace.early < 0 ? '速い' : '遅い'}予想。
                {paced.racePace.frontRunners && paced.racePace.frontRunners.length > 0 && (
                  <>
                    <br />
                    先行したい馬（{paced.racePace.frontRunners.length}頭）：
                    {paced.racePace.frontRunners
                      .map((n) => race.horses.find((h) => h.number === n))
                      .filter((h) => h)
                      .map((h) => `${h!.number} ${h!.name}`)
                      .join('、')}
                  </>
                )}
              </p>
            </div>
          )}
          <div className="note-card">
            <span>EARLY POSITION</span>
            <h4>前方をうかがう馬</h4>
            <p>
              {race.horses
                .map((h, i) => ({
                  h,
                  early: paced ? paced.horses[i].positions[0].rank : h.positions[0],
                }))
                .sort((a, b) => a.early - b.early)
                .map(({h}) => h)
                .slice(0, 3)
                .map((h) => `${h.number} ${h.name}`)
                .join('、')}
            </p>
          </div>
          {paced && (
            <div className="note-card" data-testid="contention-card">
              <span>POSITION BATTLE</span>
              <h4>位置を獲られて控える馬</h4>
              <p>
                {race.horses
                  .map((h, i) => {
                    const p = paced.horses[i].positions.find((v) => v.held);
                    return {h, p};
                  })
                  .filter((v) => v.p)
                  .sort((a, b) => a.p!.corner - b.p!.corner || a.p!.rank - b.p!.rank)
                  .slice(0, 4)
                  .map(({h, p}) => `${h.number} ${h.name}（${p!.corner}角 希望${p!.wantRank ?? '—'}→${p!.rank}番手）`)
                  .join('、') || '該当なし'}
              </p>
              <h4>仕掛けが遅れる馬</h4>
              <p>
                {race.horses
                  .map((h, i) => ({h, delay: paced.horses[i].kickDelay ?? 0}))
                  .filter((v) => v.delay > 0.05)
                  .sort((a, b) => b.delay - a.delay)
                  .slice(0, 4)
                  .map(({h, delay}) => `${h.number} ${h.name}（+${delay.toFixed(2)}秒）`)
                  .join('、') || '該当なし'}
              </p>
            </div>
          )}
          {paced && paced.horses[0].positions.length > 1 && (
            <div className="note-card" data-testid="move-card">
              <span>MID-RACE MOVE</span>
              <h4>道中で位置を上げそうな馬</h4>
              <p>
                {race.horses
                  .map((h, i) => {
                    const ps = paced.horses[i].positions;
                    return {h, ps, gain: ps[0].rank - ps.at(-1)!.rank};
                  })
                  .filter((v) => v.gain >= 2)
                  .sort((a, b) => b.gain - a.gain)
                  .slice(0, 3)
                  .map(({h, ps}) => `${h.number} ${h.name}（${ps.map((p) => p.rank).join('→')}番手）`)
                  .join('、') || '該当なし'}
              </p>
              <h4>道中で位置を下げそうな馬</h4>
              <p>
                {race.horses
                  .map((h, i) => {
                    const ps = paced.horses[i].positions;
                    return {h, ps, loss: ps.at(-1)!.rank - ps[0].rank};
                  })
                  .filter((v) => v.loss >= 2)
                  .sort((a, b) => b.loss - a.loss)
                  .slice(0, 3)
                  .map(({h, ps}) => `${h.number} ${h.name}（${ps.map((p) => p.rank).join('→')}番手）`)
                  .join('、') || '該当なし'}
              </p>
            </div>
          )}
          {paced && (
            <div className="note-card">
              <span>WIDE TRIP</span>
              <h4>外を回りそうな馬（最終角）</h4>
              <p>
                {race.horses
                  .map((h, i) => ({h, lane: paced.horses[i].positions.at(-1)!.lane}))
                  .sort((a, b) => b.lane - a.lane)
                  .slice(0, 3)
                  .map(({h, lane}) => `${h.number} ${h.name}（${laneLabel(lane)}）`)
                  .join('、')}
              </p>
            </div>
          )}
          <p className="aside-note">
            {paced
              ? '各角の順位・先頭からの馬身差・内外は展開予想モデルの値。直線は予想上がり3Fから推定します。'
              : '直線は4角予想と予想上がり3Fから相対的な前進量を補間します。入線順位の実績は使用しません。'}
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
