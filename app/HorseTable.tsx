import type {Race, Horse, ReplayRace, PlaybackMode} from './types';
export const frameColors = [
  '#fff',
  '#303534',
  '#d96859',
  '#4f7bab',
  '#e3bf58',
  '#55977e',
  '#d79653',
  '#ca88a6',
];
export const dateLabel = (d: string) => `${d.slice(0, 4)}.${d.slice(4, 6)}.${d.slice(6, 8)}`;
export default function HorseTable({
  race,
  horse,
  replay,
  mode,
  onSelect,
}: {
  race: Race;
  horse?: Horse;
  replay?: ReplayRace;
  mode: PlaybackMode;
  onSelect: (n: number) => void;
}) {
  const actual = mode === 'replay';
  if (horse)
    return (
      <>
        <div className="table-intro">
          <p>{horse.name} の馬柱</p>
          <small>対象レースより前の収録済み過去走</small>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>日付</th>
                <th>レース</th>
                <th>距離</th>
                <th>通過順位 1→4角</th>
                <th>上がり3F</th>
              </tr>
            </thead>
            <tbody>
              {horse.history.map((p, i) => (
                <tr key={i}>
                  <td>{dateLabel(p.date)}</td>
                  <td>{p.name}</td>
                  <td>{p.distance}m</td>
                  <td>{p.corners.map((v) => (v && v > 0 ? v : '—')).join(' → ')}</td>
                  <td>{p.last3?.toFixed(1) || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!horse.history.length && <p className="notice">収録済みの過去走がありません。</p>}
        </div>
      </>
    );
  return (
    <>
      <div className="table-intro">
        <p>{actual ? '対象レースの実通過順位' : '出走馬と予想位置'}</p>
        <small>
          {actual
            ? '同じ順位の馬は同一グループとして表示 · 入線順位を使用'
            : '馬柱から生成したモデル予想'}{' '}
          · ブリンカーは B 表示
        </small>
      </div>
      <div className="table-wrap">
        <table data-testid="position-table">
          <thead>
            <tr>
              <th>枠 / 馬番</th>
              <th>馬名</th>
              <th>毛色</th>
              <th>馬具</th>
              {actual ? (
                <>
                  {(replay?.corners ?? []).map((c) => (
                    <th key={c}>{c}角</th>
                  ))}
                  <th>入線順位</th>
                  <th>上がり3F</th>
                </>
              ) : (
                <>
                  <th>序盤</th>
                  <th>4角位置予想</th>
                  <th>予想上がり3F</th>
                  <th>過去走</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {race.horses.map((h) => {
              const record = replay?.horses[String(h.number)];
              return (
                <tr key={h.number}>
                  <td>
                    <span
                      className="number"
                      style={{
                        background: frameColors[h.gate - 1],
                        color: h.gate === 1 ? '#333' : '#fff',
                      }}
                    >
                      {h.number}
                    </span>
                  </td>
                  <td>
                    <button className="horse-name" onClick={() => onSelect(h.number)}>
                      {h.name} ↗
                    </button>
                    <small>
                      {h.jockey} · {h.age}歳 · {h.weight}kg
                    </small>
                  </td>
                  <td>{h.coat}</td>
                  <td>
                    {h.blinker === null ? (
                      '不明'
                    ) : h.blinker ? (
                      <span className="blinker">B</span>
                    ) : (
                      '—'
                    )}
                  </td>
                  {actual ? (
                    <>
                      {(replay?.corners ?? []).map((c) => (
                        <td key={c} className="actual-rank">
                          {record?.corners[c - 1] ?? '—'}
                        </td>
                      ))}
                      <td className="actual-rank finish-rank">{record?.finish ?? '—'}</td>
                      <td>{record?.last3?.toFixed(1) ?? '—'}秒</td>
                    </>
                  ) : (
                    <>
                      <td>
                        <span className="style-tag">
                          {h.positions[0] < 0.25 ? '前方' : h.positions[0] < 0.65 ? '中団' : '後方'}
                        </span>
                      </td>
                      <td>
                        <div className="position-track">
                          <i style={{left: `${h.positions[3] * 92}%`}} />
                        </div>
                        <small>
                          {Math.round(h.positions[3] * (race.horses.length - 1) + 1)}番手相当
                        </small>
                      </td>
                      <td>{h.last3.toFixed(1)}秒</td>
                      <td>{h.history.length}走</td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
