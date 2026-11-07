import type {Dataset} from './types';
export default function ModelNotes({data}: {data: Dataset}) {
  return (
    <div className="model-panel">
      <div>
        <p className="eyebrow">FLOW MODEL v0.1</p>
        <h3>予想と実績を分けて表示。</h3>
        <p>
          「実績再生」は対象レースの記録された通過順位・入線順位を使用。「モデル予想」はLightGBMによる1〜4角の正規化通過順位と上がり3Fを使用します。実績データをモデル予想に混ぜていません。
        </p>
        <p>
          予想の直線では4角位置と予想上がり3Fを使い、相対的な前進量を補間します。ゴール位置は表示用の推定で、別途学習した着順予測ではありません。
        </p>
        <p>
          両モードとも連続座標は未収録です。観測地点間の追い抜き位置・時刻、馬間距離、内外の進路は補間です。
        </p>
        <p>
          学習候補 {data.model.rows.toLocaleString()}頭分 / {data.model.races.toLocaleString()}
          レース。検証：{data.model.validation}。学習は2025年末で打ち切っています。
        </p>
      </div>
      <div>
        <h4>時系列検証：平均絶対誤差</h4>
        <table>
          <thead>
            <tr>
              <th>対象</th>
              <th>モデル</th>
              <th>過去平均</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.model.metrics).map(([k, v]) => (
              <tr key={k}>
                <td>{k === 'last3' ? '上がり3F（秒）' : `${k.slice(1)}角（0〜1）`}</td>
                <td>{v.mae.toFixed(3)}</td>
                <td>{v.baselineMae.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="caption">
          0〜1は先頭〜最後方の相対位置。回収率・的中率は評価対象にしていません。
        </p>
      </div>
    </div>
  );
}
