import type {Dataset} from './types';
import type {PaceForecast} from './paceForecast';
function paceLabel(k: string) {
  const c = k.replace(/[^0-9]/g, '');
  if (k === 'last3') return '上がり3F（秒）';
  if (k.startsWith('spread')) return `${c}角 隊列の長さ（馬身）`;
  if (k.startsWith('r')) return `${c}角 順位（0〜1）`;
  if (k.startsWith('g')) return `${c}角 先頭からの差（馬身）`;
  if (k.startsWith('l')) return `${c}角 内外（併走位置）`;
  return k;
}
const pct = (v: number) => `${Math.round(v * 100)}%`;
export default function ModelNotes({data, pace}: {data: Dataset; pace?: PaceForecast | null}) {
  const hasIndependent = Object.values(pace?.model.metrics ?? {}).some(
    (v) => v.independentMae !== undefined,
  );
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
      {pace && (
        <div>
          <h4>展開予想モデル：{pace.model.name}</h4>
          {(pace.model.notes ?? []).map((n) => (
            <p key={n} className="caption">
              {n}
            </p>
          ))}
          {pace.model.metrics && (
            <table data-testid="pace-metrics">
              <thead>
                <tr>
                  <th>対象（検証 {pace.model.validation ?? '—'}）</th>
                  <th>モデル</th>
                  {hasIndependent && <th>ポジション争いなし</th>}
                  <th>過去平均</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(pace.model.metrics).map(([k, v]) => (
                  <tr key={k}>
                    <td>{paceLabel(k)}</td>
                    <td>{v.mae.toFixed(3)}</td>
                    {hasIndependent && <td>{v.independentMae?.toFixed(3) ?? '—'}</td>}
                    <td>{v.baselineMae.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {pace.model.intervals && (
            <table data-testid="pace-intervals">
              <thead>
                <tr>
                  <th>予想レンジ（50%区間、検証 {pace.model.validation ?? '—'}）</th>
                  <th>実際が入った割合</th>
                  <th>平均の幅</th>
                  <th>表示</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(pace.model.intervals.targets).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k === 'last3' ? '上がり3F' : `${k.replace(/[^0-9]/g, '')}角 番手`}</td>
                    <td>{pct(v.coverage)}</td>
                    <td>
                      {k === 'last3' ? `${v.meanWidth.toFixed(2)}秒` : `${v.meanWidth.toFixed(1)}番手`}
                    </td>
                    <td>{v.shown ? 'あり' : 'なし（被覆率が外れる）'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {pace.model.moves && (
            <table data-testid="pace-moves">
              <thead>
                <tr>
                  <th>道中の位置変化（最初→最後の角）</th>
                  <th>モデル</th>
                  {pace.model.moves.withoutMoves && <th>移動モデルなし</th>}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>実際の移動との相関</td>
                  <td>{pace.model.moves.model.corr.toFixed(3)}</td>
                  {pace.model.moves.withoutMoves && <td>{pace.model.moves.withoutMoves.corr.toFixed(3)}</td>}
                </tr>
                <tr>
                  <td>大きく上げた馬の方向的中</td>
                  <td>{pct(pace.model.moves.model.upHit)}</td>
                  {pace.model.moves.withoutMoves && <td>{pct(pace.model.moves.withoutMoves.upHit)}</td>}
                </tr>
                <tr>
                  <td>大きく下げた馬の方向的中</td>
                  <td>{pct(pace.model.moves.model.downHit)}</td>
                  {pace.model.moves.withoutMoves && <td>{pct(pace.model.moves.withoutMoves.downHit)}</td>}
                </tr>
                {pace.model.moves.rankError !== undefined && (
                  <tr>
                    <td>順位誤差（全角平均, 0〜1）</td>
                    <td>{pace.model.moves.rankError.toFixed(4)}</td>
                    {pace.model.moves.withoutMoves && <td>{pace.model.moves.rankErrorWithoutMoves?.toFixed(4) ?? '—'}</td>}
                  </tr>
                )}
              </tbody>
            </table>
          )}
          {pace.model.pace && (
            <p className="caption" data-testid="pace-validation">
              前半ペース予想（検証）：実際との相関 {pace.model.pace.corr.toFixed(2)}、ハイ/平均/スローの的中{' '}
              {pct(pace.model.pace.labelHit)}（3等分なので偶然なら約33%）。平均誤差{' '}
              {pace.model.pace.mae?.toFixed(2) ?? '—'}秒（常に平均と予想した場合{' '}
              {pace.model.pace.constantMae?.toFixed(2) ?? '—'}秒）。画面のラベルは過去のGⅠとの比較。
            </p>
          )}
          {pace.model.moves && (
            <p className="caption">
              「大きく」は出走頭数の4分の1以上。過去5走の移動の平均だけを使った場合の相関は
              {pace.model.moves.pastMeanCorr?.toFixed(3) ?? '—'}。動きを出すと順位誤差は少し増えます。
            </p>
          )}
        </div>
      )}
    </div>
  );
}
