# Paddock Studio

JRA平地GⅠの展開を、馬柱を学習したLightGBMと3Dの馬で表示するWeb試作。
回収率・的中率は目的にも評価指標にも含めない。

## 現在の実装
- 指定フォルダのmykeibadbを読み取り専用で利用。DBの更新・変更は行わない。
- 各角の正規化通過順位4モデルと上がり3Fモデル。
- 完全な出走頭数と有効な馬名・レース名を持つデータを選別。
- 2025年8月までで学習、9〜12月で時系列検証。推論用は2025年末までで再学習。
- 2026年の正常なGⅠ10レースを掲載。対象レース結果は特徴量や学習に入れない。
- 過去走は対象日より前の最大5走。オッズ、人気、払戻をモデルに渡さない。
- 毛色・馬番・馬名・ブリンカー（青いメンコ）を表示。馬具不明は未取得として扱う。
- 3D再生、シーク、速度変更、追尾/立体/俯瞰、馬柱の閲覧。

## 実行
```
npm run dev
python scripts/build_forecasts.py
npx tsc --noEmit
```
Pythonはpymysql / pandas / numpy / lightgbm / python-dotenvが必要。
KEIBA_SOURCE環境変数で既存mykeibaのパスを指定できる。認証情報は既存設定をメモリー内で参照し、生成物には含めない。
モデルはmodels/、Web向け予想データはpublic/data/forecasts.json。

## 重要な未実装事項
- 最新出馬表の自動取得・差分更新。既存DBの一部に文字化けや出走頭数不一致がある。
- 当日の出走取消・馬具変更・枠順変更を反映する更新経路。
- 東京以外のJRA図面に沿うコース形状、内柵設定ごとの差分。
  東京は公式平面図の内柵と引込線を読み取り、公式周長で校正。芝・ダートの公式高低断面図を補間して立体化済み。図面読取誤差があるため測量座標ではない。他場は引き続き簡略形状。
- 実コーナー位置とモデル段階の厳密な対応、直線の加減速・進路予測。
  4角以降は隊列を保持する演出で、上がり3F予測は情報表示に使用。
- 途中に存在しない1/2角をコース別に抑制する処理（現状は仮想的な序盤位置として使う）。
- 平地GⅠ全レースの現行コース・距離マスター、十分な過去走データの充足。
- モデルの位置予測は回帰値で、表の「番手相当」は重複する。3Dでは予測値順で並べる。

## コース出典（確認 2026-11-08）
- https://www.jra.go.jp/facilities/race/tokyo/course/index.html
- https://www.jra.go.jp/facilities/race/nakayama/course/index.html
- https://www.jra.go.jp/facilities/race/chukyo/course/index.html
- https://www.jra.go.jp/facilities/race/kyoto/course/index.html
- https://www.jra.go.jp/facilities/race/hanshin/course/index.html

## 検証
TypeScript型検査・本番ビルド成功。ブラウザーで3D描画、再生/停止、馬柱選択、ダートレース切替を確認。コンソールエラーなし。
モデルの時系列検証値はmodels/report.jsonとWeb画面に表示。

2026-11-08 コース改善: 馬体長約2.4mと走路を同一縮尺に統一。3頭固定列を廃止し、連続した前後差と独立の内外位置を補間。馬群追尾は進行方向の側方から全体を収め、手動操作中は自動追尾を解除。選択時の馬体拡大を廃止しリング表示に変更。
`node scripts/check_geometry.mjs` で公式周長・高低差・経路連続性・発走〜ゴール接続・馬群配置範囲を検証。
