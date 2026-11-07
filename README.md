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
- JRA図面に沿う精密コース、距離別の引込線、内柵設定と起伏断面。
  現在の3Dは競馬場別の公式距離・高低差を参照した楕円の模式形状であり、実形状を再現したものではない。
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
