# B-展開2(阪神)

2026-11-15。再生成: `node scripts/fit_simple.mjs 09`(内回り。出力を `outputs/alignment-09-inner.json` に退避)→ `node scripts/fit_outer.mjs 09` → `node scripts/write_alignment.mjs outputs/alignment-09.json` → `node scripts/report_venue.mjs 09` / `node scripts/check_hanshin_nakayama_frames.mjs`。

## 方式(京都の当てはめを作り直した)
- 内回りは東京・中京と同じ(直線 + 緩和曲線 + 円弧の 2 つの楕円コーナー)。
- 外回りは内回りの要素列をそのまま先頭から分岐点まで写し(共有区間のレールのずれは 0)、その先に自分のコーナーを持つ。最終直線(外回り 473.6m)は内回りと同じゴール前の直線に重なる。`scripts/fit_outer.mjs`。
- 高低: 傾き連続の補間。外回りが内回りの直線から離れるところは、2 つの断面図が食い違う(内 -1.4 / 外 -0.5)ため、内回りの高さから外回り自身の高さへ 150m 程度かけて移す(`SPREAD_BLEND_VENUES`)。
- 宝塚記念の引込線は、本線の高さに必ず合流点で一致させるようにした(`app/raceRoutes.ts`)。

## 数値(`metrics.json`)
| コース | 一周(公式) | レール長 | 最終直線 | 案内図とのずれ 平均/最大 | 3px超 |
|---|---|---|---|---|---|
| 内回り | 1689.0 | 1689.01 | 356.5 | 1.3 / 4.8 px | 23/180 |
| 外回り | 2089.0 | 2089.00 | 473.6 | 1.9 / 5.0 px | 53/180 |

3px を超えるのは両コースとも公式の一周・直線を優先した結果(最大 5px)。

## 検証
- `tsc`、`check_geometry`、`check_camera`、`check_finish`、`check_pace`、`check_passing`: PASS。
- `check_hanshin_nakayama_frames`(皐月賞・桜花賞・宝塚記念・大阪杯の 8 枚。通し再生・直接シーク・逆戻りのハッシュ一致): PASS。
- 画像: `whole-turf.png`、`whole-outer.png`、`straight-low-*.png`。

## 未完了
- 曲率グラフ、案内図への重ね合わせの PNG、合流・分岐の寄り画像、低い位置の横からの坂の画像は未作成(`overlay.svg` と `metrics.json` のみ)。
- 引込線の面を経路から作る処理(東京で行ったもの)は未適用。
- 阪神の全レースの目視での通し再生は未確認。
