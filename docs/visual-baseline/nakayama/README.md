# B-展開2(中山)

2026-11-15。再生成: `node scripts/fit_simple.mjs 06`(内回り。出力を `outputs/alignment-06-inner.json` に退避)→ `node scripts/fit_outer.mjs 06` → `node scripts/write_alignment.mjs outputs/alignment-06.json` → `node scripts/report_venue.mjs 06`。方式は阪神と同じ(外回りは内回りの 1 角の途中まで共有し、その先は別の 2 コーナー)。

## 数値(`metrics.json`)
| コース | 一周(公式) | レール長 | 最終直線 | 案内図とのずれ 平均/最大 | 3px超 |
|---|---|---|---|---|---|
| 内回り | 1667.1 | 1667.11 | 310 | 0.8 / 2.1 px | 0/180 |
| 外回り | 1839.7 | 1839.74 | 310 | 4.7 / 12.0 px | 105/180 |

**外回りは案内図から大きくずれる。** 案内図の外回りは、内回りとの周長比が 1.138 で、公式の比 1.104 より 3.1% 大きい。公式の一周 1839.7m を優先して縮めたため、最大 12px(約 20m)ずれる。これは当てはめでは解消できない(図面の側の誤差)。
その影響で、スプリンターズS の外回り 1200m 発走(案内図の印に合わせる方式)は、印から終点までの距離が 1155m になり、1200m に 3.8% 引き延ばして再生する(`check_geometry` の許容を、印に合わせる方式のみ 5% にした)。

## 検証
- 阪神と共通: `tsc`、`check_geometry`、`check_camera`、`check_finish`、`check_pace`、`check_passing`: PASS。皐月賞の 2 枚を含む 8 枚の再現性ハッシュ一致。
- 画像: `whole-turf.png`、`whole-outer.png`、`straight-low-*.png`。

## 未完了
- 外回りのずれ(上記)。曲率グラフ、重ね合わせ PNG、合流・分岐の寄り画像、坂の横からの画像。
- 引込線の面を経路から作る処理は未適用。中山の全レースの目視での通し再生は未確認。
