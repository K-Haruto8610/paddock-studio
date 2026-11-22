"""パトロール映像のローカル処理手順（paddock-studio で python -m vision.patrol guide）。

1. オーナーは vision/data/videos/ に元映像を置く。ファイル名は frame_2020.pkl の
   RACE_CODE（年月日8＋場2＋回次2＋日次2＋R2）、または年月日8＋場2＋R2。
   後者は既存フレームから一意に解決する。元ファイル名と変換先は manifest に残る。
   フレームは outputs/data-expansion-v20/frame_2020.pkl、確定着順は
   outputs/v3/runs.pkl の fin を読み取る。pickle は信頼できる既存ファイルに限る。

2. この端末の Python は .\\vision\\data\\env\\Scripts\\python.exe。以下の python は
   その実行ファイルを指す。他の端末では vision/requirements.txt の固定版と、それに
   対応する CUDA 版 PyTorch を用意する。今回の既定モデルはローカルの COCO YOLOv4
   （608px）。端末の Darknet 設定と重みからの再生成は次で行う。
   python -m vision.prepare_weights <yolov4.cfg> <yolov4.weights>
   同じ出力形式の COCO YOLOv8 は run --weights <名前> --size <入力辺長> で選べる。

3. python -m vision.patrol inventory で一覧、inventory <キー> で全フレームを読む。
   寸法・公称fps・実表示時刻間隔・尺・カット候補・パン／ズーム候補を inventory.json に出す。
   オペレーターは映像を見て vision/data/races/<RACE_CODE>/setup.json の confirmed_cuts
   （各カット後の最初のフレーム）、pan、zoom、all_horses_visible_ranges（両端を含む）
   を確定する。再度 inventory <キー> で確認結果込みの棚卸しを先に報告する。
   対応するレースのない映像でも寸法等は data/unmatched/ 以下で確認できる。
   python -m vision.patrol frame <キー> <フレーム番号> で元解像度 PNG をローカルに出す。

4. setup.json の gate_open_frame で t=0、gate_anchor_frame は直後の全馬が同定可能な
   フレーム、end_frame は全馬のゴール通過後。すべて元動画の0始まり。gate_axis は馬番が
   増える方向の画像ベクトル（左から右なら [1,0]、上から下なら [0,1]）。逆向きなら
   reverse_gate_order=true。出走表の実際の馬番を用い、取消による欠番を保つ。
   高信頼の検出数が出走頭数に一致したときだけ馬番を付ける。以降は予測箱との IoU、
   高信頼→低信頼の二段階割当を使う ByteTrack 系の実装。0.8秒を超える消失は欠測にする。
   カットでは追跡を切る。カット直後に番号を独立に読める場合だけ identity_anchors に
   {"frame":番号,"points":[[馬番,u,v],...]}。点は指定馬の箱だけに入るピクセルを選ぶ。

5. 同じ setup.json の keyframes に射影の基準を記入する。各要素の形は
   {"frame":番号,"s_range":[下限m,上限m],"landmarks":[[u,v,s,y],...]}。
   u,v は元画像左上からのピクセル。s はスタートから内ラチ沿いの距離、y は内ラチから
   外への地面上の距離。残り600mの標識の s はレース距離−600。内ラチ、ハロン棒の
   地面の基部、既知の横距離を持つ地点を使う。地上高のある柵の上端やポールの先端は
   地面の基準点に使えない。局所的に同一直線上に並ばない4点以上を置く。
   s_range はそのショットに写る範囲。同一地点を複数周する場合は対象の周だけを含める。
   スタート前・ゴール後の補間用余白は -20m から距離+29m まで。
   動くカメラにはランドマークの往復光フローとフレームごとの射影再推定を使う。
   画面外に基準点が出る前とカット開始時に追加する。S0 の近似コースや奥行き誤差、
   箱の下辺中央を接地点とすることによるずれは独立の検証点で評価する。

6. python -m vision.patrol run <キー> で1レース抽出。data/races/<RACE_CODE>/current.json
   の run_id が成果物フォルダ名。frames.csv は実表示時刻・馬番・画像／コース座標、
   trajectory_5hz.csv は RACE_CODE,UMABAN,t,s,y。欠測・カットをまたぐ補間はしない。
   events.csv はS0の最終周の角・残り600m・ゴールの通過時刻（元フレーム間で補間）。
   annotated.mp4 は馬番、カメラの動きを補正した直近1秒の軌跡、元フレーム番号と
   実測時刻を表示する。再生用動画は公称fps、時刻の計算は元動画の表示時刻を使う。

7. 抽出フォルダの review.json はその run_id 専用。注釈映像を全区間見て、
   swap_intervals に {"frames":[最後の正しい番号のフレーム,入替確認フレーム],
   "horses":[関係した馬番,...]} を入れる。同時の2頭の交換は1件。ラベルが元に戻る
   交換も別の1件。全区間の確認後に identity_review_complete=true。未確認は回数不明。
   verified_identity_frames は {"1":[100,250],...} のように独立に番号を読めたフレーム。
   着順不一致は最終確認からゴールまでの再確認区間として表示する。
   ground_checks は {"frame":番号,"horse":馬番,"s":独立計測m,"y":独立計測m} の配列。
   射影に使っていない目印・実測寸法で馬の接地点を測り、直線／角・内／外・各カメラを
   含める。確認できない値は空のまま残す。

8. python -m vision.patrol qa <キー> は抽出済みの結果を再採点する。qa.json は角別の
   Kendall τ-b（公式同順位を保持）、同順位の区間内を許容する順位一致率、隊列完全一致、
   600mの±0.1秒割合（分母は全出走馬）、上がり3F誤差、着順、交換回数、横方向RMSE、
   GPU・処理秒数を記録する。公式の同順位群を考慮した grouped_tau も併記する。
   生のτ-bの中央値0.98、600m割合0.90、交換2回以下で判定し、横方向は独立検証の
   RMSE 0.5mで別判定。横方向が未測定または目標未達なら縦方向のみを候補とする。
   公式値・通過・目視確認が欠ければ未判定または不合格。角のないレースは角QA未判定。
   timing_errors.csv は馬ごとの符号付き秒差（抽出−公式）。diagnostics.json と
   frame_counts.csv で検出数不足・未割当・入替・射影欠測と座標誤差を点検する。
   処理秒数には機械抽出とQA計算を含む。棚卸しと人の注釈・確認工数は別に記録する。

9. フェーズ0を報告してから対象を拡大する。後続の明示指定分は
   python -m vision.patrol batch <キー1> <キー2> ...。完了した抽出は入力・設定・実装の
   指紋が同一なら再利用し、途中で終了したレースは次回再処理する。
   python -m vision.patrol manifest で全レースの最新QAを表示する。
   映像・画像・重み・軌跡・レビュー・manifest はすべて Git 除外の vision/data/ 内。
"""
