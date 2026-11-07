# 東京競馬場コース形状の出典

取得元: https://www.jra.go.jp/facilities/race/tokyo/course/index.html
取得日: 2026-11-08 JST

- 平面図: https://www.jra.go.jp/facilities/race/tokyo/course/img/pic_course_heimenzu.gif
- 芝高低断面図: https://www.jra.go.jp/facilities/race/tokyo/course/img/pic_course_turf.gif
- ダート高低断面図: https://www.jra.go.jp/facilities/race/tokyo/course/img/pic_course_durt.gif

app/trackGeometry.ts のピクセル座標は上記平面図の内柵を手動で読み取ったもの。
芝内柵の平面周長を2,083.1m、ダート内柵を1,899mに校正。高さは公式断面図を折れ線補間。
建物と引込線は公式平面図から読み取り。これは図面のデジタイズであって測量座標ではない。
通常の発走地点はゴールから距離を逆算。2000m・1800m・ダート1600mは引込線を別経路で接続。
コース幅は芝31〜41mの中間36m、ダート25m。芝はAコース基準。
馬のモデルは全長約2.4mとなる一定縮尺。選択しても拡大せずマーカーで示す。
前後差・内外の進路は推論済み位置取りの可視化用仮定。実測された馬間距離を再現するものではない。
