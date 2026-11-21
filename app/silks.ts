/**
 * Racing silks (勝負服), worn here by the saddle cloth since the plush horse has no jockey. They belong to the owner, so
 * they are kept by horse name and follow a horse from race to race; a horse not listed keeps the plain blue cloth.
 *
 * The cloth carries the silks as a jacket laid over each side: the body of the jacket fills the middle of the side, from
 * the spine (the collar) down to the hem (the waist), and the sleeves hang down the front and rear edges, from the
 * shoulder at the spine to the cuff at the hem. `drawSilks` paints one side that way; the cloth shader mirrors it.
 */

/** The flat dye colours of JRA silks, a little softer than the icons' pure screen colours. */
export const SILK_COLORS = {
  白: '#f2f0ea',
  黒: '#1b1b1e',
  赤: '#d9201d',
  青: '#1d47d6',
  水色: '#3bb8ec',
  黄: '#f4d60c',
  緑: '#14a64e',
  深緑: '#2b843b',
  桃: '#ef66d9',
  灰: '#bdbdbd',
  えんじ: '#9c133b',
  藤色: '#bb95e8',
} as const;
type Dye = keyof typeof SILK_COLORS;

/** The body's pattern, by its JRA name. */
export type BodyPattern =
  | {kind: '無地'}
  | {kind: '一本輪' | '二本輪'; color: Dye}
  | {kind: '山形一本輪' | '山形二本輪'; color: Dye}
  | {kind: '縦縞'; color: Dye}
  | {kind: '玉霰'; color: Dye}
  | {kind: '星散'; color: Dye}
  | {kind: '元禄'; color: Dye}
  | {kind: '十字襷'; color: Dye}
  | {kind: '鋸歯形'; color: Dye}
  | {kind: '菱'; color: Dye}
  | {kind: '肩章'; color: Dye};

/** The sleeves' pattern; '胴と同柄' repeats the body's pattern on them. */
export type SleevePattern =
  | {kind: '無地'}
  | {kind: '一本輪'; color: Dye}
  | {kind: '段違'; color: Dye}
  | {kind: '縦縞'; color: Dye}
  | {kind: '胴と同柄'};

export type Silks = {
  /** The silks as read, in the JRA way of writing them (胴の地色、柄、袖). */
  label: string;
  owner: string;
  body: Dye;
  pattern: BodyPattern;
  sleeves: Dye;
  sleevePattern: SleevePattern;
};

const silks = (owner: string, label: string, body: Dye, pattern: BodyPattern, sleeves: Dye, sleevePattern: SleevePattern = {kind: '無地'}): Silks =>
  ({owner, label, body, pattern, sleeves, sleevePattern});

const CARROT = silks('キャロットファーム', '緑、白二本輪、袖白、赤一本輪', '緑', {kind: '二本輪', color: '白'}, '白', {kind: '一本輪', color: '赤'});
const KATSUMI_YOSHIDA = silks('吉田勝己', '黄、赤縦縞、袖黒', '黄', {kind: '縦縞', color: '赤'}, '黒');
const GODOLPHIN = silks('ゴドルフィン', '青、袖水色一本輪', '青', {kind: '無地'}, '青', {kind: '一本輪', color: '水色'});

/** マイルチャンピオンシップ 2026 (read from the 出馬表 icons shared on 2026-11-22). */
const BY_HORSE: Record<string, Silks> = {
  エルデナリ: silks('Aga Khan Stud', '緑、赤肩章', '深緑', {kind: '肩章', color: '赤'}, '深緑'),
  エンブロイダリー: silks('シルクレーシング', '水色、赤玉霰、袖赤一本輪', '水色', {kind: '玉霰', color: '赤'}, '水色', {kind: '一本輪', color: '赤'}),
  ドラゴンブースト: silks('ゼットステーブル', '黄、黒山形二本輪、袖赤、青一本輪', '黄', {kind: '山形二本輪', color: '黒'}, '赤', {kind: '一本輪', color: '青'}),
  シックスペンス: CARROT,
  アドマイヤズーム: silks('近藤旬子', '水色、青鋸歯形、袖白', '水色', {kind: '鋸歯形', color: '青'}, '白'),
  ピコローズ: silks('ビジュピコ', '白、桃菱、袖同柄', '白', {kind: '菱', color: '桃'}, '白', {kind: '胴と同柄'}),
  ココナッツブラウン: silks('下河辺隆行', '白、袖赤、緑一本輪', '白', {kind: '無地'}, '赤', {kind: '一本輪', color: '緑'}),
  ロデオドライブ: KATSUMI_YOSHIDA,
  ジューンテイク: silks('吉川潤', '桃、黄星散、袖白', '桃', {kind: '星散', color: '黄'}, '白'),
  ウォーターリヒト: silks('山岡正人', '水色、青十字襷、袖桃白段違', '水色', {kind: '十字襷', color: '青'}, '白', {kind: '段違', color: '桃'}),
  ガイアフォース: silks('KRジャパン', '白、黒元禄、袖青、黄縦縞', '白', {kind: '元禄', color: '黒'}, '青', {kind: '縦縞', color: '黄'}),
  ワールズエンド: CARROT,
  レザベーション: silks('中村伊三美', '灰、青一本輪、袖青', '灰', {kind: '一本輪', color: '青'}, '青'),
  スターアニス: KATSUMI_YOSHIDA,
  エコロブルーム: silks('原村正紀', '緑、黄一本輪、袖黄、赤一本輪', '緑', {kind: '一本輪', color: '黄'}, '黄', {kind: '一本輪', color: '赤'}),
  スズハローム: silks('森達郎', 'えんじ、白玉霰、袖藤色', 'えんじ', {kind: '玉霰', color: '白'}, '藤色'),
  アンパドゥ: GODOLPHIN,
  オフトレイル: GODOLPHIN,
};

export const silksFor = (name: string): Silks | undefined => BY_HORSE[name];
export const silkedHorses = () => Object.keys(BY_HORSE);

/** Relative luminance of a dye (0..1), to pick a number colour that reads on the cloth. */
export function dyeLuminance(dye: Dye) {
  const hex = SILK_COLORS[dye];
  const [r, g, b] = [1, 3, 5].map(i => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Share of the side's length each sleeve takes, at the front and rear edges. */
const SLEEVE = 0.17;

/**
 * Paints one side of the cloth: u (x) runs from the rear edge to the front edge, v (y) from the spine to the hem. The
 * canvas has the side's own proportions, so circles stay round and stripes keep their width on the horse.
 */
export function drawSilks(ctx: CanvasRenderingContext2D, s: Silks, width: number, height: number) {
  const dye = (d: Dye) => SILK_COLORS[d];
  const unit = width / 7;                      // pattern scale: a seventh of the cloth's length, so it reads from afar
  const sleeveW = width * SLEEVE;
  const body = {x: sleeveW, w: width - 2 * sleeveW};

  ctx.save();
  ctx.fillStyle = dye(s.body);
  ctx.fillRect(0, 0, width, height);
  const bodyClip = () => {ctx.beginPath(); ctx.rect(body.x, 0, body.w, height); ctx.clip();};
  if (s.sleevePattern.kind === '胴と同柄') paintBody(0, width);
  else {
    ctx.save();
    bodyClip();
    paintBody(body.x, body.w);
    ctx.restore();
    for (const x of [0, width - sleeveW]) paintSleeve(x);
  }
  ctx.restore();

  function zigzagBand(y: number, h: number, x0: number, w: number) {
    // Both edges zigzag in step, so the band keeps its width: a row of chevrons.
    const tooth = unit * 0.9, rise = h * 0.6;
    const points = Math.ceil(w / (tooth / 2)) + 2;
    const at = (k: number): [number, number] => [x0 + (k - 1) * tooth / 2, k % 2 ? -rise / 2 : rise / 2];
    ctx.beginPath();
    for (let k = 0; k <= points; k++) {const [x, dy] = at(k); ctx.lineTo(x, y + dy);}
    for (let k = points; k >= 0; k--) {const [x, dy] = at(k); ctx.lineTo(x, y + h + dy);}
    ctx.closePath();
    ctx.fill();
  }

  function star(cx: number, cy: number, r: number) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.42 : r;
      ctx.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr);
    }
    ctx.closePath();
    ctx.fill();
  }

  /** A spaced grid over the region, centred on it, for scattered dots and stars. */
  function grid(x0: number, w: number, step: number, each: (x: number, y: number) => void) {
    const cols = Math.max(1, Math.round(w / step)), rows = Math.max(1, Math.round(height / step));
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
      each(x0 + (c + 0.5) * w / cols, (r + 0.5) * height / rows);
  }

  function paintBody(x0: number, w: number) {
    const p = s.pattern;
    if (p.kind === '無地') return;
    ctx.fillStyle = dye(p.color);
    const band = height * 0.12;
    switch (p.kind) {
      case '一本輪':
        ctx.fillRect(x0, height * 0.38, w, band);
        break;
      case '二本輪':
        ctx.fillRect(x0, height * 0.26, w, band);
        ctx.fillRect(x0, height * 0.54, w, band);
        break;
      case '山形一本輪':
        zigzagBand(height * 0.38, band, x0, w);
        break;
      case '山形二本輪':
        zigzagBand(height * 0.24, band, x0, w);
        zigzagBand(height * 0.54, band, x0, w);
        break;
      case '縦縞': {
        const stripe = unit * 0.55;
        for (let x = x0 + stripe * 0.5; x < x0 + w; x += stripe * 2) ctx.fillRect(x, 0, stripe, height);
        break;
      }
      case '玉霰':
        grid(x0, w, unit * 2, (x, y) => {ctx.beginPath(); ctx.arc(x, y, unit * 0.48, 0, Math.PI * 2); ctx.fill();});
        break;
      case '星散':
        grid(x0, w, unit * 2, (x, y) => star(x, y, unit * 0.62));
        break;
      case '元禄': {
        const sq = unit * 0.9;
        for (let r = 0; r * sq < height; r++) for (let c = 0; c * sq < w; c++)
          if ((r + c) % 2) ctx.fillRect(x0 + c * sq, r * sq, sq, sq);
        break;
      }
      case '十字襷': {
        const t = unit * 1.3;
        ctx.lineWidth = t;
        ctx.strokeStyle = dye(p.color);
        ctx.beginPath();
        ctx.moveTo(x0, 0); ctx.lineTo(x0 + w, height);
        ctx.moveTo(x0 + w, 0); ctx.lineTo(x0, height);
        ctx.stroke();
        break;
      }
      case '鋸歯形': {
        // The upper part in the pattern colour, ending in a row of teeth pointing down.
        const edge = height * 0.22, tooth = unit * 0.9;
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x0, edge);
        for (let x = x0, k = 0; x <= x0 + w + tooth; x += tooth / 2, k++) ctx.lineTo(x, edge + (k % 2 ? tooth * 0.55 : 0));
        ctx.lineTo(x0 + w, 0);
        ctx.closePath();
        ctx.fill();
        break;
      }
      case '菱': {
        // A checker turned 45°: diamonds of the pattern colour meeting at their points, the ground showing between.
        const d = unit * 0.7, step = 2 * d;
        for (let r = -1; r * step < height + step; r++) for (let c = -1; c * step < w + step; c++) {
          const cx = x0 + c * step, cy = r * step;
          ctx.beginPath();
          ctx.moveTo(cx, cy - d); ctx.lineTo(cx + d, cy); ctx.lineTo(cx, cy + d); ctx.lineTo(cx - d, cy);
          ctx.closePath();
          ctx.fill();
        }
        break;
      }
      case '肩章':
        // Epaulettes: a bar on each shoulder, where the body meets the sleeve at the spine.
        ctx.fillRect(x0, 0, unit * 0.55, height * 0.2);
        ctx.fillRect(x0 + w - unit * 0.55, 0, unit * 0.55, height * 0.2);
        break;
    }
  }

  function paintSleeve(x: number) {
    ctx.fillStyle = dye(s.sleeves);
    ctx.fillRect(x, 0, sleeveW, height);
    const p = s.sleevePattern;
    if (p.kind === '無地' || p.kind === '胴と同柄') return;
    ctx.fillStyle = dye(p.color);
    switch (p.kind) {
      case '一本輪':
        ctx.fillRect(x, height * 0.42, sleeveW, height * 0.12);
        break;
      case '段違': {
        const h = height / 9;
        for (let k = 1; k < 9; k += 2) ctx.fillRect(x, k * h, sleeveW, h);
        break;
      }
      case '縦縞': {
        const stripe = sleeveW / 5;
        for (let k = 1; k < 5; k += 2) ctx.fillRect(x + k * stripe, 0, stripe, height);
        break;
      }
    }
  }
}
