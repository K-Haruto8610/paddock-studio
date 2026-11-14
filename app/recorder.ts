/**
 * Records a forecast playback as an MP4 video: the 3D race view, the seek bar over the
 * course's height profile and the running order (隊列) below it, composed frame by frame on one canvas and captured with MediaRecorder.
 */

/** Output size. H.264 needs even dimensions. */
const WIDTH = 1920;
const SCENE_HEIGHT = 1020;
export const RECORDING_SCENE_SIZE = {width: WIDTH, height: SCENE_HEIGHT};
const SEEK_HEIGHT = 120;
const ORDER_HEIGHT = 150;
const SEEK_TOP = SCENE_HEIGHT;
const ORDER_TOP = SCENE_HEIGHT + SEEK_HEIGHT;
const HEIGHT = ORDER_TOP + ORDER_HEIGHT;
const FPS = 30;

const MP4_TYPES = [
  'video/mp4;codecs=avc1.640028',
  'video/mp4;codecs=avc1.4d0028',
  'video/mp4;codecs=avc1',
  'video/mp4',
];

/** The MP4 type this browser can record, or null when it records no MP4 (e.g. Firefox). */
export function mp4MimeType(): string | null {
  if (typeof MediaRecorder === 'undefined') return null;
  return MP4_TYPES.find((t) => MediaRecorder.isTypeSupported(t)) ?? null;
}

export type RecordingOverlay = {
  /** Race name, date and course, top right. */
  title: string;
  /** Mode and race phase, top left (e.g. "予想 · 1角 → 2角"). */
  status: string;
  /** Running order heading, e.g. "現在の隊列". */
  orderTitle: string;
  orderNote: string;
  /** Horses from the leader backwards. */
  order: {number: number; background: string; color: string}[];
  /** The seek bar as on the page. */
  seek: {
    /** 0..1 through the race. */
    progress: number;
    /** Right-handed course: the bar runs right to left, mirrored. */
    reversed: boolean;
    /** JRA height profile image, one lap per tile, repeated leftwards from the finish. */
    profileImage: string;
    /** One lap as a fraction of the race distance (tile width). */
    lapFraction: number;
    checkpoints: {label: string; at: number; active: boolean; edit: boolean; edited: boolean}[];
  };
};

const images = new Map<string, HTMLImageElement>();
/** The image once loaded (null until then; the bar is drawn without it meanwhile). */
function loadedImage(url: string) {
  let img = images.get(url);
  if (!img) {
    img = new Image();
    img.src = url;
    images.set(url, img);
  }
  return img.complete && img.naturalWidth ? img : null;
}

export class RaceRecorder {
  readonly canvas = document.createElement('canvas');
  /** File name the finished video is saved as. */
  name = 'race.mp4';
  private ctx: CanvasRenderingContext2D;
  private recorder: MediaRecorder;
  private chunks: Blob[] = [];
  private done: Promise<Blob>;

  constructor(
    readonly mimeType: string,
    private overlay: () => RecordingOverlay,
  ) {
    this.canvas.width = WIDTH;
    this.canvas.height = HEIGHT;
    this.ctx = this.canvas.getContext('2d')!;
    this.ctx.fillStyle = '#e8eddf';
    this.ctx.fillRect(0, 0, WIDTH, HEIGHT);
    const stream = this.canvas.captureStream(FPS);
    this.recorder = new MediaRecorder(stream, {mimeType, videoBitsPerSecond: 24_000_000});
    this.recorder.ondataavailable = (e) => {
      if (e.data.size) this.chunks.push(e.data);
    };
    this.done = new Promise((resolve, reject) => {
      this.recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        resolve(new Blob(this.chunks, {type: 'video/mp4'}));
      };
      this.recorder.onerror = (e) => reject((e as ErrorEvent).error ?? Error('録画に失敗しました'));
    });
    this.recorder.start(1000);
  }

  /**
   * Draws one video frame. Call it right after the WebGL scene renders, in the same task,
   * so the scene canvas still holds that frame.
   */
  drawFrame(scene: HTMLCanvasElement) {
    const c = this.ctx;
    // TrackScene renders at the recording resolution: copy pixels without resampling.
    c.drawImage(scene, 0, 0);
    const o = this.overlay();
    c.textBaseline = 'middle';
    const font = (px: number, weight = 400) =>
      `${weight} ${px}px "Hiragino Sans", "Yu Gothic UI", "Meiryo", sans-serif`;
    // Status pill, top left.
    c.font = font(22, 600);
    const pillW = c.measureText(o.status).width + 36;
    panel(c, 24, 22, pillW, 46, '#fffdf0e8');
    c.fillStyle = '#2d3a2a';
    c.fillText(o.status, 42, 46);
    // Race title, top right.
    c.font = font(22, 700);
    const titleW = c.measureText(o.title).width + 36;
    panel(c, WIDTH - 24 - titleW, 22, titleW, 46, '#ffffffe0');
    c.fillStyle = '#2d3a2a';
    c.fillText(o.title, WIDTH - 24 - titleW + 18, 46);

    this.drawSeekBar(o.seek, font);

    // Running order strip.
    c.fillStyle = '#fffefa';
    c.fillRect(0, ORDER_TOP, WIDTH, ORDER_HEIGHT);
    c.fillStyle = '#dce2d3';
    c.fillRect(0, ORDER_TOP, WIDTH, 2);
    c.font = font(22, 700);
    c.fillStyle = '#2d3a2a';
    c.fillText(o.orderTitle, 28, ORDER_TOP + 34);
    c.font = font(17);
    c.fillStyle = '#8c967e';
    c.textAlign = 'right';
    c.fillText(o.orderNote, WIDTH - 28, ORDER_TOP + 34);
    c.textAlign = 'left';
    const n = Math.max(o.order.length, 1),
      gap = 8,
      size = Math.min(56, (WIDTH - 56 - gap * (n - 1)) / n),
      y = ORDER_TOP + 66;
    c.textAlign = 'center';
    c.font = font(Math.round(size * 0.46), 700);
    o.order.forEach((h, i) => {
      const x = 28 + i * (size + gap);
      panel(c, x, y, size, size, h.background, '#00000030');
      c.fillStyle = h.color;
      c.fillText(String(h.number), x + size / 2, y + size / 2 + 1);
    });
    c.textAlign = 'left';
  }

  /** The page's seek bar: height profile, done part, checkpoint ticks and labels, current spot. */
  private drawSeekBar(seek: RecordingOverlay['seek'], font: (px: number, weight?: number) => string) {
    const c = this.ctx;
    c.fillStyle = '#fffef9';
    c.fillRect(0, SEEK_TOP, WIDTH, SEEK_HEIGHT);
    c.fillStyle = '#dce2d3';
    c.fillRect(0, SEEK_TOP, WIDTH, 2);
    const x0 = 28,
      w = WIDTH - 56,
      y0 = SEEK_TOP + 20,
      h = 56,
      at = (f: number) => x0 + (seek.reversed ? 1 - f : f) * w;
    c.save();
    c.beginPath();
    c.roundRect(x0, y0, w, h, 5);
    c.fillStyle = '#eef3ea';
    c.fill();
    c.clip();
    const img = loadedImage(seek.profileImage);
    if (img && seek.lapFraction > 0) {
      const tile = w * seek.lapFraction;
      c.save();
      if (seek.reversed) {
        // Mirror about the bar's centre, so the finish edge of each tile lands on the left.
        c.translate(2 * x0 + w, 0);
        c.scale(-1, 1);
      }
      for (let right = x0 + w; right > x0; right -= tile)
        c.drawImage(img, right - tile, y0, tile, h);
      c.restore();
    }
    const now = at(Math.max(0, Math.min(1, seek.progress)));
    c.fillStyle = '#23463626';
    c.fillRect(seek.reversed ? now : x0, y0, seek.reversed ? x0 + w - now : now - x0, h);
    c.setLineDash([4, 4]);
    c.strokeStyle = '#2e584580';
    c.lineWidth = 1.5;
    for (const cp of seek.checkpoints) {
      const x = Math.round(at(cp.at)) + 0.5;
      c.beginPath();
      c.moveTo(x, y0);
      c.lineTo(x, y0 + h);
      c.stroke();
    }
    c.restore();
    c.strokeStyle = '#d5ddca';
    c.lineWidth = 1.5;
    c.beginPath();
    c.roundRect(x0, y0, w, h, 5);
    c.stroke();
    // Current spot: a red line with a dot at its foot, as on the page.
    c.fillStyle = '#c1471f';
    c.strokeStyle = '#fff';
    c.lineWidth = 2;
    c.strokeRect(now - 2, y0 - 2, 4, h + 4);
    c.fillRect(now - 2, y0 - 2, 4, h + 4);
    c.beginPath();
    c.arc(now, y0 + h, 8, 0, Math.PI * 2);
    c.fill();
    c.stroke();
    // Checkpoint labels under the bar; the first and last stay inside the edges.
    seek.checkpoints.forEach((cp, i) => {
      c.font = font(16, cp.active ? 700 : 400);
      c.fillStyle = cp.active ? '#2e5845' : cp.edit ? '#8a5a1f' : '#8c9584';
      const edge = i === 0 ? 'left' : i === seek.checkpoints.length - 1 ? 'right' : 'center';
      c.textAlign = seek.reversed && edge !== 'center' ? (edge === 'left' ? 'right' : 'left') : edge;
      c.fillText(cp.label + (cp.edited ? '●' : ''), at(cp.at), y0 + h + 24);
    });
    c.textAlign = 'left';
  }

  /** Stops recording and resolves to the finished MP4. */
  stop(): Promise<Blob> {
    if (this.recorder.state !== 'inactive') this.recorder.stop();
    return this.done;
  }
}

function panel(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  fill: string,
  stroke = '#d7dfcd',
) {
  c.beginPath();
  c.roundRect(x, y, w, h, 7);
  c.fillStyle = fill;
  c.fill();
  c.strokeStyle = stroke;
  c.lineWidth = 1.5;
  c.stroke();
}
