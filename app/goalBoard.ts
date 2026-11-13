import * as T from 'three';
import type {Race} from './types';
import type {TrackModel} from './trackGeometry';
import {courseFor} from './courses';

// The winning post dressed for a GⅠ day: a horseshoe arch carrying the race name and date
// around the mirror post, the venue's sign beside it, flower beds along the inner rail and
// banners on the rail. An original design in the spirit of a big-race day, not a copy of
// any real course's signage.

const PURPLE = '#3a2a8c';
const DEEP_PURPLE = '#261a63';
const GOLD = '#f0cf78';
const MINCHO = '"Yu Mincho", "Hiragino Mincho ProN", "Noto Serif JP", serif';
/** Drawn larger than life, like the horses, so it reads from the pack camera. */
const SCALE = 1.3;
/** Lanes inside the inner rail where the arch stands, behind the flower beds. */
const SETBACK = 5.4;

function canvas(width: number, height: number) {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  return {c, g: c.getContext('2d')!};
}

function textureOf(c: HTMLCanvasElement) {
  const tex = new T.CanvasTexture(c);
  tex.colorSpace = T.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/** A folding fan (扇) motif, opened upwards around (x, y). */
function fan(g: CanvasRenderingContext2D, x: number, y: number, r: number, tilt: number, color: string) {
  g.save();
  g.translate(x, y);
  g.rotate(tilt);
  g.globalAlpha = 0.5;
  g.fillStyle = color;
  g.beginPath();
  g.moveTo(0, 0);
  g.arc(0, 0, r, -Math.PI * 0.85, -Math.PI * 0.15);
  g.closePath();
  g.fill();
  g.globalAlpha = 0.55;
  g.strokeStyle = '#ffffff';
  g.lineWidth = Math.max(1, r / 40);
  for (let i = 0; i <= 8; i++) {
    const a = -Math.PI * (0.85 - (0.7 * i) / 8);
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(Math.cos(a) * r, Math.sin(a) * r);
    g.stroke();
  }
  g.restore();
}

/** Text set top to bottom, centred on x, filling at most `room` pixels of height. */
function vertical(g: CanvasRenderingContext2D, text: string, x: number, top: number, room: number, max: number) {
  const chars = [...text];
  const size = Math.min(max, room / chars.length);
  g.font = `900 ${size}px ${MINCHO}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  chars.forEach((ch, i) => {
    const y = top + size * (i + 0.5);
    if ('ー－—〜~'.includes(ch)) {
      // Long-vowel marks stand upright in vertical text.
      g.save();
      g.translate(x, y);
      g.rotate(Math.PI / 2);
      g.fillText(ch, 0, 0);
      g.restore();
    } else g.fillText(ch, x, y);
  });
  return top + size * chars.length;
}

/** "天皇賞（秋）" → the name and the tag set apart below it. */
function splitName(name: string) {
  const m = name.match(/^(.*?)[（(](.+?)[）)]$/);
  return m ? {main: m[1], tag: m[2]} : {main: name, tag: ''};
}

const HORSESHOE = {width: 7, height: 8.5};
function horseshoeShape() {
  const s = new T.Shape();
  s.moveTo(-3.0, 0);
  s.lineTo(-3.5, 5);
  s.absarc(0, 5, 3.5, Math.PI, 0, true);
  s.lineTo(3.0, 0);
  s.lineTo(1.2, 0);
  s.lineTo(1.7, 5);
  s.absarc(0, 5, 1.7, 0, Math.PI, false);
  s.lineTo(-1.2, 0);
  s.closePath();
  return s;
}

/** Face of the arch: race name down the left leg, date down the right, gold rim, fans. */
function horseshoeTexture(race: Race, shape: T.Shape) {
  const W = 560,
    H = 680;
  const {c, g} = canvas(W, H);
  const px = (x: number) => ((x + HORSESHOE.width / 2) / HORSESHOE.width) * W;
  const py = (y: number) => (1 - y / HORSESHOE.height) * H;
  const wash = g.createLinearGradient(0, 0, 0, H);
  wash.addColorStop(0, '#4733a6');
  wash.addColorStop(1, DEEP_PURPLE);
  g.fillStyle = wash;
  g.fillRect(0, 0, W, H);
  fan(g, px(-2.3), py(6.9), 95, -0.5, '#e76aa5');
  fan(g, px(2.2), py(7.0), 105, 0.55, '#c9b6ff');
  fan(g, px(-2.6), py(0.5), 80, 0.3, '#c9b6ff');
  fan(g, px(2.5), py(0.6), 85, -0.35, '#e76aa5');
  fan(g, px(0), py(8.6), 90, 0, '#e76aa5');
  // Gold rim along the outline.
  g.strokeStyle = GOLD;
  g.lineWidth = 12;
  g.lineJoin = 'round';
  g.beginPath();
  shape.getPoints(40).forEach((p, i) => (i ? g.lineTo(px(p.x), py(p.y)) : g.moveTo(px(p.x), py(p.y))));
  g.closePath();
  g.stroke();
  // Race name, with a tag such as 秋 on its own plaque below.
  const {main, tag} = splitName(race.name);
  g.fillStyle = GOLD;
  g.shadowColor = 'rgba(0,0,0,0.45)';
  g.shadowBlur = 6;
  const nameTop = py(5.7),
    nameRoom = py(tag ? 1.75 : 0.6) - nameTop;
  const end = vertical(g, main, px(-2.42), nameTop, nameRoom, 96);
  if (tag) {
    const y = Math.min(end + 44, py(0.95));
    g.save();
    g.translate(px(-2.3), y);
    g.rotate(Math.PI / 4);
    g.fillStyle = GOLD;
    g.fillRect(-30, -30, 60, 60);
    g.restore();
    g.shadowBlur = 0;
    g.fillStyle = DEEP_PURPLE;
    g.font = `900 46px ${MINCHO}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText([...tag][0], px(-2.3), y + 2);
  }
  // Date: year, month, day and the day of the week.
  const y4 = race.date.slice(0, 4),
    month = String(Number(race.date.slice(4, 6))),
    day = String(Number(race.date.slice(6, 8))),
    week = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'][
      new Date(Number(y4), Number(month) - 1, Number(day)).getDay()
    ];
  g.shadowBlur = 6;
  g.fillStyle = GOLD;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  const dx = px(2.4);
  g.font = `700 44px ${MINCHO}`;
  g.fillText(y4, dx, py(5.2), 130);
  g.font = `900 96px ${MINCHO}`;
  g.fillText(month, dx, py(4.25), 130);
  g.beginPath();
  g.arc(dx, py(3.58), 7, 0, Math.PI * 2);
  g.fill();
  g.fillText(day, dx, py(2.85), 130);
  g.font = `700 40px ${MINCHO}`;
  g.fillText(week, dx, py(1.95), 130);
  return textureOf(c);
}

/** Oval badge at the crown of the arch. */
function badgeTexture() {
  const {c, g} = canvas(320, 220);
  g.fillStyle = PURPLE;
  g.fillRect(0, 0, 320, 220);
  fan(g, 60, 215, 120, -0.2, '#e76aa5');
  fan(g, 270, 60, 90, 2.6, '#c9b6ff');
  g.strokeStyle = GOLD;
  g.lineWidth = 12;
  g.beginPath();
  g.ellipse(160, 110, 150, 100, 0, 0, Math.PI * 2);
  g.stroke();
  g.fillStyle = '#ffffff';
  g.font = `900 120px ${MINCHO}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('GⅠ', 160, 116);
  return textureOf(c);
}

/** One leg of the venue sign: its name in capitals, top to bottom. */
function venueLegTexture(name: string) {
  const {c, g} = canvas(128, 640);
  g.fillStyle = '#d5ebe2';
  g.fillRect(0, 0, 128, 640);
  g.strokeStyle = '#ffffff';
  g.lineWidth = 10;
  g.strokeRect(5, 5, 118, 630);
  g.fillStyle = '#d23a3f';
  g.translate(64, 320);
  g.rotate(Math.PI / 2);
  g.font = '900 86px sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(name, 0, 6, 560);
  return textureOf(c);
}

function venueDiamondTexture(venue: string) {
  const {c, g} = canvas(256, 256);
  g.fillStyle = '#4fb3df';
  g.fillRect(0, 0, 256, 256);
  g.fillStyle = '#ffffff';
  g.font = `900 ${venue.length > 2 ? 62 : 84}px ${MINCHO}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(venue, 128, 120, 200);
  return textureOf(c);
}

function bannerTexture(text: string) {
  const {c, g} = canvas(1024, 64);
  g.fillStyle = PURPLE;
  g.fillRect(0, 0, 1024, 64);
  fan(g, 60, 80, 70, 0, '#e76aa5');
  fan(g, 964, 80, 70, 0, '#e76aa5');
  g.fillStyle = GOLD;
  g.fillRect(0, 0, 1024, 5);
  g.fillRect(0, 59, 1024, 5);
  g.font = `900 44px ${MINCHO}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, 512, 34, 820);
  return textureOf(c);
}

/** Rows of red flowers with a white row between, seen from above. */
function flowerTexture() {
  const {c, g} = canvas(128, 128);
  g.fillStyle = '#2f6b3a';
  g.fillRect(0, 0, 128, 128);
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) % 1000) / 1000;
  for (let row = 0; row < 8; row++)
    for (let i = 0; i < 9; i++) {
      g.fillStyle = row % 4 === 3 ? '#f6f2ea' : rand() > 0.5 ? '#d3182d' : '#b7102a';
      g.beginPath();
      g.arc(i * 14.2 + 7 + rand() * 4, row * 16 + 8 + rand() * 3, 5.5 + rand() * 2, 0, Math.PI * 2);
      g.fill();
    }
  const tex = textureOf(c);
  tex.wrapS = tex.wrapT = T.RepeatWrapping;
  return tex;
}

/** The same face on both sides of a flat sign, `gap` apart, so its text never reads mirrored. */
function twoSided(geometry: T.BufferGeometry, map: T.Texture, gap: number) {
  const group = new T.Group();
  const material = new T.MeshStandardMaterial({map, roughness: 0.65});
  for (const turn of [0, Math.PI]) {
    const face = new T.Mesh(geometry, material);
    face.rotation.y = turn;
    face.position.z = turn ? -gap / 2 : gap / 2;
    face.castShadow = true;
    group.add(face);
  }
  return group;
}

function box(w: number, h: number, d: number, color: string, roughness = 0.8) {
  const m = new T.Mesh(new T.BoxGeometry(w, h, d), new T.MeshStandardMaterial({color, roughness}));
  m.castShadow = m.receiveShadow = true;
  return m;
}

/** Horseshoe arch with the mirror post (ゴール板) standing in its opening. */
function arch(race: Race) {
  const group = new T.Group();
  const shape = horseshoeShape();
  const DEPTH = 0.3;
  const body = new T.Mesh(
    new T.ExtrudeGeometry(shape, {depth: DEPTH, bevelEnabled: false}),
    new T.MeshStandardMaterial({color: GOLD, roughness: 0.5, metalness: 0.3}),
  );
  body.position.z = -DEPTH / 2;
  body.castShadow = true;
  group.add(body);
  const face = new T.ShapeGeometry(shape, 24);
  const pos = face.attributes.position as T.BufferAttribute,
    uv = face.attributes.uv as T.BufferAttribute;
  for (let i = 0; i < pos.count; i++)
    uv.setXY(i, (pos.getX(i) + HORSESHOE.width / 2) / HORSESHOE.width, pos.getY(i) / HORSESHOE.height);
  group.add(twoSided(face, horseshoeTexture(race, shape), DEPTH + 0.04));
  // Badge overlapping the crown.
  const badge = twoSided(new T.CircleGeometry(1, 48), badgeTexture(), DEPTH + 0.36);
  badge.scale.set(1.75, 1.2, 1);
  badge.position.y = 8.75;
  group.add(badge);
  const rim = new T.Mesh(
    new T.CylinderGeometry(1, 1, DEPTH + 0.3, 48),
    new T.MeshStandardMaterial({color: GOLD, roughness: 0.5, metalness: 0.3}),
  );
  rim.rotation.x = Math.PI / 2;
  rim.scale.set(1.75, 1, 1.2);
  rim.position.y = 8.75;
  rim.castShadow = true;
  group.add(rim);
  // Mirror post on its plinth; the plinth also carries the arch down to the ground.
  const plinth = box(7.6, 3.4, 1.5, '#cfd2c9');
  plinth.position.y = -1.3;
  group.add(plinth);
  const post = box(0.62, 7.4, 0.42, '#2f5d48', 0.6);
  post.position.y = 3.9;
  group.add(post);
  const mirror = new T.MeshStandardMaterial({color: '#dcebf0', roughness: 0.12, metalness: 0.75});
  for (const z of [0.215, -0.215]) {
    const glass = new T.Mesh(new T.PlaneGeometry(0.34, 5.6), mirror);
    glass.position.set(0, 4.4, z);
    glass.rotation.y = z > 0 ? 0 : Math.PI;
    group.add(glass);
  }
  return group;
}

/** Venue sign: two leaning legs under a diamond carrying the course's name. */
function venueSign(race: Race) {
  const group = new T.Group();
  const leg = venueLegTexture(courseFor(race).slug.toUpperCase());
  for (const side of [-1, 1]) {
    const part = new T.Group();
    const slab = box(1.2, 6.2, 0.26, '#d5ebe2');
    part.add(slab);
    part.add(twoSided(new T.PlaneGeometry(1.2, 6.2), leg, 0.3));
    const cap = box(2.3, 0.7, 0.3, '#d5ebe2');
    cap.position.y = 3.3;
    part.add(cap);
    part.position.set(side * 1.45, 3.0, 0);
    part.rotation.z = -side * 0.13;
    group.add(part);
  }
  const diamond = new T.Shape();
  diamond.moveTo(0, -1.25);
  diamond.lineTo(1.5, 0.35);
  diamond.lineTo(0.9, 1.05);
  diamond.lineTo(-0.9, 1.05);
  diamond.lineTo(-1.5, 0.35);
  diamond.closePath();
  const face = new T.ShapeGeometry(diamond);
  const pos = face.attributes.position as T.BufferAttribute,
    uv = face.attributes.uv as T.BufferAttribute;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, (pos.getX(i) + 1.5) / 3, (pos.getY(i) + 1.25) / 2.3);
  const sign = twoSided(face, venueDiamondTexture(race.venue), 0.4);
  sign.position.y = 5.9;
  group.add(sign);
  const back = new T.Mesh(
    new T.ExtrudeGeometry(diamond, {depth: 0.34, bevelEnabled: false}),
    new T.MeshStandardMaterial({color: '#ffffff', roughness: 0.6}),
  );
  back.position.set(0, 5.9, -0.17);
  back.scale.set(1.08, 1.08, 1);
  back.castShadow = true;
  group.add(back);
  const foot = box(4.6, 3, 1.2, '#cfd2c9');
  foot.position.y = -1.45;
  group.add(foot);
  return group;
}

/**
 * Raised flower bed inside the inner rail from `from` to `to` (metres past the post),
 * banked to face the course.
 */
function flowerBed(track: TrackModel, from: number, to: number, map: T.Texture) {
  const NEAR = -0.95,
    FAR = -3.5,
    RISE = 1.25,
    n = Math.ceil((to - from) / 2);
  // Across the bed: foot of the front wall, front edge, back edge, foot of the back wall.
  const profile = [
    {lane: NEAR, y: -1.2, v: 0},
    {lane: NEAR, y: 0.12, v: 0},
    {lane: FAR, y: RISE, v: 1},
    {lane: FAR, y: -1.8, v: 1},
  ];
  const positions: number[] = [],
    uvs: number[] = [],
    indices: number[] = [];
  for (let i = 0; i <= n; i++) {
    const d = from + ((to - from) * i) / n;
    for (const {lane, y, v} of profile) {
      const p = track.sample(d, lane).point;
      positions.push(p.x, p.y + y, p.z);
      uvs.push(d / 3, v);
    }
    if (i < n)
      for (let k = 0; k < 3; k++) {
        const a = i * 4 + k;
        indices.push(a, a + 1, a + 4, a + 1, a + 5, a + 4);
      }
  }
  // Close both ends.
  for (const a of [0, n * 4]) indices.push(a, a + 1, a + 2, a, a + 2, a + 3);
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new T.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const bed = new T.Mesh(geo, new T.MeshStandardMaterial({map, roughness: 0.9, side: T.DoubleSide}));
  bed.castShadow = bed.receiveShadow = true;
  return bed;
}

/** ゴール板 and its GⅠ dressing at the winning post, inside the inner rail. */
export function addGoalBoard(scene: T.Scene, race: Race, track: TrackModel) {
  const facing = (distance: number, lane: number) => {
    const at = track.sample(distance, lane);
    const group = new T.Group();
    group.position.copy(at.point);
    group.rotation.y = -Math.atan2(at.tangent.z, at.tangent.x);
    return group;
  };
  const post = facing(0, -SETBACK);
  post.scale.setScalar(SCALE);
  post.add(arch(race));
  post.name = 'goal-post';
  scene.add(post);
  const sign = facing(13.5, -SETBACK);
  sign.scale.setScalar(SCALE);
  sign.add(venueSign(race));
  scene.add(sign);
  const flowers = flowerTexture();
  scene.add(flowerBed(track, -46, -1.2, flowers), flowerBed(track, 1.2, 34, flowers));
  // Banners hung on the course side of the inner rail, either side of the post.
  const banner = new T.PlaneGeometry(13, 0.82);
  for (const [distance, text] of [
    [-8.5, `${race.venue}競馬場`],
    [8.5, race.name],
  ] as const) {
    const hung = facing(distance, 0.14);
    hung.position.y += 0.62;
    hung.add(twoSided(banner, bannerTexture(text), 0.03));
    scene.add(hung);
  }
}
