import * as T from 'three';
import {seeded, tileNoise} from './scenery';
import {normalFromHeight} from './surfaceFinish';

/**
 * Realistic trees and clipped shrubs, built the way real-time foliage usually is: a wooden skeleton of tapering tubes
 * (trunk, scaffold limbs, branches forking down to twigs) and, over it, a few hundred leaf cards — small quads carrying a
 * drawn spray of leaves (cut out by alpha) — clustered at the branch tips and filling the species' crown outline. The cards
 * are lit with normals pointing out of the crown, so the crown shades as a volume while its edge stays leafy.
 *
 * Everything is generated in metres at a species' nominal size; instances scale it by a little. Leaves carry only lightness;
 * their colour (season, autumn patches, blossom) comes from the instance (see `foliageMaterial`).
 */

export type TreeShape = 'zelkova' | 'ginkgo' | 'cherry' | 'maple' | 'camphor' | 'pine';
export type ShrubLeaf = 'azalea' | 'enkianthus' | 'box';
export type LeafKind = TreeShape | ShrubLeaf;

type Shape = {
  /** Nominal height and crown width (m). */
  height: number; width: number;
  /** Crown outline: where it starts (fraction of height), where it is widest (fraction of the crown), how full (0.4 boxy … 1 pointed). */
  base: number; widest: number; round: number;
  /** Trunk radius at the foot (m). */
  radius: number;
  /** Scaffold trees: trunk to `trunk` (fraction of height), `limbs` limbs at `angle` from vertical, `reach` long (fraction of height). */
  trunk: number; limbs: number; angle: number; reach: number;
  /** Leader trees (ginkgo, pine): whorls of `perWhorl` branches every `step` m from `trunk`, at `angle`. */
  leader?: {step: number; perWhorl: number};
  /** Children per level below the scaffold, their angle off the parent, upward bias and length ratio. */
  children: number[]; spread: number; rise: number; ratio: number;
  /** Leaf cards per tree and their size (m). Pads: cards gather in flat pads at the tips (pine). */
  cards: number; card: number; pads?: boolean;
};

export const SHAPES: Record<TreeShape, Shape> = {
  zelkova: {height: 13, width: 12, base: 0.3, widest: 0.72, round: 0.42, radius: 0.32, trunk: 0.3, limbs: 5, angle: 0.62,
    reach: 0.5, children: [3, 2, 2], spread: 0.55, rise: 0.16, ratio: 0.62, cards: 680, card: 1.7},
  ginkgo: {height: 14, width: 5.6, base: 0.14, widest: 0.32, round: 0.9, radius: 0.27, trunk: 0.2, limbs: 0, angle: 1.0,
    reach: 0, leader: {step: 1.1, perWhorl: 3}, children: [3], spread: 0.5, rise: 0.22, ratio: 0.6, cards: 440, card: 1.2},
  cherry: {height: 8, width: 11, base: 0.26, widest: 0.5, round: 0.38, radius: 0.3, trunk: 0.22, limbs: 4, angle: 1.15,
    reach: 0.75, children: [3, 2, 2], spread: 0.6, rise: 0.05, ratio: 0.62, cards: 620, card: 1.55},
  maple: {height: 8.5, width: 8, base: 0.24, widest: 0.5, round: 0.5, radius: 0.19, trunk: 0.28, limbs: 5, angle: 0.72,
    reach: 0.45, children: [3, 2, 2], spread: 0.6, rise: 0.12, ratio: 0.62, cards: 440, card: 1.2},
  camphor: {height: 12, width: 12, base: 0.2, widest: 0.45, round: 0.5, radius: 0.36, trunk: 0.22, limbs: 6, angle: 0.8,
    reach: 0.45, children: [3, 2, 2], spread: 0.6, rise: 0.1, ratio: 0.62, cards: 700, card: 1.75},
  pine: {height: 10, width: 7, base: 0.42, widest: 0.38, round: 0.6, radius: 0.22, trunk: 0.42, limbs: 0, angle: 1.32,
    reach: 0, leader: {step: 1.15, perWhorl: 2}, children: [3], spread: 0.55, rise: 0.02, ratio: 0.55, cards: 360, card: 1.3,
    pads: true},
};

export type TreeModel = {wood: T.BufferGeometry; leaves: T.BufferGeometry; shape: Shape};

/** Tapering tubes with bark uv (u round the bark in 0.6 m repeats, v up it in 1.2 m repeats). */
class Wood {
  pos: number[] = []; nor: number[] = []; uv: number[] = []; idx: number[] = [];
  tube(points: T.Vector3[], radii: number[], sides: number) {
    const base = this.pos.length / 3;
    let along = 0;
    let reference = new T.Vector3(1, 0, 0.3).normalize();
    const wraps = Math.max(1, Math.round((2 * Math.PI * radii[0]) / 0.6));
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const dir = (i < points.length - 1 ? points[i + 1].clone().sub(p) : p.clone().sub(points[i - 1])).normalize();
      if (i > 0) along += p.distanceTo(points[i - 1]);
      let side = reference.clone().addScaledVector(dir, -reference.dot(dir));
      if (side.lengthSq() < 1e-6) side = new T.Vector3(0, 0, 1).addScaledVector(dir, -dir.z);
      side.normalize();
      reference = side.clone();
      const other = new T.Vector3().crossVectors(dir, side);
      for (let s = 0; s <= sides; s++) {
        const a = (s / sides) * Math.PI * 2;
        const n = side.clone().multiplyScalar(Math.cos(a)).addScaledVector(other, Math.sin(a));
        this.pos.push(p.x + n.x * radii[i], p.y + n.y * radii[i], p.z + n.z * radii[i]);
        this.nor.push(n.x, n.y, n.z);
        this.uv.push((s / sides) * wraps, along / 1.2);
      }
    }
    for (let i = 0; i < points.length - 1; i++)
      for (let s = 0; s < sides; s++) {
        const a = base + i * (sides + 1) + s, b = a + sides + 1;
        this.idx.push(a, a + 1, b, a + 1, b + 1, b);
      }
  }
  geometry() {
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new T.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('uv', new T.Float32BufferAttribute(this.uv, 2));
    g.setIndex(this.idx);
    return g;
  }
}

/**
 * Leaf cards: four vertices each, `aUv` in the card (0..1), `aCard` = (when it drops, which spray, which colour patch).
 * Normals point out of the crown (a little toward the card's own facing), colour holds the shade (darker inside).
 */
class Cards {
  pos: number[] = []; nor: number[] = []; uv: number[] = []; card: number[] = []; col: number[] = []; idx: number[] = [];
  add(center: T.Vector3, facing: T.Vector3, outward: T.Vector3, size: number, roll: number, card: [number, number, number], shade: number) {
    const base = this.pos.length / 3;
    const helper = Math.abs(facing.y) > 0.9 ? new T.Vector3(1, 0, 0) : new T.Vector3(0, 1, 0);
    const u = new T.Vector3().crossVectors(helper, facing).normalize().applyAxisAngle(facing, roll);
    const v = new T.Vector3().crossVectors(facing, u);
    const light = outward.clone().multiplyScalar(0.75).addScaledVector(facing, 0.25).normalize();
    const corners: [number, number][] = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
    for (const [a, b] of corners) {
      const p = center.clone().addScaledVector(u, a * size).addScaledVector(v, b * size);
      this.pos.push(p.x, p.y, p.z);
      this.nor.push(light.x, light.y, light.z);
      this.uv.push(a + 0.5, b + 0.5);
      this.card.push(...card);
      this.col.push(shade, shade, shade);
    }
    this.idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  geometry() {
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new T.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('aUv', new T.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('aCard', new T.Float32BufferAttribute(this.card, 3));
    g.setAttribute('color', new T.Float32BufferAttribute(this.col, 3));
    g.setIndex(this.idx);
    g.computeBoundingSphere();
    return g;
  }
}

const randomUnit = (random: () => number) => {
  const z = random() * 2 - 1, a = random() * Math.PI * 2, r = Math.sqrt(1 - z * z);
  return new T.Vector3(r * Math.cos(a), z, r * Math.sin(a));
};

const models = new Map<string, TreeModel>();

/** One tree of a shape (`variant` picks the random layout; a few per shape keep a row from repeating). Cached. */
export function treeModel(name: TreeShape, variant: number): TreeModel {
  const key = `${name}:${variant}`;
  const cached = models.get(key);
  if (cached) return cached;
  const shape = SHAPES[name];
  const random = seeded(1009 + variant * 7919 + name.length * 131 + name.charCodeAt(0) * 17);
  const {height: H, width: W} = shape;
  const crownFrom = shape.base * H, crownSpan = H - crownFrom, centre = crownFrom + crownSpan * 0.5;
  const p = Math.log(0.5) / Math.log(shape.widest);
  // Each tree's outline is the species' outline with a few soft bulges, so no two variants share a silhouette.
  const bulge = [0, 1, 2].map(() => ({angle: random() * Math.PI * 2, amount: 0.08 + random() * 0.1}));
  const radiusAt = (y: number, x = 0, z = 0) => {
    const t = (y - crownFrom) / crownSpan;
    if (t <= 0 || t >= 1) return 0;
    const azimuth = Math.atan2(z, x);
    const wobble = 1 + bulge.reduce((s, b) => s + b.amount * Math.cos(azimuth - b.angle), 0) - 0.04;
    return (W / 2) * wobble * Math.pow(Math.sin(Math.PI * Math.pow(t, p)), shape.round);
  };
  const inside = (v: T.Vector3, margin = 1) => Math.hypot(v.x, v.z) <= radiusAt(v.y, v.x, v.z) * margin;
  const wood = new Wood();
  const tips: {at: T.Vector3; dir: T.Vector3}[] = [];
  const up = new T.Vector3(0, 1, 0);
  const sides = (r: number) => (r > 0.12 ? 7 : r > 0.06 ? 5 : 3);
  const last = shape.children.length + 1;

  const grow = (from: T.Vector3, dir: T.Vector3, length: number, radius: number, level: number) => {
    let to = from.clone().addScaledVector(dir, length);
    if (inside(from) && !inside(to)) {
      let lo = 0, hi = 1;
      for (let i = 0; i < 10; i++) {
        const mid = (lo + hi) / 2;
        if (inside(from.clone().lerp(to, mid))) lo = mid; else hi = mid;
      }
      to = from.clone().lerp(to, Math.max(lo, 0.3));
    }
    const span = from.distanceTo(to);
    // A slight bend, and sag where the branch runs out sideways.
    const bend = randomUnit(random).multiplyScalar(span * 0.07);
    const mid = from.clone().lerp(to, 0.5).add(bend);
    mid.y -= span * 0.05 * (1 - Math.abs(dir.y));
    const end = radius * 0.7;
    if (level >= last - 1) wood.tube([from, to], [radius, end], sides(radius));
    else wood.tube([from, mid, to], [radius, radius * 0.85, end], sides(radius));
    if (level === last) {
      tips.push({at: to, dir: dir.clone()});
      return;
    }
    const n = shape.children[level - 1];
    for (let k = 0; k < n; k++) {
      const t = k === 0 ? 1 : 0.4 + 0.55 * random();
      const at = t < 0.5 ? from.clone().lerp(mid, t * 2) : mid.clone().lerp(to, t * 2 - 1);
      const across = new T.Vector3().crossVectors(dir, Math.abs(dir.y) > 0.95 ? new T.Vector3(1, 0, 0) : up).normalize()
        .applyAxisAngle(dir, (k / n) * Math.PI * 2 + random() * 1.2);
      const child = dir.clone().applyAxisAngle(across, shape.spread * (k === 0 ? 0.45 : 1) * (0.7 + 0.6 * random()));
      child.y += shape.rise;
      child.normalize();
      const r = Math.min(end, radius * (k === 0 ? 0.72 : 0.58));
      grow(at, child, length * shape.ratio * (0.8 + 0.4 * random()) * (k === 0 ? 1 : 0.88), r, level + 1);
    }
  };

  const R = shape.radius;
  if (shape.leader) {
    // A central stem to the top, bending a little, with whorls of branches that shorten with the outline.
    const points = [new T.Vector3(0, 0, 0)], radii = [R * 1.35];
    const steps = 7;
    const lean = randomUnit(random).setY(0).multiplyScalar(name === 'pine' ? 0.06 : 0.015);
    for (let i = 1; i <= steps; i++) {
      const y = (i / steps) * H * 0.97;
      const wander = name === 'pine' ? 0.12 * Math.sin(i * 1.7 + variant) : 0.03;
      points.push(new T.Vector3(lean.x * y + wander * (random() - 0.5), y, lean.z * y + wander * (random() - 0.5)));
      radii.push(R * Math.max(0.12, 1 - (i / steps) * 0.9));
    }
    wood.tube(points, radii, 8);
    const stemAt = (y: number) => {
      const f = Math.min(steps - 1e-3, (y / (H * 0.97)) * steps), i = Math.floor(f);
      return {point: points[i].clone().lerp(points[i + 1], f - i), radius: T.MathUtils.lerp(radii[i], radii[i + 1], f - i)};
    };
    for (let y = shape.trunk * H; y < H * 0.95; y += shape.leader.step * (0.8 + 0.4 * random())) {
      const turn = random() * Math.PI * 2;
      for (let k = 0; k < shape.leader.perWhorl; k++) {
        const a = turn + (k / shape.leader.perWhorl) * Math.PI * 2 + (random() - 0.5) * 0.6;
        const angle = shape.angle * (0.85 + 0.3 * random());
        const dir = new T.Vector3(Math.cos(a) * Math.sin(angle), Math.cos(angle), Math.sin(a) * Math.sin(angle));
        const {point, radius} = stemAt(y);
        const reach = Math.max(0.6, radiusAt(y + 0.5, dir.x, dir.z) * 1.05) / Math.sin(angle);
        grow(point, dir, reach * 0.6, Math.max(0.02, radius * 0.45), 1);
      }
    }
  } else {
    // A trunk to the first fork, then scaffold limbs fanning out.
    const top = shape.trunk * H;
    const lean = randomUnit(random).setY(0).multiplyScalar(0.03);
    const points = [0, 0.35, top * 0.55, top].map(y => new T.Vector3(lean.x * y, y, lean.z * y));
    wood.tube(points, [R * 1.35, R, R * 0.92, R * 0.82], 8);
    const turn = random() * Math.PI * 2;
    for (let i = 0; i < shape.limbs; i++) {
      const a = turn + (i / shape.limbs) * Math.PI * 2 + (random() - 0.5) * 0.7;
      const angle = shape.angle * (0.8 + 0.4 * random());
      const dir = new T.Vector3(Math.cos(a) * Math.sin(angle), Math.cos(angle), Math.sin(a) * Math.sin(angle));
      const from = points[3].clone().lerp(points[2], random() * 0.25);
      grow(from, dir, shape.reach * H * (0.75 + 0.35 * random()) * 0.55, R * 0.82 * (0.5 + 0.15 * random()), 1);
    }
    // The leader often carries on up the middle.
    grow(points[3].clone(), new T.Vector3(0, 1, 0), shape.reach * H * 0.5, R * 0.45, 1);
  }

  // Leaf cards in clusters: most at the twig tips, the rest filling out the outline so the silhouette is the species'.
  const cards = new Cards();
  const clusters = Math.round(shape.cards / 5);
  const fromTips = shape.pads ? clusters : Math.round(clusters * 0.5);
  const order = tips.map((tip, i) => ({tip, key: random() + i * 0})).sort((a, b) => a.key - b.key).map(t => t.tip);
  const centres: T.Vector3[] = [];
  for (let i = 0; i < fromTips && order.length; i++) centres.push(order[i % order.length].at.clone());
  for (let tries = 0; centres.length < clusters && tries < 20000; tries++) {
    const v = new T.Vector3((random() - 0.5) * W * 1.2, crownFrom + random() * crownSpan, (random() - 0.5) * W * 1.2);
    const R0 = radiusAt(v.y, v.x, v.z);
    if (R0 <= 0) continue;
    const r = Math.hypot(v.x, v.z) / R0;
    if (r <= 1 && (r > 0.6 || random() < 0.1)) centres.push(v);
  }
  const halfX = W / 2, halfY = crownSpan / 2;
  for (const c of centres) {
    const clusterDrop = random();
    const patch = random();
    for (let j = 0; j < 5; j++) {
      const offset = randomUnit(random).multiplyScalar(shape.card * (0.25 + 0.45 * random()));
      if (shape.pads) offset.y *= 0.3;
      const at = c.clone().add(offset);
      const limit = radiusAt(at.y, at.x, at.z);
      const r = Math.hypot(at.x, at.z);
      if (limit > 0 && r > limit * 1.04) at.multiply(new T.Vector3(limit * 1.04 / r, 1, limit * 1.04 / r));
      // Out of the crown: the gradient of its ellipsoid; for pads, mostly up.
      const outward = shape.pads
        ? new T.Vector3(at.x * 0.3, 1, at.z * 0.3).normalize()
        : new T.Vector3(at.x / (halfX * halfX), (at.y - centre) / (halfY * halfY), at.z / (halfX * halfX)).normalize();
      const facing = outward.clone().addScaledVector(randomUnit(random), shape.pads ? 0.5 : 0.95).normalize();
      // Darker toward the inside and the underside of the crown.
      const depth = limit > 0 ? Math.min(1, r / limit) : 0.5;
      const shade = T.MathUtils.clamp(0.42 + 0.48 * Math.pow(depth, 1.5) + 0.22 * ((at.y - crownFrom) / crownSpan), 0.35, 1.08);
      cards.add(at, facing, outward, shape.card * (0.75 + 0.5 * random()), random() * Math.PI * 2,
        [0.65 * clusterDrop + 0.35 * random(), random(), 0.7 * patch + 0.3 * random()], shade);
    }
  }
  const model = {wood: wood.geometry(), leaves: cards.geometry(), shape};
  models.set(key, model);
  return model;
}

export type ShrubForm = 'ball' | 'hedge';
const shrubs = new Map<string, {core: T.BufferGeometry; leaves: T.BufferGeometry}>();

/**
 * A clipped shrub, in metres, standing on y = 0: a rounded azalea ball (玉作り, 1.4 m across, 0.85 m tall) or a 3.6 m piece of
 * clipped hedge (1 m wide and tall, square with rounded edges; pieces overlap a little and read as one hedge). A dark core
 * keeps it from being seen through; small leaf cards sit over its surface, facing out, as a clipped face looks.
 */
export function shrubModel(form: ShrubForm, variant: number) {
  const key = `${form}:${variant}`;
  const cached = shrubs.get(key);
  if (cached) return cached;
  const random = seeded(6007 + variant * 409 + (form === 'hedge' ? 77 : 0));
  const cards = new Cards();
  let core: T.BufferGeometry;
  const card = form === 'hedge' ? 0.4 : 0.34;
  if (form === 'ball') {
    const [a, b] = [0.7 * (0.92 + 0.16 * random()), 0.46 * (0.9 + 0.2 * random())];
    core = new T.IcosahedronGeometry(1, 2);
    core.scale(a * 0.9, b * 0.9, a * 0.9);
    core.translate(0, b * 0.85, 0);
    const count = 120;
    for (let i = 0; i < count; i++) {
      const n = randomUnit(random);
      if (n.y < -0.35) n.y = -0.35 + random() * 0.2;
      n.normalize();
      const bump = 1 + 0.06 * (random() - 0.5);
      const at = new T.Vector3(n.x * a * bump, b * 0.85 + n.y * b * bump, n.z * a * bump);
      const outward = new T.Vector3(n.x / a, n.y / b, n.z / a).normalize();
      const facing = outward.clone().addScaledVector(randomUnit(random), 0.7).normalize();
      const shade = T.MathUtils.clamp(0.62 + 0.38 * outward.y, 0.4, 1.05);
      cards.add(at, facing, outward, card * (0.8 + 0.4 * random()), random() * 6.3, [random(), random(), random()], shade);
    }
  } else {
    const L = 3.6, D = 1.0, Hh = 1.0, round = 0.28;
    core = new T.BoxGeometry(L - 0.1, Hh - 0.12, D - 0.14);
    core.translate(0, (Hh - 0.12) / 2, 0);
    const half = new T.Vector3(L / 2, Hh / 2, D / 2), inner = half.clone().subScalar(round);
    // Surface area of the visible faces (top, both sides, ends); cards in proportion.
    const faces: [T.Vector3, number][] = [
      [new T.Vector3(0, 1, 0), L * D], [new T.Vector3(0, 0, 1), L * Hh], [new T.Vector3(0, 0, -1), L * Hh],
      [new T.Vector3(1, 0, 0), D * Hh], [new T.Vector3(-1, 0, 0), D * Hh],
    ];
    const total = faces.reduce((s, [, area]) => s + area, 0);
    const count = 230;
    for (let i = 0; i < count; i++) {
      let pick = random() * total, normal = faces[0][0];
      for (const [n, area] of faces) { if ((pick -= area) < 0) { normal = n; break; } }
      // A point on that face of the box, then pulled onto the rounded box.
      const p = new T.Vector3((random() * 2 - 1) * half.x, (random() * 2 - 1) * half.y, (random() * 2 - 1) * half.z);
      if (normal.x) p.x = normal.x * half.x;
      if (normal.y) p.y = half.y;
      if (normal.z) p.z = normal.z * half.z;
      if (p.y < -half.y + 0.12) p.y = -half.y + 0.12;
      const clamped = p.clone().clamp(inner.clone().negate(), inner);
      const out = p.clone().sub(clamped);
      const outward = out.lengthSq() > 1e-6 ? out.normalize() : normal.clone();
      const at = clamped.addScaledVector(outward, round * (1 + 0.05 * (random() - 0.5)));
      at.y += half.y;
      const facing = outward.clone().addScaledVector(randomUnit(random), 0.7).normalize();
      const shade = T.MathUtils.clamp(0.6 + 0.4 * outward.y + 0.12 * (at.y / Hh), 0.4, 1.08);
      cards.add(at, facing, outward, card * (0.8 + 0.4 * random()), random() * 6.3, [random(), random(), random()], shade);
    }
  }
  const model = {core, leaves: cards.geometry()};
  shrubs.set(key, model);
  return model;
}

/* ---------- Leaf sprays ---------- */

type Leaf = (g: CanvasRenderingContext2D, size: number) => void;

const ovate = (narrow: number, tip = 0.15): Leaf => (g, s) => {
  g.beginPath();
  g.moveTo(0, 0);
  g.bezierCurveTo(s * narrow, s * 0.25, s * narrow * 0.9, s * (0.75 - tip), 0, s);
  g.bezierCurveTo(-s * narrow * 0.9, s * (0.75 - tip), -s * narrow, s * 0.25, 0, 0);
  g.fill();
};
const fan: Leaf = (g, s) => {
  g.beginPath();
  g.moveTo(0, 0);
  g.lineTo(-s * 0.5, s * 0.78);
  g.quadraticCurveTo(-s * 0.25, s * 1.02, -s * 0.03, s * 0.86);
  g.lineTo(0, s * 0.74);
  g.lineTo(s * 0.03, s * 0.86);
  g.quadraticCurveTo(s * 0.25, s * 1.02, s * 0.5, s * 0.78);
  g.closePath();
  g.fill();
};
const palmate: Leaf = (g, s) => {
  g.beginPath();
  const lobes = 7;
  for (let i = 0; i <= lobes * 2; i++) {
    const a = -Math.PI * 0.85 + (i / (lobes * 2)) * Math.PI * 1.7;
    const r = i % 2 === 0 ? s * 0.55 : s * 0.2;
    g.lineTo(Math.sin(a) * r, s * 0.35 + Math.cos(a) * r);
  }
  g.closePath();
  g.fill();
};
const needles: Leaf = (g, s) => {
  g.lineWidth = Math.max(1, s * 0.035);
  for (let i = 0; i < 26; i++) {
    const a = -Math.PI * 0.9 + (i / 25) * Math.PI * 1.8;
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(Math.sin(a) * s, Math.cos(a) * s * 0.9 + s * 0.15);
    g.stroke();
  }
};
const blossom: Leaf = (g, s) => {
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2;
    g.beginPath();
    g.ellipse(Math.cos(a) * s * 0.28, Math.sin(a) * s * 0.28, s * 0.24, s * 0.18, a, 0, Math.PI * 2);
    g.fill();
  }
};
const funnel: Leaf = (g, s) => {
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2;
    g.beginPath();
    g.ellipse(Math.cos(a) * s * 0.26, Math.sin(a) * s * 0.26, s * 0.26, s * 0.2, a, 0, Math.PI * 2);
    g.fill();
  }
};

/** Per kind: leaf, leaf length (px in a 256 cell), leaves per spray, and the fourth cell (flowers, or a fourth spray). */
const SPRAYS: Record<LeafKind, {leaf: Leaf; size: number; count: number; flower?: {leaf: Leaf; size: number; count: number}}> = {
  zelkova: {leaf: ovate(0.32, 0.22), size: 30, count: 110},
  ginkgo: {leaf: fan, size: 30, count: 90},
  cherry: {leaf: ovate(0.36, 0.2), size: 38, count: 80, flower: {leaf: blossom, size: 24, count: 130}},
  maple: {leaf: palmate, size: 36, count: 80},
  camphor: {leaf: ovate(0.34, 0.12), size: 34, count: 110},
  pine: {leaf: needles, size: 46, count: 44},
  azalea: {leaf: ovate(0.3, 0.1), size: 20, count: 120, flower: {leaf: funnel, size: 30, count: 46}},
  enkianthus: {leaf: ovate(0.38, 0.08), size: 18, count: 130},
  box: {leaf: ovate(0.42, 0.02), size: 12, count: 260},
};

const atlases = new Map<LeafKind, T.Texture>();

/**
 * A 2×2 atlas of leaf sprays for a kind: cells 0–2 are sprays, cell 3 its flowers (cherry, azalea) or a fourth spray.
 * Channels are data: R the leaf's lightness, G the leaf mask, B the mask of the twigs that carry them (shown when the leaves
 * have fallen). Alpha is opaque, so the canvas keeps the values exactly.
 */
export function leafAtlas(kind: LeafKind) {
  const cached = atlases.get(kind);
  if (cached) return cached;
  const size = 512, cell = 256;
  const make = () => {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    return c;
  };
  const leaves = make(), twigs = make();
  const lg = leaves.getContext('2d')!, tg = twigs.getContext('2d')!;
  const random = seeded(311 + kind.length * 97 + kind.charCodeAt(1) * 13);
  const spray = SPRAYS[kind];
  for (let n = 0; n < 4; n++) {
    const ox = (n % 2) * cell, oy = Math.floor(n / 2) * cell;
    const flower = n === 3 && spray.flower;
    const {leaf, size: leafSize, count} = flower || spray;
    lg.save(); tg.save();
    lg.beginPath(); lg.rect(ox + 4, oy + 4, cell - 8, cell - 8); lg.clip();
    tg.beginPath(); tg.rect(ox + 4, oy + 4, cell - 8, cell - 8); tg.clip();
    // Twigs: a main twig across the cell and side twigs, thinning out.
    const stems: [number, number][] = [];
    const branch = (x: number, y: number, a: number, len: number, width: number, depth: number) => {
      const steps = 6;
      for (let i = 0; i < steps; i++) {
        const nx = x + Math.sin(a) * (len / steps), ny = y - Math.cos(a) * (len / steps);
        tg.strokeStyle = '#fff';
        tg.lineWidth = width * (1 - i / (steps * 1.6));
        tg.lineCap = 'round';
        tg.beginPath(); tg.moveTo(x, y); tg.lineTo(nx, ny); tg.stroke();
        x = nx; y = ny; a += (random() - 0.5) * 0.35;
        stems.push([x, y]);
        if (depth > 0 && random() < 0.4) branch(x, y, a + (random() < 0.5 ? -1 : 1) * (0.5 + random() * 0.5), len * 0.5, width * 0.6, depth - 1);
      }
    };
    branch(ox + cell * (0.3 + 0.4 * random()), oy + cell - 6, (random() - 0.5) * 0.8, cell * 0.95, 7, 2);
    branch(ox + cell * (0.2 + 0.6 * random()), oy + cell - 6, (random() - 0.5) * 1.4, cell * 0.6, 4, 1);
    // Leaves on the twigs, each its own lightness, with a darker midrib.
    for (let i = 0; i < count; i++) {
      const [sx, sy] = stems[Math.floor(random() * stems.length)];
      const x = sx + (random() - 0.5) * 34, y = sy + (random() - 0.5) * 34;
      const l = flower ? 200 + random() * 55 : 120 + random() * 135;
      lg.save();
      lg.translate(x, y);
      lg.rotate(random() * Math.PI * 2);
      const s = leafSize * (0.75 + 0.5 * random());
      lg.fillStyle = `rgb(${l},${l},${l})`;
      lg.strokeStyle = `rgb(${l},${l},${l})`;
      leaf(lg, s);
      if (!flower && leaf !== needles) {
        lg.strokeStyle = `rgb(${l * 0.7},${l * 0.7},${l * 0.7})`;
        lg.lineWidth = 1;
        lg.beginPath(); lg.moveTo(0, 0); lg.lineTo(0, s * 0.85); lg.stroke();
      }
      lg.restore();
    }
    lg.restore(); tg.restore();
  }
  const L = lg.getImageData(0, 0, size, size).data, Tw = tg.getImageData(0, 0, size, size).data;
  const out = document.createElement('canvas');
  out.width = out.height = size;
  const og = out.getContext('2d')!;
  const image = og.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    image.data[i * 4] = L[i * 4];
    image.data[i * 4 + 1] = L[i * 4 + 3];
    image.data[i * 4 + 2] = Tw[i * 4 + 3];
    image.data[i * 4 + 3] = 255;
  }
  og.putImageData(image, 0, 0);
  const texture = new T.CanvasTexture(out);
  texture.colorSpace = T.NoColorSpace;
  texture.anisotropy = 4;
  atlases.set(kind, texture);
  return texture;
}

/**
 * The material of leaf cards. Per instance, `aState` packs how full the crown is (x: cards drop in clusters as it falls), how
 * much of the bare twigs shows (y), the share of blossom (z) and a seed (w); `aMain`, `aPatch` are the leaf colours (autumn
 * turns in patches) and `aBlossom` the flowers'. Snow lies on the upward cards. (Packed: WebGL allows 16 attributes and the
 * instance matrix takes four.)
 */
export function foliageMaterial(atlas: T.Texture, snow: number) {
  const material = new T.MeshStandardMaterial({color: '#fff', roughness: 0.85, vertexColors: true, side: T.DoubleSide});
  material.onBeforeCompile = shader => {
    shader.uniforms.uLeaves = {value: atlas};
    shader.uniforms.uSnow = {value: snow};
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
        attribute vec2 aUv;
        attribute vec3 aCard;
        attribute vec4 aState;
        attribute vec3 aMain;
        attribute vec3 aPatch;
        attribute vec3 aBlossom;
        varying vec2 vLeafUv;
        varying vec3 vCard;
        varying float vLeaf;
        varying float vTwig;
        varying float vBloom;
        varying vec3 vMain;
        varying vec3 vPatch;
        varying vec3 vBlossom;
        varying vec3 vSpot;
        varying float vUp;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vLeafUv = aUv; vCard = aCard; vLeaf = aState.x; vTwig = aState.y; vBloom = aState.z;
        vMain = aMain; vPatch = aPatch; vBlossom = aBlossom;
        vSpot = position * 0.3 + aState.w * 23.0;
        vUp = normal.y;`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform sampler2D uLeaves;
        uniform float uSnow;
        varying vec2 vLeafUv;
        varying vec3 vCard;
        varying float vLeaf;
        varying float vTwig;
        varying float vBloom;
        varying vec3 vMain;
        varying vec3 vPatch;
        varying vec3 vBlossom;
        varying vec3 vSpot;
        varying float vUp;
        float leafHash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
        float leafNoise(vec3 p) {
          vec3 i = floor(p), f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(mix(leafHash(i), leafHash(i + vec3(1, 0, 0)), f.x), mix(leafHash(i + vec3(0, 1, 0)), leafHash(i + vec3(1, 1, 0)), f.x), f.y),
            mix(mix(leafHash(i + vec3(0, 0, 1)), leafHash(i + vec3(1, 0, 1)), f.x), mix(leafHash(i + vec3(0, 1, 1)), leafHash(i + vec3(1, 1, 1)), f.x), f.y),
            f.z);
        }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        bool flower = vCard.y < vBloom;
        float cell = flower ? 3.0 : floor(vCard.z * 2.999 + fract(vCard.y * 7.0) * 0.999);
        cell = flower ? 3.0 : mod(cell, 3.0);
        vec2 atlasUv = (clamp(vLeafUv, 0.01, 0.99) + vec2(mod(cell, 2.0), floor(cell / 2.0))) * 0.5;
        vec3 spray = texture2D(uLeaves, atlasUv).rgb;
        bool on = vCard.x < vLeaf;
        float twig = vTwig * spray.b * (1.0 - 0.75 * vLeaf);
        if (!(on && spray.g > 0.42) && twig < 0.3) discard;
        vec3 tint;
        if (on && spray.g > 0.42) {
          float patchy = smoothstep(0.3, 0.7, 0.6 * leafNoise(vSpot) + 0.4 * vCard.z);
          tint = (flower ? vBlossom : mix(vMain, vPatch, patchy)) * (0.5 + 0.62 * spray.r);
        } else {
          tint = vec3(0.085, 0.07, 0.06);
        }
        diffuseColor.rgb *= tint;
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.86, 0.88, 0.9), uSnow * smoothstep(0.15, 0.75, vUp));`)
      .replace('#include <normal_fragment_begin>', `#include <normal_fragment_begin>
        normal = normalize(vNormal);`);
  };
  material.customProgramCacheKey = () => 'leaf-cards';
  return material;
}

let barkMaps: {map: T.CanvasTexture; normalMap: T.CanvasTexture} | null = null;

/** Bark: furrows running up the trunk and flakes between them; lightness and relief, tinted per species by the instance. */
export function barkTextures() {
  if (barkMaps) return barkMaps;
  const size = 256, random = seeded(887);
  const [coarse, fine, flake] = [6, 40, 24].map(n => tileNoise(random, n));
  const height = new Float32Array(size * size);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const g = canvas.getContext('2d')!;
  const image = g.createImageData(size, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const u = x / size, v = y / size;
      // Furrows: ridges of noise squeezed across the bark (u), long up it (v).
      // (Three times across, once up, so the pattern still tiles.)
      const across = (((u * 3 + 0.12 * fine(u, v)) % 1) + 1) % 1;
      const furrow = 1 - Math.abs(coarse(across, v));
      const h = 0.7 * Math.pow(furrow, 3) + 0.25 * fine(u, v) + 0.2 * Math.max(0, flake(u, v));
      height[y * size + x] = h;
      const l = T.MathUtils.clamp(0.55 + 0.45 * h + 0.08 * (random() - 0.5), 0.25, 1) * 255;
      const i = (y * size + x) * 4;
      image.data[i] = image.data[i + 1] = image.data[i + 2] = l;
      image.data[i + 3] = 255;
    }
  g.putImageData(image, 0, 0);
  const map = new T.CanvasTexture(canvas);
  map.wrapS = map.wrapT = T.RepeatWrapping;
  map.colorSpace = T.SRGBColorSpace;
  map.anisotropy = 4;
  barkMaps = {map, normalMap: normalFromHeight(height, size, 3)};
  return barkMaps;
}
