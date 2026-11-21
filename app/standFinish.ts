import * as T from 'three';
import {seeded} from './scenery';
import {photoTextures, type PhotoSet} from './photoTextures';
import {CONCRETE_TILE, concreteMaterial} from './surfaceFinish';
import {apronTexture} from './scenery';

// Surfaces of the main stands (app/raceStands.ts), photoreal: scanned concrete, paving and timber (Poly Haven, CC0) where a
// scan exists, and drawn maps for what has no scan — façade cladding with its joints, curtain glass with the floors behind
// it, ceilings with their downlights, roof membrane. Every map is in metres: uv = metres / `userData.tile`.

const canvas = (w: number, h: number) => {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
};

function texture(c: HTMLCanvasElement, colour = true) {
  const t = new T.CanvasTexture(c);
  if (colour) t.colorSpace = T.SRGBColorSpace;
  t.wrapS = t.wrapT = T.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

/** Copies of a scan's maps for uv in metres / `tile` (the images are shared; only the repeat differs). */
function scan(set: PhotoSet, tile: number) {
  const copy = (t: T.Texture) => {
    const c = t.clone();
    c.repeat.setScalar(tile / set.size);
    c.needsUpdate = true;
    return c;
  };
  return {map: copy(set.map), roughnessMap: copy(set.roughnessMap), normalMap: copy(set.normalMap)};
}

/**
 * Façade cladding: precast or aluminium panels `panel` metres wide and `course` high, their joints a shade darker, each
 * panel a little lighter or darker than the next, with faint rain streaks below the joints. `tile` metres square.
 */
function claddingMaps(base: string, panel: number, course: number, tile: number, seed: number) {
  const size = 512, px = size / tile;
  const c = canvas(size, size), g = c.getContext('2d')!;
  const r = canvas(size, size), gr = r.getContext('2d')!;
  const random = seeded(seed);
  const colour = new T.Color(base);
  gr.fillStyle = '#9a9a9a';
  gr.fillRect(0, 0, size, size);
  for (let y = 0; y < tile; y += course)
    for (let x = 0; x < tile; x += panel) {
      const tone = colour.clone().offsetHSL(0, 0, (random() - 0.5) * 0.035);
      g.fillStyle = `#${tone.getHexString()}`;
      g.fillRect(x * px, y * px, panel * px, course * px);
      gr.fillStyle = `rgb(${130 + random() * 50 | 0},0,0)`;
      gr.fillRect(x * px, y * px, panel * px, course * px);
    }
  // Streaks: grime washed down from the joints.
  for (let i = 0; i < 90; i++) {
    const x = random() * size, y = Math.floor(random() * tile / course) * course * px;
    const grad = g.createLinearGradient(0, y, 0, y + course * px * (0.4 + random()));
    grad.addColorStop(0, 'rgba(60,55,50,0.10)');
    grad.addColorStop(1, 'rgba(60,55,50,0)');
    g.fillStyle = grad;
    g.fillRect(x, y, 2 + random() * 5, course * px * 1.4);
  }
  g.strokeStyle = `#${colour.clone().multiplyScalar(0.62).getHexString()}`;
  g.lineWidth = 2;
  gr.strokeStyle = '#ffffff';
  gr.lineWidth = 2;
  for (const [ctx] of [[g], [gr]] as const) {
    ctx.beginPath();
    for (let x = 0; x <= tile + 1e-6; x += panel) { ctx.moveTo(x * px, 0); ctx.lineTo(x * px, size); }
    for (let y = 0; y <= tile + 1e-6; y += course) { ctx.moveTo(0, y * px); ctx.lineTo(size, y * px); }
    ctx.stroke();
  }
  // Roughness lives in green (three reads G).
  const data = gr.getImageData(0, 0, size, size);
  for (let i = 0; i < data.data.length; i += 4) { data.data[i + 1] = data.data[i]; data.data[i] = data.data[i + 2] = 255; }
  gr.putImageData(data, 0, 0);
  return {map: texture(c), roughnessMap: texture(r, false)};
}

/**
 * What curtain glass shows from outside: four bays by two floors of the hall behind it — the ceiling and its lights, the
 * dim room, people and columns against the far wall — with the mullions and the floor edge across. Its emissive twin
 * holds only the lit ceiling lamps.
 */
function interiorMaps(seed: number, room: string) {
  const W = 512, H = 512, bays = 4, floors = 2;
  const c = canvas(W, H), g = c.getContext('2d')!;
  const e = canvas(W, H), ge = e.getContext('2d')!;
  ge.fillStyle = '#000';
  ge.fillRect(0, 0, W, H);
  const random = seeded(seed);
  const bw = W / bays, fh = H / floors;
  const dark = new T.Color(room);
  for (let f = 0; f < floors; f++) {
    const top = f * fh;
    const grad = g.createLinearGradient(0, top, 0, top + fh);
    grad.addColorStop(0, `#${dark.clone().offsetHSL(0, 0, 0.16).getHexString()}`);
    grad.addColorStop(0.2, `#${dark.clone().offsetHSL(0, 0, 0.06).getHexString()}`);
    grad.addColorStop(1, `#${dark.clone().offsetHSL(0, 0, -0.03).getHexString()}`);
    g.fillStyle = grad;
    g.fillRect(0, top, W, fh);
    // Ceiling: a pale band with rows of lamps.
    g.fillStyle = 'rgba(205,205,198,0.55)';
    g.fillRect(0, top, W, fh * 0.13);
    for (let x = 6; x < W; x += 21) {
      const lit = random() < 0.85;
      g.fillStyle = lit ? '#fffbe9' : '#b8b8b0';
      g.fillRect(x, top + fh * 0.06, 9, 3);
      if (lit) { ge.fillStyle = '#fff3d8'; ge.fillRect(x - 1, top + fh * 0.06 - 1, 11, 5); }
    }
    // Columns and people in the room.
    for (let x = bw * 0.5; x < W; x += bw * (1.5 + random())) {
      g.fillStyle = `rgba(30,32,34,${0.4 + random() * 0.3})`;
      g.fillRect(x, top + fh * 0.13, 10, fh * 0.87);
    }
    for (let i = 0; i < 26; i++) {
      const x = random() * W, h = fh * (0.2 + random() * 0.08), y = top + fh - 8 - h;
      g.fillStyle = `rgba(${20 + random() * 40 | 0},${20 + random() * 35 | 0},${25 + random() * 35 | 0},0.75)`;
      g.fillRect(x, y + h * 0.22, 7, h * 0.78);
      g.beginPath();
      g.arc(x + 3.5, y + h * 0.12, 4, 0, Math.PI * 2);
      g.fill();
    }
    // Slab edge at the floor.
    g.fillStyle = '#30353a';
    g.fillRect(0, top + fh - 9, W, 9);
  }
  // Mullions (aluminium) between the bays, transoms at the slabs.
  g.fillStyle = '#8d9296';
  for (let b = 0; b <= bays; b++) g.fillRect(b * bw - 3, 0, 6, H);
  for (let f = 0; f <= floors; f++) g.fillRect(0, f * fh - 5, W, 10);
  return {map: texture(c), emissiveMap: texture(e)};
}

/** A ceiling of square panels `tile` metres a side with one recessed downlight each (lit in the emissive map). */
function ceilingMaps(base: string, lamps: number, tile: number) {
  const size = 256, px = size / tile;
  const c = canvas(size, size), g = c.getContext('2d')!;
  const e = canvas(size, size), ge = e.getContext('2d')!;
  g.fillStyle = base;
  g.fillRect(0, 0, size, size);
  ge.fillStyle = '#000';
  ge.fillRect(0, 0, size, size);
  g.strokeStyle = 'rgba(0,0,0,0.25)';
  g.lineWidth = 2;
  const step = tile / lamps;
  for (let i = 0; i <= lamps; i++) {
    g.beginPath(); g.moveTo(i * step * px, 0); g.lineTo(i * step * px, size); g.stroke();
    g.beginPath(); g.moveTo(0, i * step * px); g.lineTo(size, i * step * px); g.stroke();
  }
  for (let i = 0; i < lamps; i++)
    for (let j = 0; j < lamps; j++) {
      const x = (i + 0.5) * step * px, y = (j + 0.5) * step * px;
      g.fillStyle = '#fffbea';
      ge.fillStyle = '#fff4dc';
      for (const ctx of [g, ge]) { ctx.beginPath(); ctx.arc(x, y, 3.2, 0, Math.PI * 2); ctx.fill(); }
    }
  return {map: texture(c), emissiveMap: texture(e)};
}

/** White PTFE membrane panels between roof ribs: bright where the sun comes through, seams every `seam` metres. */
function membraneMap(seam: number, tile: number) {
  const size = 256, px = size / tile;
  const c = canvas(size, size), g = c.getContext('2d')!;
  const grad = g.createLinearGradient(0, 0, size, 0);
  grad.addColorStop(0, '#d9d6cf');
  grad.addColorStop(0.5, '#f7f5f0');
  grad.addColorStop(1, '#d9d6cf');
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  g.strokeStyle = 'rgba(120,115,105,0.35)';
  g.lineWidth = 1.5;
  for (let y = 0; y <= tile; y += seam) { g.beginPath(); g.moveTo(0, y * px); g.lineTo(size, y * px); g.stroke(); }
  return texture(c);
}

export type StandLook = {
  /** Concrete of the terraces and structure, the cladding and its panel size, the trim (fascia edges, ledges). */
  concrete: string; cladding: string; panel: [number, number]; metallic: number;
  trim: string;
  /** The room seen through the glass, and the glass's own tint. */
  room: string; glass: string;
  /** Deck and roof soffits: colour, downlights per 6 m. */
  ceiling: string; lamps: number;
  /** The roof's underside: panels, timber (Kyoto), or membrane between ribs (Chukyo); the ribs' paint. */
  under: 'panel' | 'timber' | 'membrane'; underColour: string; ribs: string;
  /** Paving of the apron. */
  paving: string;
};

export type StandMaterials = Record<'concrete' | 'cladding' | 'trim' | 'glass' | 'coveredGlass' | 'mullion' | 'ceiling' | 'roofTop' | 'under' |
  'ribs' | 'membrane' | 'paving' | 'steel' | 'dark', T.MeshStandardMaterial>;

/**
 * The stand's materials for a venue's look. Vertex colours carry the shade of each piece (1 in the open, darker under a
 * deck or the roof: the sun's shadow map only reaches round the field, so the deep overhangs keep their shade baked).
 */
export function standFinish(look: StandLook, snow: number): StandMaterials {
  const photos = photoTextures();
  const shaded = <M extends T.MeshStandardMaterial>(m: M, tile: number) => {
    m.vertexColors = true;
    m.userData.tile = tile;
    return m;
  };
  const concrete = photos
    ? new T.MeshStandardMaterial({color: look.concrete, ...scan(photos.concrete, 9), roughness: 1,
      normalScale: new T.Vector2(0.7, 0.7)})
    : concreteMaterial(apronTexture(), look.concrete);
  // The concrete scan is dark: lift it toward the look's colour.
  if (photos) concrete.color.multiplyScalar(1.5);
  const cladTile = look.panel[0] * 4;
  const clad = claddingMaps(look.cladding, look.panel[0], look.panel[1], cladTile, 77);
  const cladding = new T.MeshStandardMaterial({color: '#ffffff', ...clad, roughness: 0.75 - 0.3 * look.metallic,
    metalness: 0.55 * look.metallic, envMapIntensity: 1});
  const trim = new T.MeshStandardMaterial({color: look.trim, roughness: 0.5, metalness: 0.3 * look.metallic});
  const room = interiorMaps(131, look.room);
  const glass = new T.MeshPhysicalMaterial({color: look.glass, map: room.map, emissive: '#ffffff', emissiveMap: room.emissiveMap,
    emissiveIntensity: 0.35, roughness: 0.08, metalness: 0, ior: 1.65, envMapIntensity: 0.8});
  // Glass under a deck or the roof mirrors the soffit over it, not the open sky the environment map holds.
  const coveredGlass = glass.clone();
  coveredGlass.envMapIntensity = 0.22;
  const mullion = new T.MeshStandardMaterial({color: '#9ea3a6', roughness: 0.35, metalness: 0.7});
  const lights = ceilingMaps(look.ceiling, look.lamps, 6);
  const ceiling = new T.MeshStandardMaterial({color: '#ffffff', map: lights.map, emissive: '#ffffff',
    emissiveMap: lights.emissiveMap, emissiveIntensity: 0.9, roughness: 0.85});
  const roofTop = photos
    ? new T.MeshStandardMaterial({color: new T.Color(0.95, 1.0, 1.04), ...scan(photos.roof, 3), roughness: 1, metalness: 0.2})
    : new T.MeshStandardMaterial({color: '#b9bdbf', roughness: 0.6, metalness: 0.2});
  roofTop.color.lerp(new T.Color(2.2, 2.2, 2.3), snow * 0.8);
  const under = look.under === 'timber' && photos
    ? new T.MeshStandardMaterial({color: look.underColour, ...scan(photos.cedar, 4), roughness: 1})
    : new T.MeshStandardMaterial({color: look.underColour, ...claddingMaps(look.underColour, 1.5, 6, 6, 91), roughness: 0.8});
  const ribs = new T.MeshStandardMaterial({color: look.ribs, roughness: 0.55, metalness: 0.4});
  const membrane = new T.MeshStandardMaterial({color: '#e9e6df', map: membraneMap(1.5, 6), emissive: '#fff8ec',
    emissiveIntensity: look.under === 'membrane' ? 0.22 : 0.08, roughness: 0.7, side: T.DoubleSide});
  const paving = photos
    ? new T.MeshStandardMaterial({color: look.paving, ...scan(photos.paving, 3.6), roughness: 1, normalScale: new T.Vector2(0.8, 0.8)})
    : new T.MeshStandardMaterial({color: look.paving, map: apronTexture(), roughness: 0.9});
  // The paver scan is warm and dark: grey it toward the look's colour and lift it.
  if (photos) paving.color.multiplyScalar(1.5).multiply(new T.Color(0.92, 0.98, 1.08));
  paving.userData.wetGloss = 0.12;
  paving.polygonOffset = true;
  paving.polygonOffsetFactor = -2;
  paving.polygonOffsetUnits = -2;
  const steel = new T.MeshStandardMaterial({color: '#c9ccce', roughness: 0.4, metalness: 0.6});
  const dark = new T.MeshStandardMaterial({color: '#2a2d30', roughness: 0.7});
  return {
    concrete: shaded(concrete, 9), cladding: shaded(cladding, cladTile), trim: shaded(trim, CONCRETE_TILE),
    glass: shaded(glass, 1), coveredGlass: shaded(coveredGlass, 1), mullion: shaded(mullion, 4), ceiling: shaded(ceiling, 6), roofTop: shaded(roofTop, 3),
    under: shaded(under, look.under === 'timber' ? 4 : 6), ribs: shaded(ribs, 4), membrane: shaded(membrane, 6),
    paving: shaded(paving, 3.6), steel: shaded(steel, 4), dark: shaded(dark, 4),
  };
}
