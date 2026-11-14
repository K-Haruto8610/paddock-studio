import * as T from 'three';

// Where the drawn course surface is, asked by position. Blades of grass, hoofprints and flying sand
// all have to rest on the ribbons exactly as they are drawn: the ribbons are straight between their
// own samples, so asking the route instead (which is curved and sampled elsewhere) leaves things
// floating or buried by a few centimetres, more where the route blends between two courses.

/** What lies under a point: the surface height there, the unit normal of the face (pointing up), and whether it is dirt. */
export type GroundHit = {y: number; normal: T.Vector3; dirt: boolean};

/**
 * Finds the surface under (x, z) and writes it to `out`; false when nothing is drawn there.
 * Courses cross one another at different heights (Kyoto's hill), so `hint`, the height the caller
 * expects, picks the surface it is on; surfaces within `TIE` of each other in that, or all of them
 * without a hint, are told apart by the one drawn on top.
 */
export type Ground = (x: number, z: number, out: GroundHit, hint?: number) => boolean;

/** Metres per cell of the lookup grid; a triangle of a ribbon is a few metres across. */
const CELL = 8;
/** Metres within which two surfaces count as equally near the hint. */
const TIE = 0.05;
/** How far outside a face (as a fraction of its size) a point still counts as on it, so edges leave no cracks. */
const SLACK = 0.03;

/** The surfaces to look through: indexed geometry of triangles, and the depth priority it is drawn with. */
export type GroundSurface = {geometry: T.BufferGeometry; lift: number; dirt?: boolean};

/** A lookup over the triangles of `surfaces`, built once from the very geometry that is drawn. */
export function groundOf(surfaces: GroundSurface[]): Ground {
  const positions = surfaces.map((s) => s.geometry.attributes.position as T.BufferAttribute);
  const indices = surfaces.map((s) => s.geometry.index!.array);
  // Triangles by grid cell, each packed as surface * 2^24 + triangle.
  const cells = new Map<number, number[]>();
  const key = (i: number, j: number) => i * 65536 + j;
  surfaces.forEach((_, s) => {
    const pos = positions[s],
      index = indices[s];
    for (let t = 0; t < index.length / 3; t++) {
      let minX = Infinity,
        maxX = -Infinity,
        minZ = Infinity,
        maxZ = -Infinity;
      for (let k = 0; k < 3; k++) {
        const x = pos.getX(index[t * 3 + k]),
          z = pos.getZ(index[t * 3 + k]);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minZ = Math.min(minZ, z);
        maxZ = Math.max(maxZ, z);
      }
      for (let i = Math.floor(minX / CELL); i <= Math.floor(maxX / CELL); i++)
        for (let j = Math.floor(minZ / CELL); j <= Math.floor(maxZ / CELL); j++) {
          const list = cells.get(key(i, j));
          if (list) list.push(s * 2 ** 24 + t);
          else cells.set(key(i, j), [s * 2 ** 24 + t]);
        }
    }
  });

  return (x, z, out, hint = NaN) => {
    const list = cells.get(key(Math.floor(x / CELL), Math.floor(z / CELL)));
    if (!list) return false;
    let bestLift = -Infinity,
      bestGap = Infinity,
      found = false;
    for (const id of list) {
      const s = Math.floor(id / 2 ** 24),
        t = id - s * 2 ** 24,
        pos = positions[s],
        index = indices[s],
        a = index[t * 3],
        b = index[t * 3 + 1],
        c = index[t * 3 + 2];
      const ax = pos.getX(a),
        az = pos.getZ(a),
        bx = pos.getX(b) - ax,
        bz = pos.getZ(b) - az,
        cx = pos.getX(c) - ax,
        cz = pos.getZ(c) - az;
      const det = bx * cz - cx * bz;
      if (Math.abs(det) < 1e-12) continue;
      const px = x - ax,
        pz = z - az,
        v = (px * cz - cx * pz) / det,
        w = (bx * pz - px * bz) / det,
        u = 1 - v - w;
      if (u < -SLACK || v < -SLACK || w < -SLACK) continue;
      const ya = pos.getY(a),
        y = ya + v * (pos.getY(b) - ya) + w * (pos.getY(c) - ya),
        lift = surfaces[s].lift,
        gap = Number.isNaN(hint) ? 0 : Math.abs(y - hint);
      if (found && (gap > bestGap + TIE || (gap > bestGap - TIE && lift <= bestLift))) continue;
      bestLift = lift;
      bestGap = gap;
      found = true;
      out.y = y;
      out.dirt = !!surfaces[s].dirt;
      // Normal of the face from its edges (b - a) x (c - a) in x, y, z; the faces wind either way.
      const by = pos.getY(b) - ya,
        cy = pos.getY(c) - ya,
        nx = by * cz - cy * bz,
        ny = bz * cx - bx * cz,
        nz = bx * cy - cx * by,
        flip = ny < 0 ? -1 : 1,
        length = Math.hypot(nx, ny, nz) || 1;
      out.normal.set((nx * flip) / length, (ny * flip) / length, (nz * flip) / length);
    }
    return found;
  };
}
