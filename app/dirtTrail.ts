import * as T from 'three';
import type {Ground, GroundHit} from './courseGround';
import {HOOF_COUNT, gallopAmount, hoofCount, hoofSpot, isHindHoof, stridePhase} from './plushHorse';
import type {Kick} from './closingKick';
import {seeded} from './scenery';
import {HORSE_SCALE} from './raceLayout';
import {raceSeconds} from './raceClock';

// What the horses leave on a dirt course: hoofprints that stay behind the field, and sand thrown
// back from the hooves as they push off. Both come from the same footfalls the legs are posed with
// (plushHorse.ts), and both rest on the drawn surface (courseGround.ts), so they follow the slope.
//
// The scene rebuilds the footfalls from fixed ticks after a seek. Sand already in the air is
// evaluated against the race clock, so it freezes when playback stops.

/** Hoofprints kept; the oldest are overwritten (about 300m of a full field's trail). */
const PRINTS = 1536;
/** Metres behind the last horse at which a print has faded away. */
const KEEP = 240;
/** Grains of sand in flight or settling at most. */
const GRAINS = 4096;
const PRINT_SEED = 77;
const SAND_SEED = 4101;
const footfallSeed = (seed: number, horse: number, hoof: number, step: number) =>
  seed ^ Math.imul(horse + 1, 73856093) ^ Math.imul(hoof + 1, 19349663) ^ Math.imul(step, 83492791);
const GRAVITY = 9.8;
/** Seconds a grain rests on the ground before it is gone. */
const SETTLE = 0.4;
/** A horse that goes back further than this between frames is still just running (easing jitter); more is a seek. */
const BACK_OK = 1.5;
/** A horse that goes further than this between frames has been sought forward. */
const JUMP = 45;
/** Footfalls handled per leg and frame; a long jump skips the rest. */
const MAX_STEPS = 4;
/** Hoofprint size in metres, in proportion to the drawn horse. */
const PRINT_WIDTH = 0.42 * HORSE_SCALE;
const PRINT_LENGTH = 0.5 * HORSE_SCALE;
/** Sand flies as high and as far as the drawn horse's size: speeds go with its square root. */
const THROW = Math.sqrt(HORSE_SCALE);
/** Grains thrown by a push-off. */
const GRAINS_HIND = 28;
const GRAINS_FORE = 12;
/** A footfall presses a print; a push-off throws sand. */
const STAGES = ['down', 'push'] as const;
/** The dirt's base colour, as in dirtTexture(). */
const SAND = '#ceb28a';

export type DirtTrail = ReturnType<typeof addDirtTrail>;

/** The field as the trail needs it: a group per horse, and its distance along the route and closing-kick state. */
export type TrailField = {
  horses: readonly Pick<T.Group, 'position' | 'rotation'>[];
  at: ArrayLike<number>;
  kick: ArrayLike<Kick>;
};

/** Texture pair for one print: a dark, soft-edged hoof shape, and the dent as a normal map. */
function printTextures() {
  const size = 64;
  const colour = new Uint8Array(size * size * 4);
  const height = new Float32Array(size * size);
  const hoof = (u: number, v: number) => {
    // A rounded hoof, narrower at the heel (v < 0), with the frog's ridge down the middle.
    const r = Math.hypot(u / (0.82 + 0.1 * v), (v - 0.04) / 0.98);
    const dent = T.MathUtils.smoothstep(1 - r, 0, 0.45);
    const frog = Math.exp(-((u / 0.16) ** 2)) * T.MathUtils.smoothstep(v, -0.5, 0.1) * (1 - T.MathUtils.smoothstep(v, 0.3, 0.8));
    // The sand pushed out of the dent heaps in a low rim.
    const rim = Math.exp(-(((r - 1.12) / 0.14) ** 2));
    return {r, h: -dent + 0.35 * frog * dent + 0.4 * rim};
  };
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const {r, h} = hoof(((x + 0.5) / size) * 2 - 1, ((y + 0.5) / size) * 2 - 1);
      height[y * size + x] = h;
      const i = (y * size + x) * 4;
      // Darker where deeper: damp sand from under the surface.
      colour.set([88, 68, 48, Math.round(255 * 0.75 * T.MathUtils.smoothstep(1.06 - r, 0, 0.22))], i);
    }
  const normal = new Uint8Array(size * size * 4);
  const at = (x: number, y: number) => height[Math.min(size - 1, Math.max(0, y)) * size + Math.min(size - 1, Math.max(0, x))];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * 3,
        dy = (at(x, y + 1) - at(x, y - 1)) * 3,
        k = 1 / Math.hypot(dx, dy, 1),
        i = (y * size + x) * 4;
      normal.set([(-dx * k * 0.5 + 0.5) * 255, (-dy * k * 0.5 + 0.5) * 255, (k * 0.5 + 0.5) * 255, 255], i);
    }
  const make = (data: Uint8Array, space: T.ColorSpace) => {
    const tex = new T.DataTexture(data, size, size, T.RGBAFormat);
    tex.colorSpace = space;
    tex.minFilter = T.LinearMipmapLinearFilter;
    tex.magFilter = T.LinearFilter;
    tex.generateMipmaps = true;
    tex.anisotropy = 4;
    tex.needsUpdate = true;
    return tex;
  };
  return {map: make(colour, T.SRGBColorSpace), normalMap: make(normal, T.NoColorSpace)};
}

/** Hoofprints: one instanced mesh of flat quads laid on the surface, fading out with distance behind the field. */
function makePrints(scene: T.Scene) {
  const textures = printTextures();
  const fade = {uRear: {value: 0}};
  const material = new T.MeshStandardMaterial({
    map: textures.map,
    normalMap: textures.normalMap,
    roughness: 1,
    transparent: true,
    depthWrite: false,
    // In front of the dirt ribbon (lift 3), which has to lose every depth tie to the dent drawn on it.
    polygonOffset: true,
    polygonOffsetFactor: -5,
    polygonOffsetUnits: -5,
  });
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uRear = fade.uRear;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aAt;\nuniform float uRear;\nvarying float vFade;')
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        vFade = 1.0 - smoothstep(${(KEEP * 0.55).toFixed(1)}, ${KEEP.toFixed(1)}, uRear - aAt);`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vFade;')
      .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.a *= vFade;');
  };
  const geometry = new T.PlaneGeometry(1, 1);
  const born = new T.InstancedBufferAttribute(new Float32Array(PRINTS), 1);
  born.setUsage(T.DynamicDrawUsage);
  geometry.setAttribute('aAt', born);
  const mesh = new T.InstancedMesh(geometry, material, PRINTS);
  mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
  mesh.count = 0;
  mesh.receiveShadow = true;
  // The prints are scattered along the whole trail; the geometry's own bounds say nothing.
  mesh.frustumCulled = false;
  mesh.raycast = () => {};
  scene.add(mesh);

  const matrix = new T.Matrix4(),
    forward = new T.Vector3(),
    right = new T.Vector3(),
    place = new T.Vector3();
  let made = 0;
  return {
    mesh,
    uniforms: fade,
    clear() {
      made = 0;
      mesh.count = 0;
    },
    save: () => ({made, matrices: mesh.instanceMatrix.array.slice(), born: born.array.slice()}),
    restore(snapshot: {made: number; matrices: T.TypedArray; born: T.TypedArray}) {
      made = snapshot.made;
      mesh.count = Math.min(made, PRINTS);
      mesh.instanceMatrix.array.set(snapshot.matrices);
      born.array.set(snapshot.born);
      mesh.instanceMatrix.needsUpdate = born.needsUpdate = true;
    },
    /** Presses a print at `at` (route distance) on the surface point `p`, its toe towards `heading`. */
    press(p: T.Vector3, normal: T.Vector3, heading: T.Vector3, turn: number, at: number) {
      forward.copy(heading).addScaledVector(normal, -heading.dot(normal)).normalize();
      // Prints are never quite aligned with the course.
      forward.applyAxisAngle(normal, turn);
      right.crossVectors(forward, normal);
      matrix.makeBasis(right.multiplyScalar(PRINT_WIDTH), forward.multiplyScalar(PRINT_LENGTH), normal);
      // A hair above the surface; the polygon offset does the rest.
      matrix.setPosition(place.copy(p).addScaledVector(normal, 0.01));
      const i = made % PRINTS;
      mesh.setMatrixAt(i, matrix);
      born.setX(i, at);
      made++;
      mesh.count = Math.min(made, PRINTS);
      mesh.instanceMatrix.needsUpdate = true;
      born.needsUpdate = true;
    },
    dispose() {
      scene.remove(mesh);
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      textures.map.dispose();
      textures.normalMap.dispose();
    },
  };
}

/** Sand: instanced billboards whose flight is worked out in the vertex shader from where and when each grain was thrown. */
function makeSand(scene: T.Scene) {
  const clock = {uClock: {value: 0}};
  const material = new T.MeshStandardMaterial({color: SAND, roughness: 1, metalness: 0});
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uClock = clock.uClock;
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        attribute vec3 aFrom;
        attribute vec3 aVel;
        // Race clock at the throw, size in metres, shade.
        attribute vec3 aGrain;
        // x, z of the unit normal of the ground it was thrown from.
        attribute vec2 aUp;
        uniform float uClock;
        varying vec3 vSand;
        varying vec2 vCorner;`,
      )
      .replace('#include <beginnormal_vertex>', 'vec3 objectNormal = vec3(0.0, 1.0, 0.0);')
      .replace(
        '#include <begin_vertex>',
        `float age = uClock - aGrain.x;
        vec3 ground = vec3(aUp.x, sqrt(max(1.0 - dot(aUp, aUp), 0.0)), aUp.y);
        // It flies until it is back on the plane of the ground it left, then rests there and is gone.
        float flight = max(2.0 * dot(aVel, ground) / (${GRAVITY.toFixed(1)} * max(ground.y, 0.2)), 0.0);
        float life = flight + ${SETTLE.toFixed(2)};
        float t = clamp(age, 0.0, flight);
        vec3 transformed = aFrom + aVel * t + vec3(0.0, -0.5 * ${GRAVITY.toFixed(1)} * t * t, 0.0);
        float grain = aGrain.y * step(0.0, age) * step(age, life) * (1.0 - smoothstep(flight, life, age));
        vSand = mix(vec3(0.74, 0.7, 0.66), vec3(1.12, 1.07, 1.0), aGrain.z);
        vCorner = position.xy * 2.0;`,
      )
      // Each grain is a small disc turned to face the camera.
      .replace(
        '#include <project_vertex>',
        `vec4 mvPosition = viewMatrix * vec4(transformed, 1.0);
        mvPosition.xy += position.xy * grain;
        gl_Position = projectionMatrix * mvPosition;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vSand;\nvarying vec2 vCorner;')
      .replace('#include <color_fragment>', 'if (dot(vCorner, vCorner) > 1.0) discard;\ndiffuseColor.rgb *= vSand;');
  };
  const geometry = new T.InstancedBufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const attribute = (size: number) => {
    const a = new T.InstancedBufferAttribute(new Float32Array(GRAINS * size), size);
    a.setUsage(T.DynamicDrawUsage);
    return a;
  };
  const from = attribute(3),
    vel = attribute(3),
    grain = attribute(3),
    up = attribute(2);
  geometry.setAttribute('aFrom', from);
  geometry.setAttribute('aVel', vel);
  geometry.setAttribute('aGrain', grain);
  geometry.setAttribute('aUp', up);
  geometry.instanceCount = GRAINS;
  const mesh = new T.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.raycast = () => {};
  scene.add(mesh);

  const back = new T.Vector3(),
    side = new T.Vector3(),
    v = new T.Vector3();
  let thrown = 0;
  const empty = () => {
    // Born far in the future: not yet thrown, so not drawn.
    for (let i = 0; i < GRAINS; i++) grain.setX(i, 1e9);
    grain.needsUpdate = true;
  };
  empty();
  return {
    mesh,
    uniforms: clock,
    clear() {
      thrown = 0;
      empty();
    },
    save: () => ({thrown, data: [from, vel, grain, up].map(a => a.array.slice())}),
    restore(snapshot: {thrown: number; data: T.TypedArray[]}) {
      thrown = snapshot.thrown;
      [from, vel, grain, up].forEach((a, i) => {
        a.array.set(snapshot.data[i]);
        a.needsUpdate = true;
      });
    },
    /** Throws `count` grains from the surface point `p` (normal `normal`) back along `-heading`, at the given race clock. */
    throwSand(p: T.Vector3, normal: T.Vector3, heading: T.Vector3, count: number, hind: boolean, now: number, seed: number) {
      const rand = seeded(seed);
      back.copy(heading).addScaledVector(normal, -heading.dot(normal)).normalize().negate();
      side.crossVectors(back, normal);
      for (let k = 0; k < count; k++) {
        const i = thrown++ % GRAINS;
        // Metres per second relative to the ground: the hind hooves push the hardest.
        const away = (hind ? 2 + 3.5 * rand() : 1.2 + 2.3 * rand()) * (0.6 + 0.4 * rand()) * THROW,
          lift = (hind ? 2.2 + 3 * rand() : 1.6 + 1.8 * rand()) * (0.6 + 0.4 * rand()) * THROW,
          spread = (rand() - 0.5) * (hind ? 2.4 : 1.6) * THROW;
        v.copy(back).multiplyScalar(away).addScaledVector(normal, lift).addScaledVector(side, spread);
        // Start somewhere in the hoof's reach, a hand above the ground.
        const dx = (rand() - 0.5) * 0.3 * HORSE_SCALE,
          dz = (rand() - 0.5) * 0.3 * HORSE_SCALE;
        from.setXYZ(i, p.x + dx, p.y + (0.05 + 0.1 * rand()) * HORSE_SCALE, p.z + dz);
        vel.setXYZ(i, v.x, v.y, v.z);
        grain.setXYZ(i, now, (0.09 + 0.13 * rand()) * HORSE_SCALE, rand());
        up.setXY(i, normal.x, normal.z);
      }
      from.needsUpdate = vel.needsUpdate = grain.needsUpdate = up.needsUpdate = true;
    },
    dispose() {
      scene.remove(mesh);
      geometry.dispose();
      material.dispose();
    },
  };
}

/**
 * Hoofprints and kickback for a field running on dirt, drawn on `ground`. `update` is called every frame
 * with the horses already placed and posed; it looks at what each hoof did since the last frame.
 *
 * Timing follows the race: prints are made from the stride phase the legs are posed with, and the sand's
 * flight is read from `travel` (the race's progress in metres), never the wall clock.
 */
export function addDirtTrail(scene: T.Scene, ground: Ground, reducedMotion: () => boolean) {
  const prints = makePrints(scene);
  const sand = makeSand(scene);
  const hit: GroundHit = {y: 0, normal: new T.Vector3(0, 1, 0), dirt: false};
  const spot = {x: 0, z: 0},
    where = new T.Vector3(),
    heading = new T.Vector3();

  // What the last frame looked like, per horse (and per leg for the footfall counts).
  let epoch: unknown = null;
  let armed = false;
  let seen = 0;
  let lastClock = 0;
  let shown = true;
  let lastAt = new Float64Array(0),
    lastPhase = new Float64Array(0),
    lastX = new Float64Array(0),
    lastZ = new Float64Array(0);

  const clear = () => {
    prints.clear();
    sand.clear();
    armed = false;
  };

  return {
    /** Forgets every print and grain. */
    clear,
    save: () => ({
      prints: prints.save(), sand: sand.save(), lastClock, epoch, armed, seen,
      at: lastAt.slice(), phase: lastPhase.slice(), x: lastX.slice(), z: lastZ.slice(),
    }),
    restore(snapshot: {
      prints: ReturnType<typeof prints.save>; sand: ReturnType<typeof sand.save>;
      lastClock: number; epoch: unknown; armed: boolean; seen: number;
      at: Float64Array; phase: Float64Array; x: Float64Array; z: Float64Array;
    }) {
      prints.restore(snapshot.prints);
      sand.restore(snapshot.sand);
      ({lastClock, epoch, armed, seen} = snapshot);
      lastAt = snapshot.at.slice();
      lastPhase = snapshot.phase.slice();
      lastX = snapshot.x.slice();
      lastZ = snapshot.z.slice();
    },
    /** Shows or hides what the trail has made (it keeps its history while hidden). */
    show(on: boolean) {
      shown = on;
      prints.mesh.visible = on;
      sand.mesh.visible = on;
    },
    setTime(travel: number, at: ArrayLike<number>) {
      prints.mesh.visible = shown;
      sand.mesh.visible = shown && !reducedMotion();
      sand.uniforms.uClock.value = raceSeconds(travel);
      let rear = Infinity;
      for (let i = 0; i < at.length; i++) rear = Math.min(rear, at[i]);
      prints.uniforms.uRear.value = rear;
    },
    /**
     * `travel` is the race's progress in metres. `scrubbing` is true while the horses are being moved by hand
     * (a drag in edit mode): nothing is made then. `edition` identifies the layout being shown; when it changes
     * (a hand edit) the history no longer belongs to the race and is dropped.
     */
    update(field: TrailField, travel: number, scrubbing: boolean, edition: unknown) {
      const {horses, at, kick} = field;
      if (edition !== epoch) {
        epoch = edition;
        clear();
      }
      const reduced = reducedMotion();
      this.setTime(travel, at);

      if (scrubbing) {
        armed = false;
        return;
      }
      if (seen !== horses.length) {
        seen = horses.length;
        lastAt = new Float64Array(seen);
        lastPhase = new Float64Array(seen);
        lastX = new Float64Array(seen);
        lastZ = new Float64Array(seen);
        armed = false;
      }
      // A seek, a restart or a wrap-around: nothing already made belongs to the race now shown.
      if (armed)
        for (let i = 0; i < horses.length; i++) {
          const moved = at[i] - lastAt[i];
          if (moved < -BACK_OK || moved > JUMP) {
            clear();
            break;
          }
        }
      const now = sand.uniforms.uClock.value;
      for (let i = 0; i < horses.length; i++) {
        const g = horses[i],
          phase = stridePhase(at[i], reduced, kick[i]);
        heading.set(Math.cos(g.rotation.y), 0, -Math.sin(g.rotation.y));
        // A horse that has been put somewhere else (a drag ending, the preview's wrap) did not run there.
        const ran = Math.max(at[i] - lastAt[i], 0);
        const stepped = Math.hypot(g.position.x - lastX[i], g.position.z - lastZ[i]) <= 4 * ran + 3;
        if (armed && stepped && gallopAmount(at[i], reduced) > 0.5 && phase > lastPhase[i])
          for (let n = 0; n < HOOF_COUNT; n++) {
            // Each footfall, and each push-off, that fell between the last frame and this one.
            for (const stage of STAGES) {
              if (stage === 'push' && reduced) continue;
              const before = hoofCount(n, lastPhase[i], stage),
                after = hoofCount(n, phase, stage);
              for (let c = Math.floor(before) + 1, steps = 0; c <= after && steps < MAX_STEPS; c++, steps++) {
                const f = (c - before) / (after - before);
                hoofSpot(n, lastPhase[i] + (phase - lastPhase[i]) * f, kick[i].level, spot);
                // The hoof's spot is in the model's own units.
                spot.x *= HORSE_SCALE;
                spot.z *= HORSE_SCALE;
                const cos = Math.cos(g.rotation.y),
                  sin = Math.sin(g.rotation.y);
                where.set(
                  lastX[i] + (g.position.x - lastX[i]) * f + spot.x * cos + spot.z * sin,
                  g.position.y,
                  lastZ[i] + (g.position.z - lastZ[i]) * f - spot.x * sin + spot.z * cos,
                );
                // A dirt race starts on a turf pocket and crosses turf: only dirt takes marks.
                if (!ground(where.x, where.z, hit, g.position.y) || !hit.dirt) continue;
                where.y = hit.y;
                if (stage === 'down') {
                  const turn = seeded(footfallSeed(PRINT_SEED, i, n, c))();
                  prints.press(where, hit.normal, heading, (turn - 0.5) * 0.3, at[i]);
                }
                else
                  sand.throwSand(
                    where,
                    hit.normal,
                    heading,
                    Math.round((isHindHoof(n) ? GRAINS_HIND : GRAINS_FORE) * (0.8 + 0.4 * kick[i].level)),
                    isHindHoof(n),
                    lastClock + (now - lastClock) * f,
                    footfallSeed(SAND_SEED, i, n, c),
                  );
              }
            }
          }
        lastAt[i] = at[i];
        lastPhase[i] = phase;
        lastX[i] = g.position.x;
        lastZ[i] = g.position.z;
      }
      lastClock = now;
      armed = true;
    },
    dispose() {
      prints.dispose();
      sand.dispose();
    },
  };
}
