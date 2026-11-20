import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import * as T from 'three';
import {G1_RACES} from './g1_races.mjs';

const out = path.resolve('outputs/camera-check');
await fs.mkdir(out, {recursive: true});
for (const name of ['courseData', 'alignmentData', 'alignment', 'courses', 'trackGeometry', 'raceRoutes', 'raceLayout', 'raceCamera']) {
  const source = await fs.readFile(`app/${name}.ts`, 'utf8');
  const js = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}})
    .outputText.replace(/from '(\.\/[^']+)'/g, "from '$1.js'");
  await fs.writeFile(path.join(out, `${name}.js`), js);
}
const load = (name) => import(pathToFileURL(path.join(out, `${name}.js`)));
const {raceShot, fitRaceShot, createShotTransition} = await load('raceCamera');
const {makeTrack} = await load('trackGeometry');
const {makeRoute} = await load('raceRoutes');
const {courseFor} = await load('courses');
const {HORSE_SCALE: k} = await load('raceLayout');
const {races} = JSON.parse(await fs.readFile('public/data/forecasts.json', 'utf8'));
for (const [venueCode, surface, trackCode, distance] of G1_RACES) {
  const race = {...races.find((r) => r.venueCode === venueCode), venueCode, surface, trackCode, distance};
  const route = makeRoute(race, makeTrack(race));
  const {straight} = courseFor(race);
  assert.equal(raceShot(route, 0, distance, straight), 'start');
  assert.equal(raceShot(route, Math.min(100, distance * 0.08) - 1, distance, straight), 'start');
  assert.equal(raceShot(route, distance - straight + 1, distance, straight), 'finish');
  assert.equal(raceShot(route, distance, distance, straight), 'finish');
  const shots = new Set();
  for (let travel = 110; travel < distance - straight; travel += 10) shots.add(raceShot(route, travel, distance, straight));
  assert.ok(shots.has('corner'), `${venueCode}/${distance}: corner shot`);
  assert.ok(shots.has('wide'), `${venueCode}/${distance}: wide shot`);
}
// Project every horse's padded bounds (in horse sizes): all must fit even with a long field, hills and a portrait viewport.
for (const aspect of [0.4, 1, 16 / 9, 3]) for (const shot of ['start', 'wide', 'finish']) {
  const points = Array.from({length: shot === 'finish' ? 6 : 18}, (_, i) => new T.Vector3(-i * 17, Math.sin(i) * 20, (i % 4) * 4));
  const {target, offset} = fitRaceShot(points, new T.Vector3(0, 0, 1), shot, 40, aspect);
  const camera = new T.PerspectiveCamera(40, aspect, 0.3, 6000);
  camera.position.copy(target).add(offset);
  camera.lookAt(target);
  camera.updateMatrixWorld();
  for (const point of points) for (const x of [-4 * k, 4 * k]) for (const y of [0, 6 * k]) for (const z of [-4 * k, 4 * k]) {
    const projected = point.clone().add(new T.Vector3(x, y, z)).project(camera);
    const limit = shot === 'finish' ? 0.921 : 0.821;
    assert.ok(Math.abs(projected.x) <= limit && Math.abs(projected.y) <= limit && Math.abs(projected.z) < 1, `${shot}/${aspect}: horse is in frame`);
  }
}
// A shot change must not jump, must keep moving with the leader, and must finish at its live destination.
for (const fps of [30, 60, 120]) {
  const transition = createShotTransition();
  const leader = {point: new T.Vector3(), tangent: new T.Vector3(1, 0, 0), outward: new T.Vector3(0, 0, 1)};
  const oldTarget = new T.Vector3(0, 3, 0), oldCamera = new T.Vector3(0, 50, 50);
  const target = new T.Vector3(-20, 3, 0), position = new T.Vector3(-20, 9, 35);
  transition('start', 0, true, leader, oldTarget, oldCamera, oldTarget, oldCamera);
  let result = transition('finish', 0, false, leader, oldTarget, oldCamera, target, position);
  assert.ok(result.position.distanceTo(oldCamera) < 1e-8, 'no jump at shot boundary');
  let previous = result.position.clone();
  for (let i = 1; i <= Math.ceil(2.6 * fps); i++) {
    leader.point.x = i / fps * 20;
    const liveTarget = target.clone().add(leader.point), livePosition = position.clone().add(leader.point);
    result = transition('finish', 1 / fps, false, leader, result.target, result.position, liveTarget, livePosition);
    assert.ok(result.position.distanceTo(previous) < 75 / fps, 'bounded frame movement during transition');
    previous.copy(result.position);
  }
  assert.ok(result.position.distanceTo(position.clone().add(leader.point)) < 1e-7, 'transition reaches destination');
  result = transition('corner', 0, true, leader, result.target, result.position, oldTarget, oldCamera);
  assert.ok(result.position.distanceTo(oldCamera) < 1e-8, 'seeking immediately applies the requested shot');
}
console.log(`Camera checks passed: ${G1_RACES.length} G1 routes, 12 framing scenarios and transitions at 30/60/120fps.`);
