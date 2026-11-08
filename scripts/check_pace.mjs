// Checks the race-flow (展開予想) import: schema, field matching, self-consistency of the
// bundled model output, and that playback reproduces the imported ranks / gaps / lanes / closing.
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

const out = path.resolve('outputs/pace-check');
await fs.mkdir(out, {recursive: true});
for (const name of ['courseData', 'courses', 'raceLayout', 'playback', 'paceForecast']) {
  const src = await fs.readFile(`app/${name}.ts`, 'utf8');
  const js = ts
    .transpileModule(src, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}})
    .outputText.replace(/from '(\.\/[^']+)'/g, "from '$1.js'");
  await fs.writeFile(path.join(out, name + '.js'), js);
}
const {buildPlayback, playbackFrame} = await import(pathToFileURL(path.join(out, 'playback.js')));
const {HORSE_LENGTH, HORSE_WIDTH} = await import(pathToFileURL(path.join(out, 'raceLayout.js')));
const {parsePaceForecast, paceForRace} = await import(pathToFileURL(path.join(out, 'paceForecast.js')));

const data = JSON.parse(await fs.readFile('public/data/forecasts.json', 'utf8'));
const replays = JSON.parse(await fs.readFile('public/data/replays.json', 'utf8'));
const raw = JSON.parse(await fs.readFile('public/data/pace_forecasts.json', 'utf8'));
const parsed = parsePaceForecast(raw);
assert.ok(parsed.ok, 'Bundled file matches paddock.pace-forecast/v1: ' + (parsed.error ?? ''));
const file = parsed.data;

for (const race of data.races) {
  const {pace, reason} = paceForRace(file, race);
  if (race.historical === false && !pace) continue; // upcoming race not yet modelled
  assert.ok(pace, `${race.name}: usable pace forecast (${reason ?? 'missing'})`);
  const n = race.horses.length;
  pace.corners.forEach((corner, k) => {
    const ranks = pace.horses.map((h) => h.positions[k].rank);
    assert.deepEqual([...ranks].sort((a, b) => a - b), Array.from({length: n}, (_, i) => i + 1), 'Ranks are 1..n');
    const byRank = pace.horses.map((h) => h.positions[k]).sort((a, b) => a.rank - b.rank);
    assert.equal(byRank[0].gap, 0, 'Leader has no gap');
    assert.ok(byRank.every((p, i) => i === 0 || p.gap >= byRank[i - 1].gap), 'Gaps never shrink down the order');
  });

  const plan = buildPlayback(race, 'forecast', undefined, pace);
  assert.equal(plan.source, 'pace');
  assert.deepEqual(plan.checkpoints.filter((c) => c.corner).map((c) => c.corner), pace.corners, 'Imported corners only');
  assert.ok(plan.knots.every((v, i) => i === 0 || v > plan.knots[i - 1]), 'Monotonic checkpoints');
  assert.deepEqual(buildPlayback(race, 'forecast', replays.races[race.id], pace), plan, 'Never consumes results');
  for (const c of plan.checkpoints.slice(1)) {
    const frame = playbackFrame(race, plan, c.distance / race.distance);
    assert.equal(frame.exact, c, 'Checkpoint seek is exact');
    // 前後: 3D order is exactly the imported order.
    assert.deepEqual(frame.order.map((o) => c.ranks[o.index]), Array.from({length: n}, (_, i) => i + 1), `${c.label}: 3D order = imported rank`);
    if (c.corner) {
      const k = pace.corners.indexOf(c.corner);
      // 左右: the widest and the innermost predicted horse keep their sides on the track.
      const lanes = pace.horses.map((h) => h.positions[k].lane);
      const wide = lanes.indexOf(Math.max(...lanes)), inner = lanes.indexOf(Math.min(...lanes));
      if (lanes[wide] - lanes[inner] > 0.5)
        assert.ok(frame.placements[wide].lane > frame.placements[inner].lane, `${c.label}: wide horse drawn outside`);
      // 前後: imported lengths are kept, only stretched to clear the enlarged silhouettes.
      const byRank = pace.horses.map((h, i) => ({p: h.positions[k], g: frame.placements[i].gap})).sort((a, b) => a.p.rank - b.p.rank);
      assert.ok(byRank.every(({p, g}) => g + 1e-6 >= (p.gap - byRank[0].p.gap) * HORSE_LENGTH * 1.3 + byRank[0].g), `${c.label}: gap never shortened`);
    }
  }
  // ポジション争い: horses whose spot was taken end up behind where they wanted to be.
  const held = pace.horses.flatMap((h) => h.positions.filter((p) => p.held));
  if (held.length) {
    assert.ok(held.every((p) => p.wantRank !== undefined), 'Held positions carry the wanted rank');
    assert.ok(held.reduce((s, p) => s + p.rank - p.wantRank, 0) / held.length > 0, 'Holding back costs position');
  }
  // 仕掛け遅れ: never negative, equals traffic-included minus clear-run closing, drawn in the straight.
  for (const h of pace.horses)
    if (h.kickDelay !== undefined) {
      assert.ok(h.kickDelay >= 0);
      assert.ok(Math.abs(Math.max(0, h.last3 - h.last3Free) - h.kickDelay) <= 0.1 + 1e-9, 'Delay = closing with traffic − clear run');
    }
  const delayed = pace.horses.map((h, i) => ({h, i})).filter(({h}) => (h.kickDelay ?? 0) > 0.05);
  const straight = plan.checkpoints.find((c) => c.label === '直線');
  assert.equal(Boolean(straight), delayed.length > 0, 'Straight checkpoint exactly when a move is delayed');
  if (straight) {
    const last = plan.checkpoints.find((c) => c.corner === pace.corners.at(-1)),
      fin = plan.checkpoints.at(-1);
    for (const {i} of delayed) {
      // A delayed horse has made at most half of its straight-line ranks by mid-straight.
      const made = last.ranks[i] - straight.ranks[i], total = last.ranks[i] - fin.ranks[i];
      if (total > 1) assert.ok(made <= total * 0.5 + 1, `${race.horses[i].number}: move is delayed`);
    }
  }
  // 上がり: finish order follows the closing estimate, so the straight changes the order.
  const last = plan.checkpoints.at(-2), finish = plan.checkpoints.at(-1);
  assert.notDeepEqual(last.ranks, finish.ranks, 'Closing 3F moves horses in the straight');
  for (let s = 0; s <= 200; s++) {
    const frame = playbackFrame(race, plan, s / 200);
    assert.ok(frame.placements.every((p) => Number.isFinite(p.gap) && Number.isFinite(p.lane) && p.lane > 0 && p.lane < 35));
    for (let a = 0; a < n; a++)
      for (let b = a + 1; b < n; b++) {
        const x = frame.placements[a], y = frame.placements[b];
        assert.ok(Math.abs(x.gap - y.gap) >= HORSE_LENGTH * 0.95 || Math.abs(x.lane - y.lane) >= HORSE_WIDTH * 0.6, 'No overlapping silhouettes');
      }
  }
  console.log(`${race.name}: corners=${pace.corners.join(',')} PASS`);
}

// Rejections: a file from another model must name every starter, and match the schema.
const race = data.races.find((r) => file.races[r.id]);
const broken = structuredClone(file);
delete broken.races[race.id].horses[String(race.horses[0].number)];
assert.equal(paceForRace(broken, race).pace, null, 'Missing horse is rejected');
assert.match(paceForRace(broken, race).reason, /予想がありません/);
const extra = structuredClone(file);
extra.races[race.id].horses['99'] = extra.races[race.id].horses[String(race.horses[0].number)];
assert.equal(paceForRace(extra, race).pace, null, 'Non-starter is rejected');
assert.equal(parsePaceForecast({...raw, schema: 'other/v0'}).ok, false, 'Wrong schema is rejected');
const badLane = structuredClone(raw);
Object.values(badLane.races)[0].horses[String(race.horses[0].number)].positions[0].lane = -1;
assert.equal(parsePaceForecast(badLane).ok, false, 'Negative lane is rejected');
// Minimal external file: positions + last3 only (no finish) still plays, finish from closing 3F.
const minimal = {schema: 'paddock.pace-forecast/v1', generatedAt: 'x', model: {name: 'external'}, races: {[race.id]: {corners: [4], horses: Object.fromEntries(race.horses.map((h, i) => [String(h.number), {positions: [{corner: 4, rank: i + 1, gap: i * 0.8, lane: i % 3}], last3: 34 + (i % 4) * 0.3}]))}}};
const ext = parsePaceForecast(minimal);
assert.ok(ext.ok);
const extPlan = buildPlayback(race, 'forecast', undefined, paceForRace(ext.data, race).pace);
assert.equal(extPlan.checkpoints.length, 3, 'start, 4角, finish');
console.log('PASS: schema, field matching, rank/gap/lane/closing parity, no result leakage, clearance, rejections');
