// Checks the race-flow (展開予想) import: schema, field matching, self-consistency of the
// bundled model output, and that playback reproduces the imported ranks / gaps / lanes / closing.
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

const out = path.resolve('outputs/pace-check');
await fs.mkdir(out, {recursive: true});
for (const name of ['courseData', 'alignmentData', 'alignment', 'courses', 'raceLayout', 'playback', 'paceForecast', 'closingKick']) {
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

const heldCost = [];
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
  // 予想レンジ (revision 1.1, optional): whole positions within the field and closing seconds, ordered, and each
  // range contains its published point. Ranges are only published for targets whose validation coverage passed.
  const shownTargets = new Set(
    Object.entries(file.model.intervals?.targets ?? {}).filter(([, v]) => v.shown).map(([k]) => k),
  );
  for (const h of pace.horses) {
    for (const p of h.positions) {
      if (!p.rankRange) continue;
      const [lo, hi] = p.rankRange;
      assert.ok(Number.isInteger(lo) && Number.isInteger(hi) && lo >= 1 && hi <= n && lo <= hi, `${race.name}: rank range ${lo}-${hi} in 1..${n}`);
      assert.ok(lo <= p.rank && p.rank <= hi, `${race.name}: rank ${p.rank} inside its range ${lo}-${hi}`);
      assert.ok(shownTargets.has(`r${p.corner}`), `${race.name}: range only for a corner whose coverage passed`);
    }
    if (h.last3Range) {
      const [lo, hi] = h.last3Range;
      assert.ok(lo <= h.last3 + 1e-9 && h.last3 <= hi + 1e-9, `${race.name}: last3 ${h.last3} inside ${lo}-${hi}`);
      assert.ok(shownTargets.has('last3'), `${race.name}: closing range only if its coverage passed`);
    }
  }
  for (const t of shownTargets) {
    const c = t === 'last3' ? null : Number(t.slice(1));
    if (c !== null && !pace.corners.includes(c)) continue;
    const have = pace.horses.every((h) => (c === null ? h.last3Range : h.positions.find((p) => p.corner === c)?.rankRange));
    assert.ok(have, `${race.name}: every horse has the published ${t} range`);
  }
  // ポジション争い: horses whose spot was taken end up behind where they wanted to be.
  const held = pace.horses.flatMap((h) => h.positions.filter((p) => p.held));
  if (held.length) {
    assert.ok(held.every((p) => p.wantRank !== undefined), 'Held positions carry the wanted rank');
    // A held horse can still land on its wanted rank in one race (others behind tucked in too), so the
    // cost is checked over all races: never ahead of the wish on average, and behind it overall.
    assert.ok(held.reduce((s, p) => s + p.rank - p.wantRank, 0) / held.length >= 0, `${race.name}: holding back never gains position`);
    heldCost.push(...held.map((p) => p.rank - p.wantRank));
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
  // Delay controls distance gained, not ranks gained: a tight pack can yield several passes
  // from a small gain. Check distance on an isolated fixture below, without collision padding.
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
const delayRace = {...race, horses: race.horses.slice(0, 2)};
const delayPace = structuredClone(paceForRace(file, race).pace);
delayPace.horses = delayPace.horses.slice(0, 2);
delayPace.horses.forEach((h, i) => {
  h.positions = h.positions.map((p) => ({...p, rank: i + 1, gap: i * 10, lane: i * 2}));
  h.finish = {rank: i + 1, gap: i * 2};
  h.kickDelay = 0;
});
// The trailing horse gains eight lengths in total. A 0.15 s delay permits only two
// lengths by mid-straight; 0.3 s permits none. The leader stays fixed as a reference.
for (const [delay, expectedGap] of [[0.15, 8], [0.3, 10]]) {
  delayPace.horses[1].kickDelay = delay;
  const plan = buildPlayback(delayRace, 'forecast', undefined, delayPace);
  const midIndex = plan.checkpoints.findIndex((c) => c.label === '直線');
  assert.ok(midIndex >= 0);
  const mid = plan.layouts[midIndex];
  assert.ok(Math.abs((mid[1].gap - mid[0].gap) / (HORSE_LENGTH * 1.3) - expectedGap) < 1e-9,
    `${delay}s delay: distance gained is delayed`);
  const finish = plan.layouts.at(-1);
  assert.ok(Math.abs((finish[1].gap - finish[0].gap) / (HORSE_LENGTH * 1.3) - 2) < 1e-9,
    'Delay preserves the final estimated gap');
}
// A horse gaining position takes an outside path even without a kick delay.
delayPace.horses.forEach((h, i) => {
  h.positions.forEach((p) => { p.lane = 0; });
  h.kickDelay = 0;
  h.finish = {rank: 2 - i, gap: (1 - i) * 2};
});
const outsidePlan = buildPlayback(delayRace, 'forecast', undefined, delayPace);
assert.ok(outsidePlan.layouts.at(-1)[1].lane > outsidePlan.layouts.at(-1)[0].lane,
  'Straight-line overtaker aims outside the horse it passes');
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
// Revision 1.1 is backward compatible: the same file without revision and ranges (a 1.0 file) parses and plays
// exactly the same, so the ranges are display-only.
const plain = structuredClone(raw);
delete plain.revision;
for (const r of Object.values(plain.races))
  for (const h of Object.values(r.horses)) {
    delete h.last3Range;
    h.positions.forEach((p) => delete p.rankRange);
  }
const plainParsed = parsePaceForecast(plain);
assert.ok(plainParsed.ok, 'A 1.0 file (no ranges) still parses');
assert.deepEqual(buildPlayback(race, 'forecast', undefined, paceForRace(plainParsed.data, race).pace),
  buildPlayback(race, 'forecast', undefined, paceForRace(file, race).pace), 'Ranges do not change playback');
const badRange = structuredClone(raw);
Object.values(badRange.races)[0].horses[String(race.horses[0].number)].positions[0].rankRange = [3];
assert.equal(parsePaceForecast(badRange).ok, false, 'A malformed range is rejected');
// Holding back never gains position overall; that it costs position is a property of the formation, checked
// deterministically in check_contention.py. Here it is required to be strictly positive only with enough held
// horses: v1.7 (2026-11-18) publishes 17 races with just 2 held horses, both passed by nobody (cost 0).
const meanHeldCost = heldCost.length ? heldCost.reduce((a, b) => a + b, 0) / heldCost.length : 0;
assert.ok(meanHeldCost >= 0, 'Holding back never gains position');
if (heldCost.length >= 10) assert.ok(meanHeldCost > 0, 'Holding back costs position');
console.log('PASS: schema, field matching, rank/gap/lane/closing parity, no result leakage, clearance, rejections');
