// Checks that a pass never trades sides: while one horse goes by another and the two overlap along
// the course, the one that is inside stays inside. Also reports how often the plain per-frame
// separation swaps them, and the swaps the plans themselves ask of horses running together.
import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

const out = path.resolve('outputs/passing-check');
await fs.mkdir(out, {recursive: true});
for (const name of ['courseData', 'alignmentData', 'alignment', 'courses', 'raceLayout', 'playback', 'paceForecast', 'closingKick']) {
  const src = await fs.readFile(`app/${name}.ts`, 'utf8');
  const js = ts
    .transpileModule(src, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}})
    .outputText.replace(/from '(\.\/[^']+)'/g, "from '$1.js'");
  await fs.writeFile(path.join(out, name + '.js'), js);
}
const {buildPlayback} = await import(pathToFileURL(path.join(out, 'playback.js')));
const {sampleLayouts, interpolateLayouts, separateLayouts, isPass, HORSE_LENGTH, HORSE_WIDTH} = await import(
  pathToFileURL(path.join(out, 'raceLayout.js'))
);
const {parsePaceForecast, paceForRace} = await import(pathToFileURL(path.join(out, 'paceForecast.js')));

const data = JSON.parse(await fs.readFile('public/data/forecasts.json', 'utf8'));
const replays = JSON.parse(await fs.readFile('public/data/replays.json', 'utf8'));
const paceFile = parsePaceForecast(JSON.parse(await fs.readFile('public/data/pace_forecasts.json', 'utf8'))).data;

const STEP = 0.5;
/** Side swaps between overlapping horses (during a pass / otherwise), the largest sideways jump, overlaps. */
function scan(plan, distance, sample) {
  let passSwaps = 0,
    otherSwaps = 0,
    jump = 0,
    overlaps = 0,
    prev = null;
  for (let travel = 0; travel <= distance; travel += STEP) {
    const now = sample(plan, travel);
    let k = 0;
    while (k < plan.knots.length - 2 && travel > plan.knots[k + 1]) k++;
    for (let a = 0; a < now.length; a++) {
      if (prev) jump = Math.max(jump, Math.abs(now[a].lane - prev[a].lane));
      for (let b = a + 1; b < now.length; b++) {
        const dx = Math.abs(now[a].gap - now[b].gap),
          dy = now[b].lane - now[a].lane;
        if (dx < HORSE_LENGTH * 0.95 && Math.abs(dy) < HORSE_WIDTH * 0.6) overlaps++;
        const pass = isPass(
          plan.layouts[k][a].gap - plan.layouts[k][b].gap,
          plan.layouts[k + 1][a].gap - plan.layouts[k + 1][b].gap,
        );
        if (!prev || dx >= HORSE_LENGTH || Math.abs(prev[a].gap - prev[b].gap) >= HORSE_LENGTH) continue;
        if (Math.sign(dy) !== Math.sign(prev[b].lane - prev[a].lane)) {
          if (pass) passSwaps++;
          else otherSwaps++;
        }
      }
    }
    prev = now;
  }
  return {passSwaps, otherSwaps, jump, overlaps};
}
const before = (plan, travel) => separateLayouts(interpolateLayouts(plan.layouts, travel, plan.knots));
const after = (plan, travel) => sampleLayouts(plan.layouts, travel, plan.knots);

const sum = {pass: 0, other: 0, wasPass: 0, wasOther: 0};
let plans = 0,
  worst = 0,
  wasWorst = 0;
for (const race of data.races) {
  const variants = [['legacy', buildPlayback(race, 'forecast')]];
  if (replays.races[race.id]) variants.push(['replay', buildPlayback(race, 'replay', replays.races[race.id])]);
  const {pace} = paceForRace(paceFile, race);
  if (pace) variants.push(['pace', buildPlayback(race, 'forecast', undefined, pace)]);
  for (const [kind, plan] of variants) {
    const old = scan(plan, race.distance, before),
      now = scan(plan, race.distance, after);
    sum.wasPass += old.passSwaps;
    sum.wasOther += old.otherSwaps;
    sum.other += now.otherSwaps;
    plans++;
    worst = Math.max(worst, now.jump);
    wasWorst = Math.max(wasWorst, old.jump);
    assert.equal(now.passSwaps, 0, `${race.name} (${kind}): a pass never trades sides`);
    assert.equal(now.overlaps, 0, `${race.name} (${kind}): no overlapping silhouettes`);
    assert.deepEqual(after(plan, 1234), after(plan, 1234), 'Seeking is repeatable');
  }
}
assert.ok(worst < wasWorst, 'Sideways jumps are smaller than with per-frame separation alone');
console.log(
  `PASS: ${plans} plans. Side swaps during a pass: 0 (per-frame separation alone: ${sum.wasPass}). ` +
    `Plan-driven swaps between horses running together: ${sum.other} (was ${sum.wasOther}). ` +
    `Largest sideways step ${worst.toFixed(2)} m (was ${wasWorst.toFixed(2)} m) per ${STEP} m`,
);
