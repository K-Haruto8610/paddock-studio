import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const out = 'outputs/visual-baseline';
const port = process.env.PADDOCK_CHROME_PORT ?? '9234';
const url = process.env.PADDOCK_VISUAL_URL ?? 'http://127.0.0.1:5173/visual-check';
const tabs = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json();
const ws = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject});
let next = 0;
const pending = new Map();
ws.onmessage = event => {
  const {id, result, error} = JSON.parse(event.data);
  if (!pending.has(id)) return;
  const {resolve, reject} = pending.get(id);
  pending.delete(id);
  if (error) reject(Error(error.message));
  else resolve(result);
};
ws.onclose = () => {
  for (const {reject} of pending.values()) reject(Error('Chrome closed the DevTools connection'));
  pending.clear();
};
const send = (method, params = {}) => new Promise((resolve, reject) => {
  pending.set(++next, {resolve, reject});
  ws.send(JSON.stringify({id: next, method, params}));
});
const run = async expression => {
  const result = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
  return result.result.value;
};
const queen = '2026111508050411';
const february = '2026022205010811';
const fixed = {position: [0, 1020, 720], target: [0, 0, 0]};
const shots = [
  [queen, 0, 2200, 'pack'],
  [queen, 3016, 2200, 'pack'],
  [queen, 2494, 2200, 'broadcast'],
  [queen, 3449, 2200, 'broadcast'],
  [queen, 3882, 2200, 'pack'],
  [february, 1147, 1600, 'pack'],
  [queen, 1941, 2200, 'pack', fixed],
];

try {
  await fs.mkdir(out, {recursive: true});
  await send('Page.navigate', {url});
  for (let i = 0; i < 150 && !await run("typeof captureRaceFrame === 'function'"); i++) {
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  if (process.argv.includes('--video')) {
    const first = await run('checkRaceVideo()');
    const second = await run('checkRaceVideo()');
    assert.deepEqual(first.hashes, second.hashes);
    assert.equal(first.hashes[0], first.directHash);
    assert.equal(first.width, 1920);
    assert.equal(first.height, 1290);
    assert.equal(first.stats.packetCount, first.count);
    assert.ok(Math.abs(first.duration - first.count / 60) < 1e-5);
    console.log(`PASS: ${first.count} repeatable frames, 1920×1290, 60fps, ${first.duration.toFixed(4)}s`);
  } else {
    for (const [index, [raceId, tick, distance, cameraMode, fixedCamera]] of shots.entries()) {
      const progressAt = frame => frame * 34 / (60 * distance);
      const paths = [
        [],
        Array.from({length: tick}, (_, i) => progressAt(i)),
        Array.from({length: Math.ceil(tick / 4)}, (_, i) => progressAt(i * 4)),
        [progressAt(tick + 12)],
      ];
      const hashes = [];
      if (index === 0 || raceId !== shots[index - 1][0]) {
        const warm = {raceId, progress: progressAt(tick), speed: 2, selected: null, cameraMode,
          width: 1920, height: 1020, reduced: false};
        await run(`captureRaceFrame(${JSON.stringify(warm)}).then(() => true)`);
      }
      for (const [path, history] of paths.entries()) {
        const request = {raceId, progress: progressAt(tick), speed: 2, selected: null, cameraMode,
          width: 1920, height: 1020, reduced: false, fixed: fixedCamera, history};
        const result = await run(`(async () => {
          const request = ${JSON.stringify(request)};
          const start = performance.now();
          const frame = window.checkedFrame = await captureRaceFrame(request);
          const ms = performance.now() - start;
          const pixels = frame.getContext('2d').getImageData(0, 0, frame.width, frame.height).data;
          const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', pixels));
          return {hash: Array.from(bytes, b => b.toString(16).padStart(2, '0')).join(''), ms};
        })()`);
        hashes.push(result.hash);
        if (path === 0) {
          const length = await run("window.checkedPng = checkedFrame.toDataURL().split(',')[1]; checkedPng.length");
          const chunks = [];
          for (let i = 0; i < length; i += 500000) chunks.push(await run(`checkedPng.slice(${i}, ${i + 500000})`));
          const prefix = process.env.PADDOCK_SHOT_PREFIX ?? 'frame';
          await fs.writeFile(out + '/' + prefix + '-' + (index + 1) + '.png', Buffer.from(chunks.join(''), 'base64'));
          await run('delete window.checkedPng');
        }
        console.log(index + 1, path, result.hash, Math.round(result.ms) + 'ms');
      }
      assert.deepEqual(hashes, Array(4).fill(hashes[0]), 'shot ' + (index + 1));
    }
  }
} finally {
  ws.close();
}
