// Captures the Kyoto trial images through the visual-check page (dev server + headless Chrome on 9234).
import fs from 'node:fs/promises';
const out = 'outputs/tokyo/img';
const port = process.env.PADDOCK_CHROME_PORT ?? '9234';
const url = 'http://127.0.0.1:5173/visual-check';

const tabs = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json();
const ws = new WebSocket(tabs.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let next = 0; const pending = new Map();
ws.onmessage = (e) => { const {id, result, error} = JSON.parse(e.data); const p = pending.get(id); if (!p) return; pending.delete(id); error ? p.reject(Error(error.message)) : p.resolve(result); };
const send = (method, params = {}) => new Promise((resolve, reject) => { pending.set(++next, {resolve, reject}); ws.send(JSON.stringify({id: next, method, params})); });
const run = async (expression) => { const r = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true}); if (r.exceptionDetails) throw Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text); return r.result.value; };
const shots = JSON.parse(await fs.readFile('outputs/tokyo/shots.json', 'utf8'));
await fs.mkdir(out, {recursive: true});
const only = process.argv.slice(2);
try {
  await send('Page.navigate', {url});
  for (let i = 0; i < 300 && !(await run("typeof captureRaceFrame === 'function'")); i++) await new Promise((r) => setTimeout(r, 200));
  for (const {name, race, fixed, scale} of shots) {
    if (only.length && !only.includes(name)) continue;
    const req = {raceId: race, progress: 0, speed: 2, selected: null, cameraMode: 'pack', width: 1600, height: 900, reduced: false, fixed, elevationScale: scale};
    await run(`(async () => { window.shot = await captureRaceFrame(${JSON.stringify(req)}); window.png = window.shot.toDataURL().split(',')[1]; return window.png.length; })()`);
    const length = await run('png.length'); const chunks = [];
    for (let i = 0; i < length; i += 500000) chunks.push(await run(`png.slice(${i}, ${i + 500000})`));
    await fs.writeFile(`${out}/${name}.png`, Buffer.from(chunks.join(''), 'base64'));
    console.log('wrote', name);
  }
} finally { ws.close(); }
