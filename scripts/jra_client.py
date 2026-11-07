"""Minimal polite client for public JRA pages: generic UA, >=2 s between requests,
raw responses cached under outputs/jra_cache (git-ignored) so reruns do not refetch."""
import hashlib, time, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'outputs' / 'jra_cache'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
MIN_INTERVAL = 2.5
_last = [0.0]
stats = {'requests': 0, 'cached': 0}


def _decode(raw):
    return raw.decode('cp932', 'replace') if b'shift_jis' in raw[:3000].lower() else raw.decode('utf-8', 'replace')


def fetch(url, data=None, max_age=None, label=None):
    """Return decoded HTML. data: dict for POST. max_age: seconds; None = cache forever."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((url + '|' + urllib.parse.urlencode(sorted((data or {}).items()))).encode()).hexdigest()[:16]
    path = CACHE / f'{label or "page"}_{key}.html'
    if path.exists() and (max_age is None or time.time() - path.stat().st_mtime < max_age):
        stats['cached'] += 1
        return _decode(path.read_bytes())
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={'User-Agent': UA, 'Accept-Language': 'ja'})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    _last[0] = time.time()
    stats['requests'] += 1
    path.write_bytes(raw)
    return _decode(raw)
