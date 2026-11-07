import json
from pathlib import Path
d=json.loads((Path(__file__).resolve().parents[1]/'public/data/forecasts.json').read_text(encoding='utf-8'))
assert len(d['races'])>0
assert not any(any(k in f.lower() for k in ['odds','ninki','chakujun']) for f in d['model']['features'])
for r in d['races']:
    assert r['date']>'20251231'
    if r['historical'] is False:
        assert r['entries']['source'] and r['entries']['fetchedAt'] and r['entries']['status']=='declared'
        assert not any(s['number'] in {h['number'] for h in r['horses']} for s in r['entries']['scratched'])
    assert r['venueCode'] in ['05','06','07','08','09']
    assert r['trackCode']<30
    assert len({h['number'] for h in r['horses']})==len(r['horses'])
    for h in r['horses']:
        assert all(0<=p<=1 for p in h['positions'])
        assert all(v['date']<r['date'] for v in h['history'])
        assert len(h['history'])<=5
        assert h['blinker'] in [True,False,None]
print('PASS: historical date boundaries, field uniqueness, model feature exclusions, display ranges')
