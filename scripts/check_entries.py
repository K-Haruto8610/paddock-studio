"""Offline checks (no DB, no network): race-day change handling and snapshot sanity.
  python scripts/check_entries.py"""
import copy, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_entries as fe

ENTRIES = HERE.parent / 'data' / 'entries'


def horse(n, name, **kw):
    h = {'number': n, 'gate': (n + 1) // 2, 'name': name, 'horseId': f'20210000{n:02d}', 'sex': '牝', 'age': 4, 'weight': 56.0,
         'jockey': 'テスト', 'blinker': False, 'status': 'entered', 'statusReason': None}
    return h | kw


def snap(horses, status='declared', notices=()):
    return {'raceId': 'T', 'status': status, 'source': 'test', 'race': {'date': '20261115'}, 'horses': copy.deepcopy(horses), 'notices': list(notices)}


def run():
    base = [horse(i, f'馬{i}') for i in range(1, 7)]
    first = fe.merge_with_previous(snap(base), None, 't1')
    assert [c['kind'] for c in first['changes']] == ['declared']
    # blinker added, number/jockey changed, one horse scratched via the card, one dropped from the card
    nxt = copy.deepcopy(base)
    nxt[0]['blinker'] = True
    nxt[1]['number'], nxt[1]['gate'] = 9, 5
    nxt[2]['jockey'] = '代打'
    nxt[3]['status'], nxt[3]['statusReason'] = 'scratched', '取消'
    del nxt[4]
    second = fe.merge_with_previous(snap(nxt), first, 't2')
    kinds = sorted((c['kind'], c['name']) for c in second['changes'] if c['at'] == 't2')
    assert ('blinker', '馬1') in kinds and ('number', '馬2') in kinds and ('gate', '馬2') in kinds and ('jockey', '馬3') in kinds, kinds
    assert ('scratched', '馬4') in kinds and ('scratched', '馬5') in kinds, kinds
    status = {h['name']: h['status'] for h in second['horses']}
    assert status['馬4'] == status['馬5'] == 'scratched' and status['馬1'] == 'entered'
    # a scratch is final even if a later card lists the horse as entered again
    third = fe.merge_with_previous(snap(base), second, 't3')
    assert {h['name']: h['status'] for h in third['horses']}['馬4'] == 'scratched'
    # JRA notice naming a field horse marks it scratched
    fourth = fe.merge_with_previous(snap(base, notices=['出走取消 馬6 （疾病のため）']), first, 't4')
    assert {h['name']: h['status'] for h in fourth['horses']}['馬6'] == 'scratched'
    assert fourth['notices'][0]['firstSeen'] == 't4'
    # a registration list never overwrites a declared card
    reg = fe.merge_with_previous(snap([dict(h, number=None, gate=None) for h in base], status='registered'), first, 't5')
    assert reg['status'] == 'declared' and reg['horses'][0]['number'] == 1
    # registration -> declaration transition is logged
    r0 = fe.merge_with_previous(snap([dict(h, number=None, gate=None, status='registered') for h in base], status='registered'), None, 't0')
    r1 = fe.merge_with_previous(snap(base), r0, 't1')
    assert any(c['kind'] == 'declared' for c in r1['changes'])
    print('PASS: change tracking (blinker, number/frame, jockey, scratch, notices, no regression to registration)')
    # snapshots on disk
    for p in sorted(ENTRIES.glob('*.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        assert s['status'] in ('declared', 'registered') and s['fetchedAt'] and s['source'] and s['race']['date']
        text = json.dumps(s, ensure_ascii=False).lower()
        assert 'odds' not in text and 'ninki' not in text, 'odds/popularity must never be stored'
        if s['status'] == 'declared':
            act = [h for h in s['horses'] if h['status'] != 'scratched']
            assert len({h['number'] for h in act}) == len(act) and all(h['number'] for h in act), p.name
        print('PASS: snapshot', p.name, s['status'], len(s['horses']), 'horses')


if __name__ == '__main__':
    run()
