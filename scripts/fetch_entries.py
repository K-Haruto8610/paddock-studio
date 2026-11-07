"""Fetch the declared field (出馬表) of upcoming JRA flat GI races and keep snapshots.

Source priority: (a) local mykeiba DB if it already holds pre-race card rows (read-only SELECT),
(b) JRA official public pages (no login, generic UA, >=2.5 s between requests, cached under
outputs/jra_cache). Odds and popularity are never read or stored.

Output: data/entries/<race_id>.json with source, fetchedAt, per-horse rows, a change log
(scratches, blinker, number/frame, jockey, weight) and raw JRA notices.

  python scripts/fetch_entries.py [--days 14] [--today YYYY-MM-DD] [--dry-run] [--refresh]
  python scripts/fetch_entries.py --fixture page.html --fixture-date 2026-11-15   (parser test, no network/DB)
"""
import argparse, html, json, os, re, sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_master, jra_client

ROOT = HERE.parent
ENTRIES = ROOT / 'data' / 'entries'
BASE = 'https://www.jra.go.jp/JRADB/'
VENUE_NAMES = {'東京': '05', '中山': '06', '中京': '07', '京都': '08', '阪神': '09'}
FRESH = 600  # seconds a cached volatile JRA page stays valid unless --refresh


def strip_tags(s):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()


def jra_get(page, cname, label, refresh):
    return jra_client.fetch(BASE + page, data={'cname': cname}, max_age=0 if refresh else FRESH, label=label)


# ---------------------------------------------------------------- JRA parsers
def parse_links(s, page):
    """[(cname, text)] for doAction('/JRADB/<page>', cname) anchors."""
    out = []
    for m in re.finditer(r"doAction\(\s*'/JRADB/" + page + r"'\s*,\s*'([^']+)'\s*\)[^>]*>(.*?)</a>", s, re.S):
        out.append((m.group(1), strip_tags(m.group(2))))
    return out


def track_code(surface, detail):
    right = '右' in detail
    outer = '外' in detail
    if surface == 'ダート':
        return 24 if right else 23
    return (18 if outer else 17) if right else (12 if outer else 11)


def parse_race_header(block):
    t = {}
    m = re.search(r'(\d{4})年(\d+)月(\d+)日.*?(\d+)回(\S+?)(\d+)日', strip_tags(block[:block.find('<thead')]))
    if m:
        t['date'] = f'{int(m.group(1)):04d}{int(m.group(2)):02d}{int(m.group(3)):02d}'
        t['kai'], t['venue'], t['nichi'] = int(m.group(4)), m.group(5), int(m.group(6))
    m = re.search(r'class="race_name">(.*?)<span class="grade_icon', block, re.S) or re.search(r'class="race_name">(.*?)</span>', block, re.S)
    t['name'] = strip_tags(m.group(1)) if m else ''
    t['grade1'] = 'icon_grade_g1' in block
    m = re.search(r'コース：</span>([\d,]+)<span class="unit">メートル</span><span class="detail">（([^）]*)）', block)
    if m:
        t['distance'] = int(m.group(1).replace(',', ''))
        detail = m.group(2)
        t['surface'] = 'ダート' if 'ダート' in detail else '芝'
        t['detail'] = detail
        t['trackCode'] = track_code(t['surface'], detail)
    return t


def parse_card_rows(block):
    rows = []
    for tr in re.findall(r'<tr>(.*?)</tr>', block[block.find('<tbody'):], re.S):
        num = re.search(r'<td class="num">(.*?)</td>', tr, re.S)
        horse = re.search(r'<td class="horse">(.*?)</td>', tr, re.S)
        if not num or not horse:
            continue
        n = re.search(r'\d+', strip_tags(re.sub(r'<span class="horse_icon.*?</span>', '', num.group(1), flags=re.S)))
        waku = re.search(r'alt="枠(\d)', tr)
        link = re.search(r'CNAME=([^"]+)">(.*?)</a>', horse.group(1), re.S)
        age = re.search(r'<td class="age">(.*?)</td>', tr, re.S)
        wt = re.search(r'<td class="weight">\s*([\d.]+)', tr)
        jk = re.search(r'<td class="jockey">(.*?)</td>', tr, re.S)
        sa = re.match(r'([牡牝セ騸])\D*(\d+)', strip_tags(age.group(1))) if age else None
        hid = re.search(r'pw01dud\d{2}(\d{10})', link.group(1)) if link else None
        # Cancellation is read from the non-name/non-odds cells (the marker wording is not yet seen on a real card).
        cells = strip_tags(re.sub(r'<td class="(?:horse|jockey|trainer|odds)">.*?</td>', '', tr, flags=re.S))
        gone = re.search(r'出走取消|競走除外|取消|除外', cells)
        rows.append({
            'number': int(n.group(0)) if n else None,
            'gate': int(waku.group(1)) if waku else None,
            'name': strip_tags(link.group(2)) if link else strip_tags(horse.group(1)),
            'horseId': hid.group(1) if hid else None,
            'sex': sa.group(1) if sa else None,
            'age': int(sa.group(2)) if sa else None,
            'weight': float(wt.group(1)) if wt else None,
            'jockey': re.sub(r'\s+', '', strip_tags(jk.group(1))) if jk else None,
            'blinker': True if 'icon_blinker' in num.group(1) else False,
            'status': 'scratched' if gone else 'entered',
            'statusReason': gone.group(0) if gone else None,
        })
    return rows


def parse_day_page(s):
    """All races on a 全てのレースを表示 page: list of dict(header + horses)."""
    out = []
    parts = re.split(r'<li id="syutsuba_(\d+)R">', s)
    for number, chunk in zip(parts[1::2], parts[2::2]):
        block = chunk[:chunk.find('</table>') + 8]
        h = parse_race_header(block)
        h['raceNumber'] = int(number)
        h['horses'] = parse_card_rows(block)
        out.append(h)
    return out


def parse_registered(s):
    h = parse_race_header(s)
    m = re.search(r'第(\d+)回特別登録<span class="num">(\d+)頭', s)
    h['registeredCount'] = int(m.group(2)) if m else None
    h['horses'] = []
    for hm in re.finditer(r'<td class="horse"><a href="[^"]*CNAME=([^"]+)">(.*?)</a></td>\s*<td class="weight">([\d.]+)', s, re.S):
        hid = re.search(r'pw01dud\d{2}(\d{10})', hm.group(1))
        h['horses'].append({'number': None, 'gate': None, 'name': strip_tags(hm.group(2)), 'horseId': hid.group(1) if hid else None,
                            'sex': None, 'age': None, 'weight': float(hm.group(3)), 'jockey': None, 'blinker': None, 'status': 'registered'})
    return h


def parse_notices(s, venue, kai, nichi, race_number):
    """Free-text 変更情報 for one race from the 開催お知らせ page."""
    heading = f'{kai}回{venue}{nichi}日'
    i = s.find(heading)
    if i < 0:
        return []
    block = s[i:]
    j = block.find('</table>')
    for m in re.finditer(r'<th scope="row">(\d+)</th>\s*<td>(.*?)</td>', block[:j if j > 0 else None], re.S):
        if int(m.group(1)) == race_number and strip_tags(m.group(2)):
            return [strip_tags(m.group(2))]
    return []


# ---------------------------------------------------------------- sources
def race_id(date_s, venue, kai, nichi, number):
    return f'{date_s}{venue}{kai:02d}{nichi:02d}{number:02d}'


def from_jra(today, days, refresh, master):
    """Returns list of snapshots-in-progress (no change log yet)."""
    found = []
    last = today + timedelta(days=days)
    trl = jra_get('accessT.html', 'pw03trl00/29', 'trl', refresh)
    cand = {}
    for cname, text in parse_links(trl, 'accessT.html'):
        m = re.match(r'pw03tde\d{2}(\d{2})(\d{2})(\d{2})(\d{2})(\d{8})', cname)
        mr = g1_master.find_by_name(master, re.sub(r'^第\d+回', '', text.split(' ')[0]))
        if m and mr:
            d = datetime.strptime(m.group(5), '%Y%m%d').date()
            if today <= d <= last:
                cand[mr['id']] = (mr, d, cname, m)
    if not cand:
        return found, 'JRA特別登録一覧に、期間内のGⅠはありません'
    dli = jra_get('accessD.html', 'pw01dli00/F3', 'dli00', refresh)
    day_links = {}
    for cname, _ in parse_links(dli, 'accessD.html'):
        m = re.match(r'pw01drl\d{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})', cname)
        if m:
            day_links[(m.group(5), m.group(1))] = cname
    des_cache = {}
    for mr, d, tde, tm in cand.values():
        key = (d.strftime('%Y%m%d'), tm.group(1))
        if key in day_links:
            if key not in des_cache:
                day = jra_get('accessD.html', day_links[key], 'drl', refresh)
                des = [c for c, t in parse_links(day, 'accessD.html') if c.startswith('pw01des')]
                des_cache[key] = parse_day_page(jra_get('accessD.html', des[0], 'des01', refresh)) if des else []
            card = next((r for r in des_cache[key] if g1_master.norm(r['name']) in mr['_names'] and r['grade1']), None)
            if card and card['horses']:
                card.update(status='declared', source='JRA公式 出馬表', mr=mr)
                notices = []
                if today == d or (d - today).days <= 3:
                    ide = jra_get('accessI.html', 'pw01ide01/4F', 'ide01', refresh)
                    notices = parse_notices(ide, card['venue'], card['kai'], card['nichi'], card['raceNumber'])
                card['notices'] = notices
                found.append(card)
                continue
        reg = parse_registered(jra_get('accessT.html', tde, 'tde', refresh))
        if reg['horses']:
            reg.update(status='registered', source='JRA公式 特別レース登録馬', mr=mr, raceNumber=int(tm.group(4)),
                       notices=[], kai=int(tm.group(2)), nichi=int(tm.group(3)), date=d.strftime('%Y%m%d'))
            found.append(reg)
    return found, None


def from_db(today, days, master):
    """Pre-race card rows already in mykeiba (read-only). Returns (snapshots, reason)."""
    sys.path.insert(0, os.environ.get('KEIBA_SOURCE', 'C:/Users/jouza/Downloads/keiba-yosou/mykeiba'))
    try:
        import pymysql
        from backend.config import settings
        c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    except Exception as e:  # never print connection details
        return [], 'DBに接続できません（' + type(e).__name__ + '）'
    lo, hi = today.strftime('%Y%m%d'), (today + timedelta(days=days)).strftime('%Y%m%d')
    out = []
    with c.cursor() as q:
        q.execute("""SELECT RACE_CODE,KAISAI_NEN,KAISAI_GAPPI,KEIBAJO_CODE,KAISAI_KAI,KAISAI_NICHIME,RACE_BANGO,
            KYOSOMEI_HONDAI,KYORI,TRACK_CODE FROM race_shosai WHERE BINARY GRADE_CODE='A'
            AND CONCAT(KAISAI_NEN,KAISAI_GAPPI) BETWEEN %s AND %s AND KEIBAJO_CODE BETWEEN '05' AND '09'""", (lo, hi))
        for r in q.fetchall():
            mr = g1_master.match_db_race(master, r['KYOSOMEI_HONDAI'], r['KEIBAJO_CODE'], r['KYORI'], r['TRACK_CODE'], r['KAISAI_NEN'])
            if not mr:
                continue
            q.execute("""SELECT UMABAN,WAKUBAN,KETTO_TOROKU_BANGO,BAMEI,SEIBETSU_CODE,BAREI,FUTAN_JURYO,KISHUMEI_RYAKUSHO,
                BLINKER_SHIYO_KUBUN,DATA_KUBUN,IJO_KUBUN_CODE FROM umagoto_race_joho WHERE RACE_CODE=%s ORDER BY UMABAN""", (r['RACE_CODE'],))
            rows = [x for x in q.fetchall() if str(x['UMABAN']).strip().isdigit() and x['DATA_KUBUN'] != '0']
            if not rows:
                continue
            horses = []
            for x in rows:
                gone = str(x['IJO_KUBUN_CODE'] or '').strip() in ('1', '2')  # 1 出走取消, 2 発走除外
                horses.append({'number': int(x['UMABAN']), 'gate': int(x['WAKUBAN']), 'name': x['BAMEI'].strip(), 'horseId': x['KETTO_TOROKU_BANGO'],
                               'sex': {'1': '牡', '2': '牝', '3': 'セ'}.get(str(x['SEIBETSU_CODE']).strip()), 'age': int(x['BAREI']),
                               'weight': int(x['FUTAN_JURYO']) / 10, 'jockey': x['KISHUMEI_RYAKUSHO'].strip(),
                               'blinker': {'0': False, '1': True}.get(str(x['BLINKER_SHIYO_KUBUN']).strip()),
                               'status': 'scratched' if gone else 'entered', 'statusReason': 'DB異常区分' if gone else None})
            tc = int(r['TRACK_CODE'])
            out.append({'status': 'declared', 'source': 'mykeiba DB (JV-Data)', 'mr': mr, 'name': r['KYOSOMEI_HONDAI'].strip(), 'date': r['KAISAI_NEN'] + r['KAISAI_GAPPI'],
                        'venue': mr['venue'], 'kai': int(r['KAISAI_KAI']), 'nichi': int(r['KAISAI_NICHIME']), 'raceNumber': int(r['RACE_BANGO']),
                        'distance': int(r['KYORI']), 'surface': 'ダート' if tc >= 23 else '芝', 'trackCode': tc, 'horses': horses, 'notices': []})
    c.close()
    return out, None if out else 'DBに期間内のGⅠ出馬表はありません'


# ---------------------------------------------------------------- snapshots
FIELDS = [('number', '馬番'), ('gate', '枠'), ('jockey', '騎手'), ('blinker', 'ブリンカー'), ('weight', '斤量')]


def to_snapshot(card):
    mr = card['mr']
    venue_code = mr['venueCode']
    kind_id = race_id(card['date'], venue_code, card['kai'], card['nichi'], card['raceNumber'])
    tcode = card.get('trackCode') or mr['trackCodes'][0]
    if tcode not in mr['trackCodes']:
        tcode = mr['trackCodes'][0]
    return {'raceId': kind_id, 'masterId': mr['id'], 'status': card['status'], 'source': card['source'],
            'race': {'name': card['name'] or mr['name'], 'date': card['date'], 'venueCode': venue_code, 'venue': mr['venue'], 'surface': mr['surface'],
                     'distance': mr['distance'], 'trackCode': tcode, 'kai': card['kai'], 'nichi': card['nichi'], 'raceNumber': card['raceNumber'],
                     'idProvisional': card['status'] != 'declared'},
            'horses': card['horses'], 'notices': card.get('notices', [])}


def key_of(h):
    return h.get('horseId') or h['name']


def merge_with_previous(new, old, now):
    """Carry scratches forward, build the change log against the previous snapshot."""
    changes = list(old.get('changes', [])) if old else []
    notices = list(old.get('notices', [])) if old else []
    known = {n['text'] for n in notices}
    for t in new.pop('notices', []):
        if t not in known:
            notices.append({'text': t, 'firstSeen': now})
    # A notice naming a field horse as 出走取消/競走除外 marks it scratched even if the card row is not marked.
    for n in notices:
        if re.search(r'出走取消|競走除外', n['text']):
            for h in new['horses']:
                if h['status'] == 'entered' and h['name'] in n['text']:
                    h['status'], h['statusReason'] = 'scratched', n['text']
    if old and old['status'] == 'declared' and new['status'] == 'registered':
        new = old | {'fetchedAt': now}  # never regress from a declared card to the registration list
        return new
    if old:
        prev = {key_of(h): h for h in old['horses']}
        seen = set()
        for h in new['horses']:
            k = key_of(h)
            seen.add(k)
            p = prev.get(k)
            if p is None:
                if old['status'] == 'declared':
                    changes.append({'at': now, 'kind': 'added', 'number': h['number'], 'name': h['name']})
                elif new['status'] == 'registered':
                    changes.append({'at': now, 'kind': 'registered', 'name': h['name']})
                continue
            if p['status'] == 'scratched' and h['status'] != 'scratched':
                h['status'], h['statusReason'] = 'scratched', p.get('statusReason')  # scratches are final
            if h['status'] == 'scratched' and p['status'] != 'scratched':
                changes.append({'at': now, 'kind': 'scratched', 'number': p.get('number'), 'name': h['name'], 'reason': h.get('statusReason')})
            if old['status'] == 'declared' and new['status'] == 'declared':
                for f, label in FIELDS:
                    if p.get(f) != h.get(f) and p.get(f) is not None:
                        changes.append({'at': now, 'kind': f, 'label': label, 'number': h['number'], 'name': h['name'], 'before': p.get(f), 'after': h.get(f)})
        for k, p in prev.items():
            if k not in seen:
                if old['status'] == 'declared' and new['status'] == 'declared':
                    if p['status'] != 'scratched':
                        changes.append({'at': now, 'kind': 'scratched', 'number': p.get('number'), 'name': p['name'], 'reason': 'カードから消失'})
                    new['horses'].append(dict(p, status='scratched', statusReason=p.get('statusReason') or 'カードから消失'))
                elif new['status'] == 'registered':
                    changes.append({'at': now, 'kind': 'withdrawn', 'name': p['name']})
        if old['status'] == 'registered' and new['status'] == 'declared':
            changes.append({'at': now, 'kind': 'declared', 'count': len(new['horses'])})
    elif new['status'] == 'declared':
        changes.append({'at': now, 'kind': 'declared', 'count': len(new['horses'])})
    new['horses'].sort(key=lambda h: (h['number'] is None, h['number'] or 0, h['name']))
    new.update(fetchedAt=now, changes=changes, notices=notices)
    return new


def significant(s):
    return {k: v for k, v in s.items() if k not in ('fetchedAt',)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--today')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--refresh', action='store_true', help='ignore cached JRA pages')
    ap.add_argument('--no-db', action='store_true')
    ap.add_argument('--fixture', help='parse a saved 全てのレース or 登録馬 page instead of fetching (parser test)')
    a = ap.parse_args()
    today = datetime.strptime(a.today, '%Y-%m-%d').date() if a.today else date.today()
    master = g1_master.load()
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    print(f'today={today} window={today}..{today + timedelta(days=a.days)}')
    cards, reasons = [], []
    if a.fixture:
        s = Path(a.fixture).read_text(encoding='utf-8')
        for r in parse_day_page(s):
            mr = g1_master.find_by_name(master, r['name'])
            if mr and r['grade1']:
                r.update(status='declared', source='fixture', mr=mr)
                cards.append(r)
        if not cards:
            reg = parse_registered(s)
            mr = g1_master.find_by_name(master, reg['name'])
            if mr:
                reg.update(status='registered', source='fixture', mr=mr, raceNumber=11, kai=reg.get('kai', 0), nichi=reg.get('nichi', 0), date=reg.get('date', '00000000'), notices=[])
                cards.append(reg)
    else:
        if not a.no_db:
            cards, why = from_db(today, a.days, master)
            if why: reasons.append('DB: ' + why)
        if not cards:
            try:
                cards, why = from_jra(today, a.days, a.refresh, master)
                if why: reasons.append('JRA: ' + why)
            except Exception as e:
                reasons.append(f'JRA: 取得失敗 ({type(e).__name__}: {e})')
    print(f"JRA requests={jra_client.stats['requests']} cached={jra_client.stats['cached']}")
    if not cards:
        print('No upcoming GI entries found.', *reasons, sep='\n  ')
        return 0
    ENTRIES.mkdir(parents=True, exist_ok=True)
    for card in cards:
        snap = to_snapshot(card)
        path = ENTRIES / f"{snap['raceId']}.json"
        old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        snap = merge_with_previous(snap, old, now)
        new_changes = len(snap.get('changes', [])) - len(old.get('changes', [])) if old else len(snap.get('changes', []))
        active = [h for h in snap['horses'] if h['status'] != 'scratched']
        print(f"{snap['race']['date']} {snap['race']['venue']} {snap['race']['name']} [{snap['status']}] source={snap['source']} "
              f"horses={len(snap['horses'])} active={len(active)} scratched={len(snap['horses']) - len(active)} newChanges={new_changes}")
        for h in snap['horses'][:30]:
            print('  ', h['number'], h['gate'], h['name'], h['sex'], h['age'], h['weight'], h['jockey'], h['blinker'], h['status'])
        for c in snap['changes'][-new_changes:] if new_changes else []:
            print('   change:', json.dumps(c, ensure_ascii=False))
        if not a.dry_run:
            if old and significant(old) == significant(snap):
                old['fetchedAt'] = now
                snap = old
            path.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding='utf-8')
            print('  wrote', path.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
