"""Compare data/g1_races.json against the local DB (read-only SELECT). Exit 1 on mismatches.
Reports: master races whose DB rows disagree on venue/distance/track, flat GI rows in the DB that
the master does not know, and master races with no recent DB record."""
import os, sys
from collections import defaultdict
import pymysql
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.environ.get('KEIBA_SOURCE', 'C:/Users/jouza/Downloads/keiba-yosou/mykeiba'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.config import settings
import g1_master


def readable(s):
    return bool(s) and '?' not in s and '\ufffd' not in s and '\u0152' not in s and '\u2030' not in s and '\u2018' not in s


def main(since='2023'):
    master = g1_master.load()
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute("""SELECT RACE_CODE,KAISAI_NEN,KAISAI_GAPPI,KEIBAJO_CODE,KYOSOMEI_HONDAI,KYORI,TRACK_CODE
            FROM race_shosai WHERE BINARY GRADE_CODE='A' AND KAISAI_NEN>=%s
            AND KEIBAJO_CODE BETWEEN '01' AND '10' AND TRACK_CODE BETWEEN '10' AND '29' ORDER BY RACE_CODE""", (since,))
        rows = q.fetchall()
    c.close()
    seen, problems, unknown, garbled = defaultdict(list), [], [], 0
    for r in rows:
        name = r['KYOSOMEI_HONDAI'].strip()
        m = g1_master.find_by_name(master, name)
        if m:
            seen[m['id']].append(r['KAISAI_NEN'])
            diff = g1_master.mismatches(master, m, r['KEIBAJO_CODE'], r['KYORI'], r['TRACK_CODE'], r['KAISAI_NEN'])
            if diff:
                problems.append(f"{r['KAISAI_NEN']} {name}: " + '; '.join(diff))
        elif r['KEIBAJO_CODE'] in {'05', '06', '07', '08', '09'}:
            if readable(name): unknown.append(f"{r['KAISAI_NEN']}{r['KAISAI_GAPPI']} venue{r['KEIBAJO_CODE']} {name}")
            else: garbled += 1
    print(f'DB flat GI rows since {since} (JRA venues + overseas/others): {len(rows)}')
    print(f'master races: {len(master["races"])}')
    for m in master['races']:
        ys = seen.get(m['id'], [])
        print(f"  {m['name']:<20} DB years: {','.join(ys) or '-'}")
    missing = [m['name'] for m in master['races'] if not seen.get(m['id'])]
    print('\nMISMATCH (master vs DB):'); [print('  ' + p) for p in problems] or print('  none')
    print('\nJRA-venue GI rows unknown to master:'); [print('  ' + u) for u in unknown] or print('  none')
    print(f'\nunreadable (mojibake) JRA-venue GI rows skipped: {garbled}')
    print('master races with no DB record since ' + since + ':', ', '.join(missing) or 'none')
    return 1 if problems or unknown or missing else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:2]))
