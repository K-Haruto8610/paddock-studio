"""Per-year data counts (read-only), for checking a DB import (追加指示5, 2026-11-18).

For each year 2020-2026, with the same venue/track filter as pace_model.load_frame (JRA venues 01-10, flat
tracks 10-29):
  (a) races            distinct RACE_CODE in race_shosai
  (b) named            races whose KYOSOMEI_HONDAI passes build_forecasts.clean()
  (c) named_complete   named races where every runner is present: distinct UMABAN in umagoto_race_joho (after the
                       same row filters as pace_model.load_frame) equals SHUSSO_TOSU. These are the training races.
  (d) join_rows        rows of race_shosai JOIN umagoto_race_joho (no filter beyond venue/track); join_races = races in it
      frame_rows       rows that survive pace_model.load_frame's filters (named race, clean horse name, distance,
                       field size, abnormality code), i.e. history + training rows
Also: rows of umagoto_race_joho / race_shosai per year without the join, so an import that fills only one
table shows up.

  python scripts/count_years.py                       # print
  python scripts/count_years.py --save before.json    # keep a snapshot
  python scripts/count_years.py --compare before.json # print with the difference to a snapshot
"""
import sys, json, argparse
from pathlib import Path
import pandas as pd
import pymysql

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_forecasts import settings, number, clean  # noqa: E402

YEARS = [str(y) for y in range(2020, 2027)]
FILTER = "KEIBAJO_CODE BETWEEN '01' AND '10' AND TRACK_CODE BETWEEN '10' AND '29' AND KAISAI_NEN BETWEEN '2020' AND '2026'"
KEYS = ['races', 'named', 'named_complete', 'join_races', 'join_rows', 'frame_rows', 'shosai_rows', 'umagoto_rows']


def counts():
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute(f"SELECT RACE_CODE,KAISAI_NEN,KYOSOMEI_HONDAI,SHUSSO_TOSU,KYORI FROM race_shosai WHERE {FILTER}")
        races = pd.DataFrame(q.fetchall())
        q.execute("SELECT KAISAI_NEN, COUNT(*) n FROM race_shosai GROUP BY KAISAI_NEN")
        shosai_all = {r['KAISAI_NEN']: int(r['n']) for r in q.fetchall()}
        q.execute("SELECT KAISAI_NEN, COUNT(*) n FROM umagoto_race_joho GROUP BY KAISAI_NEN")
        umagoto_all = {r['KAISAI_NEN']: int(r['n']) for r in q.fetchall()}
        q.execute(f"""SELECT r.RACE_CODE,r.KAISAI_NEN,u.UMABAN,u.BAMEI,u.IJO_KUBUN_CODE
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          AND r.KAISAI_NEN BETWEEN '2020' AND '2026'""")
        rows = pd.DataFrame(q.fetchall())
    c.close()
    races = races.drop_duplicates('RACE_CODE')
    races['named'] = races.KYOSOMEI_HONDAI.map(clean)
    races['field'] = races.SHUSSO_TOSU.map(number)
    races['dist'] = races.KYORI.map(number)
    f = rows.merge(races[['RACE_CODE', 'named', 'field', 'dist']], on='RACE_CODE')
    f['umaban'] = f.UMABAN.map(number)
    keep = (f.named & f.BAMEI.map(clean) & f.dist.between(1000, 3600) & f.field.between(5, 18)
            & f.umaban.between(1, 18) & f.IJO_KUBUN_CODE.fillna('0').str.strip().isin(['', '0', '3', '4', '5', '6', '7']))
    fr = f[keep].drop_duplicates(['RACE_CODE', 'UMABAN'])
    per = fr.groupby('RACE_CODE').agg(n=('umaban', 'nunique'), field=('field', 'first'), y=('KAISAI_NEN', 'first'))
    complete = per[per.n == per.field]
    out = {}
    for y in YEARS:
        ry = races[races.KAISAI_NEN == y]
        out[y] = {
            'races': int(len(ry)),
            'named': int(ry.named.sum()),
            'named_complete': int((complete.y == y).sum()),
            'join_races': int(rows[rows.KAISAI_NEN == y].RACE_CODE.nunique()),
            'join_rows': int((rows.KAISAI_NEN == y).sum()),
            'frame_rows': int((fr.KAISAI_NEN == y).sum()),
            'shosai_rows': shosai_all.get(y, 0),
            'umagoto_rows': umagoto_all.get(y, 0),
        }
    return out


def show(now, before=None):
    print('year  ' + ''.join(f'{k:>18}' for k in KEYS))
    for y in YEARS:
        cells = []
        for k in KEYS:
            v = now[y][k]
            if before and y in before and k in before[y]:
                d = v - before[y][k]
                cells.append(f'{v:>10} ({d:+d})' if d else f'{v:>10}      ')
            else:
                cells.append(f'{v:>18}')
        print(f'{y}  ' + ''.join(f'{c:>18}' for c in cells))
    if before:
        shrunk = [(y, k) for y in YEARS for k in KEYS if y in before and k in before[y] and now[y][k] < before[y][k]]
        print('\nDECREASES (stop and report):', shrunk if shrunk else 'none')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save')
    ap.add_argument('--compare')
    a = ap.parse_args()
    now = counts()
    before = json.loads(Path(a.compare).read_text(encoding='utf-8')) if a.compare else None
    show(now, before)
    if a.save:
        Path(a.save).write_text(json.dumps(now, indent=1, ensure_ascii=False), encoding='utf-8')
        print('saved', a.save)


if __name__ == '__main__':
    main()
