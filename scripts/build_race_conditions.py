"""Race-day conditions of the bundled races, for the 3D scene: weather and going (sky, rain, snow, wet surfaces, the
infield board) and how long the turf has been raced on since the rail was last moved (wear along the inside rail).

Reads race_shosai for every race id in public/data/forecasts.json, docs/visual-baseline/forecasts.json and the entry
cards in data/entries, and writes public/data/race_conditions.json. Read-only on the DB; no model input.

JV-Data codes: weather 1 晴, 2 曇, 3 雨, 4 小雨, 5 雪, 6 小雪; going 1 良, 2 稍重, 3 重, 4 不良.

Meeting (`meeting`): KAISAI_KAI / KAISAI_NICHIME (the race id carries both), the rail position COURSE_KUBUN (A-D) and
`courseDay`: the turf race days at the venue on that rail position up to and including the race day, counted back
across meetings (4回 into 5回 on the same B course keeps counting) and stopping at a rail move or a break of more than
60 days. Moving the rail out puts the worn strip behind it, so this, not the meeting day, is what wears the inside.
A dirt race takes the rail of the day's turf races. A race not yet in the DB (an entry card before the day) gets its
rail and count from the last turf day at the venue before it, plus one, marked `estimated`; a race without weather or going in the DB leaves them out.

    python scripts/build_race_conditions.py
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_forecasts import settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / 'public' / 'data' / 'forecasts.json', ROOT / 'docs' / 'visual-baseline' / 'forecasts.json']
ENTRIES = ROOT / 'data' / 'entries'
OUT = ROOT / 'public' / 'data' / 'race_conditions.json'
BREAK_DAYS = 60


def race_ids():
    ids = {r['id'] for f in SOURCES if f.exists() for r in json.loads(f.read_text(encoding='utf-8'))['races']}
    if ENTRIES.exists():
        ids |= {f.stem for f in ENTRIES.glob('*.json') if f.stem.isdigit() and len(f.stem) == 16}
    return sorted(ids)


def day_of(yyyymmdd):
    return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


def course_day(days, at, letter):
    """Turf days at a venue (sorted [(date, {letters})]) on rail `letter`, counted back from the last day <= `at`."""
    count, later = 0, None
    for when, letters in reversed([d for d in days if d[0] <= at]):
        if letter not in letters or (later and (later - when).days > BREAK_DAYS):
            break
        count += 1
        later = when
    return count


def main():
    ids = race_ids()
    venues = sorted({i[8:10] for i in ids})
    import pymysql
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute(
            'SELECT RACE_CODE,TRACK_CODE,TENKO_CODE,SHIBA_BABAJOTAI_CODE,DIRT_BABAJOTAI_CODE,KAISAI_KAI,KAISAI_NICHIME,'
            'COURSE_KUBUN FROM race_shosai WHERE RACE_CODE IN (' + ','.join(['%s'] * len(ids)) + ')', ids)
        rows = {r['RACE_CODE']: r for r in q.fetchall()}
        # Every turf race day at these venues, with the rail positions used that day.
        q.execute(
            "SELECT KEIBAJO_CODE, CONCAT(KAISAI_NEN, KAISAI_GAPPI) AS DAY, GROUP_CONCAT(DISTINCT COURSE_KUBUN) AS RAILS "
            "FROM race_shosai WHERE KEIBAJO_CODE IN (" + ','.join(['%s'] * len(venues)) + ") "
            "AND TRACK_CODE BETWEEN '10' AND '22' AND KAISAI_NEN >= '2019' GROUP BY KEIBAJO_CODE, DAY ORDER BY DAY", venues)
        turf_days = {}
        for r in q.fetchall():
            rails = {x.strip() for x in (r['RAILS'] or '').split(',') if x.strip()}
            turf_days.setdefault(r['KEIBAJO_CODE'], []).append((day_of(r['DAY']), rails))
    c.close()

    races = {}
    for rid in ids:
        venue, when = rid[8:10], day_of(rid[:8])
        row = rows.get(rid)
        out = {}
        if row:
            weather = int(row['TENKO_CODE'] or 0)
            dirt = int(row['TRACK_CODE']) >= 23
            going = int((row['DIRT_BABAJOTAI_CODE'] if dirt else row['SHIBA_BABAJOTAI_CODE']) or 0)
            if 1 <= weather <= 6:
                out['weather'] = weather
            if 1 <= going <= 4:
                out['going'] = going
        days = turf_days.get(venue, [])
        letter = (row['COURSE_KUBUN'] or '').strip() if row else ''
        meeting = {'kai': int(rid[10:12]), 'day': int(rid[12:14])}
        if letter:
            meeting.update(course=letter, courseDay=course_day(days, when, letter))
        else:
            # A dirt race takes the rail of the day's turf races. A race not in the DB yet takes the rail of the last turf
            # day before it, one day further on (estimated).
            before = [d for d in days if d[0] <= when]
            if before and (when - before[-1][0]).days <= BREAK_DAYS and len(before[-1][1]) == 1:
                letter = next(iter(before[-1][1]))
                same_day = before[-1][0] == when
                meeting.update(course=letter, courseDay=course_day(days, before[-1][0], letter) + (0 if same_day else 1))
                if not same_day:
                    meeting['estimated'] = True
        out['meeting'] = meeting
        races[rid] = out
    result = {
        'generatedAt': datetime.now().isoformat(timespec='seconds'),
        'source': 'race_shosai TENKO_CODE, SHIBA/DIRT_BABAJOTAI_CODE, KAISAI_KAI, KAISAI_NICHIME, COURSE_KUBUN (JV-Data)',
        'races': races,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{len(races)} races -> {OUT.relative_to(ROOT)}')
    for rid, r in races.items():
        m = r['meeting']
        print(rid, r.get('weather'), r.get('going'), f"{m['kai']}回{m['day']}日目", m.get('course'), m.get('courseDay'),
              'est' if m.get('estimated') else '')


if __name__ == '__main__':
    main()
