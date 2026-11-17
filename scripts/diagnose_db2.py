"""Deeper mykeibadb diagnosis (read-only). Run like diagnose_db.py.

Why: diagnose_db.py showed utf8mb4 everywhere and 0% garbled/missing in
the Grade-A subset, yet build_forecasts.py drops ~82% of rows and the
Takarazuka/Yasuda replays came out empty. This script finds out where
the rows actually go:

  1. Raw stored values for the two broken 2026-06 races (repr, so
     spaces / '00' / full-width digits are visible). export_replays.py
     accepts only plain digits 1..field, while the SQL missing-check in
     diagnose_db.py only tested NULL/''/'0' -- a padded or '00' value
     passes one and fails the other.
  2. The distinct CORNER4_JUNI values actually stored (top 30).
  3. A funnel replicating build_forecasts.py's filters one by one, so
     the 78,509 excluded rows are attributed to a specific step:
     date, garbled horse name, garbled race name, numeric ranges,
     abnormality codes, and the whole-field completeness rule
     (one bad horse removes the entire race).
"""
import os, sys, re
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.environ.get("KEIBA_SOURCE", "C:/Users/jouza/Downloads/keiba-yosou/mykeiba"))
import pymysql
from backend.config import settings

BROKEN = ["2026061409030411", "2026060705030211"]  # Takarazuka, Yasuda 2026

def clean(s):
    return bool(s and not re.search(r"[?@�]", str(s)) and re.search(r"[ぁ-ヿ一-龥]", str(s)))

def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def main():
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        print("== 1. raw values, broken races ==")
        for rid in BROKEN:
            q.execute(
                "SELECT UMABAN, BAMEI, CORNER1_JUNI, CORNER2_JUNI, CORNER3_JUNI, CORNER4_JUNI, "
                "NYUSEN_JUNI, KAKUTEI_CHAKUJUN, KOHAN_3F, IJO_KUBUN_CODE "
                "FROM umagoto_race_joho WHERE RACE_CODE=%s ORDER BY UMABAN LIMIT 5", (rid,))
            rows = q.fetchall()
            print(f"  {rid}: {len(rows)} sample rows")
            for r in rows:
                print("   ", {k: repr(v) for k, v in r.items()})

        print("\n== 2. distinct CORNER4_JUNI values (2025+, top 30) ==")
        q.execute(
            "SELECT CORNER4_JUNI AS v, COUNT(*) AS n FROM umagoto_race_joho "
            "WHERE KAISAI_NEN>='2025' GROUP BY CORNER4_JUNI ORDER BY n DESC LIMIT 30")
        for r in q.fetchall():
            print(f"   {repr(r['v'])}: {r['n']}")

        print("\n== 3. build_forecasts funnel (JRA flat 2020-2026) ==")
        q.execute(
            "SELECT r.RACE_CODE, r.KAISAI_NEN, r.KAISAI_GAPPI, r.SHUSSO_TOSU, r.KYOSOMEI_HONDAI, "
            "u.UMABAN, u.BAMEI, u.IJO_KUBUN_CODE, u.CORNER1_JUNI, u.KOHAN_3F, r.KYORI "
            "FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE) "
            "WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026' "
            "AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'")
        rows = q.fetchall()
        print(f"   joined rows: {len(rows)}")
        from datetime import date
        today = date.today().strftime("%Y%m%d")
        rows = [r for r in rows if (r["KAISAI_NEN"] + r["KAISAI_GAPPI"]) < today]
        print(f"   after date < today: {len(rows)}")
        bad_horse = [r for r in rows if not clean(r["BAMEI"])]
        print(f"   garbled BAMEI rows: {len(bad_horse)}")
        if bad_horse:
            venues = Counter(r["RACE_CODE"][8:10] for r in bad_horse)
            print(f"     by venue code in RACE_CODE: {dict(venues.most_common(10))}")
            print(f"     sample: {[r['BAMEI'] for r in bad_horse[:5]]}")
        rows = [r for r in rows if clean(r["BAMEI"])]
        bad_race = [r for r in rows if not clean(r["KYOSOMEI_HONDAI"])]
        print(f"   garbled KYOSOMEI rows: {len(bad_race)}")
        rows = [r for r in rows if clean(r["KYOSOMEI_HONDAI"])]
        def in_range(r):
            kyori, tosus, umaban = number(r["KYORI"]), number(r["SHUSSO_TOSU"]), number(r["UMABAN"])
            return (kyori is not None and 1000 <= kyori <= 3600 and tosus is not None
                    and 5 <= tosus <= 18 and umaban is not None and 1 <= umaban <= 18)
        kept = [r for r in rows if in_range(r)]
        print(f"   after numeric ranges: {len(kept)} (dropped {len(rows) - len(kept)})")
        rows = kept
        kept = [r for r in rows if str(r["IJO_KUBUN_CODE"] or "").strip() in ("", "0", "3", "4", "5", "6", "7")]
        print(f"   after IJO filter: {len(kept)} (dropped {len(rows) - len(kept)})")
        rows = kept
        by_race = {}
        for r in rows:
            by_race.setdefault(r["RACE_CODE"], []).append(r)
        complete = {rid: rs for rid, rs in by_race.items()
                    if len({str(x["UMABAN"]) for x in rs}) == int(number(rs[0]["SHUSSO_TOSU"]) or -1)}
        n_complete = sum(len(rs) for rs in complete.values())
        print(f"   after whole-field rule: {n_complete} rows / {len(complete)} races "
              f"(dropped {len(rows) - n_complete} rows / {len(by_race) - len(complete)} races)")
        dropped_races = [(rid, len(rs), rs[0]["SHUSSO_TOSU"]) for rid, rs in by_race.items() if rid not in complete]
        if dropped_races:
            diff = Counter((n - int(number(t) or 0)) for _, n, t in dropped_races)
            print(f"     dropped-race (actual rows - declared field) distribution: {dict(diff.most_common(10))}")
        no_target = sum(1 for rs in complete.values() for r in rs
                        if not (number(r["CORNER1_JUNI"]) and 1 <= number(r["CORNER1_JUNI"]) <= 18)
                        and not (number(r["KOHAN_3F"]) and 280 <= number(r["KOHAN_3F"]) <= 500))
        print(f"   complete-race rows with neither corner1 nor last3 usable: {no_target}")
    c.close()
    print("\nDone. Share this output back for analysis.")

if __name__ == "__main__":
    main()
