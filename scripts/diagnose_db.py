"""Diagnose mykeibadb data quality (read-only).

Run this on the machine that has mykeibadb / the mykeiba backend, e.g.:
    set KEIBA_SOURCE=C:/Users/jouza/Downloads/keiba-yosou/mykeiba
    python scripts/diagnose_db.py > outputs/db_diagnosis.txt

It never writes to the DB. It reports:
  1. Connection / table character sets (JV-Data is CP932; a charset
     mismatch on import or on the pymysql connection is the usual
     cause of mojibake turning into '?' / U+FFFD in horse/race names).
  2. Per month (YYYYMM from RACE_CODE): how many Grade-A flat races,
     how many horses have garbled names, missing corner ranks,
     missing finish, and field-count mismatches. This shows whether
     the corruption is bounded to a period (e.g. 2026-06 onward, as
     the README suspects) or spread throughout.
"""
import os, sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.environ.get("KEIBA_SOURCE", "C:/Users/jouza/Downloads/keiba-yosou/mykeiba"))
import pymysql
from backend.config import settings

GARBLED = "(u.BAMEI LIKE '%?%' OR u.BAMEI LIKE '%�%' OR u.BAMEI = '' OR u.BAMEI IS NULL)"

def main():
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        print("== character sets ==")
        q.execute("SHOW VARIABLES LIKE 'character_set%'")
        for row in q.fetchall():
            print(" ", row)
        q.execute(
            "SELECT TABLE_NAME, TABLE_COLLATION FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN ('race_shosai','umagoto_race_joho')"
        )
        for row in q.fetchall():
            print(" ", row)
        q.execute(
            "SELECT COLUMN_NAME, CHARACTER_SET_NAME, COLLATION_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME='umagoto_race_joho' AND COLUMN_NAME='BAMEI'"
        )
        for row in q.fetchall():
            print(" ", row)

        print("\n== per-month quality (Grade-A flat, venues 05-09, 2020+) ==")
        q.execute(
            "SELECT SUBSTRING(r.RACE_CODE,1,6) AS ym, COUNT(DISTINCT r.RACE_CODE) AS races, "
            "COUNT(u.UMABAN) AS horses, "
            f"SUM(CASE WHEN {GARBLED} THEN 1 ELSE 0 END) AS garbled_names, "
            "SUM(CASE WHEN u.CORNER4_JUNI IS NULL OR u.CORNER4_JUNI='' OR u.CORNER4_JUNI='0' THEN 1 ELSE 0 END) AS missing_c4, "
            "SUM(CASE WHEN u.NYUSEN_JUNI IS NULL OR u.NYUSEN_JUNI='' OR u.NYUSEN_JUNI='0' THEN 1 ELSE 0 END) AS missing_finish, "
            "SUM(CASE WHEN u.KOHAN_3F IS NULL OR u.KOHAN_3F='' OR u.KOHAN_3F='0' THEN 1 ELSE 0 END) AS missing_last3 "
            "FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE) "
            "WHERE BINARY r.GRADE_CODE='A' AND r.KEIBAJO_CODE IN ('05','06','07','08','09') "
            "AND r.TRACK_CODE BETWEEN '10' AND '29' AND r.KAISAI_NEN>='2020' "
            "GROUP BY ym ORDER BY ym"
        )
        rows = q.fetchall()
        for row in rows:
            h = int(row["horses"] or 0)
            pct = lambda v: f"{100 * int(v or 0) / h:.1f}%" if h else "-"
            print(f"  {row['ym']}: races={row['races']} horses={h} "
                  f"garbled={row['garbled_names']} ({pct(row['garbled_names'])}) "
                  f"missingC4={row['missing_c4']} ({pct(row['missing_c4'])}) "
                  f"missingFinish={row['missing_finish']} ({pct(row['missing_finish'])}) "
                  f"missingLast3={row['missing_last3']} ({pct(row['missing_last3'])})")

        print("\n== field-count mismatches per month ==")
        q.execute(
            "SELECT SUBSTRING(r.RACE_CODE,1,6) AS ym, COUNT(*) AS mismatched_races "
            "FROM race_shosai r JOIN (SELECT RACE_CODE, COUNT(*) AS n FROM umagoto_race_joho GROUP BY RACE_CODE) u "
            "USING(RACE_CODE) "
            "WHERE BINARY r.GRADE_CODE='A' AND r.KEIBAJO_CODE IN ('05','06','07','08','09') "
            "AND r.TRACK_CODE BETWEEN '10' AND '29' AND r.KAISAI_NEN>='2020' "
            "AND CAST(r.SHUSSO_TOSU AS UNSIGNED) <> u.n GROUP BY ym ORDER BY ym"
        )
        for row in q.fetchall():
            print(" ", row)

        print("\n== sample garbled rows (first 10) ==")
        q.execute(
            f"SELECT u.RACE_CODE, u.UMABAN, u.BAMEI FROM umagoto_race_joho u "
            f"WHERE {GARBLED} AND u.KAISAI_NEN>='2025' LIMIT 10"
        )
        for row in q.fetchall():
            print(" ", json.dumps(row, ensure_ascii=False, default=str))
    c.close()
    print("\nDone. Share outputs/db_diagnosis.txt back for analysis.")

if __name__ == "__main__":
    main()
