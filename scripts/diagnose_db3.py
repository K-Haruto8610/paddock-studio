"""Diagnose_db3 (read-only): what exactly fails the race-name filter?

diagnose_db2 attributed 68,231 dropped rows to KYOSOMEI_HONDAI failing
clean(), but clean() fails for three very different reasons:
  A. blank / spaces only  -> the race is simply unnamed in JV-Data
     (ordinary races). pace_model.py deliberately trains on named
     races only, so these SHOULD stay excluded.
  B. contains '?' or U+FFFD -> true mojibake in race_shosai. These are
     named races lost to an import-encoding bug and should be saved.
  C. text present but no Japanese character (e.g. romaji / digits).
This script counts distinct failing race names by category and year,
and prints samples of category B, so the fix can target B only.
"""
import os, sys, re
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.environ.get("KEIBA_SOURCE", "C:/Users/jouza/Downloads/keiba-yosou/mykeiba"))
import pymysql
from backend.config import settings

JP = re.compile(r"[ぁ-ヿ一-龥]")

def category(s):
    s = str(s or "")
    if not s.strip():
        return "A blank"
    if "?" in s or "�" in s:
        return "B mojibake"
    if not JP.search(s):
        return "C no-Japanese"
    if "@" in s:
        return "B mojibake"  # '@' is the surviving trail byte of mangled CP932
    return "ok"

def main():
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute(
            "SELECT KYOSOMEI_HONDAI AS name, KAISAI_NEN AS y, COUNT(*) AS rows_n, COUNT(DISTINCT RACE_CODE) AS races "
            "FROM race_shosai WHERE KAISAI_NEN BETWEEN '2020' AND '2026' "
            "AND KEIBAJO_CODE BETWEEN '01' AND '10' AND TRACK_CODE BETWEEN '10' AND '29' "
            "GROUP BY KYOSOMEI_HONDAI, KAISAI_NEN")
        distinct = q.fetchall()
    c.close()
    by_cat = Counter()
    by_cat_year = Counter()
    samples_b = []
    rows_by_cat = Counter()
    for r in distinct:
        cat = category(r["name"])
        by_cat[cat] += int(r["races"])
        by_cat_year[(cat, r["y"])] += int(r["races"])
        rows_by_cat[cat] += int(r["rows_n"])
        if cat == "B mojibake" and len(samples_b) < 15:
            samples_b.append(r["name"])
    print("distinct races by category:")
    for cat, n in by_cat.most_common():
        print(f"  {cat}: {n} races, {rows_by_cat[cat]} race_shosai rows")
    print("\nraces by category x year:")
    for (cat, y), n in sorted(by_cat_year.items()):
        print(f"  {cat} {y}: {n}")
    print("\ncategory B samples (true mojibake race names):")
    for s in samples_b:
        print("  ", repr(s))
    print("\nDone. Share this output back for analysis.")

if __name__ == "__main__":
    main()
