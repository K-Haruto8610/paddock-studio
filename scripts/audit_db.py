import sys
import pymysql
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'C:/Users/jouza/Downloads/keiba-yosou/mykeiba')
from backend.config import settings
c = pymysql.connect(**settings.pymysql_kwargs)
with c.cursor() as q:
    for sql in ["SELECT r.RACE_CODE,r.KYOSOMEI_HONDAI,r.SHUSSO_TOSU,COUNT(u.UMABAN) FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE) WHERE BINARY r.GRADE_CODE='A' AND r.KEIBAJO_CODE IN ('05','06','07','08','09') AND r.TRACK_CODE BETWEEN '10' AND '29' AND r.KAISAI_NEN>='2025' GROUP BY r.RACE_CODE ORDER BY r.RACE_CODE DESC LIMIT 20", "SELECT BAMEI,BLINKER_SHIYO_KUBUN,MOSHOKU_CODE,CORNER4_JUNI FROM umagoto_race_joho WHERE KAISAI_NEN='2026' LIMIT 20"]:
        q.execute(sql)
        print(q.fetchall())
