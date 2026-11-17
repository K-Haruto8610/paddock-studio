"""Export observed results separately from forecasts. Read-only; never trains a model."""
import os,sys,json
from pathlib import Path
from datetime import datetime,timezone
import pymysql
ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,os.environ.get('KEIBA_SOURCE','C:/Users/jouza/Downloads/keiba-yosou/mykeiba'))
from backend.config import settings

def positive(v,maximum=18):
    s=str(v or '').strip()
    return int(s) if s.isdigit() and 1<=int(s)<=maximum else None

def export():
    forecasts=json.loads((ROOT/'public/data/forecasts.json').read_text(encoding='utf-8'))
    c=pymysql.connect(**settings.pymysql_kwargs,cursorclass=pymysql.cursors.DictCursor)
    replays={}
    with c.cursor() as q:
        for race in forecasts['races']:
            if race.get('historical') is False: continue  # upcoming race: no result exists
            q.execute('''SELECT UMABAN,BAMEI,CORNER1_JUNI,CORNER2_JUNI,CORNER3_JUNI,CORNER4_JUNI,
                NYUSEN_JUNI,KAKUTEI_CHAKUJUN,KOHAN_3F,IJO_KUBUN_CODE FROM umagoto_race_joho WHERE RACE_CODE=%s''',(race['id'],))
            found={positive(row['UMABAN']):row for row in q.fetchall()}
            horses={}
            for h in race['horses']:
                row=found.get(h['number'])
                if not row or row['BAMEI'].strip()!=h['name']:
                    raise ValueError('Replay identity mismatch: '+race['id']+' horse '+str(h['number']))
                n=len(race['horses'])
                last=positive(row['KOHAN_3F'],999)
                horses[str(h['number'])]={'corners':[positive(row[f'CORNER{i}_JUNI'],n) for i in range(1,5)],
                    'finish':positive(row['NYUSEN_JUNI'],n),'officialFinish':positive(row['KAKUTEI_CHAKUJUN'],n),
                    'last3':last/10 if last else None,'abnormality':str(row['IJO_KUBUN_CODE'] or '').strip()}
            # A corner with no entries is absent on this route, not four fabricated zero ranks.
            available=[i+1 for i in range(4) if all(h['corners'][i] is not None for h in horses.values())]
            missing=[i+1 for i in range(4) if any(h['corners'][i] is not None for h in horses.values()) and i+1 not in available]
            full_finish=all(h['finish'] is not None for h in horses.values())
            if not available or not full_finish:
                print(f"WARNING partial result, replay skipped: {race['id']} {race['name']} corners={available} finish={full_finish}")
                continue
            replays[race['id']]={'horses':horses,'corners':available,'partialCorners':missing,'completeFinish':full_finish,
                'source':'mykeibadb.umagoto_race_joho (JV-Data)','raceDate':race['date']}
            changes=[h['number'] for h in race['horses'] if len(set(v for v in horses[str(h['number'])]['corners']+[horses[str(h['number'])]['finish']] if v is not None))>1]
            print(f"{race['name']}: corners={available} finish={full_finish}, changing horses={len(changes)}/{len(horses)}")
            if race==forecasts['races'][0]:
                for h in race['horses'][:5]:print(h['number'],h['name'],horses[str(h['number'])])
    c.close()
    # Results entered by hand (data/results/<race id>.json) for races whose DB rows are unusable.
    for race in forecasts['races']:
        p=ROOT/'data/results'/f"{race['id']}.json"
        if race['id'] in replays or not p.exists(): continue
        manual=json.loads(p.read_text(encoding='utf-8'))
        if sorted(manual['horses'],key=int)!=sorted((str(h['number']) for h in race['horses']),key=int):
            raise ValueError('Manual result field mismatch: '+race['id'])
        horses=manual['horses']
        available=[i+1 for i in range(4) if all(h['corners'][i] is not None for h in horses.values())]
        replays[race['id']]={'horses':horses,'corners':available,'partialCorners':[],
            'completeFinish':all(h['finish'] is not None for h in horses.values()),'source':manual['source'],'raceDate':manual['raceDate']}
        print(f"{race['name']}: manual result, corners={available}")
    output={'generatedAt':datetime.now(timezone.utc).isoformat(),'races':replays,
        'note':'Observed corner positions and arrival order. Between-checkpoint motion, gaps and lanes are interpolated, not recorded tracking coordinates.'}
    (ROOT/'public/data/replays.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':export()
