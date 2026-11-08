"""Read-only local DB -> chronological race-flow models -> public forecast snapshot.
No credentials or connection details are written into site artifacts.
"""
import os, sys, json, re, argparse
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
import pymysql
from lightgbm import LGBMRegressor

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('KEIBA_SOURCE', 'C:/Users/jouza/Downloads/keiba-yosou/mykeiba'))
from backend.config import settings
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_master

COATS = {'01':'栗毛','02':'栃栗毛','03':'鹿毛','04':'黒鹿毛','05':'青鹿毛','06':'青毛','07':'芦毛','08':'栗粕毛','09':'鹿粕毛','10':'青粕毛','11':'白毛'}

def number(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan

def clean(s):
    return bool(s and not re.search(r'[?@�]', str(s)) and re.search(r'[ぁ-ヿ一-龥]', str(s)))

RESULTS_DIR = ROOT / 'data' / 'results'


def load_upcoming_entries(entries_dir, today_s):
    """Latest snapshots of declared fields for GI races that have not been run yet, plus run races whose
    result is only in data/results (DB rows unusable): their pre-race forecast is kept for comparison.
    Features always use starts strictly before the race date, never the result file."""
    out = []
    for p in sorted(Path(entries_dir).glob('*.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        kept = (RESULTS_DIR / f"{s.get('raceId')}.json").exists()
        if s.get('status') == 'declared' and (s['race']['date'] >= today_s or kept):
            out.append(s)
    return sorted(out, key=lambda s: s['race']['date'])

def upcoming_features(snap, history, final, features, past_targets):
    """Feature rows for the active (non-scratched) field of an upcoming race. Same definitions as
    training: strictly prior starts only, positions normalised by the actual field size."""
    race = snap['race']
    active = [h for h in snap['horses'] if h['status'] != 'scratched']
    field = len(active)
    rows = []
    for h in active:
        past = [v for v in history.get(h['horseId'], []) if v['date'] < race['date']]
        row = {'distance': race['distance'], 'surface': int(race['surface'] == 'ダート'), 'venue': int(race['venueCode']),
               'gate': h['number'] / field, 'weight': h['weight'], 'age': h['age'], 'field': field,
               'blinker': np.nan if h['blinker'] is None else int(h['blinker']), 'starts': len(past)}
        for t in past_targets:
            vals = [v[t] for v in past[-5:]]
            row['past_' + t] = float(pd.Series(vals, dtype=float).mean()) if vals else np.nan
        rows.append(row)
    x = pd.DataFrame(rows)
    x['front_pressure'] = x.past_c4.lt(.3).mean()
    return active, past, x[features], {t: final[t].predict(x[features]) for t in final}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--today', help='YYYY-MM-DD; rows dated on/after it are not treated as history (default: today)')
    ap.add_argument('--entries-dir', default=str(ROOT / 'data' / 'entries'))
    ap.add_argument('--out', default=str(ROOT / 'public' / 'data' / 'forecasts.json'))
    args = ap.parse_args()
    today_s = (args.today or date.today().isoformat()).replace('-', '')
    master = g1_master.load()
    c = pymysql.connect(**settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute("""SELECT r.RACE_CODE,r.KAISAI_NEN,r.KAISAI_GAPPI,r.KEIBAJO_CODE,r.KYOSOMEI_HONDAI,
          r.GRADE_CODE,r.KYORI,r.TRACK_CODE,r.COURSE_KUBUN,r.SHUSSO_TOSU,r.ZENHAN_3F,
          u.UMABAN,u.WAKUBAN,u.KETTO_TOROKU_BANGO,u.BAMEI,u.MOSHOKU_CODE,u.BLINKER_SHIYO_KUBUN,
          u.BAREI,u.FUTAN_JURYO,u.KISHUMEI_RYAKUSHO,u.CORNER1_JUNI,u.CORNER2_JUNI,
          u.CORNER3_JUNI,u.CORNER4_JUNI,u.KOHAN_3F,u.IJO_KUBUN_CODE
          FROM race_shosai r JOIN umagoto_race_joho u USING(RACE_CODE)
          WHERE r.KAISAI_NEN BETWEEN '2020' AND '2026'
          AND r.KEIBAJO_CODE BETWEEN '01' AND '10' AND r.TRACK_CODE BETWEEN '10' AND '29'
          ORDER BY r.KAISAI_NEN,r.KAISAI_GAPPI,r.RACE_CODE,u.UMABAN""")
        rows = q.fetchall()
    c.close()
    d = pd.DataFrame(rows)
    d['date'] = d.KAISAI_NEN + d.KAISAI_GAPPI
    # Rows dated today or later are pre-race cards (no results): never history, never training rows.
    d = d[d.date < today_s]
    d = d[d.BAMEI.map(clean) & d.KYOSOMEI_HONDAI.map(clean)].copy()
    numeric = ['KYORI','TRACK_CODE','SHUSSO_TOSU','UMABAN','WAKUBAN','BAREI','FUTAN_JURYO','ZENHAN_3F','KOHAN_3F'] + [f'CORNER{i}_JUNI' for i in range(1,5)]
    for col in numeric: d[col] = d[col].map(number)
    d = d[(d.KYORI.between(1000,3600)) & d.SHUSSO_TOSU.between(5,18) & d.UMABAN.between(1,18)]
    d = d[d.IJO_KUBUN_CODE.fillna('0').str.strip().isin(['','0','3','4','5','6','7'])]
    # Reject incomplete fields: positions and competition features need the whole field.
    counts = d.groupby('RACE_CODE').UMABAN.transform('nunique')
    d = d[counts == d.SHUSSO_TOSU].drop_duplicates(['RACE_CODE','UMABAN']).reset_index(drop=True)
    d['surface'] = (d.TRACK_CODE >= 23).astype(int)
    d['venue'] = d.KEIBAJO_CODE.astype(int)
    d['gate'] = d.UMABAN / d.SHUSSO_TOSU
    d['blinker'] = d.BLINKER_SHIYO_KUBUN.map({'0':0,'1':1})
    d['weight'] = d.FUTAN_JURYO / 10
    d['age'] = d.BAREI
    d['distance'] = d.KYORI
    d['field'] = d.SHUSSO_TOSU
    targets = []
    for i in range(1,5):
        t = f'c{i}'
        d[t] = ((d[f'CORNER{i}_JUNI'] - 1) / (d.field-1)).where(d[f'CORNER{i}_JUNI'].between(1,d.field))
        targets.append(t)
    d['last3'] = (d.KOHAN_3F/10).where(d.KOHAN_3F.between(280,500))
    targets.append('last3')
    features = ['distance','surface','venue','gate','weight','age','field','blinker']
    # Strictly prior starts only; current-race result columns never enter features.
    d = d.sort_values(['date','RACE_CODE','UMABAN']).reset_index(drop=True)
    for t in targets:
        col = 'past_' + t
        d[col] = d.groupby('KETTO_TOROKU_BANGO')[t].transform(lambda s: s.shift().rolling(5,min_periods=1).mean())
        features.append(col)
    d['starts'] = d.groupby('KETTO_TOROKU_BANGO').cumcount()
    d['front_pressure'] = d.past_c4.lt(.3).groupby(d.RACE_CODE).transform('mean')
    features += ['starts','front_pressure']
    tr = d.date < '20250901'
    va = d.date.between('20250901','20251231')
    print(d.groupby('KAISAI_NEN')[targets].count().to_string(),flush=True)
    report = {'features':features,'trainingEnd':'2025-08-31','validation':'2025-09〜12','inferenceTrainingEnd':'2025-12-31','rows':len(d),'races':d.RACE_CODE.nunique(),'excludedRows':len(rows)-len(d),'metrics':{},'notes':['過去5走の馬柱を使用。オッズ・人気・払戻は特徴量に含めない。','位置取りは通過順位を頭数で正規化して学習。馬間距離・内外の進路は演出用の補間。','2025年9〜12月を時系列検証に使用。掲載する2026年の対象レース結果は学習に含めない。','欠損・文字化け・出走頭数の不一致を検出したレースは掲載対象外。']}
    (ROOT/'models').mkdir(exist_ok=True)
    final = {}
    for t in targets:
        fit = tr & d[t].notna()
        test = va & d[t].notna()
        if fit.sum()<100 or test.sum()<20: raise RuntimeError('Insufficient chronological data: '+t)
        model = LGBMRegressor(n_estimators=140,max_depth=5,num_leaves=20,learning_rate=.04,verbosity=-1,n_jobs=4,random_state=42)
        model.fit(d.loc[fit,features],d.loc[fit,t])
        pred = model.predict(d.loc[test,features])
        baseline = d.loc[test,'past_'+t].fillna(d.loc[fit,t].median())
        report['metrics'][t] = {'mae':float(np.mean(np.abs(pred-d.loc[test,t]))),'baselineMae':float(np.mean(np.abs(baseline-d.loc[test,t]))),'validationRows':int(test.sum())}
        fit = (d.date<'20260101') & d[t].notna()
        model.fit(d.loc[fit,features],d.loc[fit,t])
        model.booster_.save_model(str(ROOT/'models'/f'{t}.txt'))
        final[t] = model
        d['pred_'+t] = model.predict(d[features])
    races=[]
    coats=COATS
    venues={r['venueCode']:r['venue'] for r in master['races']}
    cand = d[(d.GRADE_CODE=='A') & (d.date>='20260101') & d.KEIBAJO_CODE.isin(venues)]
    # GI selection: grade A AND a master entry agreeing on name, venue, distance and track code.
    keep, skipped = [], []
    for code,g in cand.groupby('RACE_CODE',sort=False):
        r=g.iloc[0]
        if g1_master.match_db_race(master,r.KYOSOMEI_HONDAI,r.KEIBAJO_CODE,int(r.distance),int(r.TRACK_CODE),r.KAISAI_NEN): keep.append(code)
        else: skipped.append(f"{r.date} {r.KYOSOMEI_HONDAI.strip()} (venue {r.KEIBAJO_CODE} {int(r.distance)}m track {int(r.TRACK_CODE)})")
    if skipped: print('WARNING grade-A races not in g1_races.json (excluded):',skipped)
    latest = cand[cand.RACE_CODE.isin(keep)]
    history={}
    for row in d.to_dict('records'): history.setdefault(row['KETTO_TOROKU_BANGO'],[]).append(row)
    for code,g in latest.groupby('RACE_CODE',sort=False):
        r=g.iloc[0]
        horses=[]
        for _,h in g.iterrows():
            past=[v for v in history[h.KETTO_TOROKU_BANGO] if v['date']<h.date][-5:][::-1]
            def val(v): return None if pd.isna(v) else round(float(v),3)
            horses.append({'number':int(h.UMABAN),'gate':int(h.WAKUBAN),'name':h.BAMEI.strip(),'coat':coats.get(h.MOSHOKU_CODE,'不明'),'coatCode':h.MOSHOKU_CODE,'blinker':None if pd.isna(h.blinker) else bool(h.blinker),'jockey':h.KISHUMEI_RYAKUSHO.strip(),'age':int(h.age),'weight':float(h.weight),'positions':[round(float(np.clip(h['pred_c'+str(i)],0,1)),4) for i in range(1,5)],'last3':round(float(h.pred_last3),1),'history':[{'date':v['date'],'name':v['KYOSOMEI_HONDAI'].strip(),'distance':int(v['distance']),'field':int(v['field']),'corners':[val(v[f'CORNER{i}_JUNI']) for i in range(1,5)],'last3':val(v['last3'])} for v in past]})
        races.append({'id':code,'date':r.date,'name':r.KYOSOMEI_HONDAI.strip(),'venue':venues[r.KEIBAJO_CODE],'venueCode':r.KEIBAJO_CODE,'distance':int(r.distance),'surface':'ダート' if r.surface else '芝','trackCode':int(r.TRACK_CODE),'course':str(r.COURSE_KUBUN).strip(),'horses':horses,'historical':True})
    races = races[::-1]
    upcoming = []
    for snap in load_upcoming_entries(args.entries_dir, today_s):
        race = snap['race']
        active, _, x, pred = upcoming_features(snap, history, final, features, targets)
        horses = []
        for i,h in enumerate(active):
            past=[v for v in history.get(h['horseId'],[]) if v['date']<race['date']][-5:][::-1]
            def val(v): return None if pd.isna(v) else round(float(v),3)
            horses.append({'number':h['number'],'gate':h['gate'],'name':h['name'],'coat':coats.get(past[0]['MOSHOKU_CODE'],'不明') if past else '不明','coatCode':past[0]['MOSHOKU_CODE'] if past else '',
              'blinker':h['blinker'],'jockey':h['jockey'],'age':h['age'],'weight':h['weight'],
              'positions':[round(float(np.clip(pred['c'+str(k)][i],0,1)),4) for k in range(1,5)],'last3':round(float(pred['last3'][i]),1),
              'history':[{'date':v['date'],'name':v['KYOSOMEI_HONDAI'].strip(),'distance':int(v['distance']),'field':int(v['field']),'corners':[val(v[f'CORNER{k}_JUNI']) for k in range(1,5)],'last3':val(v['last3'])} for v in past]})
        scratched=[{'number':h['number'],'name':h['name'],'reason':h.get('statusReason')} for h in snap['horses'] if h['status']=='scratched']
        upcoming.append({'id':snap['raceId'],'date':race['date'],'name':race['name'],'venue':race['venue'],'venueCode':race['venueCode'],'distance':race['distance'],
          'surface':race['surface'],'trackCode':race['trackCode'],'course':'','horses':horses,'historical':False,
          'entries':{'source':snap['source'],'fetchedAt':snap['fetchedAt'],'status':snap['status'],'scratched':scratched,'changes':snap.get('changes',[])[-12:]}})
    if upcoming: print('upcoming races:',[(u['name'],len(u['horses'])) for u in upcoming])
    races = upcoming + races
    report['publishedRaces']=len(races)
    out={'generatedAt':datetime.now().isoformat(timespec='seconds'),'model':report,'races':races}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (ROOT/'models'/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

