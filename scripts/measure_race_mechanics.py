"""Measure race mechanics in the cached frame (read-only), as evidence for docs/race-simulator-design.md.

M1 finish gap vs gap at the 600 m mark and closing 3F; M2 pace x position on the closing 3F; M3 lane at the 4th
corner vs closing 3F (within horse); M4 late / early speed ratio; M5 gate vs first-corner position (within horse);
M6 persistence of running style. Run: python -I scripts/measure_race_mechanics.py [frame.pkl]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
frame = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'outputs/data-expansion-v20/frame_2020.pkl'
d = pd.read_pickle(frame)
d = d[d.complete].copy()
d['t600'] = d.soha - d.last3  # the horse's own time at the 600 m-to-go mark
d = d[d.t600.notna() & d.last3.notna() & (d.distance > 1000)]
print('rows', len(d), 'races', d.RACE_CODE.nunique())
g = d.groupby('RACE_CODE')

d['ratio'] = (600 / d.last3) / ((d.distance - 600) / d.t600)
d['dband'] = pd.cut(d.distance, [1000, 1400, 1800, 2200, 2600, 3600])
print('\n[M4] median late / early mean speed by surface (0 turf, 1 dirt) and distance')
print(d.groupby(['surface', 'dband'], observed=True).ratio.median().unstack().round(3))

d['l3rel'] = d.last3 - g.last3.transform('median')
d['pos4'] = d.r4.fillna(d.r3)


def front_back(x):
    return pd.Series({'fb': x[x.pos4 <= 1 / 3].l3rel.mean() - x[x.pos4 >= 2 / 3].l3rel.mean(), 'pace': x.pace.iloc[0]})


fb = d.dropna(subset=['pos4', 'pace']).groupby('RACE_CODE').apply(front_back).dropna()
fb['band'] = pd.qcut(fb.pace, 5)
print('\n[M2] closing 3F of the front third minus the back third (s), by early-pace quintile (negative pace = fast)')
print(fb.groupby('band', observed=True).fb.agg(['mean', 'count']).round(3))
print('slope (s per 1 s of pace):', round(np.polyfit(fb.pace, fb.fb, 1)[0], 3))

d['gap600'] = d.t600 - g.t600.transform('min')
d['fin'] = d.soha - g.soha.transform('min')
c = d[['fin', 'gap600', 'l3rel']].dropna()
print('\n[M1] corr(finish gap, gap at 600 m) %.3f, corr(finish gap, closing 3F vs race median) %.3f'
      % (c.fin.corr(c.gap600), c.fin.corr(c.l3rel)))

m = d.dropna(subset=['l4', 'r4']).copy()
m = m[m.groupby('KETTO_TOROKU_BANGO').l4.transform('size') >= 3]
for col in ['l3rel', 'l4', 'r4']:
    m[col + '_w'] = m[col] - m.groupby('KETTO_TOROKU_BANGO')[col].transform('mean')
b = np.linalg.lstsq(np.c_[m.l4_w, m.r4_w, np.ones(len(m))], m.l3rel_w, rcond=None)[0]
print('\n[M3] within horse: closing 3F %+.3f s per lane at the 4th corner (rank term %+.3f), rows %d' % (b[0], b[1], len(m)))
print('     4th-corner lane counts:', m.l4.value_counts().sort_index().to_dict())

s = d.dropna(subset=['r1']).copy()
for col in ['r1', 'gate']:
    s[col + '_w'] = s[col] - s.groupby('KETTO_TOROKU_BANGO')[col].transform('mean')
print('\n[M5] within horse: first-corner position (0..1) per unit of gate (0..1): %+.3f' % np.polyfit(s.gate_w, s.r1_w, 1)[0])
print('[M6] corr(past_r1, r1) %.3f, corr(past_last3, last3) %.3f' % (d.past_r1.corr(d.r1), d.past_last3.corr(d.last3)))
