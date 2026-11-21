"""v3 S0 check: every course against the official lap / straight, and every training race against the corners JRA recorded.

  python -I scripts/v3/check_courses.py [frame.pkl]     -> outputs/v3/s0_courses.json

(1) lap and home straight of each course's profile vs SPECS (the official figures), total turn = 2 pi;
(2) for every field-complete race of the frame: the set of corners the profile says the race passes (the last pass of
    each corner, MIN_AFTER_START metres or more after the start) vs the corners whose ranks JRA recorded (r1..r4).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import courses as cs  # noqa: E402

ROOT = cs.ROOT
OUT = ROOT / 'outputs' / 'v3'


def main():
    frame = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'outputs/data-expansion-v20/frame_2020.pkl'
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ok = [], True
    for v, spec in cs.SPECS.items():
        for kind in ('turf', 'outer', 'dirt'):
            if kind not in spec:
                continue
            c = cs.course(v, kind)
            r = {'venue': v, 'name': spec['name'], 'kind': kind, 'source': c.source, 'lap': c.lap, 'profileLap': len(c.curv),
                 'straight': c.straight, 'profileStraight': c.measured_straight, 'turnDeg': np.degrees(c.total_turn),
                 'corners': [round(float(s), 1) for s in c.corner_s]}
            r['lapOk'] = abs(r['profileLap'] - c.lap) <= .005 * c.lap  # the app's chains close to within 0.3%
            r['straightOk'] = abs(r['profileStraight'] - c.straight) <= 15  # half-degree point lies inside the 25 m easing spiral
            ok &= r['lapOk'] and r['straightOk']
            rows.append(r)
            print(f"{v} {spec['name']} {kind:5} {c.source:9} lap {c.lap:7.1f}/{len(c.curv):5d}  straight {c.straight:6.1f}/"
                  f"{r['profileStraight']:6.1f}  turn {r['turnDeg']:6.1f}  corners {r['corners']}")

    d = pd.read_pickle(frame)
    races = d[d.complete].groupby('RACE_CODE').agg(venue=('venue', 'first'), track=('track', 'first'), distance=('distance', 'first'),
                                                   **{f'has{c}': (f'r{c}', lambda s: bool(s.notna().all())) for c in (1, 2, 3, 4)})
    races['recorded'] = races[[f'has{c}' for c in (1, 2, 3, 4)]].apply(lambda r: tuple(c for c in (1, 2, 3, 4) if r[f'has{c}']), axis=1)
    best, chosen = None, cs.MIN_AFTER_START
    for gap in (0, 100, 150, 200, 250, 300, 350, 400):
        cs.MIN_AFTER_START = gap
        pred = races.apply(lambda r: cs.race_course(r.venue, r.track, r.distance).reported, axis=1)
        # A race whose recorded set is empty or a single corner tells nothing (straight courses, missing strings).
        use = races.recorded.map(len) >= 2
        rate = float((pred[use] == races.recorded[use]).mean())
        print(f'min metres after the start {gap:3d}: corner sets match {rate:.4f} of {int(use.sum())} races')
        if best is None or rate > best[1]:
            best = (gap, rate)
    cs.MIN_AFTER_START = chosen
    races['pred'] = races.apply(lambda r: cs.race_course(r.venue, r.track, r.distance).reported, axis=1)
    use = races.recorded.map(len) >= 2
    match = races.pred == races.recorded
    # recorded corners a strict subset of the profile's: a passing string missing / rejected in that race, not geometry
    subset = races.apply(lambda r: set(r.recorded) < set(r.pred), axis=1)
    best = (chosen, float(match[use].mean()))
    print(f'with the chute rules and {chosen:.0f} m: match {best[1]:.4f}, recorded a strict subset (missing strings) '
          f'{float(subset[use].mean()):.4f}, other {float((~match & ~subset)[use].mean()):.4f}')
    races['kind'] = races.apply(lambda r: cs.kind_of(r.venue, r.track), axis=1)
    races['source'] = races.apply(lambda r: 'straight' if r.kind == 'straight' else cs.course(r.venue, r.kind).source, axis=1)
    bad = races[use & ~match & ~subset]
    by_course = {k: {'races': len(g), 'recorded': str(g.recorded.mode().iloc[0]), 'profile': str(g.pred.iloc[0])}
                 for k, g in bad.groupby(['venue', 'track', 'distance', 'source'])}
    print('\nmismatching courses:')
    for k, v in by_course.items():
        print(' ', k, v)
    straight = races[races.kind == 'straight']
    res = {'courses': rows, 'allLapsAndStraightsOk': bool(ok), 'minAfterStart': best[0],
           'cornerSetMatch': {'races': int(use.sum()), 'match': best[1],
                              'bySource': races[use].assign(m=races.pred == races.recorded).groupby('source').m.agg(['mean', 'size']).to_dict('index')},
           'recordedSubset': float(subset[use].mean()), 'geometryMismatch': float((~match & ~subset)[use].mean()),
           'mismatches': [{'venue': int(k[0]), 'track': int(k[1]), 'distance': float(k[2]), 'source': k[3], **v} for k, v in by_course.items()],
           'straightRacesWithoutCorners': int((straight.recorded.map(len) == 0).sum()), 'straightRaces': int(len(straight)),
           'everyRaceHasACourse': True}
    (OUT / 's0_courses.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('allLapsAndStraightsOk', 'minAfterStart', 'cornerSetMatch')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
