"""v3 S0: course table for all 10 JRA venues (docs/race-simulator-design.md, S0).

Each course (venue x turf / outer turf / dirt) is a curvature profile along the inner rail, sampled every metre from the
winning post in the race direction. The simulator only needs two things from it:

- how much a horse turns per metre of rail (curvature, signed so the lap turns +2 pi): a horse running w metres outside
  the rail covers (1 + w * curvature) metres for each metre of rail, so the extra distance of running wide is w x (angle turned),
  whatever the radius (M3: the loss is geometric and cannot be learnt from the passing strings);
- where the four corners are, so the simulator's passing times can be read at the points JRA records (1角..4角 are
  where the turn accumulated since the post reaches 1/8, 3/8, 5/8, 7/8 of the lap's total turn, as app/courses.ts).

Sources, in order of fidelity:
  alignment  app/alignmentData.ts (clothoid / arc / line chains fitted to the JRA plans and the official lap and
             straight): Tokyo turf+dirt, Nakayama inner+outer turf, Chukyo turf+dirt, Kyoto inner+outer turf,
             Hanshin inner+outer turf.
  scaled     the venue's inner-turf alignment rescaled to the official lap and straight of a course without its own
             chain (Nakayama / Kyoto / Hanshin dirt).
  stadium    two straights and two equal bends from the official lap and straight only (Sapporo, Hakodate, Fukushima,
             Niigata, Kokura). The bend share of the lap is the mean of the aligned courses. Corner positions there are
             approximate; the check below says how well they reproduce the corners JRA records.

A race of distance D is mapped onto the closed course backwards from the post (the start is D metres before the post
along the final lap's course). Starts in chutes are therefore placed on the oval; inner->outer / outer->inner and
two-lap courses use their final-lap course for the whole race (approximation, recorded in the check).
"""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Official lap and home straight (from the last corner to the post), metres, A course. Venues 05-09 match
# app/courses.ts (retrieved 2026-11-08); 01-04 and 10 from the JRA course pages.
SPECS = {
    '01': {'name': '札幌', 'left': False, 'turf': (1640.9, 266.1), 'dirt': (1487.0, 264.3)},
    '02': {'name': '函館', 'left': False, 'turf': (1626.6, 262.1), 'dirt': (1475.8, 260.3)},
    '03': {'name': '福島', 'left': False, 'turf': (1600.0, 292.0), 'dirt': (1444.6, 295.7)},
    '04': {'name': '新潟', 'left': True, 'turf': (1623.0, 358.7), 'outer': (2223.0, 658.7), 'dirt': (1472.5, 353.9)},
    '05': {'name': '東京', 'left': True, 'turf': (2083.1, 525.9), 'dirt': (1899.0, 501.6)},
    '06': {'name': '中山', 'left': False, 'turf': (1667.1, 310.0), 'outer': (1839.7, 310.0), 'dirt': (1493.0, 308.0)},
    '07': {'name': '中京', 'left': True, 'turf': (1705.9, 412.5), 'dirt': (1530.0, 410.7)},
    '08': {'name': '京都', 'left': False, 'turf': (1782.8, 328.4), 'outer': (1894.3, 403.7), 'dirt': (1607.6, 329.1)},
    '09': {'name': '阪神', 'left': False, 'turf': (1689.0, 356.5), 'outer': (2089.0, 473.6), 'dirt': (1517.6, 352.7)},
    '10': {'name': '小倉', 'left': False, 'turf': (1615.1, 293.0), 'dirt': (1445.4, 291.3)},
}
# JV-Data TRACK_CODE -> course kind of the final lap (app/courses.ts OUTER_TRACK_CODES for the outer turf).
OUTER_TRACK_CODES = (12, 13, 16, 18, 19, 22)
STRAIGHT_TRACK_CODES = (10, 29)


def kind_of(venue, track):
    v = f'{int(venue):02d}'
    if int(track) in STRAIGHT_TRACK_CODES:
        return 'straight'
    if int(track) >= 23:
        return 'dirt'
    return 'outer' if int(track) in OUTER_TRACK_CODES and 'outer' in SPECS[v] else 'turf'


def _alignments():
    """{venue: {course: [(kind, length, kEnd)]}} parsed from app/alignmentData.ts."""
    s = (ROOT / 'app' / 'alignmentData.ts').read_text(encoding='utf-8')
    starts = [(m.group(1), m.start()) for m in re.finditer(r"'(\d\d)': \{\s*scale", s)]
    out = {}
    for i, (v, a) in enumerate(starts):
        block = s[a:starts[i + 1][1] if i + 1 < len(starts) else len(s)]
        block = block[block.index('courses:'):]
        out[v] = {m.group(1): json.loads(m.group(2).replace("'", '"'))
                  for m in re.finditer(r"(\w+): (\[\[.*?\]\])", block)}
    return out


def _curvature_from_chain(chain, step=1.0):
    """Signed curvature sampled every `step` metres along an alignment chain (same element rules as app/alignment.ts)."""
    ks, k = [], 0.0
    for el in chain:
        kind, length = el[0], float(el[1])
        k_end = float(el[2]) if len(el) > 2 else 0.0
        k0 = 0.0 if kind == 'L' else k
        k1 = 0.0 if kind == 'L' else (k if kind == 'A' else k_end)
        n = int(round(length / step))
        if n > 0:
            ks.append(k0 + (k1 - k0) * (np.arange(n) + 0.5) / n)
        k = k1
    k = np.concatenate(ks) if ks else np.zeros(0)
    return k * np.sign(k.sum())  # signed, positive = the lap's turning direction (an S-bend turns back)


def _bends(curv, thresh=1e-3):
    """[(start, end)] runs where |curvature| > thresh (metres from the post)."""
    on = curv > thresh
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def _stadium(lap, straight, bend_share):
    """Two equal bends turning pi each; home straight = official straight; bend share = bends / lap."""
    b = bend_share * lap / 2
    e = (lap - 2 * b) / 2 - straight  # straight run from the post to the first bend
    if e < 0:
        e, b = 0.0, (lap - 2 * straight) / 2
    back = lap - 2 * b - e - straight
    seg = [(e, 0.0), (b, np.pi / b), (back, 0.0), (b, np.pi / b), (straight, 0.0)]
    return np.concatenate([np.full(int(round(n)), k) for n, k in seg])


def _rescale(curv, lap, straight):
    """Rescale a curvature profile to a new lap and home straight, keeping its total turn and the shape of the bends."""
    bends = _bends(curv)
    old_last = bends[-1][1]
    body = curv[:old_last]
    new_body = lap - straight
    x = np.linspace(0, len(body) - 1, int(round(new_body)))
    k = np.interp(x, np.arange(len(body)), body) * (len(body) / new_body)
    return np.r_[k, np.zeros(int(round(straight)))]


@dataclass
class Course:
    venue: str
    kind: str
    lap: float
    straight: float
    source: str
    curv: np.ndarray        # curvature every metre from the post (len ~ lap), positive = the lap's turning direction
    corner_s: np.ndarray    # positions of 1角..4角 from the post (m)

    @property
    def measured_straight(self):
        # From the last point still more than half a degree short of the lap's turn to the post (signed: the small
        # S-bend before the post on Kyoto's outer course is a lane change that turns back, not a corner).
        turned = np.cumsum(self.curv)
        return float(len(self.curv) - 1 - np.flatnonzero(turned < turned[-1] - np.radians(.5))[-1])

    @property
    def total_turn(self):
        return float(self.curv.sum())


@lru_cache(maxsize=None)
def course(venue, kind):
    v = f'{int(venue):02d}'
    spec = SPECS[v]
    lap, straight = spec[kind]
    al = _alignments()
    if v in al and kind in al[v]:
        curv, source = _curvature_from_chain(al[v][kind]), 'alignment'
        curv = curv * (curv.sum() and 2 * np.pi / curv.sum())  # chains close to within a few degrees; make it 2 pi
    elif v in al:
        curv, source = _rescale(_curvature_from_chain(al[v]['turf']), lap, straight), 'scaled'
        curv = curv * 2 * np.pi / curv.sum()
    else:
        curv, source = _stadium(lap, straight, bend_share()), 'stadium'
    turned = np.cumsum(curv)
    corner_s = np.array([np.searchsorted(turned, turned[-1] * k / 8) for k in (1, 3, 5, 7)], float)
    return Course(v, kind, lap, straight, source, curv, corner_s)


@lru_cache(maxsize=None)
def bend_share():
    """Mean share of the lap that is bend (|curvature| > 1e-3) on the aligned courses."""
    shares = []
    for v, courses in _alignments().items():
        for chain in courses.values():
            c = _curvature_from_chain(chain)
            shares.append(sum(b - a for a, b in _bends(c)) / len(c))
    return float(np.mean(shares))


@dataclass
class RaceCourse:
    distance: float
    step: float
    turn: np.ndarray        # radians turned in each step (len = distance / step)
    corner_x: dict          # physical corner (1..4) -> race distance (m from the start) of its LAST pass
    reported: tuple         # corners JRA would report: the last pass of each, those at least MIN_AFTER_START in
    straight: bool


MIN_AFTER_START = 200.0  # chosen by the S0 check (scripts/v3/check_courses.py): JRA does not record a corner the field
#                          reaches within ~200 m of the start
# Starts in chutes that the backwards mapping onto the oval gets wrong (S0 check, 2026-11-22):
#   'straight' the chute runs straight on to the back straight: the bend the oval puts after the start is not run and its
#              corners are not recorded (Kyoto turf 1600 inner / 1800 outer start in the 2nd-corner pocket);
#   'join'     the chute joins the bend part way round: the corner is recorded although it comes early (Tokyo turf 1800).
CHUTES = {('08', 'turf', 1600): 'straight', ('08', 'outer', 1800): 'straight', ('05', 'turf', 1800): 'join'}


def race_course(venue, track, distance, step=10.0):
    """The race on the distance axis: turn per step and where each corner is last passed."""
    n = int(round(distance / step))
    kind = kind_of(venue, track)
    if kind == 'straight':
        return RaceCourse(distance, step, np.zeros(n), {}, (), True)
    c = course(venue, kind)
    L = len(c.curv)
    x = np.arange(int(round(distance)))  # metre marks from the start
    s = ((x - distance) % L).astype(int)  # rail position from the post
    k = c.curv[s].copy()
    chute = CHUTES.get((c.venue, kind, int(distance)))
    if chute == 'straight':
        # zero turn until the end of the first bend the oval mapping puts after the start
        on = k > 1e-3
        end = int(np.argmax(on)) + int(np.argmax(~on[np.argmax(on):]))
        k[:end] = 0.0
    turn = np.add.reduceat(k, np.arange(0, len(k), int(step)))[:n]
    corner_x = {}
    for j, cs in enumerate(c.corner_s, start=1):
        # last x in (0, distance] with s(x) == cs: x = distance - (L - cs) - m L for m >= 0
        last = distance - ((L - cs) % L)
        first_ok = 50.0 if chute == 'join' else MIN_AFTER_START
        if chute == 'straight' and last < end:
            continue
        if last > first_ok:
            corner_x[j] = float(last)
    reported = tuple(sorted(corner_x, key=corner_x.get))
    return RaceCourse(distance, step, turn, corner_x, reported, False)
