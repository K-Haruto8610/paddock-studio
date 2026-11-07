"""Loader for data/g1_races.json, the single source of truth for current JRA flat GI races."""
import json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / 'data' / 'g1_races.json'


def norm(name):
    s = unicodedata.normalize('NFKC', str(name or ''))
    return re.sub(r'[\s　]+', '', s)


def load():
    master = json.loads(MASTER_PATH.read_text(encoding='utf-8'))
    for r in master['races']:
        r['_names'] = {norm(r['name'])} | {norm(a) for a in r['aliases']}
    return master


def find_by_name(master, name):
    n = norm(name)
    return next((r for r in master['races'] if n in r['_names']), None)


def expected(master, race, year):
    """Expected venue/track codes for a master race in a given year (applies venueOverrides)."""
    venue, codes = race['venueCode'], race['trackCodes']
    for o in master.get('venueOverrides', []):
        if o['id'] == race['id'] and int(o['year']) == int(year):
            venue, codes = o['venueCode'], o['trackCodes']
    return venue, codes


def mismatches(master, race, venue_code, distance, track_code, year):
    """List of human-readable differences between a DB race row and the master entry."""
    v, codes = expected(master, race, year)
    out = []
    if str(venue_code) != v:
        out.append(f'venue {venue_code} != {v}')
    if int(distance) != race['distance']:
        out.append(f'distance {distance} != {race["distance"]}')
    if int(track_code) not in codes:
        out.append(f'trackCode {track_code} not in {codes}')
    return out


def match_db_race(master, name, venue_code, distance, track_code, year):
    """Select a GI race from DB columns. Returns the master race only if name, venue,
    distance and track code all agree (so a renamed/relocated race is not silently included)."""
    r = find_by_name(master, name)
    if r and not mismatches(master, r, venue_code, distance, track_code, year):
        return r
    return None
