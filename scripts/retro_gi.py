"""追加指示12 (2026-11-19): retrospective check of v2.0 (the 2020-2026 frame) on the Tenno Sho (Autumn) 2026-11-01 and the
Queen Elizabeth II Cup 2026-11-15, whose results are already in the DB.

For each race the model of THAT DAY is rebuilt: the stored labelled frame is cut at the race day (rows on or after it are dropped
before any feature is built), history features and the opponent rating (rating_pre) are rebuilt from those rows only, every model is
fitted with fit_end = the race day (`d.date < end`), and the race is forecast through the pre-race CARD path (`upcoming_rows`, the
path the Mile CS forecast takes): the field, gates, weights, ages and blinkers of the day, the horses' earlier runs and the rating
state before the day. The results of the race enter nowhere.

Nothing here touches pace_model.py's main(), public/data or models/: outputs go to docs/retro-gi/ only.

  python scripts/retro_gi.py [--races tenno,queen] [--frame <frame_2020.pkl>] [--out docs/retro-gi]

Contention parameters are the published v2.0 ones (models/pace/report.json 'contention', chosen on Jun-Aug 2025 with models fitted
before June 2025: they cannot contain the target race). The forecast ranges (rankRange) are not rebuilt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm  # noqa: E402
import rating_index as ri  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pm.ROOT
RACES = {'tenno': '2026110105041111', 'queen': '2026111508050411'}
GOING_NAME = {1: '良', 2: '稍重', 3: '重', 4: '不良'}
DEFAULT_FRAME = Path('C:/Users/jouza/Downloads/horserace_yosou/paddock-studio/outputs/data-expansion-v20/frame_2020.pkl')


def raw_frame(path):
    """The stored labelled frame without any history-derived column (what pace_model.main --frame starts from)."""
    d = pd.read_pickle(path)
    return d[[k for k in d if not (k.startswith('past') or k in ('starts', 'rating_pre'))]]


def history_frame(raw, runs_table):
    d = pm.add_traffic(pm.add_history_features(raw.copy()))
    return ri.attach(d, runs_table)


def snapshot(rows, with_going=True):
    """The race as a pre-race card (what fetch_entries stores): only what is known before the off."""
    r = rows.iloc[0]
    snap = {'raceId': r.RACE_CODE,
            'race': {'distance': int(r.distance), 'surface': 'ダート' if r.surface == 1 else '芝', 'venueCode': f'{int(r.venue):02d}',
                     'trackCode': int(r.track), 'date': r.date},
            'conditions': {'going': GOING_NAME.get(int(r.going)) if with_going and pd.notna(r.going) else None},
            'horses': [{'horseId': h.KETTO_TOROKU_BANGO, 'number': int(h.UMABAN), 'weight': float(h.weight), 'age': int(h.age),
                        'blinker': None if pd.isna(h.blinker) else bool(h.blinker), 'status': 'entered'} for h in rows.itertuples()]}
    return snap


def feature_columns():
    cols = list(dict.fromkeys(pm.MOVE_FEATURES + pm.PACE_FEATURES + ['early_wish', 'past_pace', 'past_led', 'going']))
    return [c for c in cols if c != 'field_movers' or True]


def fit_and_forecast(d, x_sets, end, params, tag):
    """Fit every model on the rows before `end` and forecast each x in x_sets {name: DataFrame} on this race's course."""
    t0 = time.time()
    models = pm.fit_models(d, end)
    print(f'  [{tag}] fitted in {time.time() - t0:.0f}s', flush=True)
    gi_rows = d[(d.date < end) & (d.grade == 'A')]
    gi_pred = models['pace'].predict(pm.race_level(gi_rows, 'RACE_CODE', 1)[pm.PACE_FEATURES])
    gi_bands = tuple(float(v) for v in np.quantile(gi_pred, [1 / 3, 2 / 3]))
    sets = pm.corner_sets(d)
    outs = {}
    for name, x in x_sets.items():
        corners = sets(int(x.venue.iloc[0]), int(x.track.iloc[0]), int(x.distance.iloc[0]))
        outs[name] = pm.race_output(models, x, 'race', corners, params, gi_bands, (), None)
    return outs, gi_bands, len(gi_rows.RACE_CODE.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--races', default='tenno,queen')
    ap.add_argument('--frame', default=str(DEFAULT_FRAME))
    ap.add_argument('--out', default=str(ROOT / 'docs' / 'retro-gi'))
    ap.add_argument('--no-ablation', action='store_true', help='skip the rating-off refit (v1.9 feature set on the same frame / fit_end)')
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_frame(args.frame)
    params = json.loads((ROOT / 'models' / 'pace' / 'report.json').read_text(encoding='utf-8'))['contention']
    print('contention parameters (published v2.0):', params, flush=True)

    # Whole-frame build: the history-path reference for the equivalence check, and the actual results (labels) of the races.
    all_runs = ri.load_runs(None)
    full_table = ri.ratings(all_runs)
    full = history_frame(raw, full_table)
    day_of = all_runs.drop_duplicates('RACE_CODE').set_index('RACE_CODE').date

    for key in args.races.split(','):
        code = RACES[key]
        rows = raw[raw.RACE_CODE == code]
        day = rows.date.iloc[0]
        print(f'== {key} {code} {day}: {len(rows)} runners', flush=True)

        # --- the day-before frame: nothing on or after the race day survives
        cut = raw[raw.date < day]
        runs = ri.load_runs(day)
        table, state = ri.ratings(runs, return_state=True)
        d = history_frame(cut, table)
        fit = (d.date < day) & d.complete
        table_days = table.RACE_CODE.map(day_of)
        info = {'raceId': code, 'raceDay': day, 'fitEnd': day,
                'frameRows': int(len(d)), 'frameFirstDay': d.date.min(), 'frameLastDay': d.date.max(),
                'frameRowsOnOrAfterRaceDay': int((d.date >= day).sum()), 'targetRaceRowsInFrame': int((d.RACE_CODE == code).sum()),
                'fitRows': int(fit.sum()), 'fitRaces': int(d[fit].RACE_CODE.nunique()),
                'fitFirstDay': d[fit].date.min(), 'fitLastDay': d[fit].date.max(),
                'ratingRunsRows': int(len(runs)), 'ratingRunsLastDay': runs.date.max(),
                'ratingRunsOnOrAfterRaceDay': int((runs.date >= day).sum()), 'ratingRunsTargetRace': int((runs.RACE_CODE == code).sum()),
                'ratingTableLastDay': table_days.max(), 'ratingTargetRaceRows': int((table.RACE_CODE == code).sum()),
                'courseNormEnd': pm.PACE_NORM_END,
                'frameRowsAfterRaceDayInSource': int((raw.date > day).sum()), 'sourceLastDay': raw.date.max()}
        fwd = pm.forward_pace(d, day)
        info['forwardPaceChunkStarts'] = fwd['chunkStarts']
        info['forwardPaceLastRaceDay'] = str(max(p for p in d[(d.date < day) & d.complete].date.unique()))

        # --- the race through the card path (rating of the day before; history of earlier runs only)
        x = pm.upcoming_rows(snapshot(rows), d, state)
        x_nogoing = pm.upcoming_rows(snapshot(rows, with_going=False), d, state)

        # --- equivalence with the history path of the whole frame (same race rows, features computed with everything in the frame)
        ref = full[full.RACE_CODE == code].sort_values('UMABAN').reset_index(drop=True)
        xs = x.sort_values('UMABAN').reset_index(drop=True)
        ref_rating = ref.rating_pre.to_numpy()
        diffs = {}
        for c in feature_columns() + ['rating_pre']:
            if c not in xs or c not in ref:
                continue
            a, b = xs[c].to_numpy(float), ref[c].to_numpy(float)
            both = ~(np.isnan(a) & np.isnan(b))
            diffs[c] = float(np.nanmax(np.abs(a - b))) if both.any() and not (np.isnan(a) ^ np.isnan(b)).any() else (None if both.any() else 0.0)
        info['cardVsHistoryPathMaxAbsDiff'] = diffs
        info['cardVsHistoryPathWorst'] = max((v for v in diffs.values() if v is not None), default=0.0)
        info['cardVsHistoryPathNaNMismatch'] = [c for c, v in diffs.items() if v is None]
        info['ratingPreCardVsFullTableMaxAbsDiff'] = float(np.nanmax(np.abs(xs.rating_pre.to_numpy(float) - ref_rating)))

        systems, bands, npast = {}, {}, {}
        outs, gb, ngi = fit_and_forecast(d, {'v20': x, 'v20_nogoing': x_nogoing}, day, params, 'v2.0 (rating_pre)')
        systems['v20'], systems['v20_nogoing'], bands['v20'] = outs['v20'], outs['v20_nogoing'], gb
        info['giBandsRaces'] = ngi
        if not args.no_ablation:
            d0, x0 = d.copy(), x.copy()
            d0['rating_pre'], x0['rating_pre'] = 0.0, 0.0  # a constant column is never split on: the v1.9 feature set
            outs, gb, _ = fit_and_forecast(d0, {'v19spec': x0}, day, params, 'v1.9 feature set (rating_pre = 0)')
            systems['v19spec'], bands['v19spec'] = outs['v19spec'], gb

        # --- actual results (labels) of the race, straight from the frame (never read by the forecasts above)
        act = ref[['UMABAN', 'WAKUBAN', 'BAMEI', 'KETTO_TOROKU_BANGO', 'field', 'last3', 'soha', 'finish_gap', 'early3', 'pace', 'going',
                   'KAKU_TSUKA_JUNI1', 'KAKU_TSUKA_JUNI2', 'KAKU_TSUKA_JUNI3', 'KAKU_TSUKA_JUNI4', 'CORNER1', 'CORNER2', 'CORNER3', 'CORNER4',
                   'CORNER1_JUNI', 'CORNER2_JUNI', 'CORNER3_JUNI', 'CORNER4_JUNI', 'TIME_SA', 'SOHA_TIME', 'ZENHAN_3F']
                  + [f'{k}{c}' for c in pm.CORNERS for k in ('r', 'g', 'l')] + ['rating_pre']]
        act.to_csv(out_dir / f'actual_{key}.csv', index=False, encoding='utf-8-sig')
        (out_dir / f'forecast_{key}.json').write_text(json.dumps(
            {'raceId': code, 'raceDay': day, 'fitEnd': day, 'params': params, 'giBands': {k: list(v) for k, v in bands.items()},
             'systems': systems, 'leakCheck': info}, ensure_ascii=False, indent=1), encoding='utf-8')
        print(json.dumps({k: v for k, v in info.items() if k not in ('cardVsHistoryPathMaxAbsDiff',)}, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
