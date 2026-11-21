"""追加指示12: scores and pictures of the retrospective forecasts written by retro_gi.py (docs/retro-gi/).

Systems compared with the actual race (the frame's labels):
  v19pub   the v1.9 forecast as published (git dd61fc0, public/data/pace_forecasts.json; trained to 2025-12-31, the 2020-2022 style frame)
  v20retro v2.0 rebuilt as of the day before the race (this branch: fit_end = race day, 2020-2026 frame, card path)
  v19spec  the same retro build with rating_pre switched off (v1.9 feature set): separates the rating from the newer data / fit_end
  v20pub   the v2.0 now on main (trained to 2025-12-31 on the 2020-2026 frame), for reference
  v20nogo  v2.0 retro with the going left unknown (a card before the going is announced)
  pastEWM  the horse's own exponentially weighted past rank at that corner, turned into places (reference)

  python scripts/retro_gi_report.py [--out docs/retro-gi]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_standard as es  # noqa: E402
import pace_model as pm  # noqa: E402
import retro_gi  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pm.ROOT
RACES = {'tenno': ('2026110105041111', '天皇賞（秋）'), 'queen': ('2026111508050411', 'エリザベス女王杯')}
LABELS = {'v19pub': 'v1.9 公開', 'v20retro': 'v2.0 前日締め', 'v19spec': 'v1.9仕様 前日締め', 'v20pub': 'v2.0 現行公開', 'v20nogo': 'v2.0 前日締め(馬場不明)',
          'pastEWM': '過去走の指数加重'}
WAKU = {1: ('#ffffff', '#222'), 2: ('#222222', '#fff'), 3: ('#d9342b', '#fff'), 4: ('#2a6fd1', '#fff'), 5: ('#f2c230', '#222'),
        6: ('#2f9e4f', '#fff'), 7: ('#f08a24', '#222'), 8: ('#f2a0c0', '#222')}


def git_json(rev, path):
    return json.loads(subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=ROOT, capture_output=True, check=True).stdout.decode('utf-8'))


def official_places(code, nums):
    """Official finishing places (KAKUTEI_CHAKUJUN) of the race, read-only; runners without one (stopped / disqualified) -> inf."""
    import pymysql
    c = pymysql.connect(**retro_gi.ri.settings.pymysql_kwargs, cursorclass=pymysql.cursors.DictCursor)
    with c.cursor() as q:
        q.execute('SELECT UMABAN, KAKUTEI_CHAKUJUN FROM umagoto_race_joho WHERE RACE_CODE=%s', (code,))
        rows = q.fetchall()
    c.close()
    place = {}
    for r in rows:
        u = pd.to_numeric(str(r['UMABAN']).strip(), errors='coerce')
        k = pd.to_numeric(str(r['KAKUTEI_CHAKUJUN']).strip(), errors='coerce')
        place[int(u)] = float(k) if pd.notna(k) and 1 <= k <= 18 else np.inf
    return np.array([place[n] for n in nums])


def pace_label(v, bands):
    return 'H' if v <= bands[0] else 'S' if v >= bands[1] else 'M'


def unpack(out):
    """A race output -> {corner: DataFrame(num, rank, gap, lane)} plus last3 / finish per horse."""
    corners = out['corners']
    rows = []
    for num, h in out['horses'].items():
        row = {'num': int(num), 'last3': h['last3'], 'finishRank': h['finish']['rank'], 'finishGap': h['finish']['gap']}
        for p in h['positions']:
            row[f'rank{p["corner"]}'] = p['rank']
            row[f'gap{p["corner"]}'] = p['gap']
            row[f'lane{p["corner"]}'] = p['lane']
        rows.append(row)
    return corners, pd.DataFrame(rows).set_index('num').sort_index(), out['pace']


def rank_places(values, index):
    """Places 1..n from a score (lower = further forward); ties by horse number."""
    order = sorted(range(len(values)), key=lambda i: (values[i], index[i]))
    r = [0] * len(values)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return np.array(r)


def corner_metrics(pred_places, act_places, n):
    err = np.abs(pred_places - act_places)
    return {'tau': es.kendall(pred_places, act_places), 'maePlaces': float(err.mean()), 'maeNorm': float(err.mean() / (n - 1)),
            'within1': float((err <= 1).mean()), 'maxErr': float(err.max())}


def lead_metrics(pred_places, act_places, nums, names, n):
    pl = int(np.argmin(pred_places))
    al = int(np.argmin(act_places))
    top_p, top_a = set(np.argsort(pred_places)[:3]), set(np.argsort(act_places)[:3])
    five_p, five_a = set(np.argsort(pred_places)[:5]), set(np.argsort(act_places)[:5])
    quarter = max(1, n // 4)
    return {'predLeader': f'{nums[pl]} {names[pl]}', 'predLeaderActualPlace': int(act_places[pl]),
            'predLeaderIsLeader': bool(act_places[pl] == 1), 'predLeaderTop3': bool(act_places[pl] <= 3), 'predLeaderTopQuarter': bool(act_places[pl] <= quarter),
            'actualLeader': f'{nums[al]} {names[al]}', 'actualLeaderPredPlace': int(pred_places[al]),
            'top3Overlap': len(top_p & top_a), 'top5Overlap': len(five_p & five_a), 'quarter': quarter}


def build(key, out_dir, hist_full):
    code, name = RACES[key]
    fc = json.loads((out_dir / f'forecast_{key}.json').read_text(encoding='utf-8'))
    act = pd.read_csv(out_dir / f'actual_{key}.csv', encoding='utf-8-sig').sort_values('UMABAN').reset_index(drop=True)
    n = len(act)
    nums, names = act.UMABAN.astype(int).tolist(), act.BAMEI.tolist()
    pub19 = git_json('dd61fc0', 'public/data/pace_forecasts.json')
    pub20 = git_json('HEAD', 'public/data/pace_forecasts.json')
    outs = {'v19pub': pub19['races'][code], 'v20retro': fc['systems']['v20'], 'v19spec': fc['systems'].get('v19spec'),
            'v20pub': pub20['races'][code], 'v20nogo': fc['systems']['v20_nogoing']}
    outs = {k: v for k, v in outs.items() if v}
    for tag, pub in (('v19', pub19), ('v20now', pub20)):  # the published forecasts of this race, for the record
        (out_dir / f'published_{tag}_{key}.json').write_text(json.dumps({'version': pub['model']['version'], 'generatedAt': pub['generatedAt'],
                                                                         'contention': pub['model']['contention'], 'race': pub['races'][code]},
                                                                        ensure_ascii=False), encoding='utf-8')
    bands = {'v19pub': pub19['races'][code]['pace']['bands'], 'v20pub': pub20['races'][code]['pace']['bands'],
             'v20retro': fc['giBands']['v20'], 'v20nogo': fc['giBands']['v20'], 'v19spec': fc['giBands'].get('v19spec')}
    corners = outs['v20retro']['corners']
    for k, o in outs.items():
        assert o['corners'] == corners, (k, o['corners'], corners)
    pred = {k: unpack(o) for k, o in outs.items()}
    hist = hist_full[hist_full.RACE_CODE == code].sort_values('UMABAN').reset_index(drop=True)
    assert hist.UMABAN.astype(int).tolist() == nums

    act_places = {}
    for c in corners:
        r = act[f'r{c}']
        assert r.notna().all(), (key, c)
        act_places[c] = np.rint(r.to_numpy() * (n - 1) + 1).astype(int)

    res = {'race': name, 'raceId': code, 'date': fc['raceDay'], 'field': n, 'corners': corners, 'ranks': {}, 'lead': {}, 'gaps': {}, 'pace': {}, 'last3': {}, 'finish': {}}
    # --- ranks
    systems = [k for k in outs] + ['pastEWM']
    for c in corners:
        res['ranks'][c] = {}
        for k in systems:
            if k == 'pastEWM':
                v = hist[f'past_r{c}'].fillna(hist_full[hist_full.date < hist.date.iloc[0]][f'r{c}'].median()).to_numpy()
                p = rank_places(list(v), nums)
            else:
                p = pred[k][1].loc[nums, f'rank{c}'].to_numpy().astype(int)
            res['ranks'][c][k] = corner_metrics(p, act_places[c], n)
    # --- who is in front
    for tag, c in (('firstCorner', corners[0]), ('lastCorner', corners[-1])):
        res['lead'][tag] = {'corner': c, 'actualTop5': [f'{nums[i]}' for i in np.argsort(act_places[c])[:5]]}
        for k in outs:
            p = pred[k][1].loc[nums, f'rank{c}'].to_numpy().astype(int)
            res['lead'][tag][k] = lead_metrics(p, act_places[c], nums, names, n)
            res['lead'][tag][k]['predTop5'] = [f'{nums[i]}' for i in np.argsort(p)[:5]]
    res['frontRunners'] = {k: o['pace']['frontRunners'] for k, o in outs.items()}
    # --- gaps (lengths)
    for c in corners:
        res['gaps'][c] = {}
        for k in outs:
            g = pred[k][1].loc[nums, f'gap{c}'].to_numpy()
            a = act[f'g{c}'].to_numpy(float)
            q = lambda v: float(np.quantile(v, pm.SPREAD_Q))  # noqa: E731
            res['gaps'][c][k] = {'mae': float(np.nanmean(np.abs(g - a))), 'bias': float(np.nanmean(g - a)), 'predSpread80': q(g), 'actualSpread80': q(a[~np.isnan(a)]),
                                 'predMax': float(g.max()), 'actualMax': float(np.nanmax(a))}
    # --- pace: seconds against the course norm and the class
    actual_pace = float(act.pace.iloc[0])
    res['pace'] = {'actualSeconds': actual_pace, 'actualEarly3': float(act.early3.iloc[0]), 'zenhan3f': str(act.ZENHAN_3F.iloc[0])}
    for k, o in outs.items():
        b = bands[k]
        res['pace'][k] = {'predSeconds': o['pace']['early'], 'label': o['pace']['label'], 'bands': b, 'error': o['pace']['early'] - actual_pace,
                          'actualLabelOnSameBands': pace_label(actual_pace, b)}
    res['pace']['actualLabelFixedBands'] = pace_label(actual_pace, pm.PACE_BANDS)
    # --- last 3F (seconds)
    a3 = act.last3.to_numpy(float)
    ok = ~np.isnan(a3)
    for k in outs:
        p3 = pred[k][1].loc[nums, 'last3'].to_numpy(float)
        res['last3'][k] = {'mae': float(np.mean(np.abs(p3[ok] - a3[ok]))), 'bias': float(np.mean(p3[ok] - a3[ok])),
                           'corr': float(np.corrcoef(p3[ok], a3[ok])[0, 1]), 'tau': es.kendall(p3[ok], a3[ok]),
                           'predBest': float(p3.min()), 'actualBest': float(np.nanmin(a3)), 'predMean': float(p3[ok].mean()), 'actualMean': float(a3[ok].mean()),
                           'predFastest': nums[int(np.argmin(p3))], 'actualFastest': nums[int(np.nanargmin(a3))]}
    # --- finish (reference: from the final-corner gap and the closing difference, not a finishing-order model)
    act_fin = official_places(code, nums)
    res['finish']['actualTop3'] = [str(nums[i]) for i in np.argsort(act_fin, kind='stable')[:3]]
    res['finish']['actualPlaces'] = {str(n_): (None if np.isinf(p) else int(p)) for n_, p in zip(nums, act_fin)}
    for k in outs:
        pf = pred[k][1].loc[nums, 'finishRank'].to_numpy().astype(int)
        res['finish'][k] = {'tau': es.kendall(pf, act_fin), 'top3': [str(nums[i]) for i in np.argsort(pf)[:3]],
                            'top3Overlap': len(set(np.argsort(pf)[:3]) & set(np.argsort(act_fin, kind='stable')[:3]))}
    act['finish'] = act_fin
    return res, act, pred, nums, names, corners, act_places


def horse_table(key, act, pred, nums, names, corners, act_places):
    n = len(nums)
    ks = ['v19pub', 'v20retro']
    head = ['馬番', '馬名'] + [f'{c}角 実→1.9→2.0' for c in corners] + ['上がり3F 実/1.9/2.0', '実着順']
    fin = act.finish.to_numpy(float)
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    order = np.argsort(fin, kind='stable')
    for i in order:
        num = nums[i]
        cells = [str(num), names[i]]
        for c in corners:
            cells.append(f'{act_places[c][i]} → {pred["v19pub"][1].loc[num, f"rank{c}"]} → {pred["v20retro"][1].loc[num, f"rank{c}"]}')
        cells.append(f'{act.last3.iloc[i]:.1f} / {pred["v19pub"][1].loc[num, "last3"]:.1f} / {pred["v20retro"][1].loc[num, "last3"]:.1f}')
        cells.append('-' if np.isinf(fin[i]) else str(int(fin[i])))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def svg(key, act, pred, nums, corners, act_places, name, date):
    """Formation strips: one panel per corner, rows = actual / v1.9 published / v2.0 retro. x = lengths behind the leader (leader on the
    right), y = lane (0 = inside). Actual gaps and lanes come from the JRA passing notation (approximate widths)."""
    rows = [('実績', None), ('v1.9 公開', 'v19pub'), ('v2.0 前日締め', 'v20retro')]
    W, left, panel_h, row_h, lane_h = 1180, 118, 0, 78, 17
    ys, y = {}, 56
    for c in corners:
        ys[c] = y
        y += 26 + 3 * row_h + 18
    H = y + 8
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Meiryo, Yu Gothic, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="#f7f7f4"/>',
           f'<text x="12" y="22" font-size="16" font-weight="bold" fill="#222">{name} {date[:4]}-{date[4:6]}-{date[6:]}  隊列の比較（右が先頭・上が外）</text>',
           '<text x="12" y="40" fill="#555">位置は先頭からの馬身差、縦は内外（実績はJRA通過順表記の近似幅、予想は進路スロット）。丸の色は枠、数字は馬番。</text>']
    for c in corners:
        gaps = [act[f'g{c}'].to_numpy(float)] + [pred[k][1][f'gap{c}'].to_numpy(float) for k in ('v19pub', 'v20retro')]
        gmax = max(8.0, float(np.nanmax(np.concatenate(gaps)))) * 1.04
        plot_w = W - left - 24
        sx = lambda g, gmax=gmax, plot_w=plot_w: left + plot_w - g / gmax * plot_w  # noqa: E731
        y0 = ys[c]
        out.append(f'<text x="12" y="{y0 + 14}" font-size="14" font-weight="bold" fill="#222">{c}角</text>')
        for i, (label, k) in enumerate(rows):
            ry = y0 + 26 + i * row_h
            out.append(f'<rect x="{left - 8}" y="{ry}" width="{plot_w + 16}" height="{row_h - 6}" fill="{"#fff" if i % 2 == 0 else "#fbfbf8"}" stroke="#ddd"/>')
            out.append(f'<text x="14" y="{ry + 32}" fill="#333">{label}</text>')
            for gm in range(0, int(gmax) + 1, 5):
                out.append(f'<line x1="{sx(gm):.1f}" x2="{sx(gm):.1f}" y1="{ry}" y2="{ry + row_h - 6}" stroke="#eee"/>')
            for j, num in enumerate(nums):
                if k is None:
                    g, lane = act[f'g{c}'].iloc[j], act[f'l{c}'].iloc[j]
                else:
                    g, lane = pred[k][1].loc[num, f'gap{c}'], pred[k][1].loc[num, f'lane{c}']
                if pd.isna(g) or pd.isna(lane):
                    continue
                cy = ry + row_h - 6 - 11 - min(float(lane), 3.5) * lane_h
                wk = int(act.WAKUBAN.iloc[j])
                fill, txt = WAKU.get(wk, ('#ccc', '#000'))
                out.append(f'<circle cx="{sx(g):.1f}" cy="{cy:.1f}" r="9" fill="{fill}" stroke="#555" stroke-width="0.8"/>')
                out.append(f'<text x="{sx(g):.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-size="10.5" fill="{txt}">{num}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'docs' / 'retro-gi'))
    ap.add_argument('--frame', default=str(retro_gi.DEFAULT_FRAME))
    args = ap.parse_args()
    out_dir = Path(args.out)
    hist_full = pm.add_history_features(retro_gi.raw_frame(args.frame).copy())
    allres = {}
    for key in RACES:
        if not (out_dir / f'forecast_{key}.json').exists():
            continue
        res, act, pred, nums, names, corners, act_places = build(key, out_dir, hist_full)
        allres[key] = res
        (out_dir / f'formation_{key}.svg').write_text(svg(key, act, pred, nums, corners, act_places, RACES[key][1], res['date']), encoding='utf-8')
        (out_dir / f'horses_{key}.md').write_text(horse_table(key, act, pred, nums, names, corners, act_places), encoding='utf-8')
    (out_dir / 'metrics.json').write_text(json.dumps(allres, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps(allres, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
