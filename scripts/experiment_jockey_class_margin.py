"""Experiment (追加指示3, 2026-11-18): jockey / class-change / last-start margin features for the per-horse models.
Body-weight change is deliberately not used (announced too late for a published forecast).

Groups (definitions and DB columns: see the JMC block in pace_model.py):
  jockey  jockey_same, jockey_rides, jockey_mean_r1, jockey_mean_fin
  class   class_today, class_last, class_diff
  margin  last_margin
  +variants  class_mean5, margin_mean3 are tried on top of the selected set.

Same discipline as experiment_rel_features.py: selection on Jun-Aug 2025 (models fitted before June),
Sep-Dec 2025 report-only, baseline recomputed on the same frame. Adoption rule (same as before): continuous
blended rank MAE better by >= 0.002 at 3+ corners on validation, and last3 / lane not worse by more than 0.01.

Result 2026-11-18 (tuning 230 / validation 273 races) -> NOT ADOPTED; FEATURES unchanged.
  Tuning, mean continuous rank MAE r1-r4: base 0.2330, seed 7 0.2332 (noise ~0.0010 per corner). +jockey 0.2337,
  +class 0.2329, +margin 0.2332, +all 0.2334, +all+variants 0.2332: nothing beats the seed noise; +class selected by 0.0001.
  Validation (+class vs base), continuous: r1 0.2168/0.2175, r2 0.2179/0.2180, r3 0.2240/0.2237, r4 0.2287/0.2284
  -> gains +0.0008/+0.0001/-0.0003/-0.0003: fails the ">= 0.002 at 3 corners" rule. last3 +0.0002, lanes within 0.01.
  Likely reasons: jockey stats rest on a thin history (DB is dense only from 2025), and the past-5 position means
  already carry most of what class and margin say about running style.

Run:  python scripts/experiment_jockey_class_margin.py [--cache frame.pkl] [--out result.json]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm  # noqa: E402
pm.legacy_procedure('experiment_jockey_class_margin.py')  # pre-E0: reproduces its record on the v1.8 procedure (E0 review finding 2)
import experiment_rel_features as rel  # noqa: E402

BASE = list(pm.FEATURES)
GROUPS = pm.JMC_GROUPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache')
    ap.add_argument('--out')
    args = ap.parse_args()
    today = date.today().isoformat().replace('-', '')
    if args.cache and Path(args.cache).exists():
        d = pd.read_pickle(args.cache)
    else:
        d, _ = pm.load_frame(today)
        d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))
        d = pm.add_jockey_class_margin(d, pm.load_history(today))
        if args.cache:
            d.to_pickle(args.cache)
    allf = pm.JMC_FEATURES + pm.JMC_VARIANTS
    print(d[allf].describe().T[['count', 'mean', 'std', 'min', 'max']].to_string(), flush=True)
    tr = d[d.complete]
    print('complete-race rows with jockey stats:', int(tr.jockey_rides.notna().sum()), '/', len(tr), flush=True)

    configs = {'base': BASE, 'base-seed7': BASE, **{f'+{g}': BASE + f for g, f in GROUPS.items()}, '+all': BASE + pm.JMC_FEATURES,
               '+all+variants': BASE + allf}
    tune = rel.window(d, '20250601', '20250601', '20250831', configs, 'TUNING (selection)')
    base = rel.mean_cont(tune['base'])
    cands = {k: v for k, v in configs.items() if not k.startswith('base')}
    best = min(cands, key=lambda k: rel.mean_cont(tune[k]))
    print(f'selected on tuning: {best} (mean cont {rel.mean_cont(tune[best]):.4f} vs base {base:.4f})', flush=True)
    val = rel.window(d, '20250901', '20250901', '20251231', {'base': BASE, best: cands[best]}, 'VALIDATION (report only)')
    b, s = val['base'], val[best]
    gains = {t: b['cont'][t] - s['cont'][t] for t in rel.RANKS}
    worse = {t: s['contention'][t] - b['contention'][t] for t in ['last3', 'l1', 'l2', 'l3', 'l4']}
    adopt = sum(v >= .002 for v in gains.values()) >= 3 and all(v <= .01 for v in worse.values())
    print('\nadoption check (validation):', flush=True)
    print('  continuous rank gain per corner:', {t: round(v, 4) for t, v in gains.items()}, flush=True)
    print('  last3 / lane worsening:', {t: round(v, 4) for t, v in worse.items()}, flush=True)
    print('  ADOPT:', adopt, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({'groups': GROUPS, 'selected': best, 'selectedFeatures': cands[best], 'adopt': adopt,
                                              'tuning': tune, 'validation': val}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
