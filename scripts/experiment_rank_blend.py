"""Experiment (read-only, trains nothing persistent): two candidate fixes
for the pace model's rank predictions, scored on the usual validation
window (2025-09..12, complete-label races only, same as validate()).

Background (2026-11-17): after richer history, the independent per-horse
rank predictions improved (r4 0.267 -> 0.260) but still lose to the
past-5-run mean (0.237), and the contention layer makes several targets
worse (l1 0.432 independent -> 0.611 with contention). Candidates:

  A. Blend the independent continuous rank prediction with the past-mean
     (weights 0.3 / 0.5 / 0.7 on the model). The 2026 holdout suggested
     blending helps; this tests it on the larger validation window.
  B. Learning-to-rank: LGBMRanker (lambdarank), one model per corner,
     trained per race group with relevance = reversed finishing position
     at that corner. Rank within race by predicted score.

Run:  python scripts/experiment_rank_blend.py
Prints one line per corner per method. Nothing is written.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace_model as pm
from lightgbm import LGBMRanker

def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))

WEIGHTS = (0.3, 0.5, 0.7)


def run_window(d, fit_end, start, end, label):
    """Fit on data before fit_end, score independent / blends / past-mean / ranker on [start, end]."""
    models = pm.fit_models(d, fit_end)
    races = pm.evaluation_races(models, d, start, end)
    tr = (d.date < fit_end) & d.complete
    print(f"\n==={label}: fit<{fit_end}, eval {start}..{end}, {len(races)} races ===")
    res = {}
    for c in pm.CORNERS:
        t = f"r{c}"
        med = d.loc[tr, t].median()
        act, ind, past = [], [], []
        for x, corners, numbers, pred, spreads, early in races:
            if c not in corners:
                continue
            act.extend(x[t].to_numpy())
            ind.extend(pred[t])
            past.extend(x[f"past_{t}"].fillna(med).to_numpy())
        if not act:
            continue
        r = {"independent": mae(ind, act), "past-mean": mae(past, act)}
        for w in WEIGHTS:
            r[f"blend{w}"] = mae([w * i + (1 - w) * p for i, p in zip(ind, past)], act)
        train = d.loc[tr & d[t].notna()].sort_values("RACE_CODE")
        rel = np.rint((1 - train[t].to_numpy()) * (train.field.to_numpy() - 1)).astype(int)
        ranker = LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05,
                            num_leaves=31, min_child_samples=20, verbosity=-1, n_jobs=4, random_state=42)
        ranker.fit(train[pm.FEATURES], rel, group=train.groupby("RACE_CODE").size().tolist())
        errs = []
        for x, corners, numbers, pred, spreads, early in races:
            if c not in corners:
                continue
            scores = ranker.predict(x[pm.FEATURES])
            order = np.argsort(-scores, kind="stable")
            assigned = np.empty(len(x))
            assigned[order] = np.arange(len(x)) / max(1, len(x) - 1)
            errs.extend(np.abs(assigned - x[t].to_numpy()))
        r["ranker"] = float(np.mean(errs))
        res[t] = r
        print(f"  {t} (n={len(act)}): " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
    return res


def main():
    today_s = date.today().isoformat().replace("-", "")
    d, raw = pm.load_frame(today_s)
    d = pm.add_traffic(pm.add_history_features(pm.add_labels(d)))
    # Selection window: Jun-Aug 2025 (fit before June). Validation window is report-only.
    tune = run_window(d, "20250601", "20250601", "20250831", "TUNING (selection)")
    val = run_window(d, "20250901", "20250901", "20251231", "VALIDATION (report only)")
    cands = ["independent", "blend0.3", "blend0.5", "blend0.7", "ranker"]
    pick = {t: min(cands, key=lambda k: tune[t][k]) for t in tune}
    print("\nSelected on tuning window (per corner):")
    for t in tune:
        print(f"  {t}: {pick[t]}  -> validation {val[t][pick[t]]:.4f} vs past-mean {val[t]['past-mean']:.4f}")
    print("\nValidation reference (contention, from pace_model.validate): r1 0.242 / r2 0.247 / r3 0.257 / r4 0.265; "
          "past-mean 0.225 / 0.225 / 0.231 / 0.237.")


if __name__ == "__main__":
    main()
