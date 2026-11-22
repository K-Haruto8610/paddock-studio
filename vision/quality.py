"""Independent QA from the original frame-rate observations, never from the 5 Hz export."""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from .footage import ranges


def continuous(group):
    valid = group.dropna(subset=["s", "y"])
    split = (valid.frame.diff() != 1) | (valid.shot.diff() != 0) | (valid.t.diff() > .2 + 1e-8)
    return valid.groupby(split.cumsum())


def products(rows, rail):
    marks = {f"corner{c}": s for c, s in rail.course.corner_x.items()}
    marks.update(last600=rail.course.distance - 600, finish=rail.course.distance)
    events, samples = [], []
    for (code, horse), group in rows.groupby(["RACE_CODE", "UMABAN"]):
        # Keep an explicit missing row on the common race clock.
        grid = np.arange(max(0, int(np.floor(group.t.max()*5 + 1e-8))) + 1) / 5
        sy = np.full((len(grid), 2), np.nan)
        found = set()
        for _, run in continuous(group):
            t, s, y = run[["t", "s", "y"]].to_numpy().T
            inside = (grid >= t[0]-1e-8) & (grid <= t[-1]+1e-8)
            sy[inside] = np.c_[np.interp(grid[inside], t, s), np.interp(grid[inside], t, y)]
            for mark, value in marks.items():
                idx = np.flatnonzero((s[:-1] < value) & (value <= s[1:]))
                if mark in found or not len(idx):
                    continue
                i = idx[0]
                at = t[i] + (t[i+1]-t[i]) * (value-s[i])/(s[i+1]-s[i])
                events.append((code, horse, mark, value, at, int(run.frame.iloc[i+1])))
                found.add(mark)
        samples.append(pd.DataFrame(dict(RACE_CODE=code, UMABAN=horse, t=grid, s=sy[:, 0], y=sy[:, 1])))
    columns = ["RACE_CODE", "UMABAN", "event", "s", "t", "frame"]
    return pd.concat(samples, ignore_index=True), pd.DataFrame(events, columns=columns)


def order(times, official):
    official = official.where(official > 0)
    observed = times.reindex(official.index)
    if official.isna().any() or observed.isna().any():
        return dict(n=int((official.notna() & observed.notna()).sum()), tau=None, grouped_tau=None,
                    exact_fraction=None, complete_match=None, mismatch_horses=[])
    # Official tied ranks occupy a block of positions. Compatible orders within
    # a block remain tied for tau-b. An unresolved predicted tie across blocks
    # is not given an arbitrary horse-number tie breaker.
    ranks = rankdata(observed, method="average")
    truth = rankdata(official, method="average")
    sorted_ranks = np.sort(truth)
    model = []
    for rank in ranks:
        positions = np.flatnonzero(np.sort(ranks) == rank)
        model.append(sorted_ranks[positions].mean())
    matched = np.isclose(model, truth)
    tau = kendalltau(official, observed).statistic
    grouped = kendalltau(truth, model).statistic
    return dict(n=len(observed), tau=float(tau) if np.isfinite(tau) else None,
                grouped_tau=float(grouped) if np.isfinite(grouped) else None,
                exact_fraction=float(matched.mean()), complete_match=bool(matched.all()),
                mismatch_horses=[int(h) for h in official.index[~matched]])


def assess(rows, events, official, finish_ranks, review):
    horses = official.index
    by_event = {name: part.set_index("UMABAN").t for name, part in events.groupby("event")}
    empty = pd.Series(dtype=float)
    corners = {}
    for c in range(1, 5):
        ranks = official[f"CORNER{c}_JUNI"]
        if (ranks > 0).any():
            corners[c] = order(by_event.get(f"corner{c}", empty), ranks)
    taus = [v["tau"] for v in corners.values()]
    median = float(np.median(taus)) if taus and None not in taus else None
    last600 = by_event.get("last600", empty).reindex(horses)
    finish = by_event.get("finish", empty).reindex(horses)
    timing = pd.DataFrame(dict(t600_error_s=last600-(official.soha-official.last3),
                               last3_error_s=finish-last600-official.last3))
    hit = float((timing.t600_error_s.abs() <= .1+1e-9).sum()/len(horses))
    finish_order = order(finish, pd.Series(finish_ranks, dtype=float).reindex(horses))
    keyed = rows.set_index(["frame", "UMABAN"])
    errors = []
    for check in review.get("ground_checks", []):
        row = keyed.loc[(check["frame"], check["horse"])]
        errors.append((row.s-float(check["s"]), row.y-float(check["y"])))
    def rmse(col):
        values = np.array([e[col] for e in errors])
        return float(np.sqrt(np.mean(values**2))) if len(values) and np.isfinite(values).all() else None
    lateral = rmse(1)
    reviewed = review.get("identity_review_complete") is True
    swaps = review.get("swap_intervals", [])
    for event in swaps:
        a, b = event["frames"]
        if not rows.frame.min() <= a <= b <= rows.frame.max() or not set(event["horses"]) <= set(horses) or not event["horses"]:
            raise ValueError("Invalid identity switch interval")
    failures = {}
    for h, part in rows.groupby("UMABAN"):
        failures[int(h)] = dict(
            missing_box=ranges(part.loc[part.u.isna(), "frame"]),
            missing_projection=ranges(part.loc[part.u.notna() & part.s.isna(), "frame"]))
    status = "pass"
    timing_testable = (official.soha-official.last3).notna().all()
    if median is None or not reviewed or finish_order["complete_match"] is None or not timing_testable:
        status = "incomplete"
    if (median is not None and median < .98) or len(swaps) > 2 or (timing_testable and hit < .9):
        status = "fail"
    last_seen = review.get("verified_identity_frames", {})
    possible = {}
    for horse in finish_order["mismatch_horses"]:
        end = int(events.loc[(events.UMABAN == horse) & (events.event == "finish"), "frame"].iloc[0])
        ids = rows.loc[(rows.UMABAN == horse) & rows.u.notna(), "frame"]
        first = int(ids.min())
        verified = [i for i in last_seen.get(str(horse), []) if first <= i <= end]
        possible[horse] = [max(verified, default=first), end]
    return dict(status=status, corner=corners, corner_tau_median=median,
                corner_complete_match_fraction=(sum(v["complete_match"] is True for v in corners.values()) / len(corners)
                                                if corners else None),
                t600_within_01_fraction=hit if timing_testable else None, t600_n=int(timing.t600_error_s.count()),
                last3_n=int(timing.last3_error_s.count()),
                last3_mae_s=float(timing.last3_error_s.abs().mean()) if timing.last3_error_s.count() else None,
                finish=finish_order, identity_switches=len(swaps) if reviewed else None,
                identity_switches_confirmed_min=len(swaps),
                ground_check_n=len(errors), s_rmse_m=rmse(0), y_rmse_m=lateral,
                lateral_target_met=lateral <= .5 if lateral is not None else None,
                suggested_use=("hold" if status != "pass" else
                               "longitudinal_and_lateral" if lateral is not None and lateral <= .5 else "longitudinal_only"),
                missing_box_fraction=float(rows.u.isna().mean()),
                missing_projection_fraction=float((rows.u.notna() & rows.s.isna()).mean())), timing, dict(
                    detection_or_tracking_gaps=failures, identity_switch_intervals=swaps,
                    finish_mismatch_review_windows=possible,
                    ground_check_errors_m=[[float(v) if np.isfinite(v) else None for v in e] for e in errors])
