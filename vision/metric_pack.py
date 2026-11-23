"""Export the reviewed pack run and grade it against the local database.

Run python -m vision.metric_pack RACE_CODE [--qa-only]. Copy verified cut pairs
to cuts/identity_anchors.json with confirmed=true and horse=<read number>.
Ground calibration uses setup.json keyframes of [u,v,s,y] and s_range, as in
the original patrol runner. Put independent checks and switches in review.json.
"""
import argparse
from collections import deque
import time

import cv2
import numpy as np
import pandas as pd

from .footage import frames, ranges, reader
from .geometry import Rail, Registration
from .io_local import mykeibadb_results, read_json, write_json
from .patrol import manifest, race_lock
from .quality import assess, products
from .track_pack import BOX, EXPERIMENT, source


def id_maps(tracks, field, folder, finish_anchored):
    maps = {"forward": {}, "reverse": {}}
    final_shot = int(tracks.shot.max())
    maps["forward"].update({(0, h): h for h in field.index})
    if finish_anchored:
        maps["reverse"].update({(final_shot, h): h for h in field.index})
    path = folder / "cuts/identity_anchors.json"
    if not path.exists():
        return maps
    starts = tracks[tracks.direction == "forward"].groupby("shot").frame.min()
    shot_at = {int(i): int(s) for s, i in starts.items() if s > 0}
    for pair in read_json(path)["pairs"]:
        if pair.get("confirmed") is not True:
            continue
        horse = pair["horse"]
        shot = shot_at[pair["frame"]]
        if horse not in field.index:
            raise ValueError("Invalid reviewed horse number")
        for direction, segment, key in [("forward", shot, "post_forward_id"),
                                        ("reverse", shot - 1, "pre_reverse_id")]:
            local = pair.get(key)
            if local is None:
                continue
            mapping = maps[direction]
            subset = tracks[(tracks.shot == segment) & (tracks.direction == direction)]
            if local not in set(subset.id):
                raise ValueError("Reviewed ID is absent from this shot")
            if ((segment, local) in mapping and mapping[segment, local] != horse
                    or any(s == segment and k != local and h == horse for (s, k), h in mapping.items())):
                raise ValueError("Conflicting reviewed identities")
            mapping[segment, local] = horse
    return maps


def add_names(tracks, maps):
    tracks = tracks.copy()
    tracks["horse"] = pd.array([maps[d].get((s, h)) for d, s, h in
                                tracks[["direction", "shot", "id"]].itertuples(index=False, name=None)], dtype="Int64")
    return tracks


def named_differences(tracks):
    direct = tracks[tracks.horse.notna() & (tracks.box_id >= 0) & tracks.evidence.isin(["observed", "anchor"])]
    keys = ["frame", "box_id", "shot"]
    common = direct[direct.direction == "forward"].merge(direct[direct.direction == "reverse"], on=keys)
    wrong = common[common.horse_x != common.horse_y]
    return dict(shared_named_observations=len(common), intervals=[
        dict(shot=int(s), forward_horse=int(h), reverse_horse=int(k), frames=span)
        for (s, h, k), part in wrong.groupby(["shot", "horse_x", "horse_y"])
        for span in ranges(sorted(part.frame.unique()))])


def export_pixels(clock, tracks, dest):
    # Each sample corresponds to the closest actual video frame on the gate clock.
    grid = pd.DataFrame({"t": np.arange(int(np.floor(clock.t.max()*5))+1) / 5})
    selected = pd.merge_asof(grid, clock[["t", "frame"]].rename(columns={"t": "source_t"}),
                             left_on="t", right_on="source_t", direction="nearest", tolerance=.1)
    positions = tracks.copy()
    positions["u"] = (positions.left + positions.right) / 2
    positions["v"] = positions.bottom
    cols = ["frame", "shot", "direction", "id", "horse", "evidence", "u", "v"]
    selected.merge(positions[cols], on="frame", how="left").to_csv(dest/"pixel_tracks_5hz.csv", index=False)


def extract(field, video, cfg, rail, clock, tracks, dest):
    inv = read_json(dest.parent.parent/"inventory.json")
    reg = Registration(rail, cfg["keyframes"], cfg["confirmed_cuts"])
    groups = dict(tuple(tracks[tracks.direction == "forward"].groupby("frame")))
    trails = {}
    records = []
    with reader(video) as cap:
        width, height = inv["width"], inv["height"]
        writer = cv2.VideoWriter(str(dest/"annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 inv["fps"], (width, height))
        if not writer.isOpened():
            raise ValueError("Cannot open the annotated video")
        try:
            for i, pts, im in frames(cap):
                if i < clock.frame.iloc[0]:
                    continue
                if i > clock.frame.iloc[-1]:
                    break
                if i in cfg["confirmed_cuts"]:
                    trails.clear()
                if reg.keyframes:
                    reg.update(i, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
                group = groups.get(i)
                if group is not None:
                    for row in group.itertuples():
                        l, top, r, b = (int(getattr(row, k)) for k in BOX)
                        u, v = (row.left+row.right)/2, row.bottom
                        color = (0, 220, 100) if pd.notna(row.horse) else (255, 190, 70)
                        direct = row.evidence in {"anchor", "observed"}
                        if not direct:
                            color = (0, 175, 255)
                        point = reg.locate([u, v]) if direct and pd.notna(row.horse) else None
                        if direct and pd.notna(row.horse):
                            s, y = point if point is not None else (np.nan, np.nan)
                            records.append((i, int(row.horse), u, v, s, y))
                        label = f"#{row.horse}" if pd.notna(row.horse) else f"S{row.shot}.{row.id}"
                        p = (round(u), round(v))
                        trail = trails.setdefault(row.id, deque(maxlen=25))
                        if trail and trail[-1][0] != i-1:
                            trail.clear()
                        trail.append((i, p))
                        if len(trail) > 1:
                            cv2.polylines(im, [np.int32([xy for _, xy in trail])], False, color, 2)
                        cv2.rectangle(im, (l,top), (r,b), color, 2)
                        cv2.putText(im, f"{label} {row.evidence[0]}", (l,max(18,top)), 0,.65,color,2)
                cv2.putText(im, f"frame {i}   t={pts-clock.pts.iloc[0]:.3f}s   # gate/review ID | S shot-local"
                            " | o observed  a anchor  m merged  p prediction",
                            (12,34), 0,.75,(0,255,255),2)
                writer.write(im)
            else:
                if i < clock.frame.iloc[-1]:
                    raise ValueError("Truncated input")
        finally:
            writer.release()
    points = pd.DataFrame(records, columns=["frame", "UMABAN", "u", "v", "s", "y"])
    rows = clock[["frame", "t"]].merge(pd.DataFrame({"UMABAN": field.index}), how="cross")
    rows["shot"] = np.searchsorted([c for c in cfg["confirmed_cuts"] if c > cfg["gate_open_frame"]],
                                    rows.frame, side="right")
    rows["RACE_CODE"] = str(field.RACE_CODE.iloc[0])
    rows = rows.merge(points, on=["frame", "UMABAN"], how="left", validate="one_to_one")
    samples, events = products(rows, rail)
    for name, table in [("frames", rows), ("trajectory_5hz", samples), ("events", events)]:
        table.to_csv(dest/f"{name}.csv", index=False)
    return int(samples.s.notna().sum())


def run(key, experiment, qa_only=False):
    field, video, cfg, dest = source(key, experiment)
    code = str(field.RACE_CODE.iloc[0])
    with race_lock(dest.parent.parent):
        tracking = read_json(dest/"track_summary.json")
        detection = read_json(dest/"summary.json")
        tracker_input = read_json(dest/"tracker_counts.json")
        if not qa_only:
            start = time.perf_counter()
            clock = pd.read_csv(dest/"counts.csv")
            clock["t"] = clock.pts - clock.pts.iloc[0]
            tracks = pd.read_csv(dest/"tracks.csv")
            maps = id_maps(tracks, field, dest, tracking["official_finish_anchored"])
            tracks = add_names(tracks, maps)
            write_json(dest/"named_direction_review.json", named_differences(tracks))
            export_pixels(clock, tracks, dest)
            rail = Rail(int(field.venue.iloc[0]),int(field.track.iloc[0]),int(field.distance.iloc[0]))
            count = extract(field, video, cfg, rail, clock, tracks, dest)
            write_json(dest/"export.json", dict(valid_metric_samples_5hz=count, seconds=time.perf_counter()-start))
        exported = read_json(dest/"export.json")
        official, finish = mykeibadb_results(code, field)
        rows = pd.read_csv(dest/"frames.csv", dtype={"RACE_CODE": str})
        events = pd.read_csv(dest/"events.csv")
        review = read_json(dest/"review.json") if (dest/"review.json").exists() else {}
        qa, timing, details = assess(rows, events, official, finish, review)
        qa.update(RACE_CODE=code, experiment=experiment, official_finish_horses=len(finish),
                  official_timed_horses=int((official.soha-official.last3).count()),
                  field=len(field), valid_metric_samples_5hz=exported["valid_metric_samples_5hz"],
                  detection=detection, tracker_input=tracker_input, tracking=tracking, export_seconds=exported["seconds"],
                  processing_seconds=detection["seconds"]+tracking["seconds"]+exported["seconds"])
        checks = dest.parent/"box_checks.json"
        if checks.exists():
            qa["draft_box_validation"] = next((r for r in read_json(checks) if r["model"]=="reviewed"
                and detection["weights"]=="patrol11s-reviewed.pt" and r["split"]=="val"
                and r["confidence"]==detection["confidence"]), None)
        timing.to_csv(dest/"timing_errors.csv", index_label="UMABAN")
        write_json(dest/"diagnostics.json", details)
        write_json(dest/"qa.json", qa)
        manifest(code, qa)
        return qa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args()
    print(run(args.key, args.experiment, args.qa_only))
