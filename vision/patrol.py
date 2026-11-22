"""One-race patrol teacher-data spike. Run from paddock-studio with python -m vision.patrol."""
import argparse
from collections import deque
from contextlib import closing, contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time

import cv2
import numpy as np
import pandas as pd

from . import __doc__ as GUIDE
from .footage import frames, inspect, ranges, reader
from .geometry import Rail, Registration, transform
from .io_local import DATA, HERE, ROOT, digest, official_finish, race_input, read_json, videos, write_json
from .quality import assess, products
from .tracking import Detector, Tracker


def inventory(key):
    if key is None:
        return {"folder": str(DATA / "videos"), "files": [p.name for p in videos()]}
    try:
        code, _, path = race_input(key)
        folder = DATA / "races" / code
        join_issue = None
    except (FileNotFoundError, ValueError) as exc:
        matches = [v for v in videos() if v.stem == key]
        if len(matches) != 1:
            raise
        path, code = matches[0], None
        folder = DATA / "unmatched" / path.stem
        join_issue = str(exc)
    checksum = digest(path)
    inspector = digest(HERE/"footage.py")
    report_path = folder/"inventory.json"
    result = read_json(report_path) if report_path.is_file() else {}
    if result.get("video_sha256") != checksum or result.get("inspector_sha256") != inspector:
        result = inspect(path)
    result.update(RACE_CODE=code, join_issue=join_issue, video_sha256=checksum, inspector_sha256=inspector)
    if not (folder / "setup.json").exists():
        write_json(folder / "setup.json", dict(video_sha256=result["video_sha256"], confirmed_cuts=None,
            pan=None, zoom=None, all_horses_visible_ranges=None, gate_open_frame=None, gate_anchor_frame=None,
            end_frame=None, gate_axis=[1, 0], reverse_gate_order=False, keyframes=[], identity_anchors=[]))
    cfg = read_json(folder / "setup.json")
    if cfg.get("video_sha256") == result["video_sha256"]:
        for field in ("confirmed_cuts", "pan", "zoom", "all_horses_visible_ranges"):
            result[field] = cfg.get(field)
    write_json(report_path, result)
    return result


def checked_setup(folder, video, inv):
    cfg = read_json(folder / "setup.json")
    if not (cfg["video_sha256"] == inv["video_sha256"] == digest(video)):
        raise ValueError("Source changed. Repeat inventory and review the setup for the new footage.")
    start, anchor, end = (cfg[k] for k in ("gate_open_frame", "gate_anchor_frame", "end_frame"))
    if not all(type(i) is int for i in (start, anchor, end)) or not 0 <= start <= anchor < end < inv["frames"]:
        raise ValueError("Mark gate opening, full-field gate anchor and race end in setup.json")
    cuts = cfg["confirmed_cuts"]
    if not isinstance(cuts, list) or cuts != sorted(set(cuts)) or any(not 0 < i < inv["frames"] for i in cuts):
        raise ValueError("Review cut candidates across the video")
    if not all(isinstance(cfg[k], bool) for k in ("pan", "zoom", "reverse_gate_order")):
        raise ValueError("Review pan, zoom and gate numbering direction")
    visible = cfg["all_horses_visible_ranges"]
    if not isinstance(visible, list) or not any(a <= anchor <= b for a, b in visible):
        raise ValueError("Gate anchor must be within a confirmed all-horses-visible interval")
    if any(not 0 <= a <= b < inv["frames"] for a, b in visible):
        raise ValueError("Invalid all-horses-visible interval")
    axis = np.asarray(cfg["gate_axis"], dtype=float)
    if axis.shape != (2,) or not np.isfinite(axis).all() or not np.any(axis):
        raise ValueError("Invalid image-space gate axis")
    if not cfg["keyframes"] and not cfg.get("pixel_review_only"):
        raise ValueError("Mark inner rail and ground-plane landmarks in setup.json")
    return cfg


def manual_anchor(tracker, frame, boxes, points, horses, t):
    # Visible numbers after an edit must independently identify each supplied box.
    tracker.tracks.clear()
    used = set()
    for h, u, v in points:
        if h not in horses or h in tracker.tracks:
            raise ValueError(f"Invalid independently read horse number at frame {frame}")
        hits = np.flatnonzero((boxes[:, 0] <= u) & (u <= boxes[:, 2]) &
                              (boxes[:, 1] <= v) & (v <= boxes[:, 3]))
        if len(hits) != 1 or hits[0] in used:
            continue
        used.add(hits[0])
        tracker.tracks[h] = boxes[hits[0], :4], np.zeros(4), t
    return {h: state[0] for h, state in tracker.tracks.items()}


def annotate(image, horse, box, point, registration, trail):
    color = ((int(horse)*83)%170+85, (int(horse)*47)%170+85, (int(horse)*109)%170+85)
    x, y, r, b = box.astype(int)
    cv2.rectangle(image, (x, y), (r, b), color, 1)
    cv2.putText(image, str(horse), (x, max(16, y-2)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    if point is not None:
        ground = registration.rail.world(*point)
        trail.append(ground)
        line = transform(trail, np.linalg.inv(registration.matrix))
        cv2.polylines(image, [line.clip(-100000, 100000).astype(np.int32)], False, color, 2)
    elif not registration.keyframes:
        trail.append([(x+r)//2, b])
        cv2.polylines(image, [np.array(trail, np.int32)], False, color, 2)
    else:
        trail.clear()


def extract(path, inv, cfg, rail, official, detector, dest):
    horses = official.index.tolist()
    tracker = Tracker()
    reg = Registration(rail, cfg["keyframes"], cfg["confirmed_cuts"])
    anchors = {a["frame"]: a["points"] for a in cfg.get("identity_anchors", [])}
    if any(i not in cfg["confirmed_cuts"] or i <= cfg["gate_anchor_frame"] for i in anchors):
        raise ValueError("Independent identity anchors are only used at post-gate cuts")
    trails = {h: deque(maxlen=max(2, round(inv["fps"]))) for h in horses}
    rows = []
    diagnostics = []
    shot = 0
    previous_t = None
    with ExitStack() as stack:
        cap = stack.enter_context(reader(path))
        writer = cv2.VideoWriter(str(dest/"annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 inv["fps"], (inv["width"], inv["height"]))
        stack.callback(writer.release)
        if not writer.isOpened():
            raise ValueError("Cannot open local annotated video writer")
        for i, pts, image in frames(cap):
            if i < cfg["gate_open_frame"]:
                continue
            if i == cfg["gate_open_frame"]:
                zero = pts
            t = pts-zero
            if i in cfg["confirmed_cuts"]:
                shot += 1
                tracker.tracks.clear()
                for trail in trails.values():
                    trail.clear()
            reg.update(i, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            gate_tiles = cfg.get("gate_tiles")
            if gate_tiles and gate_tiles["frames"][0] <= i <= gate_tiles["frames"][1]:
                sections = []
                for x, y, r, b, low, high in gate_tiles["regions"]:
                    part = detector(image[y:b, x:r])
                    part[:, :4] += [x, y, x, y]
                    cx = (part[:, 0] + part[:, 2])/2
                    ymin, ymax = gate_tiles.get("foot_y_range", [y, b])
                    sections.append(part[(cx >= low) & (cx < high) & (part[:, 3] > ymin) & (part[:, 3] < ymax)])
                boxes = np.concatenate(sections)
            else:
                boxes = detector(image)
            if i == cfg["gate_anchor_frame"] and (boxes[:, 4] >= .5).sum() == len(horses):
                assigned = tracker.seed(boxes, horses, cfg["gate_axis"], cfg["reverse_gate_order"], t)
            elif i in anchors:
                assigned = manual_anchor(tracker, i, boxes, anchors[i], horses, t)
            else:
                assigned = tracker.update(boxes, t, previous_t)
            diagnostics.append((i, t, len(boxes), len(assigned), reg.matrix is not None))
            if cfg.get("pixel_review_only"):
                for x, y, r, b in boxes[:, :4].astype(int):
                    cv2.rectangle(image, (x, y), (r, b), (150, 150, 150), 1)
                cv2.putText(image, "Pixel tracks | gray: detection", (8, 49), 0, .65, (0, 255, 255), 2)
            for h in horses:
                box = assigned.get(h)
                u = v = np.nan
                point = None
                if box is not None:
                    u, v = (box[0]+box[2])/2, box[3]
                    point = reg.locate([u, v])
                    annotate(image, h, box, point, reg, trails[h])
                else:
                    trails[h].clear()
                s, y = point if point is not None else (np.nan, np.nan)
                rows.append((str(official.RACE_CODE.iloc[0]), h, i, shot, t, u, v, s, y))
            cv2.putText(image, f"frame {i}   t={t:.3f}s   ID {len(assigned)}/{len(horses)}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
            writer.write(image)
            previous_t = t
            if i == cfg["end_frame"]:
                break
    if not diagnostics or diagnostics[-1][0] != cfg["end_frame"]:
        raise ValueError("Video ended before the specified race end")
    counts = pd.DataFrame(diagnostics, columns=["frame", "t", "detections", "identified", "projection_ok"])
    counts.to_csv(dest/"frame_counts.csv", index=False)
    return pd.DataFrame(rows, columns=["RACE_CODE", "UMABAN", "frame", "shot", "t", "u", "v", "s", "y"])


def manifest(code, record):
    with closing(sqlite3.connect(DATA / "manifest.sqlite3", timeout=60)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS races (RACE_CODE TEXT PRIMARY KEY, qa_json TEXT NOT NULL)")
        db.execute("INSERT OR REPLACE INTO races VALUES (?, ?)", (code, json.dumps(record, allow_nan=False)))


@contextmanager
def race_lock(folder):
    # SQLite releases the per-race writer lock even when a process is killed.
    with closing(sqlite3.connect(folder/"writer.sqlite3", timeout=1)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        yield


def review_result(code, official, run):
    qa_start = time.perf_counter()
    meta = read_json(run/"run.json")
    review_file = run / "review.json"
    review = read_json(review_file) if review_file.is_file() else {}
    if review.get("run_id") != run.name:
        review = {}
    rows = pd.read_csv(run/"frames.csv", dtype={"RACE_CODE": str})
    events = pd.read_csv(run/"events.csv", dtype={"RACE_CODE": str})
    finish = official_finish(code, official.index)
    summary, timing, detail = assess(rows, events, official, finish, review)
    counts = pd.read_csv(run/"frame_counts.csv")
    detail["detection_shortfall_intervals"] = ranges(counts.loc[counts.detections < len(official), "frame"])
    detail["unassigned_box_intervals"] = ranges(counts.loc[counts.detections > counts.identified, "frame"])
    detail["unregistered_intervals"] = ranges(counts.loc[~counts.projection_ok, "frame"])
    summary.update(meta, RACE_CODE=code, field=len(official), venue=int(official.venue.iloc[0]),
                   going=int(official.going.iloc[0]) if pd.notna(official.going.iloc[0]) else None,
                   frames_with_detection_shortfall=int((counts.detections < len(official)).sum()),
                   frames_with_unassigned_detections=int((counts.detections > counts.identified).sum()),
                   unregistered_frames=int((~counts.projection_ok).sum()))
    timing.to_csv(run/"timing_errors.csv", index_label="UMABAN")
    write_json(run/"diagnostics.json", detail)
    summary["qa_seconds"] = time.perf_counter()-qa_start
    summary["processing_seconds"] = summary["extraction_seconds"] + summary["qa_seconds"]
    write_json(run/"qa.json", summary)
    manifest(code, summary)
    return summary


def process(key, weights_name, device, size):
    started = time.perf_counter()
    code, official, video = race_input(key)
    folder = DATA / "races" / code
    if not (folder/"inventory.json").is_file():
        raise ValueError(f"Run inventory {code} and review the footage first")
    with race_lock(folder):
        return process_locked(code, official, video, folder, weights_name, device, size, started)


def process_locked(code, official, video, folder, weights_name, device, size, started):
    inv = read_json(folder/"inventory.json")
    cfg = checked_setup(folder, video, inv)
    if any(inv[field] != cfg[field] for field in ("confirmed_cuts", "all_horses_visible_ranges", "pan", "zoom")):
        raise ValueError("Re-run inventory after reviewing cuts, camera movement and full-field visibility")
    weights = DATA / "weights" / weights_name
    if Path(weights_name).name != weights_name or not weights.is_file():
        raise ValueError("Place local COCO detector weights under vision/data/weights/")
    detector_id = digest(weights)
    # Bind cache to source, setup, runner/S0 code, model, joined race rows and inference options.
    sig = hashlib.sha256()
    for item in [inv["video_sha256"], digest(folder/"setup.json"), detector_id, device, str(size),
                 official.to_json(), *(digest(p) for p in sorted(HERE.glob("*.py"))),
                 digest(ROOT/"scripts/v3/courses.py"), digest(ROOT/"app/alignmentData.ts")]:
        sig.update(item.encode())
    run_id = sig.hexdigest()[:24]
    run = folder / run_id
    if not run.exists():
        rail = Rail(int(official.venue.iloc[0]), int(official.track.iloc[0]), int(official.distance.iloc[0]))
        model = Detector(weights, device, size)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            dest = Path(temp)/"output"
            dest.mkdir()
            rows = extract(video, inv, cfg, rail, official, model, dest)
            trajectory, events = products(rows, rail)
            for name, table in [("frames", rows), ("trajectory_5hz", trajectory), ("events", events)]:
                table.to_csv(dest/f"{name}.csv", index=False)
            from torch import cuda
            write_json(dest/"run.json", dict(run_id=run_id, video_filename=video.name,
                input_key=video.stem, device=device, gpu=cuda.get_device_name(device) if device.startswith("cuda") else None,
                weights_file=weights.name, weights_sha256=detector_id, extraction_seconds=time.perf_counter()-started))
            try:
                dest.rename(run)
            except OSError:
                if not (run/"run.json").is_file():
                    raise
    result = review_result(code, official, run)
    review_path = run / "review.json"
    if not review_path.exists():
        write_json(review_path, dict(run_id=run_id, identity_review_complete=False,
                                    swap_intervals=[], verified_identity_frames={}, ground_checks=[]))
    write_json(folder/"current.json", {"run_id": run_id})
    return result


def view(path, at, folder):
    with reader(path) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, at)
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"No frame {at}")
    out = folder / f"frame-{at}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise ValueError("Frame export failed")
    return {"local_frame": str(out)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    sub.add_parser("inventory").add_argument("key", nargs="?")
    run = sub.add_parser("run")
    run.add_argument("key")
    run.add_argument("--weights", default="yolov4-608.torchscript")
    run.add_argument("--size", type=int, default=608)
    run.add_argument("--device", default="cuda:0")
    batch = sub.add_parser("batch", help="Process only the listed reviewed races, reusing completed extraction")
    batch.add_argument("keys", nargs="+")
    sub.add_parser("qa").add_argument("key")
    frame = sub.add_parser("frame")
    frame.add_argument("key")
    frame.add_argument("index", type=int)
    sub.add_parser("manifest")
    sub.add_parser("guide", help="Show Japanese setup and review instructions")
    args = cli.parse_args()
    if args.action == "guide":
        print(GUIDE)
        return
    if args.action == "inventory":
        result = inventory(args.key)
    elif args.action == "run":
        result = process(args.key, args.weights, args.device, args.size)
    elif args.action == "batch":
        result = []
        for key in args.keys:
            summary = process(key, "yolov4-608.torchscript", "cuda:0", 608)
            result.append(summary)
    elif args.action == "manifest":
        path = DATA/"manifest.sqlite3"
        if path.exists():
            with closing(sqlite3.connect(path)) as db:
                result = [json.loads(row[0]) for row in db.execute("SELECT qa_json FROM races ORDER BY RACE_CODE")]
        else:
            result = []
    else:
        code, official, video = race_input(args.key)
        folder = DATA/"races"/code
        if args.action == "frame":
            result = view(video, args.index, folder/"frames")
        else:
            with race_lock(folder):
                run = folder / read_json(folder/"current.json")["run_id"]
                result = review_result(code, official, run)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
