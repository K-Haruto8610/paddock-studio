"""Video timestamps and independent inventory. Frame zero is the first displayed frame."""
from contextlib import contextmanager
import math

import cv2
import numpy as np


@contextmanager
def reader(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode {path.name}")
        yield cap
    finally:
        cap.release()


def frames(cap):
    previous = -1.0
    index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not math.isfinite(t) or t <= previous:
            raise ValueError(f"Invalid presentation timestamp at frame {index}")
        yield index, t, image
        index += 1
        previous = t


def motion(a, b):
    pts = cv2.goodFeaturesToTrack(a, 160, .02, 8)
    if pts is None or len(pts) < 8:
        return None
    target, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
    ok = status.ravel().astype(bool)
    if ok.sum() < 8:
        return None
    m, inliers = cv2.estimateAffinePartial2D(pts[ok], target[ok], ransacReprojThreshold=2)
    if m is None or inliers.sum() < max(8, len(pts) * .35):
        return None
    return m


def inspect(path):
    previous = None
    cuts, pans, zooms = [], [], []
    timestamps = []
    with reader(path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Missing nominal FPS")
        for i, t, image in frames(cap):
            h, w = image.shape[:2]
            timestamps.append(t)
            small = cv2.resize(image, (320, max(32, round(h * 320 / w))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                m = motion(previous, gray)
                if m is None and cv2.absdiff(previous, gray).mean() > 28:
                    cuts.append(i)
                if m is not None:
                    center = np.array([gray.shape[1] / 2, gray.shape[0] / 2])
                    if np.linalg.norm(m[:, :2] @ center + m[:, 2] - center) > 1.0:
                        pans.append(i)
                    if abs(np.linalg.norm(m[:, 0]) - 1) > .002:
                        zooms.append(i)
            previous = gray
    if not timestamps:
        raise ValueError("Empty video")
    intervals = np.diff(timestamps)
    tail = float(np.median(intervals)) if len(intervals) else 1 / fps
    return dict(file=path.name, width=w, height=h, fps=fps, frames=len(timestamps),
                duration_s=timestamps[-1] - timestamps[0] + tail,
                min_frame_s=float(min(intervals, default=tail)), max_frame_s=float(max(intervals, default=tail)),
                candidate_cuts=cuts, candidate_pan_ranges=ranges(pans), candidate_zoom_ranges=ranges(zooms),
                confirmed_cuts=None, pan=None, zoom=None, all_horses_visible_ranges=None)


def ranges(indices):
    result = []
    for i in indices:
        if result and i == result[-1][1] + 1:
            result[-1][1] = i
        else:
            result.append([i, i])
    return result
