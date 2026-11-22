"""Metric inner rail from S0, and shot-local registration to marked ground points."""
import cv2
import numpy as np

from scripts.v3.courses import race_course


class Rail:
    def __init__(self, venue, track, distance):
        self.course = race_course(venue, track, distance, step=1)
        # A local right-handed course plane with positive curvature turning left.
        # An actual right-turn camera view is reflected by the fitted homography.
        k = np.r_[np.zeros(20), self.course.turn, np.zeros(30)]
        theta = np.cumsum(k) - .5 * k
        self.tangent = np.column_stack([np.cos(theta), np.sin(theta)])
        self.outward = self.tangent[:, [1, 0]] * [1, -1]
        self.xy = np.vstack([[0, 0], np.cumsum(self.tangent, axis=0)])
        self.start = -20

    def world(self, s, y):
        d = np.asarray(s) - self.start
        if np.any(d < 0) or np.any(d >= len(self.tangent)):
            raise ValueError("Landmark outside the modelled race")
        i = np.floor(d).astype(int)
        return self.xy[i] + (d-i)[..., None]*self.tangent[i] + np.asarray(y)[..., None]*self.outward[i]

    def coordinates(self, xy, limits):
        lo = max(0, int(np.floor(limits[0])) - self.start)
        hi = min(len(self.tangent), int(np.ceil(limits[1])) - self.start)
        if hi <= lo:
            raise ValueError("Empty shot distance range")
        diff = xy - self.xy[lo:hi]
        along = (diff * self.tangent[lo:hi]).sum(axis=1).clip(0, 1)
        off = diff - along[:, None] * self.tangent[lo:hi]
        j = np.argmin((off**2).sum(axis=1))
        s = self.start + lo + j + along[j]
        if not limits[0] < s < limits[1]:
            return None
        return float(s), float(off[j] @ self.outward[lo+j])


def transform(points, matrix):
    p = np.c_[np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))] @ matrix.T
    if np.any(np.abs(p[:, 2]) < 1e-9):
        raise ValueError("Point projects to infinity")
    return p[:, :2] / p[:, 2, None]


class Registration:
    def __init__(self, rail, keyframes, cuts):
        self.rail = rail
        self.keyframes = {k["frame"]: k for k in keyframes}
        self.cuts = set(cuts)
        self.previous = self.pixels = self.matrix = None

    def update(self, frame, gray):
        if frame in self.cuts:
            self.pixels = None
        if frame in self.keyframes:
            key = self.keyframes[frame]
            marks = np.asarray(key["landmarks"], np.float32)
            if marks.ndim != 2 or marks.shape[1] != 4 or len(marks) < 4 or not np.isfinite(marks).all():
                raise ValueError(f"Need at least four [u,v,s,y] ground landmarks at {frame}")
            self.pixels = np.ascontiguousarray(marks[:, :2])
            self.world = self.rail.world(*marks[:, 2:].T)
            self.limits = key["s_range"]
        elif self.pixels is not None:
            pts = self.pixels
            h, w = gray.shape
            levels = max(0, min(3, int(np.log2(min(h, w) / 64))))
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, pts, None, maxLevel=levels)
            prev, back, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, nxt, None, maxLevel=levels)
            good = ok.ravel().astype(bool) & back.ravel().astype(bool)
            good &= np.linalg.norm(prev - pts, axis=1) < 1
            good &= (nxt[:, 0] >= 0) & (nxt[:, 0] < w) & (nxt[:, 1] >= 0) & (nxt[:, 1] < h)
            self.pixels, self.world = nxt[good], self.world[good]
        self.previous = gray
        self.matrix = None
        if self.pixels is not None and len(self.pixels) >= 4:
            # RANSAC threshold is in pixels. Invert the fit for horse positions.
            image_from_ground, inliers = cv2.findHomography(self.world, self.pixels, cv2.RANSAC, 2.)
            if image_from_ground is not None and inliers.sum() >= 4 and np.linalg.cond(image_from_ground) < 1e12:
                keep = inliers.ravel().astype(bool)
                self.pixels, self.world = self.pixels[keep], self.world[keep]
                self.matrix = np.linalg.inv(image_from_ground)
                return
        self.pixels = None

    def locate(self, uv):
        if self.matrix is None:
            return None
        return self.rail.coordinates(transform([uv], self.matrix)[0], self.limits)
