"""Local COCO YOLO inference and a ByteTrack-style two-score association."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


class Detector:
    def __init__(self, weights, device, size):
        if not weights.is_file():
            raise ValueError(f"Missing local COCO TorchScript weights: {weights}")
        self.model = torch.jit.load(str(weights), map_location=device).eval()
        self.device, self.size = device, size

    @torch.inference_mode()
    def __call__(self, image):
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        sw, sh = round(w * scale), round(h * scale)
        l, t = (self.size-sw)//2, (self.size-sh)//2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[t:t+sh, l:l+sw] = cv2.resize(image, (sw, sh))
        rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
        pred = self.model(torch.from_numpy(rgb).to(self.device).float()[None] / 255)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if pred.ndim != 3 or pred.shape[:2] != (1, 84):
            raise ValueError("Expected COCO xywh and 80 class scores with shape [1,84,N]")
        pred = pred[0].T
        conf, label = pred[:, 4:].max(1)
        horse = (label == 17) & (conf >= .1)
        boxes = pred[horse, :4].cpu().numpy()
        scores = conf[horse].cpu().numpy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), .099, .65), int).ravel()
        boxes = boxes[keep]
        boxes[:, 2:] += boxes[:, :2]
        boxes = (boxes - [l, t, l, t]) / scale
        return np.column_stack([boxes.clip(0, [w, h, w, h]), scores[keep]])


class Tracker:
    """Constant-velocity boxes, global IoU assignment, then low-score recovery.

    Only gate identities are tracked. A lost horse stays unknown after timeout.
    """
    def __init__(self):
        self.tracks = {}

    def seed(self, detections, horses, axis, reverse, t):
        boxes = detections[detections[:, 4] >= .5, :4]
        if len(boxes) != len(horses):
            raise ValueError(f"Gate anchor has {len(boxes)} detections for {len(horses)} starters")
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        order = centers @ np.asarray(axis, float)
        self.tracks.clear()
        indices = np.argsort(order)
        for horse, i in zip(horses, indices[::-1] if reverse else indices):
            self.tracks[horse] = (boxes[i], np.zeros(4), t)
        return {h: entry[0] for h, entry in self.tracks.items()}

    def update(self, det, t, previous_t):
        active = [h for h, (_, _, last) in self.tracks.items() if t-last <= .8]
        matched = {}
        for high in (True, False):
            candidates = det[det[:, 4] >= .5] if high else det[det[:, 4] < .5]
            ids = [h for h in active if h not in matched and (high or self.tracks[h][2] == previous_t)]
            if not ids or not len(candidates):
                continue
            predicted = np.array([box + vel*(t-last) for box, vel, last in (self.tracks[h] for h in ids)])
            boxes = candidates[:, :4]
            p = predicted[:, None]
            inter = (np.minimum(p[:, :, 2:], boxes[:, 2:]) -
                     np.maximum(p[:, :, :2], boxes[:, :2])).clip(min=0).prod(axis=2)
            area = (predicted[:, 2:]-predicted[:, :2]).clip(min=0).prod(axis=1)
            union = area[:, None] + (boxes[:, 2:]-boxes[:, :2]).prod(axis=1) - inter
            iou = inter / np.maximum(union, 1e-6)
            cost = np.where(iou >= (.2 if high else .4), 1-iou, 1e4)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e4:
                    continue
                h = ids[i]
                old, vel, last = self.tracks[h]
                box = boxes[j]
                self.tracks[h] = box, .5*vel + .5*(box-old)/(t-last), t
                matched[h] = box
        return matched
