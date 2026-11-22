"""Regression checks for the spike's timing, missing-data and local publication contract."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from . import io_local, patrol
from .geometry import Rail, Registration
from .quality import assess, order, products
from .tracking import Tracker


class SpikeTests(unittest.TestCase):
    def test_moving_camera_and_cut_reset(self):
        rail = Rail(4, 10, 1000)
        reg = Registration(rail, [{"frame": 0, "s_range": [0, 200], "landmarks":
            [[40, 40, 20, 0], [160, 40, 140, 0], [40, 100, 20, 12], [160, 100, 140, 12]]}], [2])
        gray = np.random.default_rng(9).integers(0, 255, (140, 200), dtype=np.uint8)
        camera = np.array([[1.01, 0, 3.], [0, 1.01, -2.]])
        reg.update(0, gray)
        reg.update(1, cv2.warpAffine(gray, camera, (200, 140)))
        np.testing.assert_allclose(reg.locate(camera @ [100, 65, 1]), [80, 5], atol=.1)
        reg.update(2, gray)
        self.assertIsNone(reg.locate([100, 65]))

    def test_curved_rail(self):
        rail = Rail(5, 11, 2000)
        for s in (12.6, 760.2, 1456.7, 1900.1):
            xy = rail.world(s, 3.5)
            measured = rail.coordinates(xy, [s-40, s+40])
            np.testing.assert_allclose(measured, [s, 3.5], atol=.04)

    def test_no_interpolation_over_losses_or_cuts(self):
        d = pd.DataFrame(dict(RACE_CODE=["K"]*4, UMABAN=1, frame=[0, 1, 3, 4], shot=[0, 1, 1, 1],
                              t=[0., .2, .6, .8], s=[399, 401, 999, 1001], y=1.))
        grid, events = products(d, Rail(4, 10, 1000))
        self.assertTrue(grid.loc[grid.t == .4, "s"].isna().all())
        self.assertEqual(events.event.tolist(), ["finish"])
        self.assertAlmostEqual(events.iloc[0].t, .7)

    def test_tied_corner_and_missing_horse(self):
        official = pd.Series([1, 1, 3], index=[1, 2, 3])
        observed = pd.Series([2., 1., 3.], index=[1, 2, 3])
        metrics = order(observed, official)
        self.assertTrue(metrics["complete_match"])
        self.assertLess(metrics["tau"], 1)
        self.assertAlmostEqual(metrics["grouped_tau"], 1)
        self.assertIsNone(order(observed.iloc[:2], official)["tau"])

    def test_gate_order_low_confidence_and_expiry(self):
        boxes = np.array([[0, 0, 10, 10, .8], [15, 0, 25, 10, .9]])
        tr = Tracker()
        start = tr.seed(boxes, [2, 5], [1, 0], True, 0)
        self.assertEqual(start[2][0], 15)
        boxes[:, 4] = .22
        self.assertEqual(set(tr.update(boxes, .05, 0)), {2, 5})
        self.assertEqual(tr.update(boxes, 1, .05), {})

    def test_acceptance_and_lateral_downgrade(self):
        official = pd.DataFrame(dict(CORNER1_JUNI=[1, 2], CORNER2_JUNI=0, CORNER3_JUNI=0,
                                     CORNER4_JUNI=0, soha=[90, 91], last3=[30, 30]), index=[1, 2])
        events = pd.DataFrame([(h, label, t+h-1, 100) for h in (1, 2)
                               for label, t in [("corner1", 20), ("last600", 60), ("finish", 90)]],
                              columns=["UMABAN", "event", "t", "frame"])
        rows = pd.DataFrame(dict(frame=[0, 0, 100, 100], UMABAN=[1, 2, 1, 2], u=1, s=100., y=1.))
        review = dict(identity_review_complete=True, ground_checks=[dict(frame=0, horse=1, s=100, y=2)])
        grade, _, _ = assess(rows, events, official, {1: 1, 2: 2}, review)
        self.assertEqual((grade["status"], grade["suggested_use"]), ("pass", "longitudinal_only"))
        review["swap_intervals"] = [dict(frames=[i, i+1], horses=[1, 2]) for i in (0, 20, 40)]
        self.assertEqual(assess(rows, events, official, {1: 1, 2: 2}, review)[0]["status"], "fail")
        review["identity_review_complete"] = False
        lower_bound = assess(rows, events, official, {1: 1, 2: 2}, review)[0]
        self.assertEqual((lower_bound["status"], lower_bound["identity_switches_confirmed_min"]), ("fail", 3))
        self.assertIsNone(lower_bound["identity_switches"])
        review["swap_intervals"] = []
        events = events[~((events.event == "last600") & (events.UMABAN == 2))]
        grade, _, _ = assess(rows, events, official, {1: 1, 2: 2}, review)
        self.assertEqual((grade["status"], grade["t600_within_01_fraction"]), ("fail", .5))

    def test_one_race_publication_resume_review(self):
        # Artificial motion and independent official labels exercise the file pipeline.
        io_local.DATA.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=io_local.DATA) as tmp:
            data = Path(tmp)
            code = "2020010506010109"
            video = data/"videos"/"202001050609.mp4"
            video.parent.mkdir()
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (240, 120))
            self.assertTrue(writer.isOpened())
            img = np.random.default_rng(1).integers(0, 255, (120, 240, 3), dtype=np.uint8)
            for _ in range(24):
                writer.write(img)
            writer.release()
            def boxes(i):
                x = 27 + 5*i
                return np.array([[x-10, 40, x+10, 53, .9], [x-11, 55, x+9, 68, .8]])
            class FixtureDetector:
                def __init__(self, *args):
                    self.i = 0
                def __call__(self, image):
                    result = boxes(self.i)
                    self.i += 1
                    return result
            official = pd.DataFrame(dict(RACE_CODE=[code]*2, UMABAN=[1, 2], field=2, distance=1000,
                venue=4, track=10, going=1, soha=[2.06, 2.08], last3=[1.2, 1.2],
                CORNER1_JUNI=0, CORNER2_JUNI=0, CORNER3_JUNI=0, CORNER4_JUNI=0))
            frame = data/"frame.pkl"
            official.to_pickle(frame)
            finish = data/"finish.pkl"
            pd.DataFrame(dict(RACE_CODE=[code]*2, UMABAN=[1, 2], fin=[1, 2])).to_pickle(finish)
            weights = data/"weights"/"fixture"
            weights.parent.mkdir()
            weights.touch()
            with (patch.object(io_local, "DATA", data), patch.object(patrol, "DATA", data),
                  patch.object(io_local, "FRAME", frame), patch.object(io_local, "FINISH", finish),
                  patch.object(patrol, "Detector", FixtureDetector)):
                inv = patrol.inventory(video.stem)
                self.assertEqual(inv["frames"], 24)
                folder = data/"races"/code
                cfg = io_local.read_json(folder/"setup.json")
                marks = [[30, 50, 0, 0], [130, 50, 1000, 0], [30, 80, 0, 10], [130, 80, 1000, 10]]
                cfg.update(gate_open_frame=0, gate_anchor_frame=0, end_frame=23, gate_axis=[0, 1],
                           confirmed_cuts=[], pan=False, zoom=False, all_horses_visible_ranges=[[0, 23]],
                           keyframes=[dict(frame=0, landmarks=marks, s_range=[-20, 1029])])
                io_local.write_json(folder/"setup.json", cfg)
                self.assertEqual(patrol.inventory(video.stem)["confirmed_cuts"], [])
                res = patrol.process(code, "fixture", "cpu", 608)
                self.assertEqual(res["status"], "incomplete")  # Straight course has no corner evidence.
                self.assertEqual(res["t600_within_01_fraction"], 1)
                self.assertLess(res["last3_mae_s"], .001)
                self.assertTrue(res["finish"]["complete_match"])
                self.assertIsNone(res["identity_switches"])
                run = folder / res["run_id"]
                with patrol.reader(run/"annotated.mp4") as cap:
                    self.assertEqual(len(list(patrol.frames(cap))), 24)
                grid = pd.read_csv(run/"trajectory_5hz.csv")
                self.assertEqual(len(grid), 24)
                review = io_local.read_json(run/"review.json")
                review.update(identity_review_complete=True, ground_checks=[dict(frame=5, horse=1, s=220, y=1)])
                io_local.write_json(run/"review.json", review)
                with patch.object(patrol, "Detector", side_effect=AssertionError("unexpected rerun")):
                    again = patrol.process(video.stem, "fixture", "cpu", 608)
                self.assertEqual(again["identity_switches"], 0)
                self.assertLess(again["y_rmse_m"], .02)
                self.assertEqual(again["run_id"], res["run_id"])


if __name__ == "__main__":
    unittest.main()
