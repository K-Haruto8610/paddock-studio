import unittest

import numpy as np
import pandas as pd

from .metric_pack import named_differences
from .track_pack import MotionTracks, finish_seeds


class PackTests(unittest.TestCase):
    def test_merged_observation_and_expiry_in_both_directions(self):
        a = np.array([5., 5., 25., 30.])
        b = a + [8., 0., 8., 0.]
        color = np.eye(14)[4]
        one_box = np.array([np.r_[a, .9, color]])
        for direction in (-1, 1):
            tracker = MotionTracks()
            for horse, box in enumerate([a, b], 1):
                tracker.seed(horse, box, color, 5.)
            first = tracker.match(one_box, 5.+direction*.2)
            self.assertEqual([first[h][1] for h in [1, 2]], ["observed", "merged"])
            missing = np.empty((0, 19))
            self.assertIn(2, tracker.match(missing, 5.+direction))
            self.assertNotIn(2, tracker.match(missing, 5.+direction*1.01))
            again = tracker.match(np.array([np.r_[a, .9, color]]), 5.+direction*1.05)
            self.assertEqual(again[1][1], "observed")

    def test_finish_requires_independently_reviewed_unique_crossings(self):
        field = pd.DataFrame(index=[2, 7])
        events = [{"frame": 20, "box_id": 1, "rank": 1}, {"frame": 20, "box_id": 0, "rank": 2}]
        payload = {"confirmed": True, "crossings": events}
        obs = {20: np.arange(38).reshape(2,19)}
        self.assertFalse(finish_seeds(payload, {2: 1}, field, obs, 10, 21))
        self.assertFalse(finish_seeds({**payload, "confirmed": False}, {7: 1, 2: 2}, field, obs, 10, 21))
        seeds = finish_seeds(payload, {7: 1, 2: 2}, field, obs, 10, 21)
        self.assertEqual([s[0] for s in seeds[20]], [7,2])
        with self.assertRaises(ValueError):
            finish_seeds({**payload, "crossings": [events[0], events[0]]}, {7: 1, 2: 2}, field, obs, 10, 21)

    def test_named_disagreement_keeps_independent_names(self):
        rows = pd.DataFrame([(3, 10, 0, "forward", 8, "observed"), (3, 10, 0, "reverse", 11, "observed"),
                             (3, 11, 1, "forward", None, "observed"), (3, 11, 1, "reverse", 5, "observed")],
                            columns=["shot", "frame", "box_id", "direction", "horse", "evidence"])
        result = named_differences(rows)
        self.assertEqual(result["shared_named_observations"], 1)
        self.assertEqual(result["intervals"][0]["forward_horse"], 8)
        self.assertEqual(result["intervals"][0]["reverse_horse"], 11)
