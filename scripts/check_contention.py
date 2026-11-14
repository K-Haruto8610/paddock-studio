"""Deterministic formation checks; no database or model fitting required."""
import unittest

import numpy as np

from pace_model import simulate


class ContentionTests(unittest.TestCase):
    def formation(self, ranks, early=None, lanes=None):
        n = len(next(iter(ranks.values())))
        pred = {}
        for c, values in ranks.items():
            pred.update({f'r{c}': np.array(values), f'g{c}': np.arange(n, dtype=float),
                         f'l{c}': np.zeros(n) if lanes is None else np.array(lanes)})
        return simulate(list(range(1, n + 1)), np.ones(n) if early is None else early,
                        pred, {c: 6 for c in ranks}, list(ranks),
                        {'alpha': 0, 'delta': .04, 'width': 4, 'front': .35, 'hold': .06})

    def test_normal_and_front_runner_limits(self):
        normal = self.formation({1: [0] * 12})[1]
        front = self.formation({1: [0] * 12}, np.zeros(12))[1]
        self.assertEqual(max(normal['lane']), 3)
        self.assertEqual(max(front['lane']), 4)
        self.assertTrue(any(normal['held']))
        self.assertGreater(normal['gap'][4], normal['gap'][0])

    def test_fourth_corner_has_no_headcount_limit(self):
        form = self.formation({4: [0] * 12})[4]
        self.assertEqual(max(form['lane']), 11)
        self.assertFalse(any(form['held']))

    def test_advancing_horses_go_outside_without_headcount_limit(self):
        form = self.formation({2: np.linspace(0, 1, 12), 3: np.arange(12)[::-1] * .001})
        first, later = form[2], form[3]
        # The last six horses all try to pass the front half before corner 4.
        for i in range(6, 12):
            self.assertFalse(later['held'][i])
            self.assertGreater(later['lane'][i], max(first['lane'][:i]))
            self.assertLess(later['rank'][i], first['rank'][i])
        self.assertEqual(len({later['gap'][i] for i in range(6, 12)}), 1)
        self.assertTrue(all(v >= 0 for v in later['out']))

    def test_separate_rows_keep_their_lateral_offset(self):
        form = self.formation({2: [0, 1], 3: [0, 1]}, lanes=[2, 4])
        self.assertEqual(form[2]['lane'], [2, 4])
        self.assertEqual(form[3]['lane'], [2, 4])
        self.assertEqual(form[3]['out'], [0, 0])
        self.assertEqual(form[3]['localLane'], [0, 0])

    def test_yielding_horse_drops_back_while_overtaker_moves_outside(self):
        form = self.formation({2: [0, 1], 3: [1, 0]})[3]
        self.assertEqual(form['rank'], [2, 1])
        self.assertGreater(form['gap'][0], form['gap'][1])
        self.assertEqual(form['lane'][0], 0)
        self.assertGreater(form['lane'][1], form['lane'][0])


if __name__ == '__main__':
    unittest.main()
