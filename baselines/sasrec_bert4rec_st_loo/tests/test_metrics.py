import math
import unittest

from utils.metrics import cal_metrics


class MetricsTest(unittest.TestCase):
    def test_cal_metrics_returns_hr_ndcg_at_5_and_10_plus_mrr(self):
        ranks = [1, 6, 11]

        hr5, hr10, ndcg5, ndcg10, mrr = cal_metrics(ranks)

        self.assertAlmostEqual(hr5, 1 / 3)
        self.assertAlmostEqual(hr10, 2 / 3)
        self.assertAlmostEqual(ndcg5, 1 / 3)
        self.assertAlmostEqual(ndcg10, (1 + 1 / math.log2(7)) / 3)
        self.assertAlmostEqual(mrr, (1 + 1 / 6 + 1 / 11) / 3)


if __name__ == "__main__":
    unittest.main()
