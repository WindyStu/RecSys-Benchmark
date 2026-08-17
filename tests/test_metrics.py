import math
import unittest

from recsys_benchmark.evaluator.metrics import evaluate_recommendations


class MetricsTest(unittest.TestCase):
    def test_computes_accuracy_metrics_for_ranked_recommendations(self):
        recommendations = {
            "u1": ["i1", "i2", "i3"],
            "u2": ["i4", "i5", "i6"],
        }
        ground_truth = {
            "u1": {"i2", "i4"},
            "u2": {"i4"},
        }

        result = evaluate_recommendations(
            recommendations=recommendations,
            ground_truth=ground_truth,
            cutoffs=[1, 2, 3],
        )

        self.assertAlmostEqual(result["recall@1"], 0.5)
        self.assertAlmostEqual(result["precision@1"], 0.5)
        self.assertAlmostEqual(result["hitrate@1"], 0.5)
        self.assertAlmostEqual(result["recall@2"], 0.75)
        expected_ndcg_2 = ((1 / math.log2(3)) / (1 + 1 / math.log2(3)) + 1.0) / 2
        self.assertAlmostEqual(result["ndcg@2"], expected_ndcg_2)
        self.assertAlmostEqual(result["mrr@3"], (1 / 2 + 1) / 2)
        self.assertAlmostEqual(result["map@3"], ((1 / 2) / 2 + 1.0) / 2)

    def test_computes_coverage_novelty_and_domain_metrics_when_metadata_exists(self):
        recommendations = {
            "u1": ["i1", "i2"],
            "u2": ["i2", "i3"],
        }
        ground_truth = {
            "u1": {"i2"},
            "u2": {"i3"},
        }
        item_domains = {"i1": "A", "i2": "B", "i3": "B"}
        item_popularity = {"i1": 1, "i2": 3, "i3": 1}

        result = evaluate_recommendations(
            recommendations=recommendations,
            ground_truth=ground_truth,
            cutoffs=[2],
            catalog_items={"i1", "i2", "i3", "i4"},
            item_domains=item_domains,
            item_popularity=item_popularity,
        )

        self.assertAlmostEqual(result["itemcoverage@2"], 3)
        self.assertAlmostEqual(result["catalogcoverage@2"], 0.75)
        self.assertGreater(result["novelty@2"], 0)
        self.assertAlmostEqual(result["domainmixratio@2:A"], 0.25)
        self.assertAlmostEqual(result["domainmixratio@2:B"], 0.75)
        self.assertAlmostEqual(result["domainrecall@2:B"], 1.0)
        self.assertAlmostEqual(result["domaincoverage@2:B"], 1.0)
        self.assertAlmostEqual(result["crossdomaintransfergap@2"], 0.0)


if __name__ == "__main__":
    unittest.main()
