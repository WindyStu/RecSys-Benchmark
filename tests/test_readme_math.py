import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReadmeMathCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        cls.metrics = readme.split("## Metrics", 1)[1]

    def test_metrics_avoid_github_rejected_macros(self):
        rejected_macros = (r"\operatorname", r"\DeclareMathOperator", r"\newcommand")

        for macro in rejected_macros:
            with self.subTest(macro=macro):
                self.assertNotIn(macro, self.metrics)

    def test_inline_math_uses_backtick_delimiters(self):
        bare_inline_dollars = re.findall(r"(?<![\$`])\$(?![\$`])", self.metrics)

        self.assertEqual([], bare_inline_dollars)

    def test_display_math_delimiters_are_balanced(self):
        delimiter_count = self.metrics.count("$$")

        self.assertGreater(delimiter_count, 0)
        self.assertEqual(0, delimiter_count % 2)


if __name__ == "__main__":
    unittest.main()
