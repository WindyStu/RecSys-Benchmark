import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = ROOT / "data_preprocessing"
CONFIGS = ROOT / "configs" / "methods"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataPreprocessingTest(unittest.TestCase):
    def test_component_configs_use_independent_preprocessing_directory(self):
        for method_id in (
            "data_preprocess_amazon",
            "data_preprocess_cross_domain",
            "text_embedding_aliyun",
        ):
            with self.subTest(method_id=method_id):
                config = yaml.safe_load((CONFIGS / f"{method_id}.yaml").read_text(encoding="utf-8"))
                self.assertEqual(config["source"], "data_preprocessing")

    def test_amazon_preprocessor_uses_raw_data_environment_default(self):
        module = load_module("benchmark_preprocess_amazon", PREPROCESSING / "preprocess_amazon.py")
        with patch.dict(os.environ, {"RECSYS_RAW_DATA_ROOT": "/datasets/raw"}), patch.object(
            sys, "argv", ["preprocess_amazon.py"]
        ):
            args = module.parse_args()

        self.assertEqual(args.data_root, "/datasets/raw")

    def test_cross_domain_paths_are_derived_from_configured_roots(self):
        module = load_module(
            "benchmark_preprocess_cross_domain", PREPROCESSING / "preprocess_cross_domain.py"
        )

        configs = module.build_cross_domain_configs(Path("/amazon"), Path("/douban"))

        self.assertEqual(configs["asc"]["a"]["reviews"], str(Path("/amazon/Sports_and_Outdoors_5.json")))
        self.assertEqual(configs["dbm"]["b"]["reviews"], str(Path("/douban/moviereviews_cleaned.txt")))

    def test_aliyun_embedding_uses_environment_credentials(self):
        module = load_module("benchmark_aliyun_embedding", PREPROCESSING / "aliyun_text_emb.py")
        with patch.dict(
            os.environ,
            {"ALIYUN_API_KEY": "test-key", "ALIYUN_BASE_URL": "https://example.invalid/v1"},
        ), patch.object(sys, "argv", ["aliyun_text_emb.py"]):
            args = module.parse_args()

        self.assertEqual(args.api_key, "test-key")
        self.assertEqual(args.base_url, "https://example.invalid/v1")


if __name__ == "__main__":
    unittest.main()
