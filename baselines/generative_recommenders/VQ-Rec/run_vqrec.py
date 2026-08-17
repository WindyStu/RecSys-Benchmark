#!/usr/bin/env python3
"""
VQ-Rec fine-tuning and evaluation on SDSR single-domain datasets.

Usage:
    python3 run_vqrec.py --dataset Cell_Phones
    python3 run_vqrec.py --dataset Electronics --lr 0.001 --epochs 300
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color

# VQ-Rec lives as a subdirectory: generative-recommenders/VQ-Rec/
_VQREC_DIR = os.path.dirname(os.path.abspath(__file__))
if _VQREC_DIR not in sys.path:
    sys.path.insert(0, _VQREC_DIR)

from vqrec import VQRec
from utils import create_dataset
from trainer import VQRecTrainer
from recbole_compat import patch_recbole_token_seq_remap
from vqrec_checkpoint import checkpoint_dataset_name, load_trusted_checkpoint
from vqrec_preflight import validate_prepared_inter_files

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQ-Rec fine-tuning for single-domain recommendation"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., Cell_Phones)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Maximum fine-tuning epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.003, help="Learning rate"
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="pretrained/VQRec-FHCKM-300-20230315.pth",
        help="Path to pre-trained VQ-Rec model",
    )
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="fix_enc",
        choices=["fix_enc", "full"],
        help="Fine-tune mode: fix_enc (freeze encoder) or full",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        default=False,
        help="Show tqdm progress bars",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_patch_config(dataset: str, args: argparse.Namespace) -> Config:
    """Load RecBole config with SDSR-specific overrides."""
    props = [
        os.path.join(_VQREC_DIR, "props", "VQRec.yaml"),
        os.path.join(_VQREC_DIR, "props", "finetune.yaml"),
    ]

    config_dict = {
        "epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "eval_batch_size": 1024,
        "MAX_ITEM_LIST_LENGTH": 50,
        "learning_rate": args.lr,
        "stopping_step": args.early_stop_patience,
        "topk": [5, 10],
        "metrics": ["HIT", "NDCG"],
        "valid_metric": "NDCG@10",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "seed": args.seed,
        "show_progress": args.show_progress,
        "gpu_id": 0,
        "use_gpu": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    config = Config(
        model=VQRec,
        dataset=dataset,
        config_file_list=props,
        config_dict=config_dict,
    )

    # Override paths to use SDSR-prepared data
    config["data_path"] = os.path.join(_VQREC_DIR, "dataset", "downstream")
    config["index_path"] = os.path.join(_VQREC_DIR, "dataset", "downstream")
    config["index_suffix"] = "OPQ32,IVF1,PQ32x8.strict.index"
    config["plm_suffix"] = "feat1CLS"
    config["plm_size"] = 768
    config["index_pretrain_dataset"] = None

    return config


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    patch_recbole_token_seq_remap()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset: {args.dataset}")

    # ---- Config ----
    config = load_and_patch_config(args.dataset, args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    validate_prepared_inter_files(
        config["data_path"],
        args.dataset,
        config["MAX_ITEM_LIST_LENGTH"],
    )

    # ---- Dataset ----
    dataset = create_dataset(config)
    logger.info(f"Dataset: {dataset}")

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # ---- Model ----
    model = VQRec(config, train_data.dataset).to(device)
    model.pq_codes = model.pq_codes.to(device)

    # Load pre-trained
    pretrained_path = os.path.join(_VQREC_DIR, args.pretrained_model)
    if os.path.exists(pretrained_path):
        checkpoint = load_trusted_checkpoint(pretrained_path, map_location=device)
        logger.info(f"Loading pre-trained model from {pretrained_path}")
        logger.info(
            f"Transfer [{checkpoint_dataset_name(checkpoint)}] -> [{args.dataset}]"
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if args.finetune_mode == "fix_enc":
            logger.info("[Fine-tune mode] Freezing sequence encoder")
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    else:
        logger.warning(
            f"Pre-trained model not found at {pretrained_path}, training from scratch"
        )

    logger.info(f"Model: VQRec")

    # ---- Trainer ----
    trainer = VQRecTrainer(config, model)

    logger.info(f"{'='*60}")
    logger.info(
        f"Starting fine-tuning: max {args.epochs} epochs, "
        f"patience={args.early_stop_patience}, lr={args.lr}"
    )
    logger.info(f"{'='*60}")

    # ---- Train ----
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=args.show_progress,
    )

    # ---- Test ----
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=args.show_progress,
    )

    # ---- Output (consistent with HSTU format) ----
    logger.info(f"{'='*60}")
    logger.info("Final Test Results:")
    logger.info(
        f"  HR@5:     {test_result.get('HIT@5', 0):.10f}"
    )
    logger.info(
        f"  HR@10:    {test_result.get('HIT@10', 0):.10f}"
    )
    logger.info(
        f"  NDCG@5:   {test_result.get('NDCG@5', 0):.10f}"
    )
    logger.info(
        f"  NDCG@10:  {test_result.get('NDCG@10', 0):.10f}"
    )
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
