#!/usr/bin/env python3
"""
Data preparation for VQ-Rec fine-tuning on SDSR datasets.

For each npy-format dataset:
1. Project 2048-dim Qwen embeddings → 768-dim (compatible with pre-trained codebook)
2. Build FAISS OPQ32 index
3. Save PLM features as .feat1CLS binary
4. Convert JSON interactions to RecBole .inter format

Usage:
    python3 prepare_data.py --dataset Cell_Phones
    python3 prepare_data.py --dataset Cell_Phones --data_dir /path/to/SDSR
    python3 prepare_data.py --all
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# RecBole format constants
USER_ID = "user_id"
ITEM_ID = "item_id"
ITEM_ID_LIST = "item_id_list"
TIMESTAMP = "timestamp:float"
DEFAULT_MAX_ITEM_LIST_LENGTH = 50
INTER_HEADER = (
    "user_id:token\titem_id_list:token_seq\titem_id:token\ttimestamp:float"
)

# Datasets with npy files only
NPY_DATASETS = [
    "Cell_Phones",
    "Clothing",
    "Douban_Book",
    "Douban_Movie",
    "Electronics",
    "Grocery",
    "Home_Kitchen",
    "Sports",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for VQ-Rec fine-tuning")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--all", action="store_true", help="Process all npy datasets")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfsshare/home/zhangyuqi/data/SDSR",
        help="Root directory of SDSR datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,  # auto: VQ-Rec/dataset/downstream/
        help="Output directory for RecBole-format data",
    )
    parser.add_argument(
        "--proj_seed",
        type=int,
        default=42,
        help="Random seed for projection matrix",
    )
    parser.add_argument(
        "--max_item_list_length",
        type=int,
        default=DEFAULT_MAX_ITEM_LIST_LENGTH,
        help="Maximum history length for item_id_list",
    )
    return parser.parse_args()


def build_projection_matrix(
    in_dim: int, out_dim: int, seed: int = 42
) -> np.ndarray:
    """Build a fixed random projection matrix (Gaussian, orthonormalized)."""
    rng = np.random.RandomState(seed)
    W = rng.randn(in_dim, out_dim).astype(np.float32)
    # Orthonormalize columns via QR
    Q, _ = np.linalg.qr(W)
    return Q[:, :out_dim]


def project_embeddings(
    emb: np.ndarray,  # [N, 2048]
    proj: np.ndarray,  # [2048, 768]
) -> np.ndarray:
    """Project embeddings from 2048-dim to 768-dim."""
    return (emb.astype(np.float32) @ proj).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> str:
    """Build FAISS OPQ32,IVF1,PQ32x8 index and save to disk.

    Args:
        embeddings: [N, 768] float32

    Returns:
        Path to saved index file.
    """
    import faiss

    d = embeddings.shape[1]  # 768
    M = 32
    nbits = 8

    # OPQ pre-transform
    opq = faiss.OPQMatrix(d, M)
    opq.train(embeddings)

    # Apply OPQ
    transformed = opq.apply(embeddings)

    # PQ index with IVF
    nlist = 1  # IVF1
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

    index.train(transformed)
    index.add(transformed)

    # Build full pipeline: OPQ → IndexIVFPQ
    full_index = faiss.IndexPreTransform(opq, index)

    # Extract PQ codes for verification
    logger.info(f"  Index trained: {index.ntotal} vectors, {d}-dim, M={M}, nbits={nbits}")
    logger.info(f"  OPQ transform: {M} components")

    return full_index


def convert_to_recbole_inter(
    inter_data: dict,
    output_path: str,
    max_item_list_length: int = DEFAULT_MAX_ITEM_LIST_LENGTH,
) -> int:
    """Convert JSON interactions to RecBole .inter format.

    RecBole atomic format:
        user_id:token\titem_id_list:token_seq\titem_id:token\ttimestamp:float

    Returns:
        Number of interactions written.
    """
    lines = []
    for user_id_str, item_ids in inter_data.items():
        user_id = int(user_id_str)
        for i in range(1, len(item_ids)):
            item_id = item_ids[i]
            item_id = int(item_id)
            # item_id_list: space-separated history up to (but not including) current item
            history_start = max(0, i - max_item_list_length)
            history = [str(x) for x in item_ids[history_start:i]]
            history_str = " ".join(history)
            timestamp = float(i + 1)
            line = f"{user_id}\t{history_str}\t{item_id}\t{timestamp}"
            lines.append(line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join([INTER_HEADER] + lines))

    logger.info(f"  Written {len(lines)} interactions to {output_path}")
    return len(lines)


def prepare_dataset(
    dataset_name: str,
    data_dir: str,
    output_dir: str,
    proj_matrix: np.ndarray,
    max_item_list_length: int,
) -> None:
    """Prepare one SDSR dataset for VQ-Rec fine-tuning."""
    logger.info(f"{'='*60}")
    logger.info(f"Preparing dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    src_dir = os.path.join(data_dir, dataset_name)
    # Inter/feat files go flat: dataset/downstream/{dataset}.inter
    # Index files go in subdir: dataset/downstream/{dataset}/{dataset}.index
    idx_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(idx_dir, exist_ok=True)

    # 1. Load inter.json
    inter_path = os.path.join(src_dir, f"{dataset_name}.inter.json")
    with open(inter_path, "r") as f:
        inter_data = json.load(f)
    logger.info(f"  Loaded {len(inter_data)} users from inter.json")

    # 2. Load npy embedding
    import glob
    npy_pattern = os.path.join(src_dir, f"{dataset_name}.emb-*.npy")
    npy_matches = glob.glob(npy_pattern)
    if not npy_matches:
        raise FileNotFoundError(f"No .npy file found matching {npy_pattern}")
    npy_path = npy_matches[0]
    emb_2048 = np.load(npy_path)  # [N, 2048]
    logger.info(f"  Loaded embeddings: {emb_2048.shape}, {emb_2048.dtype}")

    # 3. Project 2048 → 768
    emb_768 = project_embeddings(emb_2048, proj_matrix)
    logger.info(f"  Projected to 768-dim: {emb_768.shape}")

    # 4. Save PLM feature file (.feat1CLS) — flat under output_dir
    feat_path = os.path.join(output_dir, f"{dataset_name}.feat1CLS")
    emb_768.tofile(feat_path)
    logger.info(f"  Saved PLM features: {feat_path}")

    # 5. Build FAISS index — inside subdirectory
    logger.info("  Building FAISS OPQ32,IVF1,PQ32x8 index...")
    full_index = build_faiss_index(emb_768)

    index_path = os.path.join(
        idx_dir, f"{dataset_name}.OPQ32,IVF1,PQ32x8.strict.index"
    )
    import faiss
    faiss.write_index(full_index, index_path)
    logger.info(f"  Saved FAISS index: {index_path}")

    # 6. Convert to RecBole .inter format — flat under output_dir
    inter_out = os.path.join(output_dir, f"{dataset_name}.inter")
    convert_to_recbole_inter(inter_data, inter_out, max_item_list_length)

    # 7. Create train/valid/test splits — flat under output_dir
    _create_splits(output_dir, dataset_name)

    logger.info(f"  Done: {dataset_name}\n")


def _create_splits(dst_dir: str, dataset_name: str) -> None:
    """Create train/valid/test .inter files using leave-one-out strategy.

    For each user:
      - test:  last interaction
      - valid: second-to-last interaction
      - train: remaining interactions
    """
    inter_path = os.path.join(dst_dir, f"{dataset_name}.inter")
    with open(inter_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if lines and lines[0] == INTER_HEADER:
        lines = lines[1:]

    # Group by user
    user_lines: dict = {}
    for line in lines:
        parts = line.split("\t")
        uid = parts[0]
        if uid not in user_lines:
            user_lines[uid] = []
        user_lines[uid].append(line)

    train_lines = []
    valid_lines = []
    test_lines = []

    for uid, ulines in user_lines.items():
        if len(ulines) >= 3:
            train_lines.extend(ulines[:-2])
            valid_lines.append(ulines[-2])
            test_lines.append(ulines[-1])
        elif len(ulines) == 2:
            train_lines.append(ulines[0])
            valid_lines.append(ulines[1])
            test_lines.append(ulines[1])
        else:
            train_lines.append(ulines[0])
            valid_lines.append(ulines[0])
            test_lines.append(ulines[0])

    _write_lines(os.path.join(dst_dir, f"{dataset_name}.train.inter"), train_lines)
    _write_lines(os.path.join(dst_dir, f"{dataset_name}.valid.inter"), valid_lines)
    _write_lines(os.path.join(dst_dir, f"{dataset_name}.test.inter"), test_lines)


def _write_lines(path: str, lines: list) -> None:
    with open(path, "w") as f:
        f.write("\n".join([INTER_HEADER] + lines))


def main() -> None:
    args = parse_args()

    if not args.all and not args.dataset:
        logger.error("Must specify --dataset or --all")
        sys.exit(1)

    datasets = NPY_DATASETS if args.all else [args.dataset]

    # Validate
    for ds in datasets:
        if ds not in NPY_DATASETS:
            logger.error(f"Unknown dataset: {ds}. Available: {NPY_DATASETS}")
            sys.exit(1)

    # Default output_dir relative to the VQ-Rec directory
    output_dir = args.output_dir
    if output_dir is None:
        vqrec_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(vqrec_dir, "dataset", "downstream")

    # Build shared projection matrix
    proj = build_projection_matrix(2048, 768, seed=args.proj_seed)
    logger.info(f"Projection matrix: {proj.shape}, seed={args.proj_seed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max item list length: {args.max_item_list_length}")

    for ds in datasets:
        prepare_dataset(ds, args.data_dir, output_dir, proj, args.max_item_list_length)

    logger.info(f"\nAll {len(datasets)} dataset(s) prepared.")


if __name__ == "__main__":
    main()
