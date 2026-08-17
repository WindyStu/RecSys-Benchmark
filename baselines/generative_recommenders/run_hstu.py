# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HSTU training and evaluation script for single-domain sequential recommendation.

Usage:
    python3 run_hstu.py --dataset Beauty
    python3 run_hstu.py --dataset Beauty --max_seq_len 150 --epochs 300 --lr 5e-4
    python3 run_hstu.py --dataset Beauty --data_dir /path/to/data --seed 123
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import Dict, Optional

import gin
import numpy as np
import torch

# ---- Patch fbgemm ops with pure PyTorch fallbacks (fbgemm_gpu not available) ----
import torch.utils._pytree as _pytree


def _async_cumsum_fallback(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)])


def _dense_to_jagged_fallback(
    dense: torch.Tensor,
    x_offsets: list,
) -> list:
    """Pure PyTorch fallback for fbgemm.dense_to_jagged.

    dense: [B, N, ...] padded tensor
    x_offsets: list of [B+1] offset tensors
    Returns: [values] where values is [total, ...] (jagged/flattened)
    """
    B = x_offsets[0].size(0) - 1
    total = int(x_offsets[0][-1].item())
    trailing_dims = dense.shape[2:]  # everything after the seq_len dim
    values = torch.empty(
        (total, *trailing_dims), dtype=dense.dtype, device=dense.device
    )
    for b in range(B):
        s = int(x_offsets[0][b].item())
        e = int(x_offsets[0][b + 1].item())
        length = e - s
        if length > 0:
            values[s:e] = dense[b, :length]
    return [values]


def _jagged_to_padded_dense_fallback(
    values: torch.Tensor,
    offsets: list,
    max_lengths: list,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Pure PyTorch fallback for fbgemm.jagged_to_padded_dense."""
    B = offsets[0].size(0) - 1
    N = max_lengths[0]
    out = torch.full(
        (B, N, *values.shape[1:]),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    for b in range(B):
        s, e = int(offsets[0][b]), int(offsets[0][b + 1])
        L = min(e - s, N)
        out[b, :L] = values[s : s + L]
    return out


def _pytree_dense_to_jagged_fallback(
    dense_values: list,  # list of tensors
    x_offsets: list,
) -> list:
    """Pure PyTorch fallback for fbgemm.dense_to_jagged (pytree variant)."""
    return [_dense_to_jagged_fallback(dv, x_offsets)[0] for dv in dense_values]


# Monkey-patch torch.ops.fbgemm
import torch._ops as _ops

_FBGEMM_NS_NAME = "fbgemm"
_has_fbgemm = hasattr(torch.ops, _FBGEMM_NS_NAME)

if not _has_fbgemm:
    # Create a lightweight namespace-like object
    class _FbgemmCompat:
        @staticmethod
        def asynchronous_complete_cumsum(lengths):
            return _async_cumsum_fallback(lengths)

        @staticmethod
        def dense_to_jagged(dense, x_offsets):
            return _dense_to_jagged_fallback(dense, x_offsets)

        @staticmethod
        def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
            return _jagged_to_padded_dense_fallback(
                values, offsets, max_lengths, padding_value
            )

        @staticmethod
        def pytree_dense_to_jagged(dense_values, x_offsets):
            return _pytree_dense_to_jagged_fallback(dense_values, x_offsets)

    torch.ops.fbgemm = _FbgemmCompat()  # type: ignore
else:
    # fbgemm exists but some ops may be missing
    if not hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"):
        torch.ops.fbgemm.asynchronous_complete_cumsum = (  # type: ignore
            _async_cumsum_fallback
        )
    if not hasattr(torch.ops.fbgemm, "dense_to_jagged"):
        torch.ops.fbgemm.dense_to_jagged = _dense_to_jagged_fallback  # type: ignore
    if not hasattr(torch.ops.fbgemm, "jagged_to_padded_dense"):
        torch.ops.fbgemm.jagged_to_padded_dense = (  # type: ignore
            _jagged_to_padded_dense_fallback
        )
# -----------------------------------------------------------------

from generative_recommenders.research.data.eval import (
    _avg,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from generative_recommenders.research.data.sdsr_dataset import (
    SDSRDataset,
    infer_max_seq_len,
    load_sdsr_data,
)
from generative_recommenders.research.indexing.utils import get_top_k_module
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    InBatchNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    FeatureAugmentedEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    hstu_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    sdsr_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.trainer.data_loader import create_data_loader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HSTU training for single-domain sequential recommendation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., Beauty, Electronics, Clothing)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfsshare/home/zhangyuqi/data/SDSR",
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--gin_config_file",
        type=str,
        default="configs/sdsr/hstu-large.gin",
        help="Path to gin config file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Max sequence length; auto-inferred from data P95 if not set",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without NDCG@10 improvement)",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=128,
        help="Number of negatives for sampled softmax",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Softmax temperature",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpts",
        help="Directory to save checkpoints",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    # Load gin config for model structure
    if os.path.exists(args.gin_config_file):
        logger.info(f"Loading gin config from {args.gin_config_file}")
        gin.parse_config_file(args.gin_config_file)
    else:
        logger.warning(f"Gin config not found: {args.gin_config_file}, using defaults")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---- Load data ----
    logger.info(f"Loading dataset: {args.dataset}")
    sdsr_data = load_sdsr_data(args.data_dir, args.dataset)

    max_seq_len = args.max_seq_len or infer_max_seq_len(sdsr_data.seq_len_stats)
    logger.info(
        f"Sequence length stats: {sdsr_data.seq_len_stats}"
    )
    logger.info(f"Using max_sequence_length: {max_seq_len}")
    logger.info(
        f"Dataset: {sdsr_data.num_items} unique items, "
        f"{len(sdsr_data.user_sequences)} users, "
        f"max_item_id={sdsr_data.max_item_id}"
    )

    # Train/eval datasets via leave-one-out split
    train_dataset = SDSRDataset(
        sdsr_data=sdsr_data,
        max_sequence_length=max_seq_len,
        ignore_last_n=1,
    )
    eval_dataset = SDSRDataset(
        sdsr_data=sdsr_data,
        max_sequence_length=max_seq_len,
        ignore_last_n=0,
    )

    # Data loaders (single GPU, world_size=1, rank=0)
    _, train_data_loader = create_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        world_size=1,
        rank=0,
        shuffle=True,
        drop_last=False,
    )
    _, eval_data_loader = create_data_loader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        world_size=1,
        rank=0,
        shuffle=False,
        drop_last=False,
    )

    # ---- Build model ----
    item_embedding_dim = 200
    gr_output_length = 10

    # num_items = max_item_id + 1 (accounts for +1 shift in SDSRDataset)
    num_items = sdsr_data.max_item_id + 1

    # Build embedding module based on feature type
    if sdsr_data.feature_type == "categorical":
        logging.info("Using FeatureAugmentedEmbeddingModule (categorical features)")
        embedding_module = FeatureAugmentedEmbeddingModule(
            num_items=num_items,
            item_embedding_dim=item_embedding_dim,
            feat_vocab_sizes=sdsr_data.feat_vocab_sizes,
            item_to_feat_a=sdsr_data.item_to_feat_a,
            item_to_feat_b=sdsr_data.item_to_feat_b,
            item_to_feat_c=sdsr_data.item_to_feat_c,
            item_to_feat_d=sdsr_data.item_to_feat_d,
        )
    elif sdsr_data.feature_type == "dense":
        logging.info(
            f"Using DenseFeatureEmbeddingModule (dense dim={sdsr_data.dense_dim})"
        )
        from generative_recommenders.research.modeling.sequential.embedding_modules import (
            DenseFeatureEmbeddingModule,
        )
        embedding_module = DenseFeatureEmbeddingModule(
            num_items=num_items,
            item_embedding_dim=item_embedding_dim,
            pretrained_emb=sdsr_data.pretrained_emb,
            dense_dim=sdsr_data.dense_dim,
        )
    else:
        logging.info("Using LocalEmbeddingModule (no side features)")
        from generative_recommenders.research.modeling.sequential.embedding_modules import (
            LocalEmbeddingModule,
        )
        embedding_module = LocalEmbeddingModule(
            num_items=num_items,
            item_embedding_dim=item_embedding_dim,
        )

    interaction_module, _ = get_similarity_function(
        module_type="DotProduct",
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )

    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_seq_len + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=0.2,
    )

    output_postproc_module = L2NormEmbeddingPostprocessor(
        embedding_dim=item_embedding_dim,
        eps=1e-6,
    )

    # Build HSTU via gin-configured hstu_encoder
    model = hstu_encoder(
        max_sequence_length=max_seq_len,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        similarity_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=False,
        verbose=True,
    )
    model = model.to(device)
    logger.info(f"Model: {model.debug_str()}")

    # ---- Loss & sampler ----
    ar_loss = SampledSoftmaxLoss(
        num_to_sample=args.num_negatives,
        softmax_temperature=args.temperature,
        model=model,
        activation_checkpoint=False,
    ).to(device)

    negatives_sampler = InBatchNegativesSampler(
        l2_norm=True,
        l2_norm_eps=1e-6,
        dedup_embeddings=True,
    ).to(device)

    # ---- Optimizer ----
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0,
    )

    # ---- Training state ----
    best_ndcg10 = 0.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0
    best_state_dict = None

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Starting training: {args.epochs} max epochs, "
                f"early stopping patience={args.early_stop_patience}")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for row in iter(train_data_loader):
            seq_features, target_ids, target_ratings = sdsr_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1,
            )

            # Scatter target into past_ids for teacher forcing
            B, N = seq_features.past_ids.shape
            seq_features.past_ids.scatter_(
                dim=1,
                index=seq_features.past_lengths.view(-1, 1),
                src=target_ids.view(-1, 1),
            )

            opt.zero_grad()
            input_embeddings = model.get_item_embeddings(seq_features.past_ids)
            seq_embeddings = model(
                past_lengths=seq_features.past_lengths,
                past_ids=seq_features.past_ids,
                past_embeddings=input_embeddings,
                past_payloads=seq_features.past_payloads,
            )

            # In-batch negative sampling
            supervision_ids = seq_features.past_ids
            in_batch_ids = supervision_ids.view(-1)
            negatives_sampler.process_batch(
                ids=in_batch_ids,
                presences=(in_batch_ids != 0),
                embeddings=model.get_item_embeddings(in_batch_ids),
            )

            ar_mask = supervision_ids[:, 1:] != 0
            loss, _ = ar_loss(
                lengths=seq_features.past_lengths,
                output_embeddings=seq_embeddings[:, :-1, :],
                supervision_ids=supervision_ids[:, 1:],
                supervision_embeddings=input_embeddings[:, 1:, :],
                supervision_weights=ar_mask.float(),
                negatives_sampler=negatives_sampler,
                **seq_features.past_payloads,
            )

            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        # ---- Eval (full epoch) ----
        model.eval()
        eval_state = get_eval_state(
            model=model,
            all_item_ids=sdsr_data.all_item_ids,
            negatives_sampler=negatives_sampler,
            top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                top_k_method="MIPSBruteForceTopK",
                model=model,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            ),
            device=device,
        )

        eval_dict_all: Dict[str, list] = {}
        for row in iter(eval_data_loader):
            seq_features, target_ids, target_ratings = sdsr_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1,
            )
            eval_dict = eval_metrics_v2_from_tensors(
                eval_state,
                model,
                seq_features,
                target_ids=target_ids,
                target_ratings=target_ratings,
            )
            for k, v in eval_dict.items():
                if k not in eval_dict_all:
                    eval_dict_all[k] = []
                eval_dict_all[k].append(v)

        # Aggregate all eval metrics
        for k in eval_dict_all:
            eval_dict_all[k] = torch.cat(eval_dict_all[k], dim=-1)

        hr5 = float(_avg(eval_dict_all["hr@5"], world_size=1))
        hr10 = float(_avg(eval_dict_all["hr@10"], world_size=1))
        ndcg5 = float(_avg(eval_dict_all["ndcg@5"], world_size=1))
        ndcg10 = float(_avg(eval_dict_all["ndcg@10"], world_size=1))

        epoch_time = time.time() - epoch_start

        # ---- Early stopping check ----
        is_best = ndcg10 > best_ndcg10
        marker = " *" if is_best else ""

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {avg_train_loss:.4f} | "
            f"HR@5: {hr5:.10f} HR@10: {hr10:.10f} "
            f"NDCG@5: {ndcg5:.10f} NDCG@10: {ndcg10:.10f} | "
            f"best: epoch {best_epoch}{marker}"
        )

        if is_best:
            best_ndcg10 = ndcg10
            best_epoch = epoch
            best_metrics = {
                "hr@5": hr5,
                "hr@10": hr10,
                "ndcg@5": ndcg5,
                "ndcg@10": ndcg10,
            }
            epochs_without_improvement = 0
            # Save best checkpoint
            best_state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "metrics": best_metrics,
                "args": vars(args),
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {args.early_stop_patience} epochs)"
            )
            break

    # ---- Final results ----
    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info(
        f"Best epoch: {best_epoch} | "
        f"HR@5: {best_metrics['hr@5']:.10f} | "
        f"HR@10: {best_metrics['hr@10']:.10f} | "
        f"NDCG@5: {best_metrics['ndcg@5']:.10f} | "
        f"NDCG@10: {best_metrics['ndcg@10']:.10f}"
    )

    # Save best checkpoint
    if best_state_dict is not None:
        ckpt_path = os.path.join(args.output_dir, args.dataset, "best_model.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(best_state_dict, ckpt_path)
        logger.info(f"Best checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
