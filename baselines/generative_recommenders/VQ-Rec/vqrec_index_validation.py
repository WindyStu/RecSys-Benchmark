"""Index bound checks for VQ-Rec CUDA indexing operations."""

import torch


def _tensor_min_max(values: torch.Tensor) -> tuple[int, int]:
    detached = values.detach()
    return int(detached.min().item()), int(detached.max().item())


def _raise_out_of_bounds(
    name: str,
    min_value: int,
    max_value: int,
    lower_bound: int,
    upper_bound: int,
) -> None:
    raise ValueError(
        f"{name} index out of bounds: min={min_value}, max={max_value}, "
        f"expected {lower_bound} <= index < {upper_bound}"
    )


def validate_forward_indices(
    item_seq: torch.Tensor,
    item_seq_len: torch.Tensor,
    pq_codes: torch.Tensor,
    max_position_embeddings: int,
    code_embedding_rows: int,
) -> None:
    """Validate VQ-Rec forward indices before CUDA gather/embedding kernels."""
    if item_seq.size(1) > max_position_embeddings:
        raise ValueError(
            f"item_seq width {item_seq.size(1)} exceeds MAX_ITEM_LIST_LENGTH/"
            f"position embeddings {max_position_embeddings}. Regenerate data with "
            f"--max_item_list_length {max_position_embeddings} and remove old "
            "RecBole processed caches."
        )

    seq_len_min, seq_len_max = _tensor_min_max(item_seq_len)
    if seq_len_min < 1 or seq_len_max > item_seq.size(1):
        _raise_out_of_bounds(
            "item_seq_len",
            seq_len_min,
            seq_len_max,
            1,
            item_seq.size(1) + 1,
        )

    item_min, item_max = _tensor_min_max(item_seq)
    if item_min < 0 or item_max >= pq_codes.size(0):
        _raise_out_of_bounds("item_seq", item_min, item_max, 0, pq_codes.size(0))

    used_pq_codes = pq_codes[item_seq]
    pq_min, pq_max = _tensor_min_max(used_pq_codes)
    if pq_min < 0 or pq_max >= code_embedding_rows:
        _raise_out_of_bounds("pq_codes", pq_min, pq_max, 0, code_embedding_rows)


def validate_ce_targets(
    targets: torch.Tensor,
    num_classes: int,
    field_name: str,
) -> None:
    """Validate CrossEntropyLoss targets before CUDA loss kernels."""
    target_min, target_max = _tensor_min_max(targets)
    if target_min < 0 or target_max >= num_classes:
        _raise_out_of_bounds(f"{field_name} target", target_min, target_max, 0, num_classes)
