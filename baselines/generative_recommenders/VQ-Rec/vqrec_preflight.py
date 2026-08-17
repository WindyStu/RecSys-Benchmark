"""Preflight checks for prepared VQ-Rec RecBole files."""

import os


INTER_HEADER = (
    "user_id:token\titem_id_list:token_seq\titem_id:token\ttimestamp:float"
)


def validate_prepared_inter_file(path: str, max_item_list_length: int) -> None:
    """Validate one prepared .inter file before RecBole loads it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prepared RecBole file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != INTER_HEADER:
            raise ValueError(
                f"{path}: invalid RecBole header {header!r}; expected {INTER_HEADER!r}"
            )

        rows = 0
        for line_no, raw_line in enumerate(f, start=2):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{line_no}: expected 4 tab-separated columns, got {len(parts)}"
                )
            history = parts[1].split()
            if not history:
                raise ValueError(
                    f"{path}:{line_no}: empty item_id_list. Regenerate data with the "
                    "latest VQ-Rec/prepare_data.py so first interactions are skipped."
                )
            if len(history) > max_item_list_length:
                raise ValueError(
                    f"{path}:{line_no}: item_id_list length {len(history)} exceeds "
                    f"MAX_ITEM_LIST_LENGTH={max_item_list_length}. Regenerate data with "
                    f"--max_item_list_length {max_item_list_length}."
                )
            rows += 1

    if rows == 0:
        raise ValueError(f"{path}: no interaction rows found after the header")


def validate_prepared_inter_files(
    data_path: str,
    dataset: str,
    max_item_list_length: int,
) -> None:
    """Validate benchmark train/valid/test files for one dataset."""
    for suffix in ("train", "valid", "test"):
        validate_prepared_inter_file(
            os.path.join(data_path, f"{dataset}.{suffix}.inter"),
            max_item_list_length,
        )
