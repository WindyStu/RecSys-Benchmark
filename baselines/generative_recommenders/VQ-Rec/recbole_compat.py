"""Compatibility patches for running RecBole 1.0.x with newer pandas."""

import numpy as np
import pandas as pd


def token_seq_split_points(series: pd.Series) -> np.ndarray:
    """Return row split points for a RecBole TOKEN_SEQ column.

    RecBole 1.0.x used ``series.agg(len)`` here. With newer pandas versions,
    that can return the length of the whole Series instead of per-row lengths.
    """
    return np.cumsum(series.apply(len).to_numpy(dtype=np.int64))[:-1]


def token_seq_lengths(series: pd.Series) -> np.ndarray:
    """Return per-row lengths for a RecBole TOKEN_SEQ column."""
    return series.apply(len).to_numpy(dtype=np.int64)


def apply_benchmark_presets(dataset, feature_type, feature_source) -> None:
    """Apply RecBole benchmark presets with pandas-safe sequence lengths.

    RecBole 1.2.x stores the item-list length field on the instance as
    ``item_list_length_field``. Older VQ-Rec-compatible versions expose similar
    behavior through the instance too, so avoid class-level constants here.
    """
    list_suffix = dataset.config["LIST_SUFFIX"]
    for field in dataset.inter_feat:
        if field + list_suffix in dataset.inter_feat:
            list_field = field + list_suffix
            setattr(dataset, f"{field}{list_suffix}_field", list_field)

    dataset.set_field_property(
        dataset.item_list_length_field,
        feature_type.TOKEN,
        feature_source.INTERACTION,
        1,
    )
    dataset.inter_feat[dataset.item_list_length_field] = token_seq_lengths(
        dataset.inter_feat[dataset.item_id_list_field]
    )


def patch_recbole_token_seq_remap() -> None:
    """Patch RecBole TOKEN_SEQ handling for pandas 2.x/3.x."""
    from recbole.data.dataset.dataset import Dataset
    from recbole.data.dataset.sequential_dataset import SequentialDataset
    from recbole.utils import FeatureSource, FeatureType

    if getattr(Dataset._remap, "_vqrec_pandas_compat", False):
        remap_patched = True
    else:
        remap_patched = False

    def _remap(self, remap_list):
        if len(remap_list) == 0:
            return
        tokens, split_point = self._concat_remaped_tokens(remap_list)
        new_ids_list, mp = pd.factorize(tokens)
        new_ids_list = np.split(new_ids_list + 1, split_point)
        mp = np.array(["[PAD]"] + list(mp))
        token_id = {t: i for i, t in enumerate(mp)}

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = token_id
                self.field2token_id[field] = mp
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = token_seq_split_points(feat[field])
                feat[field] = np.split(new_ids, split_point)

    _remap._vqrec_pandas_compat = True
    if not remap_patched:
        Dataset._remap = _remap

    if getattr(SequentialDataset._benchmark_presets, "_vqrec_pandas_compat", False):
        return

    def _benchmark_presets(self):
        apply_benchmark_presets(self, FeatureType, FeatureSource)

    _benchmark_presets._vqrec_pandas_compat = True
    SequentialDataset._benchmark_presets = _benchmark_presets
