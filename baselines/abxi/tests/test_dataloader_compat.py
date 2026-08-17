from argparse import Namespace

import numpy as np

from models.data.dataloader import EvalDataset, TrainDataset


def make_args() -> Namespace:
    return Namespace(
        len_trim=3,
        n_neg=2,
        n_item_a=3,
        n_item=6,
        n_mtc=2,
        full_eval=False,
    )


def test_train_dataset_accepts_legacy_four_field_samples() -> None:
    args = make_args()
    sample = (
        np.array([0, 1, 4], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 0, 4], dtype=np.int32),
        np.array([1, 4, 2], dtype=np.int32),
    )
    dataset = TrainDataset(args, [sample])

    seq_x, seq_a, seq_b, gt, gt_neg = dataset[0]

    assert seq_x.tolist() == [0, 1, 4]
    assert seq_a.tolist() == [0, 1, 0]
    assert seq_b.tolist() == [0, 0, 4]
    assert gt.tolist() == [1, 4, 2]
    assert list(gt_neg.shape) == [3, 4]


def test_train_dataset_trims_legacy_samples_to_current_len_trim() -> None:
    args = make_args()
    args.n_item = 8
    sample = (
        np.array([5, 6, 0, 1, 4], dtype=np.int32),
        np.array([0, 0, 0, 1, 0], dtype=np.int32),
        np.array([5, 6, 0, 0, 4], dtype=np.int32),
        np.array([6, 0, 1, 4, 2], dtype=np.int32),
    )
    dataset = TrainDataset(args, [sample])

    seq_x, seq_a, seq_b, gt, gt_neg = dataset[0]

    assert seq_x.tolist() == [0, 1, 4]
    assert seq_a.tolist() == [0, 1, 0]
    assert seq_b.tolist() == [0, 0, 4]
    assert gt.tolist() == [1, 4, 2]
    assert list(gt_neg.shape) == [3, 4]


def test_eval_dataset_accepts_legacy_four_field_samples() -> None:
    args = make_args()
    sample = (
        np.array([0, 1, 4], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 0, 4], dtype=np.int32),
        np.array([2], dtype=np.int32),
    )
    dataset = EvalDataset(args, [sample])

    seq_x, seq_a, seq_b, gt, gt_mtc = dataset[0]

    assert seq_x.tolist() == [0, 1, 4]
    assert seq_a.tolist() == [0, 1, 0]
    assert seq_b.tolist() == [0, 0, 4]
    assert gt.tolist() == [2]
    assert len(gt_mtc) == 2
