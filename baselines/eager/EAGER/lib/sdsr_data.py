import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SdsrSummary:
    dataset: str
    data_dir: Path
    output_dir: Path
    user_num: int
    item_num: int
    train_instances: int
    validation_instances: int
    test_instances: int


def _load_sequences(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sequences = []
    for user, seq in raw.items():
        items = [int(item) for item in seq]
        sequences.append((int(user), items))
    sequences.sort(key=lambda pair: pair[0])
    return sequences


def _infer_item_num(data_dir, dataset, sequences):
    max_item = max((item for _, seq in sequences for item in seq), default=-1)
    item_file = data_dir / f"{dataset}.item.json"
    if item_file.exists():
        with open(item_file, "r", encoding="utf-8") as f:
            item_json = json.load(f)
        item_keys = [int(key) for key in item_json.keys()]
        if item_keys:
            max_item = max(max_item, max(item_keys))
    return max_item + 1


def _format_history(history, history_len):
    history = list(history[-history_len:])
    if len(history) < history_len:
        history = [-1] * (history_len - len(history)) + history
    return ",".join(str(item) for item in history)


def _write_segmented_train(rows, output_dir, train_sample_seg_cnt, seed):
    rng = random.Random(seed)
    rng.shuffle(rows)
    segment_rows = [[] for _ in range(train_sample_seg_cnt)]
    for idx, row in enumerate(rows):
        segment_rows[idx % train_sample_seg_cnt].append(row)
    for idx, rows_for_segment in enumerate(segment_rows):
        with open(output_dir / f"train_instances_{idx}", "w", encoding="utf-8") as f:
            for row in rows_for_segment:
                f.write(row)


def _write_eval(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row)


def prepare_sdsr_domain(
    data_dir,
    output_dir,
    dataset,
    seq_len=20,
    min_seq_len=5,
    train_sample_seg_cnt=10,
    seed=2024,
    force=False,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    inter_file = data_dir / f"{dataset}.inter.json"
    if not inter_file.exists():
        raise FileNotFoundError(f"missing SDSR interaction file: {inter_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    item_num_file = output_dir / "item_node_num.txt"
    expected_files = [output_dir / f"train_instances_{idx}" for idx in range(train_sample_seg_cnt)]
    expected_files += [output_dir / "validation_instances", output_dir / "test_instances", item_num_file]
    if not force and all(path.exists() for path in expected_files):
        user_num, item_num = np.loadtxt(item_num_file, dtype=np.int32, delimiter=",")
        return SdsrSummary(dataset, data_dir, output_dir, int(user_num), int(item_num), -1, -1, -1)

    sequences = _load_sequences(inter_file)
    item_num = _infer_item_num(data_dir, dataset, sequences)
    history_len = seq_len - 1
    train_rows = []
    validation_rows = []
    test_rows = []

    for user, seq in sequences:
        if len(seq) < min_seq_len:
            continue

        for label_idx in range(min_seq_len - 1, len(seq) - 2):
            history = _format_history(seq[:label_idx], history_len)
            train_rows.append(f"{user}|{history}|{seq[label_idx]}\n")

        validation_history = _format_history(seq[:-2], history_len)
        validation_rows.append(f"{user}|{validation_history}|{seq[-2]}\n")

        test_history = _format_history(seq[:-1], history_len)
        test_rows.append(f"{user}|{test_history}|{seq[-1]}\n")

    _write_segmented_train(train_rows, output_dir, train_sample_seg_cnt, seed)
    _write_eval(validation_rows, output_dir / "validation_instances")
    _write_eval(test_rows, output_dir / "test_instances")
    np.savetxt(item_num_file, np.array([len(sequences), item_num]), fmt="%d", delimiter=",")

    return SdsrSummary(
        dataset=dataset,
        data_dir=data_dir,
        output_dir=output_dir,
        user_num=len(sequences),
        item_num=item_num,
        train_instances=len(train_rows),
        validation_instances=len(validation_rows),
        test_instances=len(test_rows),
    )


def load_semantic_features(data_dir, dataset, item_num):
    data_dir = Path(data_dir)
    embedding_file = data_dir / f"{dataset}.emb-qwen-api-td.npy"
    if embedding_file.exists():
        features = np.load(embedding_file).astype(np.float32)
        if features.shape[0] < item_num:
            pad = np.zeros((item_num - features.shape[0], features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)
        return features[:item_num]

    index_file = data_dir / f"{dataset}.index.json"
    if not index_file.exists():
        ids = np.arange(item_num, dtype=np.float32).reshape(-1, 1)
        return ids / max(item_num - 1, 1)

    with open(index_file, "r", encoding="utf-8") as f:
        index_json = json.load(f)
    width = max((len(value) for value in index_json.values()), default=1)
    features = np.zeros((item_num, width), dtype=np.float32)
    token_re = re.compile(r"_(\d+)>$")
    for key, tokens in index_json.items():
        item_id = int(key)
        if item_id >= item_num:
            continue
        values = []
        for token in tokens:
            match = token_re.search(str(token))
            values.append(float(match.group(1)) if match else 0.0)
        features[item_id, : len(values)] = values
    max_value = max(float(features.max()), 1.0)
    return features / max_value
