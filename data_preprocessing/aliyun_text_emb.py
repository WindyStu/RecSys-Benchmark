import argparse
import html
import json
import os
import re
import time
from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def clean_text(raw_text) -> str:
    if raw_text is None:
        return ""
    if isinstance(raw_text, list):
        raw_text = " ".join(str(x) for x in raw_text)
    elif isinstance(raw_text, dict):
        raw_text = " ".join(f"{k}: {v}" for k, v in raw_text.items())
    else:
        raw_text = str(raw_text)

    cleaned_text = html.unescape(raw_text.strip())
    cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
    cleaned_text = re.sub(r'["\n\r]+', " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    if not cleaned_text or len(cleaned_text) >= 2000:
        return ""
    return cleaned_text if cleaned_text.endswith(".") else cleaned_text + "."


def load_item_fields(item_path: str, fields: Iterable[str]) -> List[Tuple[int, List[str]]]:
    with open(item_path, "r", encoding="utf-8") as f:
        item2feature = json.load(f)

    item_texts = []
    for item_id, feature in item2feature.items():
        texts = []
        for field in fields:
            if field in feature:
                text = clean_text(feature[field])
                if text:
                    texts.append(text)
        if not texts:
            texts.append(" ")
        item_texts.append((int(item_id), texts))

    item_texts.sort(key=lambda x: x[0])
    actual_ids = [item_id for item_id, _ in item_texts]
    expected_ids = list(range(len(item_texts)))
    if actual_ids != expected_ids:
        raise ValueError("Item ids must be dense and zero-based, so .npy row i aligns with item_id i.")
    return item_texts


def batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def request_embeddings_once(client: OpenAI, args, texts: List[str]) -> np.ndarray:
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.embeddings.create(
                model=args.model,
                input=texts,
                dimensions=args.dimensions,
                timeout=args.request_timeout,
            )
            embeddings = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding
            if any(embedding is None for embedding in embeddings):
                raise RuntimeError("Embedding response is missing one or more inputs.")
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            sleep_seconds = args.retry_sleep * (2**attempt)
            print(
                f"Request failed at attempt {attempt + 1}/{args.max_retries + 1}: {exc}. "
                f"Retry in {sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Embedding request failed after retries: {last_error}") from last_error


def request_embeddings(client: OpenAI, args, texts: List[str]) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), args.api_batch_size):
        chunks.append(request_embeddings_once(client, args, texts[start : start + args.api_batch_size]))
    return np.concatenate(chunks, axis=0)


def embed_batch(client: OpenAI, args, item_batch: List[Tuple[int, List[str]]]):
    item_ids = [item_id for item_id, _ in item_batch]

    if args.field_mode == "join":
        texts = [args.field_separator.join(fields) for _, fields in item_batch]
        return item_ids, request_embeddings(client, args, texts)

    flat_texts = []
    flat_to_item = []
    for item_pos, (_, fields) in enumerate(item_batch):
        for text in fields:
            flat_to_item.append(item_pos)
            flat_texts.append(text)

    flat_embeddings = request_embeddings(client, args, flat_texts)
    grouped = defaultdict(list)
    for flat_idx, item_pos in enumerate(flat_to_item):
        grouped[item_pos].append(flat_embeddings[flat_idx])

    embeddings = []
    for item_pos in range(len(item_batch)):
        embeddings.append(np.stack(grouped[item_pos], axis=0).mean(axis=0))
    return item_ids, np.stack(embeddings, axis=0).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark item text embeddings with Aliyun Bailian API.")
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--root", type=str, default="data", help="Data root containing <dataset>/<dataset>.item.json.")
    parser.add_argument("--item_path", type=str, default="", help="Override item json path.")
    parser.add_argument("--output_path", type=str, default="", help="Override output .npy path.")
    parser.add_argument("--api_key", type=str, default=os.getenv("ALIYUN_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=os.getenv("ALIYUN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", type=str, default="text-embedding-v4")
    parser.add_argument("--dimensions", type=int, default=2048)
    parser.add_argument("--plm_name", type=str, default="qwen-api")
    parser.add_argument("--fields", type=str, nargs="+", default=["title", "description"])
    parser.add_argument("--field_mode", type=str, choices=["average", "join"], default="average")
    parser.add_argument("--field_separator", type=str, default="\n")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of items processed per loop.")
    parser.add_argument("--api_batch_size", type=int, default=10, help="Aliyun embedding API allows at most 10 inputs per request.")
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("Set ALIYUN_API_KEY or pass --api_key before requesting embeddings.")
    if args.api_batch_size <= 0 or args.api_batch_size > 10:
        raise ValueError("--api_batch_size must be in [1, 10].")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    dataset_dir = os.path.join(args.root, args.dataset)
    item_path = args.item_path or os.path.join(dataset_dir, f"{args.dataset}.item.json")
    output_path = args.output_path or os.path.join(dataset_dir, f"{args.dataset}.emb-{args.plm_name}-td.npy")
    progress_path = output_path + ".progress.npy"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    item_texts = load_item_fields(item_path, args.fields)

    if os.path.exists(output_path) and not args.overwrite and not os.path.exists(progress_path):
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite true to regenerate.")

    if args.overwrite or not os.path.exists(output_path):
        embeddings = open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(item_texts), args.dimensions))
        done = np.zeros(len(item_texts), dtype=bool)
    else:
        embeddings = open_memmap(output_path, mode="r+")
        if embeddings.shape != (len(item_texts), args.dimensions):
            raise ValueError(f"Existing output shape mismatch: {embeddings.shape}")
        done = np.load(progress_path).astype(bool)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(f"Dataset: {args.dataset}")
    print(f"Items: {len(item_texts)}")
    print(f"Model: {args.model}, dimensions={args.dimensions}")
    print(f"Fields: {args.fields}, field_mode={args.field_mode}")
    print(f"Output: {output_path}")

    processed_batches = 0
    for item_batch in batched(item_texts, args.batch_size):
        batch_ids = [item_id for item_id, _ in item_batch]
        if all(done[item_id] for item_id in batch_ids):
            continue

        pending = [(item_id, fields) for item_id, fields in item_batch if not done[item_id]]
        item_ids, batch_embeddings = embed_batch(client, args, pending)
        for row, item_id in enumerate(item_ids):
            embeddings[item_id] = batch_embeddings[row]
            done[item_id] = True

        processed_batches += 1
        if processed_batches % args.save_every == 0:
            embeddings.flush()
            np.save(progress_path, done)
        print(f"Processed {int(done.sum())}/{len(item_texts)} items")

    embeddings.flush()
    np.save(progress_path, done)
    if not done.all():
        missing = np.where(~done)[0][:20].tolist()
        raise RuntimeError(f"Embedding generation incomplete. Missing examples: {missing}")

    os.remove(progress_path)
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print(f"Saved item embeddings to {output_path}")


if __name__ == "__main__":
    main()
