import math


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        item = int(item)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recall_at_k(predictions, labels, k):
    total = 0.0
    count = 0
    for pred, gt in zip(predictions, labels):
        gt_set = {int(item) for item in gt}
        if not gt_set:
            continue
        pred_set = set(_dedupe(pred)[:k])
        total += len(pred_set & gt_set) / len(gt_set)
        count += 1
    return total / count if count else 0.0


def ndcg_at_k(predictions, labels, k):
    total = 0.0
    count = 0
    for pred, gt in zip(predictions, labels):
        gt_set = {int(item) for item in gt}
        if not gt_set:
            continue
        ranked = _dedupe(pred)[:k]
        dcg = 0.0
        for idx, item in enumerate(ranked):
            if item in gt_set:
                dcg += 1.0 / math.log2(idx + 2)
        ideal_hits = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
        total += dcg / idcg if idcg else 0.0
        count += 1
    return total / count if count else 0.0


def compute_metrics(predictions, labels, cutoffs=(5, 10)):
    metrics = {}
    for k in cutoffs:
        metrics[f"recall@{k}"] = recall_at_k(predictions, labels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, labels, k)
    return metrics


def format_metrics(metrics):
    keys = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
    return " ".join(f"{key}={metrics[key]:.6f}" for key in keys)
