from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence


def evaluate_recommendations(
    recommendations: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Iterable[str]],
    cutoffs: Sequence[int],
    catalog_items: Iterable[str] | None = None,
    item_domains: Mapping[str, str] | None = None,
    item_popularity: Mapping[str, int | float] | None = None,
    item_vectors: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, float | str]:
    """Compute ranking metrics from per-user ranked recommendation lists."""
    normalized_truth = {user: set(items) for user, items in ground_truth.items()}
    users = sorted(normalized_truth)
    if not users:
        raise ValueError("ground_truth must contain at least one user")
    if not cutoffs:
        raise ValueError("cutoffs must not be empty")

    max_k = max(cutoffs)
    metrics: dict[str, float | str] = {}

    for k in sorted(set(cutoffs)):
        recall_values = []
        precision_values = []
        hit_values = []
        ndcg_values = []
        ap_values = []
        reciprocal_ranks = []

        for user in users:
            truth = normalized_truth[user]
            ranked = list(recommendations.get(user, []))[:k]
            hits = [1 if item in truth else 0 for item in ranked]
            num_hits = sum(hits)

            recall_values.append(num_hits / len(truth) if truth else 0.0)
            precision_values.append(num_hits / k)
            hit_values.append(1.0 if num_hits > 0 else 0.0)
            ndcg_values.append(_ndcg(hits, min(len(truth), k)))
            ap_values.append(_average_precision(hits, min(len(truth), k)))

            rr = 0.0
            for index, hit in enumerate(hits, start=1):
                if hit:
                    rr = 1.0 / index
                    break
            reciprocal_ranks.append(rr)

        metrics[f"recall@{k}"] = _mean(recall_values)
        metrics[f"precision@{k}"] = _mean(precision_values)
        metrics[f"hitrate@{k}"] = _mean(hit_values)
        metrics[f"ndcg@{k}"] = _mean(ndcg_values)
        metrics[f"map@{k}"] = _mean(ap_values)
        metrics[f"mrr@{k}"] = _mean(reciprocal_ranks)

        unique_recommended = _unique_topk_items(recommendations, users, k)
        metrics[f"itemcoverage@{k}"] = float(len(unique_recommended))
        if catalog_items is not None:
            catalog = set(catalog_items)
            metrics[f"catalogcoverage@{k}"] = len(unique_recommended) / len(catalog) if catalog else 0.0

        if item_popularity is not None:
            metrics[f"novelty@{k}"] = _novelty(recommendations, users, k, item_popularity)

        if item_vectors is not None:
            metrics[f"intralistdiversity@{k}"] = _intra_list_diversity(recommendations, users, k, item_vectors)
        else:
            metrics[f"intralistdiversity@{k}"] = "N/A"

        if item_domains is not None:
            metrics.update(_domain_metrics(recommendations, normalized_truth, users, k, item_domains, catalog_items))

    if max_k not in cutoffs:
        metrics[f"mrr@{max_k}"] = metrics[f"mrr@{max_k}"]
    return metrics


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ndcg(hits: Sequence[int], ideal_hits: int) -> float:
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _average_precision(hits: Sequence[int], ideal_hits: int) -> float:
    if ideal_hits == 0:
        return 0.0
    precision_sum = 0.0
    hit_count = 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / ideal_hits


def _unique_topk_items(recommendations: Mapping[str, Sequence[str]], users: Sequence[str], k: int) -> set[str]:
    unique_items: set[str] = set()
    for user in users:
        unique_items.update(list(recommendations.get(user, []))[:k])
    return unique_items


def _novelty(
    recommendations: Mapping[str, Sequence[str]],
    users: Sequence[str],
    k: int,
    item_popularity: Mapping[str, int | float],
) -> float:
    total_popularity = sum(max(float(value), 0.0) for value in item_popularity.values())
    if total_popularity <= 0:
        return 0.0

    novelty_values = []
    fallback = 1.0 / total_popularity
    for user in users:
        for item in list(recommendations.get(user, []))[:k]:
            probability = max(float(item_popularity.get(item, 0.0)) / total_popularity, fallback)
            novelty_values.append(-math.log2(probability))
    return _mean(novelty_values)


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return 1.0 - dot / (left_norm * right_norm)


def _intra_list_diversity(
    recommendations: Mapping[str, Sequence[str]],
    users: Sequence[str],
    k: int,
    item_vectors: Mapping[str, Sequence[float]],
) -> float | str:
    diversity_values = []
    for user in users:
        ranked = [item for item in list(recommendations.get(user, []))[:k] if item in item_vectors]
        if len(ranked) < 2:
            continue
        pair_distances = []
        for left_index, left_item in enumerate(ranked):
            for right_item in ranked[left_index + 1 :]:
                pair_distances.append(_cosine_distance(item_vectors[left_item], item_vectors[right_item]))
        diversity_values.append(_mean(pair_distances))
    return _mean(diversity_values) if diversity_values else "N/A"


def _domain_metrics(
    recommendations: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, set[str]],
    users: Sequence[str],
    k: int,
    item_domains: Mapping[str, str],
    catalog_items: Iterable[str] | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    recommended_domain_counts: Counter[str] = Counter()
    recommended_by_domain: defaultdict[str, set[str]] = defaultdict(set)
    total_recommended = 0

    for user in users:
        for item in list(recommendations.get(user, []))[:k]:
            domain = item_domains.get(item)
            if domain is None:
                continue
            recommended_domain_counts[domain] += 1
            recommended_by_domain[domain].add(item)
            total_recommended += 1

    for domain, count in sorted(recommended_domain_counts.items()):
        metrics[f"domainmixratio@{k}:{domain}"] = count / total_recommended if total_recommended else 0.0

    domain_recall_values: dict[str, float] = {}
    domains_with_truth = sorted({item_domains[item] for truth in ground_truth.values() for item in truth if item in item_domains})
    for domain in domains_with_truth:
        recalls = []
        ndcgs = []
        for user in users:
            domain_truth = {item for item in ground_truth[user] if item_domains.get(item) == domain}
            if not domain_truth:
                continue
            ranked = list(recommendations.get(user, []))[:k]
            hits = [1 if item in domain_truth else 0 for item in ranked]
            recalls.append(sum(hits) / len(domain_truth))
            ndcgs.append(_ndcg(hits, min(len(domain_truth), k)))
        if recalls:
            recall = _mean(recalls)
            domain_recall_values[domain] = recall
            metrics[f"domainrecall@{k}:{domain}"] = recall
            metrics[f"domainndcg@{k}:{domain}"] = _mean(ndcgs)

    if catalog_items is not None:
        catalog_by_domain: defaultdict[str, set[str]] = defaultdict(set)
        for item in catalog_items:
            domain = item_domains.get(item)
            if domain is not None:
                catalog_by_domain[domain].add(item)
        for domain, domain_catalog in sorted(catalog_by_domain.items()):
            recommended = recommended_by_domain.get(domain, set())
            metrics[f"domaincoverage@{k}:{domain}"] = len(recommended) / len(domain_catalog) if domain_catalog else 0.0

    if len(domain_recall_values) > 1:
        metrics[f"crossdomaintransfergap@{k}"] = max(domain_recall_values.values()) - min(domain_recall_values.values())
    else:
        metrics[f"crossdomaintransfergap@{k}"] = 0.0
    return metrics
