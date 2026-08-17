# Metrics

For each user `u`, `R_u^K` is the top-K recommendation list and `G_u` is the ground-truth set.

## Accuracy

- `Recall@K = |R_u^K intersect G_u| / |G_u|`
- `Precision@K = |R_u^K intersect G_u| / K`
- `HitRate@K = 1[|R_u^K intersect G_u| > 0]`
- `DCG@K = sum_i rel_i / log2(i + 1)`
- `NDCG@K = DCG@K / IDCG@K`
- `MRR@K = 1 / rank(first relevant item)`, or `0` if no relevant item appears before K.
- `MAP@K` averages precision values at ranks where relevant items appear, normalized by `min(|G_u|, K)`.

Metrics are averaged over users.

## Coverage, Novelty, Diversity

- `ItemCoverage@K`: number of unique recommended items.
- `CatalogCoverage@K`: `ItemCoverage@K / catalog size`.
- `Novelty@K`: average `-log2(popularity(item) / total interactions)`.
- `IntraListDiversity@K`: average pairwise distance between items in each top-K list. It requires item vectors, categories, or another similarity source; otherwise it is `N/A`.

## Cross-domain

- `DomainRecall@K:<domain>`: recall over ground-truth items from a domain.
- `DomainNDCG@K:<domain>`: NDCG over ground-truth items from a domain.
- `DomainCoverage@K:<domain>`: recommended unique items in a domain divided by catalog items in that domain.
- `DomainMixRatio@K:<domain>`: fraction of recommended top-K items assigned to a domain.
- `CrossDomainTransferGap@K`: maximum domain recall minus minimum domain recall.

## Engineering

Adapters may record:

- `TrainTime`
- `EvalTime`
- `PeakGPU/CPU Memory`
- `Params`

Unavailable values are stored as `N/A`.
