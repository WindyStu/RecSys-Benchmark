import numpy as np


def cal_norm_mask(mask):
    """ calculate normalized mask """
    return mask * mask.sum(1).reciprocal().unsqueeze(-1)


def cal_metrics(ranks: list) -> list:
    """ calculate hr/recall, ndcg at k = 5, 10, and mrr. """
    N = len(ranks)
    hr5, hr10, ndcg5, ndcg10, mrr = 0., 0., 0., 0., 0.

    for rank in ranks:
        mrr += 1 / rank
        if rank <= 10:
            hr10 += 1
            ndcg10 += 1 / np.log2(rank + 1)
            if rank <= 5:
                hr5 += 1
                ndcg5 += 1 / np.log2(rank + 1)
    return [x / N for x in (hr5, hr10, ndcg5, ndcg10, mrr)]
