# RecSys-Benchmark

RecSys-Benchmark is a unified wrapper benchmark for single-domain sequential recommendation (SDSR) and cross-domain sequential recommendation (CDSR). It keeps each paper implementation mostly intact, while standardizing experiment configuration, adapter execution, prediction formats, evaluation, and result aggregation.

## Highlights

- Unified `ModelAdapter` layer without forcing every paper into one `Dataset / Model / Trainer` API.
- YAML experiment configs with CLI overrides.
- Unified evaluator for full-ranking and sampled-ranking protocols.
- Unified result aggregator with multi-seed `mean +/- std`.
- Explicit `ranker` and `component` method registry.
- Tiny toy SDSR/CDSR datasets for smoke tests.
- Open-source checklist for licenses, secrets, data, checkpoints, and large files.

## Supported Tasks

| Task | Description | Default protocol |
| --- | --- | --- |
| SDSR | Single-domain sequential recommendation | Full ranking |
| CDSR | Cross-domain sequential recommendation over mixed-domain user sequences | Full ranking |

Sampled ranking is also supported for reproducing paper-specific protocols such as 999 negative samples.

## Method Zoo

Integrated rankers are methods intended to enter leaderboard tables after their adapters export `candidate_scores` or `topk` predictions.

| Method ID | Method | Type | Source tree | Status |
| --- | --- | --- | --- | --- |
| `abxi` | ABXI | ranker | `baselines/abxi` | adapter-ready |
| `merit` | MERIT | ranker | `baselines/merit` | adapter-ready |
| `eager` | EAGER | ranker | `baselines/eager` | adapter-ready |
| `tri_cdr` | Tri-CDR | ranker | `baselines/tri_cdr` | source-integrated |
| `sasrec` | SASRec | ranker | `baselines/sasrec_bert4rec_st_loo` | adapter-ready |
| `bert4rec` | BERT4Rec | ranker | `baselines/sasrec_bert4rec_st_loo` | adapter-ready |
| `sr_gnn` | SR-GNN | ranker | `baselines/sasrec_bert4rec_st_loo` | partial |
| `stosa` | STOSA | ranker | `baselines/sasrec_bert4rec_st_loo` | adapter-ready |
| `cf_sasrec` | CF-SASRec | ranker | `baselines/letter/CF-SASRec` | adapter-ready |
| `letter_tiger` | LETTER-TIGER | ranker | `baselines/letter/LETTER-TIGER` | adapter-ready |
| `letter_lc_rec` | LETTER-LC-Rec | ranker | `baselines/letter/LETTER-LC-Rec` | adapter-ready |
| `hstu` | HSTU / Generative Recommenders | ranker | `baselines/generative_recommenders` | adapter-ready |
| `gencdr` | GenCDR | ranker | `baselines/gencdr` | adapter-ready |

Status meanings:

- `source-integrated`: sanitized source, method YAML, command adapter, and train command are present; unified prediction export is still missing.
- `adapter-ready`: the method/component has a complete adapter contract and can be invoked through the benchmark interface.
- `runnable`: at least one real dataset/seed has completed train or predict through the benchmark and produced unified artifacts.
- `reproduced`: multi-seed results have been checked against local or paper numbers.
- `partial`: only part of the method pipeline is integrated, or the copied source lacks a complete ranker entrypoint.

Pipeline components are tracked but do not directly enter Recall/NDCG leaderboards.

| Method ID | Component | Purpose |
| --- | --- | --- |
| `letter_rqvae_tokenizer` | LETTER RQ-VAE tokenizer | Learn semantic item IDs |
| `gencdr_rqvae` | GenCDR RQ-VAE | Generate quantized semantic codes |
| `gencdr_adapter_tokenizer` | GenCDR adapter tokenizer | Train domain-adaptive tokenizer stages |
| `text_embedding_aliyun` | Aliyun text embedding | Encode item text vectors |
| `data_preprocess_amazon` | Amazon preprocessing | Extract SDSR sequences and item text |
| `data_preprocess_cross_domain` | CDSR preprocessing | Build mixed-domain sequences and ID maps |

## Dataset Preparation

Real datasets are not committed to this repository.

SDSR raw sources:

- Amazon category 5-core review files and category metadata from [UCSD Amazon datasets](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
- Douban rating/review/side-information data from [Kaggle Douban Dataset](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information).

CDSR datasets combine two related domains into one mixed sequence dataset. For example, `asc` means Amazon Sports and Clothing. `map_item.txt` maps raw item IDs to mapped item IDs and domain labels `0/1`; `map_user.txt` maps raw platform user IDs to benchmark user IDs.

Expected processed layouts:

```text
data/SDSR/<dataset>/
├── <dataset>.inter.json
├── <dataset>.item.json
├── <dataset>.index.json
└── <dataset>.emb-*.npy        # optional

data/CDSR/<pair>/
├── <pair>.inter.json
├── <pair>.item.json
├── <pair>.index.json          # optional
├── <pair>.emb-*.npy           # optional
├── map_item.txt
└── map_user.txt
```

Preprocessing scripts come from the local `my_letter/data_process` codebase and are copied in sanitized form. Any text embedding key must be passed through environment variables such as `ALIYUN_API_KEY`; never hard-code keys in source files.

## Quick Start

Install:

```bash
python -m pip install -e .
```

Run tests:

```bash
python -m unittest discover -s tests
```

Evaluate the toy SDSR prediction file:

```bash
python -m recsys_benchmark evaluate \
  --predictions examples/toy_sdsr/BeautyToy/topk.csv \
  --ground-truth examples/toy_sdsr/BeautyToy/ground_truth.csv \
  --output outputs/runs/toy_sdsr/metrics.json \
  --input-type topk \
  --cutoffs 5 10 \
  --method-id toy_topk \
  --dataset BeautyToy \
  --task sdsr \
  --protocol full \
  --seed 1 \
  --item-metadata examples/toy_sdsr/BeautyToy/item_metadata.csv
```

Aggregate results:

```bash
python -m recsys_benchmark aggregate \
  --results outputs/runs \
  --output-csv results/leaderboard.csv \
  --output-md results/leaderboard.md
```

Dry-run a configured method adapter:

```bash
python -m recsys_benchmark run \
  --config configs/experiments/toy_sdsr_sasrec_dryrun.yaml \
  --dry-run
```

Run a SASRec-family method on local Beauty data after setting `RECSYS_DATA_ROOT` to the directory that contains `data_SDSR`:

```bash
python -m recsys_benchmark run \
  --config configs/experiments/beauty_sasrec.yaml \
  --override method.defaults.cuda=0
```

Use `beauty_bert4rec.yaml` or `beauty_stosa.yaml` for the other two methods. These adapters prepare the baseline-specific sequence files, invoke the original trainer, and collect the final native HR/NDCG/MRR row into `metrics.json`. Their status remains below `runnable` until a real server run succeeds.

Run a dual-domain ranker on ASC data with `configs/experiments/asc_abxi.yaml` or `configs/experiments/asc_merit.yaml`. The prepare stage remaps the processed mixed sequence into each original implementation's contiguous, 1-based two-domain format. Native A/B metrics are retained as per-domain metrics and macro-averaged for the primary leaderboard columns.

Run EAGER with `configs/experiments/beauty_eager.yaml`. Its adapter executes DIN pretraining, EAGER training, and a separate full-ranking evaluation with seen-item filtering. Override `method.defaults.din_batches` and `method.defaults.eager_batches` for smoke runs; benchmark results should use the documented defaults or explicitly record the overrides.

Run CF-SASRec with `configs/experiments/beauty_cf_sasrec.yaml`. The adapter links the local dataset into the layout expected by the original code, trains with an explicit seed, exports item embeddings, and collects the test metrics selected by the best validation NDCG.

Run LETTER-TIGER or LETTER-LC-Rec with `beauty_letter_tiger.yaml` or `beauty_letter_lc_rec.yaml`. Supply the external pretrained model without committing a private path, for example `--override method.defaults.base_model=/models/t5-base`. Their train and predict stages are separate, and the generated `mean_results` are normalized into the benchmark metric schema.

Run HSTU with `configs/experiments/beauty_hstu.yaml`. This uses the repository's SDSR-specific entrypoint and gin architecture config, then reads full-ranking HR/NDCG from the best checkpoint. Install the baseline's CUDA/fbgemm dependencies in a dedicated environment before the server run.

Run GenCDR with `configs/experiments/asc_gencdr.yaml`. The prepare stage trains its RQ-VAE tokenizer from the processed item embeddings and writes `asc.index.json` beside the local dataset. The train and predict stages then run the included cross-domain generative ranker and normalize its generated-ranking HR/NDCG JSON. Supply the external T5 model with `--override method.defaults.base_model=/models/t5-base`; no model or private path is committed.

Inspect method readiness:

```bash
python -m recsys_benchmark inspect-methods \
  --methods configs/methods \
  --output outputs/method_readiness.json
```

## Unified Configuration

Configs are split into:

- `configs/datasets/*.yaml`: dataset paths, task type, domains, and metadata files.
- `configs/methods/*.yaml`: method ID, method type, source tree, adapter class, supported tasks, and command templates.
- `configs/experiments/*.yaml`: dataset-method pairing, seed, protocol, cutoffs, and output root.

CLI overrides use dotted keys:

```bash
python -m recsys_benchmark run --config configs/experiments/toy_sdsr_sasrec_dryrun.yaml --override seed=0 --override evaluation.protocol=sampled
```

## Prediction Formats

`candidate_scores`:

```text
user_id,item_id,score,domain(optional),split
```

`topk`:

```text
user_id,rank,item_id,score(optional),domain(optional),split
```

`candidate_scores` is preferred because the unified evaluator can apply consistent filtering and ranking. `topk` is supported for generative methods or original code that only exports ranked lists. Leaderboards must keep these input types separate.

## Evaluation Protocols

- `full`: rank all candidate items in the evaluation candidate pool, filter training-history seen items, then compute metrics. This is the default.
- `sampled`: evaluate one positive target against sampled negatives. Record negative count, seed, and sampling strategy.

Do not compare `full` and `sampled` rows in the same primary leaderboard.

## Metrics

For user `u`, let `R_u^K` be the top-K recommended list and `G_u` be the ground-truth relevant set.

- `Recall@K = |R_u^K intersect G_u| / |G_u|`
- `Precision@K = |R_u^K intersect G_u| / K`
- `HitRate@K = 1[|R_u^K intersect G_u| > 0]`
- `DCG@K = sum_i rel_i / log2(i + 1)`
- `NDCG@K = DCG@K / IDCG@K`
- `MRR@K = 1 / rank(first relevant item)`, or `0` if there is no hit.
- `MAP@K` averages precision values at relevant hit positions up to K.
- `ItemCoverage@K` is the count of unique recommended items.
- `CatalogCoverage@K = unique recommended items / catalog size`.
- `Novelty@K` averages `-log2(item popularity probability)`.
- `IntraListDiversity@K` averages pairwise item distance when item vectors or categories exist; otherwise `N/A`.
- `DomainRecall@K`, `DomainNDCG@K`, and `DomainCoverage@K` compute the same ideas inside each domain.
- `DomainMixRatio@K` reports the recommendation share of each domain.
- `CrossDomainTransferGap@K` is the difference between the best and worst domain recall.

See `docs/metrics.md` for details.

## Adding a New Method

1. Copy or sanitize the source into `baselines/<method>/`.
2. Add `configs/methods/<method>.yaml`.
3. Implement or reuse an adapter under `recsys_benchmark/adapters/`.
4. Make the adapter export `candidate_scores` or `topk`.
5. Run the unified evaluator and aggregator.
6. Document dependencies and citation metadata.

## Open-source Safety

Before publishing to GitHub:

- Run `python scripts/check_secrets.py .`.
- Confirm no real data, embeddings, checkpoints, logs, or private configs are tracked.
- Audit licenses for every copied baseline.
- Rotate any API key that was ever hard-coded locally.
- Keep generated outputs outside Git.

See `docs/open_source_checklist.md`.
