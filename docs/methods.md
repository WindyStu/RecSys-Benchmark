# Methods

Methods are registered by `method_id`, not by folder. One source tree may contain multiple benchmark methods.

## Rankers

Rankers can produce recommendation predictions and enter leaderboard tables.

- ABXI
- MERIT
- EAGER
- Tri-CDR
- SASRec
- BERT4Rec
- SR-GNN
- STOSA
- CF-SASRec
- LETTER-TIGER
- LETTER-LC-Rec
- HSTU / Generative Recommenders
- GenCDR when a runnable recommendation stage is available

## Components

Components are pipeline stages and do not directly enter Recall/NDCG leaderboards.

- LETTER RQ-VAE tokenizer
- GenCDR RQ-VAE/tokenizer stages
- Aliyun text embedding
- Amazon preprocessing
- Cross-domain preprocessing

## Registry

Each method has a YAML file in `configs/methods/` with:

- `method_id`
- `method_type`
- `status`
- `source`
- `adapter`
- `supported_tasks`
- `supported_eval_inputs`
- command templates
- notes and citation metadata when available

## Status

- `source-integrated`: sanitized source, YAML registry, adapter class, and command templates exist. Unified prediction export is not complete yet.
- `adapter-ready`: the method or component has a complete benchmark contract and can be invoked through the adapter interface.
- `runnable`: a real dataset and seed have completed through the benchmark interface and produced artifacts.
- `reproduced`: multi-seed results have been compared with local historical or paper numbers.
- `partial`: only a component or incomplete ranker path is integrated.
