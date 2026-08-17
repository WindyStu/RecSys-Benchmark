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
- `source`
- `adapter`
- `supported_tasks`
- `supported_eval_inputs`
- command templates
- notes and citation metadata when available
