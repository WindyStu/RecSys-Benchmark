# Preprocessing

Reusable preprocessing scripts live in the repository-level `data_preprocessing/` directory. They are independent of paper baseline implementations under `baselines/`.

## SDSR

1. Download Amazon 5-core review files and metadata, or Douban data.
2. Extract sequential interactions into `<dataset>.inter.json`.
3. Extract item text and metadata into `<dataset>.item.json`.
4. Optionally generate semantic IDs in `<dataset>.index.json`.
5. Optionally encode item text into `<dataset>.emb-*.npy`.

Set `RECSYS_RAW_DATA_ROOT` or pass `--data_root` to `data_preprocessing/preprocess_amazon.py`.

## CDSR

1. Choose two related domains.
2. Map raw users into a shared user ID space.
3. Map raw items into a shared item ID space plus a domain label.
4. Build mixed-domain user sequences.
5. Save `map_item.txt` and `map_user.txt`.
6. Optionally encode the merged item text file into text embeddings.

Set `AMAZON_RAW_ROOT` and/or `DOUBAN_RAW_ROOT`, then run `data_preprocessing/preprocess_cross_domain.py`.

## Secrets

Embedding APIs must read keys from environment variables:

```powershell
$env:ALIYUN_API_KEY="..."
$env:ALIYUN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" # optional
```

Never store keys in source code or YAML configs.
