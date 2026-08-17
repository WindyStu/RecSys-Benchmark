# Data

This benchmark documents reproducible data preparation but does not redistribute real datasets.

## SDSR

Single-domain datasets are prepared from:

- Amazon 5-core category reviews and category metadata.
- Douban rating/review/side-information data.

The processed SDSR format is:

```text
<dataset>.inter.json
<dataset>.item.json
<dataset>.index.json
<dataset>.emb-*.npy optional
```

## CDSR

Cross-domain datasets merge two domains into a mixed user sequence. Example: `asc` means Amazon Sports and Clothing.

The processed CDSR format is:

```text
<pair>.inter.json
<pair>.item.json
<pair>.index.json optional
<pair>.emb-*.npy optional
map_item.txt
map_user.txt
```

`map_item.txt` stores raw item ID, mapped item ID, and domain label. `map_user.txt` stores raw user ID and mapped user ID.

## Local Paths

Example local paths used during development:

```powershell
$env:RECSYS_DATA_ROOT="D:\data\data"
```

Then SDSR and CDSR are expected under:

```text
D:\data\data\data_SDSR
D:\data\data\data_CDSR
```
