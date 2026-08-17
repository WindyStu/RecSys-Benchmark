# MERIT-pytorch

This is the ***official*** Pytorch implementation of paper "Multi-Domain Enhancement via Residual Interwoven Transfer in Cross-Domain Sequential Recommendation" accepted by ACM Multimedia 2025.

**NOTICE**: we largely refractor the code to improve speed, training stability, and code readability. Hence, the tuned hyperparameter provided in `demo.sh` may no longer be optimal.

If you encounter performance issues, please submit an issue as soon as possible, as I currently do not have enough time to fully test the new code.

## 1. Data
In argument `--data`, `afk` refers to Amazon Food-Kitchen dataset, `amb` refers to Amazon Movie-Book dataset, and `abe` refers to Amazon Beauty-Electronics dataset.

Processed data are stored in `/data/`. If you wanna process your own data, please put the data under /data/raw/, and check the preprocess scripts `/utils/preprocess.py`.

## 2. Usage
Please check demo.sh on running on different datasets.

This local reproduction defaults to full-ranking evaluation (`--eval_mode full`) instead of sampled
999-negative evaluation. By default it looks for datasets in a sibling `../data` directory, so from
`E:\CHORD\MERIT` it will read processed files such as `E:\CHORD\data\ape\ape_50_seq.pkl`.

Example:
```
python main.py --data ape --cuda 0
python main.py --data asc --cuda 0
python main.py --data dbm --cuda 0
```

Use `--eval_bs` to reduce evaluation memory usage during full ranking. Use
`--eval_mode sampled --n_mtc 999` only if you need to reproduce the original sampled protocol.


## 3. File Tree
```
MERIT/
├── data/
│   ├── abe/
│   │   ├── abe_50_preprocessed.txt
│   │   ├── abe_50_seq.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   ├── afk/
│   │   ├── afk_50_preprocessed.txt
│   │   ├── afk_50_seq.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   ├── amb/
│   │   ├── amb_50_preprocessed.txt
│   │   ├── amb_50_seq.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   ├── mapper_raw_file.py
│   └── prepare_amazon_data.py
├── models/
│   ├── data/
│   │   ├── dataloader.py
│   │   └── evaluation.py
│   ├── utils/
│   │   ├── initialization.py
│   │   └── position.py
│   ├── attention.py
│   ├── ffn.py
│   └── MERIT.py
├── demo.sh
├── main.py
├── noter.py
├── README.md
├── requirements.txt
└── trainer.py
```
