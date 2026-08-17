#!/bin/bash
#BSUB -J merit_train
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -M 64000
#BSUB -R "rusage[mem=64000]"
#BSUB -o logs/merit_%J.out
#BSUB -e logs/merit_%J.err

set -euo pipefail

DATASET="${DATASET:-ape}"
LR_G="${LR_G:-0.5}"

cd "$HOME/honglu/research/CHORD/MERIT"
mkdir -p logs checkpoints

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate merit

# auto-preprocess if seq.pkl not found
if [ ! -f "$HOME/honglu/research/CHORD/data/${DATASET}/${DATASET}_50_seq.pkl" ]; then
  echo "[info] seq.pkl not found, running preprocessing..."
  python data/prepare_mapped_seq_data.py \
    --data "$DATASET" \
    --path_data_root "$HOME/honglu/research/CHORD/data" \
    --len_max 50
fi

python main.py \
  --data "$DATASET" \
  --cuda 0 \
  --eval_bs 2 \
  --n_worker 2 \
  --lr_g "$LR_G"
