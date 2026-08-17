#!/bin/bash
#BSUB -J merit_ghk
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -M 64000
#BSUB -R "rusage[mem=64000]"
#BSUB -o logs/merit_ghk.%J.out
#BSUB -e logs/merit_ghk.%J.err

set -euo pipefail

cd "$HOME/honglu/research/CHORD/MERIT"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate merit

if [ ! -f "$HOME/honglu/research/CHORD/data/ghk/ghk_50_seq.pkl" ]; then
  python data/prepare_mapped_seq_data.py --data ghk --path_data_root "$HOME/honglu/research/CHORD/data" --len_max 50
fi

python main.py \
  --data ghk \
  --cuda 0 \
  --eval_bs 2 \
  --n_worker 2
