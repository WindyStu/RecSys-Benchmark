#!/bin/bash
#BSUB -J merit_eval_all
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -n 2
#BSUB -M 32000
#BSUB -R "rusage[mem=32000]"
#BSUB -o logs/merit_eval_all.%J.out
#BSUB -e logs/merit_eval_all.%J.err

set -euo pipefail

cd "$HOME/honglu/research/CHORD/MERIT"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate merit

echo "========================================"
echo "  MERIT - Batch Evaluation (with NDCG@5)"
echo "========================================"
echo ""

DATASETS=("ape" "asc" "dbm" "ghk")
# 如果有其他数据集，加进去即可:
# DATASETS=("ape" "asc" "dbm" "ghk" "afk" "abe" "amb")

for dataset in "${DATASETS[@]}"; do
  ckpt="checkpoints/${dataset}-50-d256-seed3407-best.pth"
  
  if [ ! -f "$ckpt" ]; then
    echo "[skip] $dataset - checkpoint not found: $ckpt"
    echo ""
    continue
  fi
  
  echo "----------------------------------------"
  echo "Evaluating: $dataset"
  echo "Checkpoint: $ckpt"
  echo "----------------------------------------"
  
  python eval.py \
    --ckpt "$ckpt" \
    --data "$dataset" \
    --cuda 0 \
    --eval_bs 2 \
    --n_worker 2
  
  echo ""
done

echo "========================================"
echo "  All evaluations completed!"
echo "========================================"
