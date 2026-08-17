#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$ROOT_DIR/.." && pwd)
cd "$REPO_ROOT"
mkdir -p ./CF-SASRec/log ./CF-SASRec/lsflog ./CF-SASRec/runs ./CF-SASRec/results ./CF-SASRec/cache ./RQ-VAE/ckpt

DATASET=Beauty
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./CF-SASRec/log/${DATASET}_${RUN_TS}.log"
LSF_OUT="./CF-SASRec/lsflog/${DATASET}_${RUN_TS}.out"
LSF_ERR="./CF-SASRec/lsflog/${DATASET}_${RUN_TS}.err"
: > "$LOG_FILE"

bsub -N -q volta \
-e "$LSF_ERR" \
-o "$LSF_OUT" \
-n 1 \
-gpu "num=1:mode=exclusive_process" \
"bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=CF-SASRec train dataset=${DATASET} time=${RUN_TS}\"; python CF-SASRec/main.py \
  --dataset ${DATASET} \
  --train_dir qwen_letter \
  --device cuda:0 \
  --num_epochs 100 \
  --eval_step 1 \
  --batch_size 256 \
  --hidden_units 32 \
  --maxlen 50 \
  --num_blocks 2 \
  --num_heads 1 \
  --dropout_rate 0.2 \
  --lr 1e-3 \
  --eval_batch_size 256 \
  --time_span 256 \
  --n_workers 3 \
  --patience 10 \
  --topk 5 10 \
  --output_path ./RQ-VAE/ckpt/${DATASET}-32d-sasrec.pt \
  --metrics_path ./CF-SASRec/results/${DATASET}_sasrec_metrics.json \
  --log_file ${LOG_FILE}'"
