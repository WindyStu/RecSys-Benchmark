#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$ROOT_DIR/.." && pwd)
cd "$REPO_ROOT"
mkdir -p ./RQ-VAE/log ./RQ-VAE/lsflog ./checkpoint/Beauty_qwen_letter_no_cf

DATASET=Beauty
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./RQ-VAE/log/${DATASET}_${RUN_TS}.log"
LSF_OUT="./RQ-VAE/lsflog/${DATASET}_${RUN_TS}.out"
LSF_ERR="./RQ-VAE/lsflog/${DATASET}_${RUN_TS}.err"
: > "$LOG_FILE"

bsub -N -q gpu \
-m gpu02 \
-e "$LSF_ERR" \
-o "$LSF_OUT" \
-n 1 \
-gpu "num=1:mode=exclusive_process" \
"bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=RQ-VAE train no_cf dataset=${DATASET} time=${RUN_TS} text_embedding=./data/${DATASET}/${DATASET}.emb-qwen-api-td.npy ckpt_dir=./checkpoint/${DATASET}_qwen_letter_no_cf\"; \
 python RQ-VAE/main.py \
  --device cuda:0 \
  --data_path ./data/${DATASET}/${DATASET}.emb-qwen-api-td.npy \
  --ckpt_dir ./checkpoint/${DATASET}_qwen_letter_no_cf \
  --no_cf \
  --alpha 0.0 \
  --beta 0.0001 \
  --epochs 20000 \
  --eval_step 100 \
  --patience 10 \
  --batch_size 1024 \
  --num_workers 4 \
  --log_file ${LOG_FILE}'"
