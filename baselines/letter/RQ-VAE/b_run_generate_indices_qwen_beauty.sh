#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$ROOT_DIR/.." && pwd)
cd "$REPO_ROOT"
mkdir -p ./RQ-VAE/log ./RQ-VAE/lsflog ./data/Beauty

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
"bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=generate_indices dataset=${DATASET} time=${RUN_TS} output_file=./data/${DATASET}/${DATASET}.index.qwen-letter.json\"; \
 LOG_FILE=${LOG_FILE} DEVICE=cuda:0 bash RQ-VAE/generate_indices_qwen_beauty.sh'"
