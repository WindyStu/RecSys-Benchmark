#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

DATASET=${DATASET:-Beauty}
DEVICE=${DEVICE:-auto}
EPOCHS=${EPOCHS:-100}
EVAL_STEP=${EVAL_STEP:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
HIDDEN_SIZE=${HIDDEN_SIZE:-32}
MAX_LEN=${MAX_LEN:-50}
NUM_LAYERS=${NUM_LAYERS:-2}
NUM_HEADS=${NUM_HEADS:-1}
DROPOUT=${DROPOUT:-0.2}
LR=${LR:-1e-3}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
TOPK=${TOPK:-"5 10"}
TIME_SPAN=${TIME_SPAN:-256}
TRAIN_DIR=${TRAIN_DIR:-qwen_letter}
N_WORKERS=${N_WORKERS:-3}
PATIENCE=${PATIENCE:-10}

OUTPUT_PATH=${OUTPUT_PATH:-"${REPO_ROOT}/RQ-VAE/ckpt/${DATASET}-${HIDDEN_SIZE}d-sasrec.pt"}
METRICS_PATH=${METRICS_PATH:-"${SCRIPT_DIR}/results/${DATASET}_sasrec_metrics.json"}

cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/runs" "${SCRIPT_DIR}/results" "${SCRIPT_DIR}/cache" "${REPO_ROOT}/RQ-VAE/ckpt"

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE=$(python - <<'PY'
import torch
print("cuda:0" if torch.cuda.is_available() else "cpu")
PY
)
fi

echo "CF-SASRec device=${DEVICE}"

python CF-SASRec/main.py \
  --dataset "${DATASET}" \
  --train_dir "${TRAIN_DIR}" \
  --device "${DEVICE}" \
  --num_epochs "${EPOCHS}" \
  --eval_step "${EVAL_STEP}" \
  --batch_size "${BATCH_SIZE}" \
  --hidden_units "${HIDDEN_SIZE}" \
  --maxlen "${MAX_LEN}" \
  --num_blocks "${NUM_LAYERS}" \
  --num_heads "${NUM_HEADS}" \
  --dropout_rate "${DROPOUT}" \
  --lr "${LR}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --time_span "${TIME_SPAN}" \
  --n_workers "${N_WORKERS}" \
  --patience "${PATIENCE}" \
  --topk ${TOPK} \
  --output_path "${OUTPUT_PATH}" \
  --metrics_path "${METRICS_PATH}"
