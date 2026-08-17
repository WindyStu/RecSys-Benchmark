#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

DATASET=Beauty
CKPT_ROOT="${REPO_ROOT}/checkpoint/${DATASET}_qwen_letter_no_cf"
CKPT_PATH=${CKPT_PATH:-}
OUTPUT_FILE="${REPO_ROOT}/data/${DATASET}/${DATASET}.index.qwen-letter-no-cf.json"
DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
LOG_ARGS=()
if [[ -n "${LOG_FILE:-}" ]]; then
  LOG_ARGS=(--log_file "${LOG_FILE}")
fi

cd "${REPO_ROOT}"

if [[ -z "${CKPT_PATH}" ]]; then
  CKPT_PATH=$(find "${CKPT_ROOT}" -path "*/best_collision_model.pth" -type f | sort | tail -n 1)
fi

if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
  echo "Missing no-cf tokenizer checkpoint under ${CKPT_ROOT}" >&2
  echo "Run RQ-VAE/b_run_train_tokenizer_qwen_beauty_no_cf.sh first." >&2
  exit 1
fi

python RQ-VAE/generate_indices.py \
  --dataset "${DATASET}" \
  --checkpoint_path "${CKPT_PATH}" \
  --output_file "${OUTPUT_FILE}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  "${LOG_ARGS[@]}"
