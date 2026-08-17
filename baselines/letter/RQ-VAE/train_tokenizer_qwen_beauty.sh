#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

DATASET=${DATASET:-Beauty}
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data/${DATASET}"}
EMB_PATH=${EMB_PATH:-"${DATA_DIR}/${DATASET}.emb-qwen-api-td.npy"}
CF_EMB=${CF_EMB:-"${SCRIPT_DIR}/ckpt/${DATASET}-32d-sasrec.pt"}
CKPT_DIR=${CKPT_DIR:-"${REPO_ROOT}/checkpoint/${DATASET}_qwen_letter"}

DEVICE=${DEVICE:-auto}
EPOCHS=${EPOCHS:-20000}
EVAL_STEP=${EVAL_STEP:-100}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-4}
ALPHA=${ALPHA:-0.01}
BETA=${BETA:-0.0001}
PATIENCE=${PATIENCE:-10}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_FILE=${LOG_FILE:-"${SCRIPT_DIR}/log/${DATASET}_${RUN_TS}.log"}

if [[ ! -f "${EMB_PATH}" ]]; then
  echo "Missing qwen text embedding: ${EMB_PATH}" >&2
  echo "Generate it with data_process/aliyun_text_emb.py first." >&2
  exit 1
fi

if [[ -f "${EMB_PATH}.progress.npy" ]]; then
  echo "Embedding generation is incomplete: ${EMB_PATH}.progress.npy" >&2
  echo "Resume data_process/aliyun_text_emb.py until the progress file disappears." >&2
  exit 1
fi

if [[ ! -f "${CF_EMB}" ]]; then
  echo "Missing LETTER CF embedding: ${CF_EMB}" >&2
  echo "Original LETTER still needs item collaborative embedding from SASRec/LightGCN." >&2
  echo "Set CF_EMB=/path/to/Beauty-32d-sasrec.pt if it is stored elsewhere." >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${CKPT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE=$(python - <<'PY'
import torch
print("cuda:0" if torch.cuda.is_available() else "cpu")
PY
)
fi

echo "Train LETTER tokenizer with qwen text embedding"
echo "dataset=${DATASET}"
echo "text_embedding=${EMB_PATH}"
echo "cf_embedding=${CF_EMB}"
echo "device=${DEVICE}"
echo "ckpt_dir=${CKPT_DIR}"

python RQ-VAE/main.py \
  --device "${DEVICE}" \
  --data_path "${EMB_PATH}" \
  --cf_emb "${CF_EMB}" \
  --ckpt_dir "${CKPT_DIR}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --epochs "${EPOCHS}" \
  --eval_step "${EVAL_STEP}" \
  --patience "${PATIENCE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --log_file "${LOG_FILE}"
