#!/usr/bin/env bash
# ============================================================
# Stage 1: Generate .index.json from Qwen embeddings via RQ-VAE
# Usage:
#   bash scripts/b_run_export_index.sh asc
#   bash scripts/b_run_export_index.sh ape
#   bash scripts/b_run_export_index.sh dbm
#   bash scripts/b_run_export_index.sh all
#
# Env vars (optional):
#   LSF_QUEUE  - LSF queue name (default: gpu)
#   NUM_GPUS   - number of GPUs (default: 1)
# ============================================================
set -e

PAIR=${1:?missing pair name (asc/ape/dbm/all)}

LSF_QUEUE=${LSF_QUEUE:-gpu}
NUM_GPUS=${NUM_GPUS:-1}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"
mkdir -p ./lsflog ./log ./ckpt/rqvae_export

DATA_PATH="/nfsshare/home/liujingyan/data/CDSR/data"

if [ "$PAIR" = "all" ]; then
    PAIR_ARG="--all"
else
    PAIR_ARG="--pair ${PAIR}"
fi

echo "[$(date '+%F %T')] Submitting Stage1 RQ-VAE job..."
echo "  pair=${PAIR}"
echo "  queue=${LSF_QUEUE}  gpus=${NUM_GPUS}"
echo "  data_path=${DATA_PATH}"
echo "  log: ./log/rqvae_${PAIR}.log"

bsub -N -q "${LSF_QUEUE}" \
  -e ./lsflog/%J.err \
  -o ./lsflog/%J.out \
  -n "${NUM_GPUS}" \
  -gpu "num=${NUM_GPUS}:mode=exclusive_process" \
  "export WANDB_DISABLED=true; \
   echo \"[\$(date '+%F %T')] Job started on \$(hostname)\"; \
   echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"; \
   python export_index.py \
    ${PAIR_ARG} \
    --data_path ${DATA_PATH} \
    --device cuda:0 \
    --epochs 1000 \
    >> ./log/rqvae_${PAIR}.log 2>&1; \
   echo \"[\$(date '+%F %T')] Job finished\""

echo "Stage1 RQ-VAE job for '${PAIR}' submitted."
