#!/usr/bin/env bash
# ============================================================
# Step 2: Tri-CDR training with TCA + TCL
# Usage: bash scripts/b_run_tricdr.sh <pair> <source> <target>
#
# Examples:
#   bash scripts/b_run_tricdr.sh asc Sports Clothing
#   bash scripts/b_run_tricdr.sh ghk Home_Kitchen Grocery
# Env vars: LSF_QUEUE (default: gpu), NUM_GPUS (default: 1)
# ============================================================
set -e
PAIR=${1:?missing pair name}
SOURCE=${2:?missing source domain}
TARGET=${3:?missing target domain}
LSF_QUEUE=${LSF_QUEUE:-gpu}
NUM_GPUS=${NUM_GPUS:-1}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"
mkdir -p ./lsflog ./log

DATA_ROOT="/nfsshare/home/liujingyan/data/CDSR/data"

echo "[$(date '+%F %T')] Submitting Tri-CDR job: ${PAIR} ${SOURCE}→${TARGET}"
echo "  queue=${LSF_QUEUE} gpus=${NUM_GPUS}"
echo "  log: ./log/tricdr_${PAIR}_${SOURCE}2${TARGET}.log"

bsub -N -q "${LSF_QUEUE}" \
  -e ./lsflog/%J.err -o ./lsflog/%J.out \
  -n "${NUM_GPUS}" \
  -gpu "num=${NUM_GPUS}:mode=exclusive_process" \
  "export WANDB_DISABLED=true; \
   echo \"[\$(date '+%F %T')] Job started on \$(hostname)\"; \
   python train_tricdr.py \
    --cross_dataset ${PAIR} --source ${SOURCE} --target ${TARGET} \
    --dataset ${TARGET} \
    --data_root ${DATA_ROOT} \
    --maxlen 200 --batch_size 32 \
    >> ./log/tricdr_${PAIR}_${SOURCE}2${TARGET}.log 2>&1; \
   echo \"[\$(date '+%F %T')] Job finished\""

echo "Tri-CDR job for ${PAIR} ${SOURCE}→${TARGET} submitted."
