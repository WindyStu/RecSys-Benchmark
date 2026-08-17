#!/usr/bin/env bash
# ============================================================
# Step 0: Convert GenCDR .inter.json to Tri-CDR .pkl format
# Usage: bash scripts/b_run_convert.sh <pair>
#        bash scripts/b_run_convert.sh all
# Env vars: LSF_QUEUE (default: gpu)
# ============================================================
set -e
PAIR=${1:?missing pair name (asc/ape/dbm/ghk/all)}
LSF_QUEUE=${LSF_QUEUE:-gpu}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"
mkdir -p ./lsflog ./log

DATA_ROOT="/nfsshare/home/liujingyan/data/CDSR/data"

if [ "$PAIR" = "all" ]; then
    PAIR_ARG="--all"
    JOB_NAME="convert_all"
else
    PAIR_ARG="--pair ${PAIR}"
    JOB_NAME="convert_${PAIR}"
fi

echo "[$(date '+%F %T')] Submitting convert job for ${PAIR}"
echo "  data_root=${DATA_ROOT}"
echo "  log: ./log/${JOB_NAME}.log"

bsub -N -q "${LSF_QUEUE}" \
  -e ./lsflog/%J.err -o ./lsflog/%J.out \
  -n 1 \
  "export WANDB_DISABLED=true; \
   echo \"[\$(date '+%F %T')] Job started\"; \
   python convert_data.py ${PAIR_ARG} --data_root ${DATA_ROOT} --direction both \
   >> ./log/${JOB_NAME}.log 2>&1; \
   echo \"[\$(date '+%F %T')] Job finished\""

echo "Convert job for '${PAIR}' submitted."
