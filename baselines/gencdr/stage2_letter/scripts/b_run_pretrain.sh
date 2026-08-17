#!/usr/bin/env bash
# ============================================================
# Stage 2 Phase 1: Joint pretraining on all cross-domain pairs
# Submit: bash scripts/b_run_pretrain.sh
#
# Env vars (optional):
#   LSF_QUEUE  - LSF queue name (default: gpu)
#   NUM_GPUS   - number of GPUs (default: 1)
# ============================================================
set -e

LSF_QUEUE=${LSF_QUEUE:-gpu}
NUM_GPUS=${NUM_GPUS:-1}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"
mkdir -p ./lsflog ./log ./ckpt/letter_pretrain

MASTER_PORT=$((20000 + RANDOM % 20000))

echo "[$(date '+%F %T')] Submitting pretrain job..."
echo "  queue=${LSF_QUEUE}  gpus=${NUM_GPUS}"
echo "  root_dir=$ROOT_DIR  master_port=$MASTER_PORT"
echo "  log: ./log/pretrain.log"

bsub -N -q "${LSF_QUEUE}" \
  -e ./lsflog/%J.err \
  -o ./lsflog/%J.out \
  -n "${NUM_GPUS}" \
  -gpu "num=${NUM_GPUS}:mode=exclusive_process" \
  "export WANDB_DISABLED=true; export WANDB_MODE=disabled; \
   echo \"[\$(date '+%F %T')] Job started on \$(hostname)\"; \
   torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} ./finetune.py \
    --stage pretrain \
    --base_model ./ckpt/TIGER \
    --data_path /nfsshare/home/liujingyan/data/CDSR/data \
    --datasets asc ape dbm ghk \
    --output_dir ./ckpt/letter_pretrain \
    --per_device_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --temperature 1.0 \
    --index_file .index.json \
    --logging_step 10 \
    >> ./log/pretrain.log 2>&1; \
   echo \"[\$(date '+%F %T')] Job finished\""

echo "Pretrain job submitted."
