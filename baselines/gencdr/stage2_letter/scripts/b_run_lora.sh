#!/usr/bin/env bash
# ============================================================
# Stage 2 Phase 2: Domain-specific LoRA fine-tuning
# Usage: bash scripts/b_run_lora.sh <pair> <target_domain>
#
# Domain label mapping (from map_item.txt):
#   asc: Sports(label 0), Clothing(label 1)
#   ape: Phone/Cell_Phones(label 0), Electronics(label 1)
#   dbm: Book(label 0), Movies(label 1)
#   ghk: Grocery(label 0), Home_Kitchen(label 1)
#
# Examples:
#   bash scripts/b_run_lora.sh asc Sports
#   bash scripts/b_run_lora.sh asc Clothing
#   bash scripts/b_run_lora.sh ape Phone
#   bash scripts/b_run_lora.sh ape Electronics
#   bash scripts/b_run_lora.sh dbm Book
#   bash scripts/b_run_lora.sh dbm Movies
#   bash scripts/b_run_lora.sh ghk Grocery
#   bash scripts/b_run_lora.sh ghk Home_Kitchen
#
# Env vars (optional):
#   LSF_QUEUE  - LSF queue name (default: gpu)
#   NUM_GPUS   - number of GPUs (default: 1)
# ============================================================
set -e

PAIR=${1:?missing pair name (asc/ape/dbm/ghk)}
TARGET_DOMAIN=${2:?missing target domain (Sports/Clothing/Phone/Electronics/Book/Movies/Grocery/Home_Kitchen)}

LSF_QUEUE=${LSF_QUEUE:-gpu}
NUM_GPUS=${NUM_GPUS:-1}

# Map target domain to domain_label (0 or 1) based on map_item.txt
case "${TARGET_DOMAIN}" in
  Sports|Phone|Book|Grocery)
    DOMAIN_LABEL=0
    ;;
  Clothing|Electronics|Movies|Home_Kitchen)
    DOMAIN_LABEL=1
    ;;
  *)
    echo "Unknown target domain: ${TARGET_DOMAIN}"
    echo "Known domains: Sports(0) Clothing(1) Phone(0) Electronics(1) Book(0) Movies(1) Grocery(0) Home_Kitchen(1)"
    exit 1
    ;;
esac

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"
mkdir -p ./lsflog ./log ./ckpt/letter_lora_${TARGET_DOMAIN} ./results/${TARGET_DOMAIN}

MASTER_PORT=$((20000 + RANDOM % 20000))

echo "[$(date '+%F %T')] Submitting LoRA job..."
echo "  pair=${PAIR} target=${TARGET_DOMAIN} label=${DOMAIN_LABEL}"
echo "  queue=${LSF_QUEUE}  gpus=${NUM_GPUS}"
echo "  log: ./log/lora_${TARGET_DOMAIN}.log"

bsub -N -q "${LSF_QUEUE}" \
  -e ./lsflog/%J.err \
  -o ./lsflog/%J.out \
  -n "${NUM_GPUS}" \
  -gpu "num=${NUM_GPUS}:mode=exclusive_process" \
  "export WANDB_DISABLED=true; export WANDB_MODE=disabled; \
   echo \"[\$(date '+%F %T')] Job started on \$(hostname)\"; \
   torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} ./finetune.py \
    --stage lora \
    --base_model ./ckpt/TIGER \
    --pretrained_model ./ckpt/letter_pretrain \
    --data_path /nfsshare/home/liujingyan/data/CDSR/data \
    --dataset ${PAIR} \
    --target_domain ${TARGET_DOMAIN} \
    --domain_label ${DOMAIN_LABEL} \
    --output_dir ./ckpt/letter_lora_${TARGET_DOMAIN} \
    --per_device_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --temperature 1.0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --index_file .index.json \
    --logging_step 10 \
    --test_batch_size 16 \
    --eval_log_step 100 \
    >> ./log/lora_${TARGET_DOMAIN}.log 2>&1; \
   echo \"[\$(date '+%F %T')] Job finished\""

echo "LoRA job for ${TARGET_DOMAIN} submitted."
