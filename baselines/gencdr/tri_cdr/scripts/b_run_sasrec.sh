#!/usr/bin/env bash
# ============================================================
# Step 1: SASRec single-domain pretraining
# Usage: bash scripts/b_run_sasrec.sh <pair> <domain>
#
# domain = Sports / Clothing / Phone / Electronics /
#          Book / Movies / Grocery / Home_Kitchen / Mix
#
# Examples:
#   bash scripts/b_run_sasrec.sh asc Sports
#   bash scripts/b_run_sasrec.sh asc Mix
# Env vars: LSF_QUEUE (default: gpu), NUM_GPUS (default: 1)
# ============================================================
set -e
PAIR=${1:?missing pair name}
DOMAIN=${2:?missing domain name}
LSF_QUEUE=${LSF_QUEUE:-gpu}
NUM_GPUS=${NUM_GPUS:-1}

# Determine source/target for direction directory
case "${DOMAIN}" in
  Sports|Clothing)
    SOURCE=Sports; TARGET=Clothing ;;
  Phone|Electronics)
    SOURCE=Phone; TARGET=Electronics ;;
  Book|Movies)
    SOURCE=Book; TARGET=Movies ;;
  Grocery|Home_Kitchen)
    SOURCE=Grocery; TARGET=Home_Kitchen ;;
  Mix)
    # Need pair to know source/target
    case "${PAIR}" in
      asc) SOURCE=Sports; TARGET=Clothing ;;
      ape) SOURCE=Phone; TARGET=Electronics ;;
      dbm) SOURCE=Book; TARGET=Movies ;;
      ghk) SOURCE=Grocery; TARGET=Home_Kitchen ;;
    esac
    ;;
esac

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"
mkdir -p ./lsflog ./log

DATA_ROOT="/nfsshare/home/liujingyan/data/CDSR/data"

echo "[$(date '+%F %T')] Submitting SASRec job: ${PAIR}/${DOMAIN}"
echo "  direction: ${SOURCE}→${TARGET}"
echo "  queue=${LSF_QUEUE} gpus=${NUM_GPUS}"
echo "  log: ./log/sasrec_${PAIR}_${DOMAIN}.log"

bsub -N -q "${LSF_QUEUE}" \
  -e ./lsflog/%J.err -o ./lsflog/%J.out \
  -n "${NUM_GPUS}" \
  -gpu "num=${NUM_GPUS}:mode=exclusive_process" \
  "export WANDB_DISABLED=true; \
   echo \"[\$(date '+%F %T')] Job started on \$(hostname)\"; \
   python train_sasrec.py \
    --pair ${PAIR} --domain ${DOMAIN} \
    --source ${SOURCE} --target ${TARGET} \
    --data_root ${DATA_ROOT} \
    >> ./log/sasrec_${PAIR}_${DOMAIN}.log 2>&1; \
   echo \"[\$(date '+%F %T')] Job finished\""

echo "SASRec job for ${PAIR}/${DOMAIN} submitted."
