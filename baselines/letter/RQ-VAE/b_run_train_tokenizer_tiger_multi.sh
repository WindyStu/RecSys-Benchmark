#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$ROOT_DIR/.." && pwd)
cd "$REPO_ROOT"
mkdir -p ./RQ-VAE/log ./RQ-VAE/lsflog ./checkpoint

DATASETS=${1:-"Cell_Phones Clothing Douban_Book Douban_Movie Sports Electronics"}
DATASET_LIST=($DATASETS)

for IDX in "${!DATASET_LIST[@]}"; do
  DATASET=${DATASET_LIST[$IDX]}
  TEXT_EMB="./data/${DATASET}/${DATASET}.emb-qwen-api-td.npy"
  CKPT_DIR="./checkpoint/${DATASET}_tiger"
  RUN_TS=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="./RQ-VAE/log/${DATASET}_tiger_${RUN_TS}.log"
  LSF_OUT="./RQ-VAE/lsflog/${DATASET}_tiger_${RUN_TS}.out"
  LSF_ERR="./RQ-VAE/lsflog/${DATASET}_tiger_${RUN_TS}.err"
  : > "$LOG_FILE"

  if [[ ! -f "$TEXT_EMB" ]]; then
    echo "Skip ${DATASET}: missing ${TEXT_EMB}"
    continue
  fi

  mkdir -p "$CKPT_DIR"

  bsub -J "rqvae_tiger_${DATASET}" -N -q volta \
  -e "$LSF_ERR" \
  -o "$LSF_OUT" \
  -n 1 \
  -gpu "num=1:mode=exclusive_process" \
  "bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=RQ-VAE TIGER tokenizer train dataset=${DATASET} time=${RUN_TS} text_embedding=${TEXT_EMB} ckpt_dir=${CKPT_DIR}\"; \
   python RQ-VAE/main.py \
    --device cuda:0 \
    --data_path ${TEXT_EMB} \
    --ckpt_dir ${CKPT_DIR} \
    --no_cf \
    --no_diversity \
    --alpha 0.0 \
    --beta 0.0 \
    --epochs 20000 \
    --eval_step 100 \
    --patience 10 \
    --batch_size 1024 \
    --num_workers 4 \
    --log_file ${LOG_FILE}'"

  if [[ "$IDX" -lt $((${#DATASET_LIST[@]} - 1)) ]]; then
    sleep 3
  fi
done
