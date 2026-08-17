#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$ROOT_DIR/.." && pwd)
cd "$REPO_ROOT"
mkdir -p ./RQ-VAE/log ./RQ-VAE/lsflog ./checkpoint

#DATASETS=${1:-"Cell_Phones Clothing Douban_Book Douban_Movie Sports"}
DATASETS=${1:-"Electronics"}
DATASET_LIST=($DATASETS)

for IDX in "${!DATASET_LIST[@]}"; do
  DATASET=${DATASET_LIST[$IDX]}
  TEXT_EMB="./data/${DATASET}/${DATASET}.emb-qwen-api-td.npy"
  CF_EMB="./RQ-VAE/ckpt/${DATASET}-32d-sasrec.pt"
  CKPT_DIR="./checkpoint/${DATASET}_qwen_letter"
  RUN_TS=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="./RQ-VAE/log/${DATASET}_${RUN_TS}.log"
  LSF_OUT="./RQ-VAE/lsflog/${DATASET}_${RUN_TS}.out"
  LSF_ERR="./RQ-VAE/lsflog/${DATASET}_${RUN_TS}.err"
  : > "$LOG_FILE"

  if [[ ! -f "$TEXT_EMB" ]]; then
    echo "Skip ${DATASET}: missing ${TEXT_EMB}"
    continue
  fi

  if [[ ! -f "$CF_EMB" ]]; then
    echo "Skip ${DATASET}: missing ${CF_EMB}"
    continue
  fi

  mkdir -p "$CKPT_DIR"

  bsub -J "rqvae_${DATASET}" -N -q volta \
  -e "$LSF_ERR" \
  -o "$LSF_OUT" \
  -n 1 \
  -gpu "num=1:mode=exclusive_process" \
  "bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=RQ-VAE train dataset=${DATASET} time=${RUN_TS} text_embedding=${TEXT_EMB} cf_embedding=${CF_EMB} ckpt_dir=${CKPT_DIR}\"; \
   python RQ-VAE/main.py \
    --device cuda:0 \
    --data_path ${TEXT_EMB} \
    --cf_emb ${CF_EMB} \
    --ckpt_dir ${CKPT_DIR} \
    --alpha 0.01 \
    --beta 0.0001 \
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
