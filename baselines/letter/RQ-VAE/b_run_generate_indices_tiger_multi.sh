#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$ROOT_DIR/.." && pwd)
cd "$REPO_ROOT"
mkdir -p ./RQ-VAE/log ./RQ-VAE/lsflog

DATASETS=${1:-"Cell_Phones Clothing Douban_Book Douban_Movie Sports Electronics"}
DATASET_LIST=($DATASETS)

for IDX in "${!DATASET_LIST[@]}"; do
  DATASET=${DATASET_LIST[$IDX]}
  CKPT_ROOT="./checkpoint/${DATASET}_tiger"
  OUTPUT_FILE="./data/${DATASET}/${DATASET}.index.tiger.json"
  RUN_TS=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="./RQ-VAE/log/${DATASET}_tiger_generate_indices_${RUN_TS}.log"
  LSF_OUT="./RQ-VAE/lsflog/${DATASET}_tiger_generate_indices_${RUN_TS}.out"
  LSF_ERR="./RQ-VAE/lsflog/${DATASET}_tiger_generate_indices_${RUN_TS}.err"
  : > "$LOG_FILE"

  if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "Skip ${DATASET}: missing RQ-VAE tokenizer checkpoint dir ${CKPT_ROOT}"
    echo "  Run: bash RQ-VAE/b_run_train_tokenizer_tiger_multi.sh \"${DATASET}\""
    continue
  fi

  if [[ ! -d "./data/${DATASET}" ]]; then
    echo "Skip ${DATASET}: missing data dir ./data/${DATASET}"
    continue
  fi

  bsub -J "idx_tiger_${DATASET}" -N -q volta \
  -e "$LSF_ERR" \
  -o "$LSF_OUT" \
  -n 1 \
  -gpu "num=1:mode=exclusive_process" \
  "bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=generate_indices TIGER dataset=${DATASET} time=${RUN_TS} checkpoint_root=${CKPT_ROOT} output_file=${OUTPUT_FILE}\"; \
   CKPT_PATH=\$(find ${CKPT_ROOT} -path \"*/best_collision_model.pth\" -type f | sort | tail -n 1); \
   if [ -z \"\$CKPT_PATH\" ]; then echo \"Missing best_collision_model.pth under ${CKPT_ROOT}. RQ-VAE training is not finished or failed.\" >&2; exit 1; fi; \
   python RQ-VAE/generate_indices.py \
    --dataset ${DATASET} \
    --checkpoint_path \"\$CKPT_PATH\" \
    --output_file ${OUTPUT_FILE} \
    --device cuda:0 \
    --batch_size 64 \
    --num_workers 4 \
    --log_file ${LOG_FILE}'"

  if [[ "$IDX" -lt $((${#DATASET_LIST[@]} - 1)) ]]; then
    sleep 3
  fi
done
