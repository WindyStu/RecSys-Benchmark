#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"
mkdir -p ./log ./lsflog ./ckpt ./results

DATASETS=${1:-"Cell_Phones Clothing Douban_Book Douban_Movie Sports Electronics"}
DATASET_LIST=($DATASETS)

for IDX in "${!DATASET_LIST[@]}"; do
  DATASET=${DATASET_LIST[$IDX]}
  INDEX_FILE="../data/${DATASET}/${DATASET}.index.tiger.json"
  OUTPUT_DIR="./ckpt/${DATASET}_tiger"
  RUN_TS=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="./log/${DATASET}_tiger_${RUN_TS}.log"
  LSF_OUT="./lsflog/${DATASET}_tiger_${RUN_TS}.out"
  LSF_ERR="./lsflog/${DATASET}_tiger_${RUN_TS}.err"
  MASTER_PORT=$((29610 + IDX))

  if [[ ! -f "$INDEX_FILE" ]]; then
    echo "Skip ${DATASET}: missing ${INDEX_FILE}"
    echo "Run RQ-VAE/b_run_generate_indices_tiger_multi.sh \"${DATASET}\" first."
    continue
  fi

  mkdir -p "$OUTPUT_DIR" "./results/${DATASET}"
  : > "$LOG_FILE"

  bsub -J "tiger_${DATASET}" -N -q volta \
  -e "$LSF_ERR" \
  -o "$LSF_OUT" \
  -n 1 \
  -gpu "num=1:mode=exclusive_process" \
  "bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=TIGER train_and_test dataset=${DATASET} time=${RUN_TS} index_file=${INDEX_FILE} output_dir=${OUTPUT_DIR}\"; \
   export WANDB_DISABLED=true; export WANDB_MODE=disabled; python -c \"import sentencepiece\" || { echo \"Missing dependency: sentencepiece. Run: pip install sentencepiece\" >&2; exit 1; }; torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} ./finetune.py \
    --base_model ./ckpt/TIGER \
    --data_path ../data \
    --output_dir ${OUTPUT_DIR} \
    --dataset ${DATASET} \
    --index_file .index.tiger.json \
    --per_device_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --temperature 1.0 \
    --logging_step 10 \
    --test_batch_size 16 \
    --eval_log_step 100 \
    --log_file ${LOG_FILE}'"

  if [[ "$IDX" -lt $((${#DATASET_LIST[@]} - 1)) ]]; then
    sleep 3
  fi
done
