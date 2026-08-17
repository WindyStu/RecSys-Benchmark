#!/usr/bin/env bash
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"
mkdir -p ./log ./lsflog ./ckpt/Beauty_qwen_letter_no_cf ./results/Beauty

DATASET=Beauty
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./log/${DATASET}_${RUN_TS}.log"
LSF_OUT="./lsflog/${DATASET}_${RUN_TS}.out"
LSF_ERR="./lsflog/${DATASET}_${RUN_TS}.err"

if [[ ! -f ../data/${DATASET}/${DATASET}.index.qwen-letter-no-cf.json ]]; then
  echo "Missing ../data/${DATASET}/${DATASET}.index.qwen-letter-no-cf.json"
  echo "Run RQ-VAE/b_run_generate_indices_qwen_beauty_no_cf.sh first."
  exit 1
fi
: > "$LOG_FILE"

bsub -N -q gpu \
-m gpu02 \
-e "$LSF_ERR" \
-o "$LSF_OUT" \
-n 1 \
-gpu "num=1:mode=exclusive_process" \
"bash -lc 'export PYTHONUNBUFFERED=1; echo \"stage=LETTER-TIGER train_and_test no_cf dataset=${DATASET} time=${RUN_TS} index_file=../data/${DATASET}/${DATASET}.index.qwen-letter-no-cf.json output_dir=./ckpt/${DATASET}_qwen_letter_no_cf\"; \
 export WANDB_DISABLED=true; export WANDB_MODE=disabled; python -c \"import sentencepiece\" || { echo \"Missing dependency: sentencepiece. Run: pip install sentencepiece\" >&2; exit 1; }; torchrun --nproc_per_node=1 --master_port=29502 ./finetune.py \
  --base_model ./ckpt/TIGER \
  --data_path ../data \
  --output_dir ./ckpt/${DATASET}_qwen_letter_no_cf \
  --dataset ${DATASET} \
  --index_file .index.qwen-letter-no-cf.json \
  --per_device_batch_size 256 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --epochs 200 \
  --temperature 1.0 \
  --logging_step 10 \
  --test_batch_size 16 \
  --eval_log_step 100 \
  --log_file ${LOG_FILE}'"
