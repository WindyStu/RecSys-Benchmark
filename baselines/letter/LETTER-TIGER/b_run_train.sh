# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET/
NUM_GPUS=${NUM_GPUS:-1}
LSF_QUEUE=${LSF_QUEUE:-gpu}
LSF_HOST=${LSF_HOST:-gpu02}
MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"
mkdir -p ./log ./lsflog

RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./log/${DATASET}_${RUN_ID}.log"
LSF_OUT="./lsflog/${DATASET}_${RUN_ID}.out"
LSF_ERR="./lsflog/${DATASET}_${RUN_ID}.err"
: > "$LOG_FILE"


bsub -N -q "$LSF_QUEUE" \
-e "$LSF_ERR" \
-o "$LSF_OUT" \
-n "$NUM_GPUS" \
-m "$LSF_HOST" \
-gpu "num=${NUM_GPUS}:mode=exclusive_process" \
"bash -lc 'cd \"${ROOT_DIR}\"; export PYTHONUNBUFFERED=1; echo \"stage=LETTER-TIGER train dataset=${DATASET} time=${RUN_ID} output_dir=${OUTPUT_DIR} num_gpus=${NUM_GPUS} master_port=${MASTER_PORT}\"; export WANDB_DISABLED=true; export WANDB_MODE=disabled; torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.json \
    --temperature 1.0 \
    --log_file ${LOG_FILE}'"
