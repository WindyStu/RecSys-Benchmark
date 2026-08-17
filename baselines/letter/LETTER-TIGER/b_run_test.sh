# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

DATASET=Instruments
DATA_PATH=../data
CKPT_PATH=${CKPT_PATH:-./ckpt/$DATASET/}
LSF_QUEUE=${LSF_QUEUE:-gpu}
LSF_HOST=${LSF_HOST:-gpu02}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
NUM_BEAMS=${NUM_BEAMS:-20}

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"
mkdir -p ./log ./lsflog ./results/$DATASET

RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./log/${DATASET}_${RUN_ID}.log"
LSF_OUT="./lsflog/${DATASET}_${RUN_ID}.out"
LSF_ERR="./lsflog/${DATASET}_${RUN_ID}.err"
RESULTS_FILE="./results/${DATASET}/test_${RUN_ID}.json"
: > "$LOG_FILE"

echo "results file: $ROOT_DIR/${RESULTS_FILE#./}"

bsub -N -q "$LSF_QUEUE" \
-e "$LSF_ERR" \
-o "$LSF_OUT" \
-n 1 \
-m "$LSF_HOST" \
-gpu "num=1:mode=exclusive_process" \
"bash -lc 'cd \"${ROOT_DIR}\"; export PYTHONUNBUFFERED=1; echo \"stage=LETTER-TIGER test dataset=${DATASET} time=${RUN_ID} data_path=${DATA_PATH} ckpt_path=${CKPT_PATH} results_file=${RESULTS_FILE} test_batch_size=${TEST_BATCH_SIZE} num_beams=${NUM_BEAMS}\"; python ./test.py \
    --gpu_id 0 \
    --ckpt_path \"${CKPT_PATH}\" \
    --dataset \"${DATASET}\" \
    --data_path \"${DATA_PATH}\" \
    --results_file \"${RESULTS_FILE}\" \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_beams $NUM_BEAMS \
    --test_prompt_ids 0 \
    --index_file .index.json \
    --log_file ${LOG_FILE}'"
