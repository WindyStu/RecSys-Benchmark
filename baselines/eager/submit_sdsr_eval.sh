#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
export DATA_ROOT="${DATA_ROOT:-/nfsshare/home/liujingyan/data/CDSR/data}"
export RUN_ROOT="${RUN_ROOT:-/nfsshare/home/liujingyan/eager_runs}"
export WORK_ROOT="${WORK_ROOT:-${RUN_ROOT}/sdsr_work}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/sdsr_outputs}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export DATASET_FILE="${DATASET_FILE:-${PROJECT_ROOT}/lsf/datasets_sdsr.txt}"
MAX_CONCURRENT_GPU="${MAX_CONCURRENT_GPU:-6}"
NUM_DATASETS="$(grep -v '^[[:space:]]*#' "${DATASET_FILE}" | sed '/^[[:space:]]*$/d' | awk 'END {print NR}')"

if [[ "${NUM_DATASETS}" -lt 1 ]]; then
  echo "No active datasets in ${DATASET_FILE}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/lsf_logs"

echo "Submitting EAGER SDSR evaluation array"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "DATASET_FILE=${DATASET_FILE}"
echo "NUM_DATASETS=${NUM_DATASETS}"
echo "MAX_CONCURRENT_GPU=${MAX_CONCURRENT_GPU}"

bsub \
  -J "eager_sdsr_eval[1-${NUM_DATASETS}]%${MAX_CONCURRENT_GPU}" \
  -q gpu \
  -gpu "num=1:mode=exclusive_process" \
  -n "${CPU_CORES:-1}" \
  -R "rusage[mem=${MEM_MB:-32000}]" \
  -W "${WALLTIME:-12:00}" \
  -o "${PROJECT_ROOT}/lsf_logs/eager_eval_%J_%I.out" \
  -e "${PROJECT_ROOT}/lsf_logs/eager_eval_%J_%I.err" \
  < "${PROJECT_ROOT}/lsf/eval_sdsr.lsf"
