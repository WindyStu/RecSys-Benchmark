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

mkdir -p "${PROJECT_ROOT}/lsf_logs"

mapfile -t DATASETS < <(grep -v '^[[:space:]]*#' "${DATASET_FILE}" | sed '/^[[:space:]]*$/d')
if [[ "${#DATASETS[@]}" -lt 1 ]]; then
  echo "No active datasets in ${DATASET_FILE}" >&2
  exit 1
fi

echo "Submitting ${#DATASETS[@]} normal jobs, one dataset per bsub"
for DATASET in "${DATASETS[@]}"; do
  export DATASET
  JOB_SAFE_DATASET="${DATASET//[^A-Za-z0-9_]/_}"
  echo "Submitting ${DATASET}"
  bsub \
    -J "eager_sdsr_train_${JOB_SAFE_DATASET}" \
    -q gpu \
    -gpu "num=1:mode=exclusive_process" \
    -n "${CPU_CORES:-1}" \
    -R "rusage[mem=${MEM_MB:-64000}]" \
    -W "${WALLTIME:-72:00}" \
    -o "${PROJECT_ROOT}/lsf_logs/eager_train_${JOB_SAFE_DATASET}_%J.out" \
    -e "${PROJECT_ROOT}/lsf_logs/eager_train_${JOB_SAFE_DATASET}_%J.err" \
    < "${PROJECT_ROOT}/lsf/train_sdsr.lsf"
  sleep "${SUBMIT_SLEEP_SECONDS:-2}"
done
