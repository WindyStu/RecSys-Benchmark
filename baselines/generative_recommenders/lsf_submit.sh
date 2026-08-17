#!/bin/bash
# LSF submission script for HSTU single-domain training.
#
# Usage:
#   bash lsf_submit.sh --dataset Beauty
#   bash lsf_submit.sh --dataset Electronics --epochs 300 --lr 5e-4
#   LSF_QUEUE=gpu_priority bash lsf_submit.sh --dataset Clothing
#
# All arguments after the script name are forwarded to run_hstu.py.

set -e

# Extract dataset name for log naming
DATASET=""
PREV=""
for ARG in "$@"; do
    if [ "$PREV" = "--dataset" ]; then
        DATASET="$ARG"
        break
    fi
    case "$ARG" in
        --dataset=*) DATASET="${ARG#*=}" ;;
    esac
    PREV="$ARG"
done

DATASET="${DATASET:-unknown}"

mkdir -p logs

bsub -N -q "${LSF_QUEUE:-gpu}" \
    -gpu "num=1" \
    -o "logs/%J_${DATASET}.out" \
    -e "logs/%J_${DATASET}.err" \
    python3 run_hstu.py "$@"
