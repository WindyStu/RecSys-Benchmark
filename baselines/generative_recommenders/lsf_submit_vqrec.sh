#!/bin/bash
# LSF submission script for VQ-Rec fine-tuning on SDSR datasets.
#
# Usage:
#   bash lsf_submit_vqrec.sh --dataset Cell_Phones
#   bash lsf_submit_vqrec.sh --dataset Electronics --lr 0.001 --epochs 300
#
# Prerequisites:
#   1. Run prepare_data.py first: python3 VQ-Rec/prepare_data.py --dataset Cell_Phones

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

# VQ-Rec is a subdirectory of generative-recommenders
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VQREC_DIR="${SCRIPT_DIR}/VQ-Rec"

mkdir -p logs

bsub -N -q "${LSF_QUEUE:-gpu}" \
    -gpu "num=1" \
    -o "logs/%J_vqrec_${DATASET}.out" \
    -e "logs/%J_vqrec_${DATASET}.err" \
    python3 "${VQREC_DIR}/run_vqrec.py" "$@"
