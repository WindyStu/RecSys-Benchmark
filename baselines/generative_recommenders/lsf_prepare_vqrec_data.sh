#!/bin/bash
# LSF submission script for VQ-Rec data preparation.
# Builds FAISS index, projects embeddings, converts to RecBole format.
#
# Usage:
#   bash lsf_prepare_vqrec_data.sh --dataset Cell_Phones
#   bash lsf_prepare_vqrec_data.sh --all      # all 8 npy datasets

set -e

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

DATASET="${DATASET:-all}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VQREC_DIR="${SCRIPT_DIR}/VQ-Rec"

mkdir -p logs

bsub -N -q "${LSF_QUEUE:-gpu}" \
    -gpu "num=1" \
    -o "logs/%J_prepare_vqrec_${DATASET}.out" \
    -e "logs/%J_prepare_vqrec_${DATASET}.err" \
    python3 "${VQREC_DIR}/prepare_data.py" "$@"
