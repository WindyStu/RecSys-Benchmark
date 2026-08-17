#!/bin/bash
#BSUB -J merit_smoke
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -M 16000
#BSUB -R "rusage[mem=16000]"
#BSUB -o logs/lsf_smoke.%J.out
#BSUB -e logs/lsf_smoke.%J.err

set -euo pipefail

cd "$HOME/honglu/research/CHORD/MERIT"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate merit

python smoke_test.py --data ape --cuda 0 --eval_bs 2 --n_worker 0
