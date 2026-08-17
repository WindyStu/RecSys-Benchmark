#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Submitting all dataset training jobs to LSF..."

# 格式: DATASET=<数据集名> LR_G=<学习率衰减系数> bsub -J merit_<数据集名> < jobs/lsf_train_one.sh

# Amazon 系列数据集
DATASET=ape LR_G=0.5 bsub -J merit_ape < jobs/lsf_train_one.sh
DATASET=asc LR_G=0.5 bsub -J merit_asc < jobs/lsf_train_one.sh
DATASET=dbm LR_G=0.5 bsub -J merit_dbm < jobs/lsf_train_one.sh
DATASET=ghk LR_G=0.4 bsub -J merit_ghk < jobs/lsf_train_one.sh

# 如果你还要跑 demo.sh 里的其他数据集，取消下面注释即可:
# DATASET=afk LR_G=0.4 bsub -J merit_afk < jobs/lsf_train_one.sh
# DATASET=abe LR_G=0.8 bsub -J merit_abe < jobs/lsf_train_one.sh
# DATASET=amb LR_G=0.1 bsub -J merit_amb < jobs/lsf_train_one.sh

echo ""
echo "All jobs submitted. Check status with: bjobs"
echo "Output logs: logs/merit_*.out / logs/merit_*.err"
echo "Best model checkpoints will be saved to: checkpoints/"
