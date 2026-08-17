#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash lsf/submit_bert4rec.sh <dataset> [options]

Single-domain examples:
  bash lsf/submit_bert4rec.sh Beauty
  bash lsf/submit_bert4rec.sh Electronics --batch-size 256 --epoch 300

Pair-domain examples:
  bash lsf/submit_bert4rec.sh abe --task dt
  bash lsf/submit_bert4rec.sh afk --task mt --merge-strategy concat

Options:
  --task st|dt|mt              Default: st for single domains, dt for pair aliases
  --model bert4rec|sasrec|stosa
                               Default: bert4rec
  --seed N                     Default: 3407
  --batch-size N               Default: 128
  --epoch N                    Default: 500
  --worker N                   Default: 1
  --lr FLOAT                   Default: 1e-4
  --l2 FLOAT                   Default: 1e-3
  --margin FLOAT               Default: 0.0, used by STOSA
  --source-data-root PATH      Default: /nfsshare/home/liujingyan/data/CDSR/data
  --eval-mode full|sampled     Default: full
  --python CMD                 Default: python
  --cuda-index N               Default: 0
  --force-serialize            Rebuild the project pickle dataset before training
  --dry-run                    Print resolved settings without submitting bsub
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

DATASET=$1
shift

TASK_INPUT=""
export MODEL=${MODEL:-bert4rec}
export SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-/nfsshare/home/liujingyan/data/CDSR/data}
export PYTHON=${PYTHON:-python}
export SEED=${SEED:-3407}
export BATCH_SIZE=${BATCH_SIZE:-128}
export N_WORKER=${N_WORKER:-1}
export N_EPOCH=${N_EPOCH:-500}
export LR=${LR:-1e-4}
export L2=${L2:-1e-3}
export MARGIN=${MARGIN:-0.0}
export EVAL_MODE=${EVAL_MODE:-full}
export CUDA_INDEX=${CUDA_INDEX:-0}
export MERGE_STRATEGY=${MERGE_STRATEGY:-round_robin}
export FORCE_SERIALIZE=${FORCE_SERIALIZE:-0}
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --task)
      TASK_INPUT=$2
      shift 2
      ;;
    --model)
      export MODEL=$2
      shift 2
      ;;
    --seed)
      export SEED=$2
      shift 2
      ;;
    --batch-size)
      export BATCH_SIZE=$2
      shift 2
      ;;
    --epoch|--epochs|--n-epoch)
      export N_EPOCH=$2
      shift 2
      ;;
    --worker|--workers|--n-worker)
      export N_WORKER=$2
      shift 2
      ;;
    --lr)
      export LR=$2
      shift 2
      ;;
    --l2)
      export L2=$2
      shift 2
      ;;
    --margin)
      export MARGIN=$2
      shift 2
      ;;
    --source-data-root)
      export SOURCE_DATA_ROOT=$2
      shift 2
      ;;
    --eval-mode)
      export EVAL_MODE=$2
      shift 2
      ;;
    --python)
      export PYTHON=$2
      shift 2
      ;;
    --cuda-index)
      export CUDA_INDEX=$2
      shift 2
      ;;
    --merge-strategy)
      export MERGE_STRATEGY=$2
      shift 2
      ;;
    --force-serialize)
      export FORCE_SERIALIZE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$DATASET" in
  Beauty|abeauty)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Beauty
    export DATA=abeauty
    ;;
  Electronics|ae)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Electronics
    export DATA=ae
    ;;
  Grocery|af)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Grocery
    export DATA=af
    ;;
  Home_Kitchen|ak)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Home_Kitchen
    export DATA=ak
    ;;
  Douban_Movie|am)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Douban_Movie
    export DATA=am
    ;;
  Douban_Book|abook)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Douban_Book
    export DATA=abook
    ;;
  Cell_Phones|cell_phones)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Cell_Phones
    export DATA=cell_phones
    ;;
  Clothing|clothing)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Clothing
    export DATA=clothing
    ;;
  Instruments|instruments)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Instruments
    export DATA=instruments
    ;;
  Sports|sports)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Sports
    export DATA=sports
    ;;
  Yelp|yelp)
    export TASK=${TASK_INPUT:-st}
    export DOMAIN=Yelp
    export DATA=yelp
    ;;
  abe)
    export TASK=${TASK_INPUT:-dt}
    export DATA=abe
    export DOMAIN_A=Beauty
    export DOMAIN_B=Electronics
    ;;
  afk)
    export TASK=${TASK_INPUT:-dt}
    export DATA=afk
    export DOMAIN_A=Grocery
    export DOMAIN_B=Home_Kitchen
    ;;
  amb)
    export TASK=${TASK_INPUT:-dt}
    export DATA=amb
    export DOMAIN_A=Douban_Movie
    export DOMAIN_B=Douban_Book
    ;;
  *)
    echo "Unknown dataset alias: $DATASET" >&2
    echo "Use a known domain name such as Beauty, Electronics, Grocery, or pair alias abe/afk/amb." >&2
    exit 2
    ;;
esac

echo "Submitting LSF job:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  SOURCE_DATA_ROOT=$SOURCE_DATA_ROOT"
echo "  TASK=$TASK MODEL=$MODEL DATA=$DATA EVAL_MODE=$EVAL_MODE"
if [ "$TASK" = "st" ]; then
  echo "  DOMAIN=$DOMAIN"
else
  echo "  DOMAIN_A=$DOMAIN_A DOMAIN_B=$DOMAIN_B MERGE_STRATEGY=$MERGE_STRATEGY"
fi
echo "  SEED=$SEED BATCH_SIZE=$BATCH_SIZE N_EPOCH=$N_EPOCH"
echo "  LR=$LR L2=$L2 MARGIN=$MARGIN"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

bsub < "$PROJECT_ROOT/lsf/train_bert4rec.lsf"
