#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-eager-py311-cu124}"

conda create -n "${ENV_NAME}" python=3.11 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

python - <<'PY'
import torch
import transformers
import numpy
import pandas

print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
PY
