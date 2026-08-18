#!/usr/bin/env bash
# Creates a conda environment for this repo from scratch.
#
# Usage:
#   ./setup_env.sh [env-name]
#
# env-name defaults to "sfl-backdoors" if not given.
#
# torch/torchvision need to come from PyTorch's own cu121 wheel index (see
# requirements.txt), which conda's dependency solver doesn't handle well.
# Using conda only for the Python interpreter and pip for everything else
# is the reliable way to reproduce this stack.

set -euo pipefail

ENV_NAME="${1:-sfl-backdoors}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating conda env '$ENV_NAME' with Python 3.10..."
conda create -y -n "$ENV_NAME" python=3.10

conda activate "$ENV_NAME"

echo "Installing dependencies (torch/torchvision from the cu121 wheel index)..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "Done. Activate with: conda activate $ENV_NAME"
python - <<'EOF'
import torch, torchvision
print(f"torch {torch.__version__} (cuda available: {torch.cuda.is_available()})")
print(f"torchvision {torchvision.__version__} (has GTSRB: {hasattr(torchvision.datasets, 'GTSRB')})")
EOF
