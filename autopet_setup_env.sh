#!/usr/bin/env bash
# Create a separate WSL environment for the official autoPET III submission.
# The Microsoft baseline environment (~/venvs/lymphoma_seg) is not touched.
#
# Run inside WSL from the project root:
#     bash autopet_setup_env.sh
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$PROJECT/autopet-3-submission"
VENV="$HOME/venvs/autopet3"

# Same torch build as ~/venvs/lymphoma_seg: the RTX 5060 (sm_120) needs a cu128 wheel.
TORCH_VERSION="2.11.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python 3.11 "$VENV"
fi

uv pip install --python "$VENV/bin/python" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"

# The readme's install step ("pip install -e ."). The repo's own nnunetv2 2.5.1 fork is
# used as it is; this only adds nnunetv2.egg-info/, which the repo's .gitignore ignores.
uv pip install --python "$VENV/bin/python" -e "$REPO"

"$VENV/bin/python" -c "import torch, nnunetv2; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print('nnunetv2 from', nnunetv2.__path__[0])"
