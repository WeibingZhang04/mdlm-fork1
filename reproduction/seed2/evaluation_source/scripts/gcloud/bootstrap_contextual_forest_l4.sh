#!/usr/bin/env bash
set -euo pipefail

readonly MOUNT_POINT='/mnt/contextual-forest'
readonly REPO_DIR="${MOUNT_POINT}/mdlm"
readonly BUNDLE='/tmp/mdlm-contextual-forest.bundle'
readonly BRANCH='codex/crf-recovery'
readonly VENV="${MOUNT_POINT}/venv"

if [[ ! -f "${BUNDLE}" ]]; then
  echo "Copy the git bundle to ${BUNDLE} before bootstrapping" >&2
  exit 2
fi
if [[ -e "${REPO_DIR}" ]]; then
  echo "Refusing to overwrite existing ${REPO_DIR}" >&2
  exit 2
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git jq python3-venv tmux
git clone "${BUNDLE}" "${REPO_DIR}"
git -C "${REPO_DIR}" checkout "${BRANCH}"
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install \
  numpy==1.26.4 \
  pytest==8.3.5
"${VENV}/bin/python" -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

"${VENV}/bin/python" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
cd "${REPO_DIR}"
PYTHONDONTWRITEBYTECODE=1 "${VENV}/bin/python" -m pytest -q \
  tests/test_structured_forest.py \
  tests/test_structured_objective.py \
  tests/test_neural_g1.py \
  test_synthetic_distributions.py \
  test_g1_benchmark.py
