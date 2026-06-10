#!/usr/bin/env bash
# ============================================================================
# setup_envs.sh -- conda envs for the LOTUS / Palimpzest baselines (HOST side).
# ============================================================================
# One env per system (their dependency pins conflict: lotus needs numpy<2 and
# py<3.13, palimpzest needs numpy==2.0.2 and py>=3.12).
#
# torch is installed FIRST from the cu118 index (host driver is CUDA 11.8;
# default cu12 wheels fall back to CPU), then the framework, then tabpfn --
# same versions as the NeurDB AI engine (torch 2.4.1, tabpfn 2.2.1).
#
# Usage:  bash setup_envs.sh [lotus|pz|all(default)]
# ============================================================================
set -euo pipefail

LOTUS_REPO="$HOME/r/neurdb/neurdb-dev/.local/baselines/lotus"
PZ_REPO="$HOME/r/neurdb/neurdb-dev/.local/baselines/palimpzest"
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
COMMON_PKGS=(tabpfn==2.2.1 psycopg2-binary scikit-learn pyarrow)

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

setup_lotus() {
  conda env list | grep -q '^bl_lotus ' || conda create -y -n bl_lotus python=3.11
  conda activate bl_lotus
  pip install "torch==2.4.1" --index-url "$TORCH_INDEX"
  pip install -e "$LOTUS_REPO"
  pip install "${COMMON_PKGS[@]}"
  python - <<'EOF'
import torch, lotus, tabpfn
print("lotus env OK | torch", torch.__version__, "| cuda:", torch.cuda.is_available())
EOF
  conda deactivate
}

setup_pz() {
  conda env list | grep -q '^bl_pz ' || conda create -y -n bl_pz python=3.12
  conda activate bl_pz
  pip install "torch==2.4.1" --index-url "$TORCH_INDEX"
  pip install -e "$PZ_REPO"
  pip install "${COMMON_PKGS[@]}"
  python - <<'EOF'
import torch, palimpzest, tabpfn
print("pz env OK | torch", torch.__version__, "| cuda:", torch.cuda.is_available())
EOF
  conda deactivate
}

case "${1:-all}" in
  lotus) setup_lotus ;;
  pz)    setup_pz ;;
  all)   setup_lotus; setup_pz ;;
  *)     echo "usage: $0 [lotus|pz|all]"; exit 1 ;;
esac
echo "== setup done =="
