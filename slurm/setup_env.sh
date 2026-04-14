#!/bin/bash
# Local development environment setup.
#
# NOTE: On the MPI cluster this script is NOT needed — run_experiment.sh
# executes inside the pre-built singularity container:
#   /ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
# which already includes PyTorch, JAX, and all dependencies.
#
# Use this script only to set up a local (CPU) venv for running tests and
# iterating on the code before submitting cluster jobs.
#
# Usage:
#   bash slurm/setup_env.sh       # CPU venv for local dev + testing
#   bash slurm/setup_env.sh --gpu # CPU venv + CUDA JAX (if local GPU present)

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
GPU_MODE=0

for arg in "$@"; do
    if [[ "$arg" == "--gpu" ]]; then GPU_MODE=1; fi
done

echo "============================================"
echo "HIM-HER local dev environment setup"
echo "PROJECT_DIR : $PROJECT_DIR"
echo "VENV_DIR    : $VENV_DIR"
echo "GPU mode    : $GPU_MODE"
echo "Python      : $(python3 --version 2>&1)"
echo "============================================"
echo "NOTE: For cluster runs, use singularity container directly."
echo "      See slurm/run_experiment.sh."
echo "============================================"

# ---- create venv ----
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment already exists — skipping creation."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools --quiet

# ---- install dependencies ----
if [[ $GPU_MODE -eq 1 ]]; then
    echo "[2/4] Installing GPU requirements (CUDA 12)..."
    pip install "jax[cuda12]>=0.4.26" --upgrade --quiet
    grep -v "^jax" "$PROJECT_DIR/requirements.txt" | pip install -r /dev/stdin --quiet
    pip install "wandb>=0.17.0" "nvitop>=1.3.0" --quiet
else
    echo "[2/4] Installing CPU requirements..."
    pip install -r "$PROJECT_DIR/requirements.txt" --quiet
fi

# ---- install package in editable mode ----
echo "[3/4] Installing him_her package (editable)..."
pip install -e "$PROJECT_DIR" --quiet

# ---- smoke test ----
echo "[4/4] Verifying installation..."
python -c "
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
print(f'  JAX version : {jax.__version__}')
print(f'  Devices     : {jax.devices()}')
print(f'  Platform    : {jax.default_backend()}')

from him_her.envs.predator_prey import PredatorPreyEnv
from him_her.models.base_model import ModelSet
print('  him_her package : OK')
"

echo ""
echo "Setup complete! Activate with: source $VENV_DIR/bin/activate"
echo "Run tests with:                pytest tests/ -v"
echo "Submit cluster jobs with:      bash slurm/submit_all.sh"
