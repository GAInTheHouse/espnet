#!/usr/bin/env bash
# gcp_scripts/setup_gcp_vm.sh
#
# One-shot setup for a GCP GPU VM to run the ESPnet2 RL fine-tuning experiment.
#
# Tested on: Ubuntu 22.04 LTS, CUDA 12.x (a2-highgpu-1g / n1-standard-8 + V100)
#
# Usage (run as the default GCP user, NOT root):
#   bash gcp_scripts/setup_gcp_vm.sh
#
# After completion, start a new shell or run:
#   source ~/.espnet_env
# then proceed to run_espnet_rl_experiment.sh.

set -euo pipefail

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') [setup] $*"; }

ESPNET_ROOT="${ESPNET_ROOT:-${HOME}/espnet}"
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# 1. Verify CUDA is present
# ---------------------------------------------------------------------------
log "=== Step 1: Verifying CUDA ==="
if ! command -v nvcc &>/dev/null; then
    log "ERROR: nvcc not found. Ensure the VM has CUDA drivers installed."
    log "On a2-highgpu-1g / V100 Deep Learning VM images, CUDA is pre-installed."
    log "For a plain VM: sudo apt-get install -y cuda-toolkit-12-3"
    exit 1
fi
nvcc --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# 2. System packages
# ---------------------------------------------------------------------------
log "=== Step 2: Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 sox bc git wget curl \
    build-essential libssl-dev zlib1g-dev \
    python3-pip python3-dev python3-venv

# ---------------------------------------------------------------------------
# 3. Clone ESPnet (or update if already present)
# ---------------------------------------------------------------------------
log "=== Step 3: Setting up ESPnet source ==="
if [ -d "${ESPNET_ROOT}/.git" ]; then
    log "ESPnet already cloned at ${ESPNET_ROOT}; pulling latest."
    git -C "${ESPNET_ROOT}" pull --ff-only || log "Pull skipped (local changes present)."
else
    log "Cloning ESPnet to ${ESPNET_ROOT} ..."
    git clone https://github.com/espnet/espnet.git "${ESPNET_ROOT}"
fi

# ---------------------------------------------------------------------------
# 4. Install ESPnet Python package
# ---------------------------------------------------------------------------
log "=== Step 4: Installing ESPnet (pip editable install) ==="
cd "${ESPNET_ROOT}"

# Bootstrap tools (installs Kaldi-ish utils and sets up Python venv)
if [ ! -f tools/activate_python.sh ]; then
    log "Running ESPnet tools installer ..."
    # Lightweight: skip Kaldi if not needed (raw features only)
    cd tools
    make TH_VERSION=2.1.0 || true   # ignore non-fatal failures
    cd ..
fi

# Install ESPnet itself with all extras
${PYTHON} -m pip install --upgrade pip
${PYTHON} -m pip install -e ".[all]" --quiet || \
    ${PYTHON} -m pip install -e "." --quiet

# Install espnet_model_zoo for pretrained model downloads
${PYTHON} -m pip install espnet_model_zoo --quiet

# ---------------------------------------------------------------------------
# 5. Install RL-specific Python dependencies
# ---------------------------------------------------------------------------
log "=== Step 5: Installing RL dependencies ==="
if [ -f "${ESPNET_ROOT}/requirements_rl.txt" ]; then
    ${PYTHON} -m pip install -r "${ESPNET_ROOT}/requirements_rl.txt"
else
    log "WARNING: requirements_rl.txt not found; installing known deps directly."
    ${PYTHON} -m pip install \
        "jiwer>=3.0.0" \
        "google-generativeai>=0.5.0" \
        "datasets>=2.14.0" \
        "soundfile>=0.12.0" \
        "librosa>=0.10.0" \
        "huggingface_hub>=0.20.0" \
        "loralib>=0.1.2"
fi

# ---------------------------------------------------------------------------
# 6. Verify key imports
# ---------------------------------------------------------------------------
log "=== Step 6: Verifying Python imports ==="
${PYTHON} - <<'PYEOF'
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
import jiwer; print(f"jiwer {jiwer.__version__}")
import soundfile; print(f"soundfile OK")
import datasets; print(f"datasets {datasets.__version__}")
try:
    import google.generativeai as genai
    print("google-generativeai OK")
except ImportError:
    print("WARNING: google-generativeai not importable (Gemini LLM reward unavailable)")
try:
    import loralib; print("loralib OK")
except ImportError:
    print("WARNING: loralib not importable (LoRA unavailable)")
PYEOF

# ---------------------------------------------------------------------------
# 7. Write environment file
# ---------------------------------------------------------------------------
log "=== Step 7: Writing ~/.espnet_env ==="
cat > "${HOME}/.espnet_env" <<EOF
# ESPnet environment — sourced by run_espnet_rl_experiment.sh
export ESPNET_ROOT="${ESPNET_ROOT}"
export PATH="\${ESPNET_ROOT}/tools/bin:\${ESPNET_ROOT}/egs2/TEMPLATE/asr1/utils:\${PATH}"
export PYTHONPATH="\${ESPNET_ROOT}:\${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
EOF

log "Environment written to ~/.espnet_env"
log ""
log "=== Setup complete ==="
log "Run 'source ~/.espnet_env' then 'bash gcp_scripts/run_espnet_rl_experiment.sh'"
