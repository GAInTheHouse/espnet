#!/usr/bin/env bash
# local/capture_env.sh
#
# Captures the full environment snapshot for the espnet_rl conda environment.
# Run from egs2/afrispeech_rl/asr1/ inside the espnet_rl environment:
#
#   conda activate espnet_rl
#   bash local/capture_env.sh
#
# Output files are written to espnet-docs/env/:
#   espnet_rl_env_summary.txt  — package versions, git hash, nvidia-smi
#   espnet_rl_pip_freeze.txt   — full pip freeze output

set -euo pipefail

OUT_DIR="espnet-docs/env"
mkdir -p "${OUT_DIR}"

SUMMARY="${OUT_DIR}/espnet_rl_env_summary.txt"
PIP_FREEZE="${OUT_DIR}/espnet_rl_pip_freeze.txt"

{
    echo "=== Environment snapshot: $(date '+%Y-%m-%dT%H:%M:%S') ==="
    echo ""

    echo "--- Python ---"
    python3 -c "import sys, platform; print(f'python: {sys.version}'); print(f'platform: {platform.platform()}')"
    echo ""

    echo "--- PyTorch ---"
    python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda_available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda_version: {torch.version.cuda}')
    print(f'gpu_count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  gpu[{i}]: {torch.cuda.get_device_name(i)}')
"
    echo ""

    echo "--- ESPnet ---"
    python3 -c "import espnet; print(f'espnet: {espnet.__version__}')" 2>/dev/null || echo "espnet: not importable"
    echo ""

    echo "--- jiwer ---"
    python3 -c "import jiwer; print(f'jiwer: {jiwer.__version__}')" 2>/dev/null || echo "jiwer: not installed"
    echo ""

    echo "--- datasets ---"
    python3 -c "import datasets; print(f'datasets: {datasets.__version__}')" 2>/dev/null || echo "datasets: not installed"
    echo ""

    echo "--- Git hash ---"
    git -C "$(dirname "${BASH_SOURCE[0]}")/../../.." rev-parse HEAD 2>/dev/null || echo "git: not available"
    echo ""

    echo "--- nvidia-smi ---"
    nvidia-smi 2>/dev/null || echo "nvidia-smi: not available"

} | tee "${SUMMARY}"

echo ""
echo "--- pip freeze ---"
pip freeze | tee "${PIP_FREEZE}"

echo ""
echo "Saved:"
echo "  ${SUMMARY}"
echo "  ${PIP_FREEZE}"
