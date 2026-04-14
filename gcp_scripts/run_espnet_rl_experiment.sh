#!/usr/bin/env bash
# gcp_scripts/run_espnet_rl_experiment.sh
#
# End-to-end driver for the ESPnet2 RL fine-tuning experiment on GCP.
# Runs all 8 recipe stages and prints final WER results.
#
# Usage
# -----
#   # Full run (default: mwer reward, penalty loss, no LoRA)
#   bash gcp_scripts/run_espnet_rl_experiment.sh
#
#   # REINFORCE with domain-weighted WER reward
#   bash gcp_scripts/run_espnet_rl_experiment.sh \
#       --reward_mode wwer --reward_loss_type reinforce \
#       --domain_terms "hypertension arrhythmia tachycardia"
#
#   # LLM reward with real Gemini key
#   bash gcp_scripts/run_espnet_rl_experiment.sh \
#       --reward_mode llm --gemini_api_key "YOUR_KEY"
#
#   # With LoRA, single GPU
#   bash gcp_scripts/run_espnet_rl_experiment.sh --use_lora true --ngpu 1
#
#   # Resume from a specific stage
#   bash gcp_scripts/run_espnet_rl_experiment.sh --stage 6 --stop_stage 7
#
# Prerequisites
# -------------
#   source ~/.espnet_env        (written by setup_gcp_vm.sh)
#   pip install -r requirements_rl.txt

set -euo pipefail

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') [run_experiment] $*"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
stage=1
stop_stage=8
ngpu=1

reward_mode="mwer"
reward_loss_type="penalty"   # NeMo default
gemini_api_key="${GEMINI_API_KEY:-}"
domain_terms=""
use_lora=false
mock_llm=false

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)              stage="$2";           shift 2 ;;
        --stop_stage)         stop_stage="$2";      shift 2 ;;
        --ngpu)               ngpu="$2";            shift 2 ;;
        --reward_mode)        reward_mode="$2";     shift 2 ;;
        --reward_loss_type)   reward_loss_type="$2";shift 2 ;;
        --gemini_api_key)     gemini_api_key="$2";  shift 2 ;;
        --domain_terms)       domain_terms="$2";    shift 2 ;;
        --use_lora)           use_lora="$2";        shift 2 ;;
        --mock_llm)           mock_llm=true;        shift   ;;
        *) log "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------
log "=== Environment check ==="

if [ -z "${ESPNET_ROOT:-}" ]; then
    if [ -f "${HOME}/.espnet_env" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/.espnet_env"
        log "Sourced ~/.espnet_env"
    else
        log "ERROR: ESPNET_ROOT is not set and ~/.espnet_env does not exist."
        log "Run gcp_scripts/setup_gcp_vm.sh first."
        exit 1
    fi
fi

if [ ! -d "${ESPNET_ROOT}" ]; then
    log "ERROR: ESPNET_ROOT=${ESPNET_ROOT} does not exist."
    exit 1
fi

log "ESPNET_ROOT : ${ESPNET_ROOT}"
log "Python      : $(python3 --version 2>&1)"

if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    if [ "${ngpu}" -gt 0 ]; then
        log "WARNING: CUDA not available but --ngpu=${ngpu}. Training will be very slow on CPU."
        log "Set --ngpu 0 to suppress this warning on a CPU-only VM."
    fi
else
    log "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

# Verify key Python imports
python3 - <<'PYEOF'
missing = []
for pkg in ["jiwer", "datasets", "soundfile"]:
    try: __import__(pkg)
    except ImportError: missing.append(pkg)
if missing:
    print(f"WARNING: missing packages: {', '.join(missing)}. Run: pip install -r requirements_rl.txt")
else:
    print("All required packages available.")
PYEOF

# ---------------------------------------------------------------------------
# Navigate to recipe directory
# ---------------------------------------------------------------------------
RECIPE_DIR="${ESPNET_ROOT}/egs2/afrispeech_rl/asr1"
if [ ! -d "${RECIPE_DIR}" ]; then
    log "ERROR: Recipe directory not found: ${RECIPE_DIR}"
    exit 1
fi
cd "${RECIPE_DIR}"
log "Working directory: $(pwd)"

# ---------------------------------------------------------------------------
# Build pass-through args for run.sh
# ---------------------------------------------------------------------------
RUN_ARGS=(
    "--stage"            "${stage}"
    "--stop_stage"       "${stop_stage}"
    "--ngpu"             "${ngpu}"
    "--reward_mode"      "${reward_mode}"
    "--reward_loss_type" "${reward_loss_type}"
    "--use_lora"         "${use_lora}"
)

if [ -n "${gemini_api_key}" ]; then
    RUN_ARGS+=("--gemini_api_key" "${gemini_api_key}")
fi
if [ -n "${domain_terms}" ]; then
    RUN_ARGS+=("--domain_terms" "${domain_terms}")
fi
if [ "${mock_llm}" = true ]; then
    RUN_ARGS+=("--mock_llm" "true")
fi

# ---------------------------------------------------------------------------
# Run the recipe
# ---------------------------------------------------------------------------
log "=== Starting recipe ==="
log "  reward_mode      : ${reward_mode}"
log "  reward_loss_type : ${reward_loss_type}"
log "  use_lora         : ${use_lora}"
log "  ngpu             : ${ngpu}"
log "  stages           : ${stage} → ${stop_stage}"

bash run.sh "${RUN_ARGS[@]}"

# ---------------------------------------------------------------------------
# Print final results summary
# ---------------------------------------------------------------------------
log ""
log "=== RESULTS SUMMARY ==="
found_any=false
for result_file in exp/asr_*/decode_*/result.txt; do
    if [ -f "${result_file}" ]; then
        log "--- ${result_file} ---"
        cat "${result_file}"
        echo ""
        found_any=true
    fi
done

if [ "${found_any}" = false ]; then
    log "No result.txt files found yet. Run stage 8 to produce WER scores."
fi

log "=== Experiment complete ==="
log "Checkpoints:"
log "  SFT : exp/asr_sft/valid.loss.best.pth"
log "  RL  : exp/asr_rl/valid.loss.best.pth"
