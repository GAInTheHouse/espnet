#!/usr/bin/env bash
# egs2/afrispeech_rl/asr1/local/data.sh
#
# Downloads AfriSpeech-200 (clinical), VoxPopuli EN (10k), and LibriSpeech
# dev-clean from HuggingFace and writes kaldi-format data directories.
#
# Expected to be called from egs2/afrispeech_rl/asr1/ with utils/ in PATH.
#
# Options:
#   --max_voxpopuli  N   max VoxPopuli utterances (default 10000)
#   --seed           N   random seed for subsampling (default 42)
#   --stage          N   start from this sub-stage (1-7)
#   --stop_stage     N   stop after this sub-stage

set -euo pipefail

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') (data.sh) $*"; }

# Defaults
max_voxpopuli=10000
seed=42
stage=1
stop_stage=100

. utils/parse_options.sh 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR="${RECIPE_DIR}/data/downloads"

# ---------------------------------------------------------------------------
# Stage 1: AfriSpeech-200 clinical — train split
# ---------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: AfriSpeech-200 clinical train"
    python "${SCRIPT_DIR}/data_hf.py" \
        --dataset afrispeech \
        --split train \
        --output_dir "${RECIPE_DIR}/data/afrispeech_train" \
        --audio_dir "${DOWNLOAD_DIR}/afrispeech/train" \
        --seed "${seed}"
fi

# ---------------------------------------------------------------------------
# Stage 2: AfriSpeech-200 clinical — dev split
# ---------------------------------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: AfriSpeech-200 clinical dev"
    python "${SCRIPT_DIR}/data_hf.py" \
        --dataset afrispeech \
        --split dev \
        --output_dir "${RECIPE_DIR}/data/afrispeech_dev" \
        --audio_dir "${DOWNLOAD_DIR}/afrispeech/dev" \
        --seed "${seed}"
fi

# ---------------------------------------------------------------------------
# Stage 3: AfriSpeech-200 clinical — test split
# ---------------------------------------------------------------------------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: AfriSpeech-200 clinical test"
    python "${SCRIPT_DIR}/data_hf.py" \
        --dataset afrispeech \
        --split test \
        --output_dir "${RECIPE_DIR}/data/afrispeech_test" \
        --audio_dir "${DOWNLOAD_DIR}/afrispeech/test" \
        --seed "${seed}"
fi

# ---------------------------------------------------------------------------
# Stage 4: VoxPopuli EN — train (10k utterances)
# ---------------------------------------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: VoxPopuli EN train (max ${max_voxpopuli} utterances)"
    python "${SCRIPT_DIR}/data_hf.py" \
        --dataset voxpopuli \
        --split train \
        --output_dir "${RECIPE_DIR}/data/voxpopuli_train" \
        --audio_dir "${DOWNLOAD_DIR}/voxpopuli/train" \
        --max_samples "${max_voxpopuli}" \
        --seed "${seed}"
fi

# ---------------------------------------------------------------------------
# Stage 5: VoxPopuli EN — dev
# ---------------------------------------------------------------------------
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: VoxPopuli EN dev"
    python "${SCRIPT_DIR}/data_hf.py" \
        --dataset voxpopuli \
        --split validation \
        --output_dir "${RECIPE_DIR}/data/voxpopuli_dev" \
        --audio_dir "${DOWNLOAD_DIR}/voxpopuli/dev" \
        --seed "${seed}"
fi

# ---------------------------------------------------------------------------
# Stage 6: LibriSpeech dev-clean (catastrophic-forgetting eval only)
# ---------------------------------------------------------------------------
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: LibriSpeech dev-clean (forgetting eval)"
    python "${SCRIPT_DIR}/data_hf.py" \
        --dataset librispeech \
        --split "validation.clean" \
        --output_dir "${RECIPE_DIR}/data/librispeech_dev_clean" \
        --audio_dir "${DOWNLOAD_DIR}/librispeech/dev_clean" \
        --seed "${seed}"
fi

# ---------------------------------------------------------------------------
# Stage 7: Combine AfriSpeech + VoxPopuli train into a single train set
# ---------------------------------------------------------------------------
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Combining AfriSpeech + VoxPopuli into data/train_combined"
    if command -v utils/combine_data.sh &>/dev/null; then
        utils/combine_data.sh \
            "${RECIPE_DIR}/data/train_combined" \
            "${RECIPE_DIR}/data/afrispeech_train" \
            "${RECIPE_DIR}/data/voxpopuli_train"
    else
        # Fallback: simple concatenation without Kaldi utils
        log "utils/combine_data.sh not found; concatenating manually."
        mkdir -p "${RECIPE_DIR}/data/train_combined"
        for f in wav.scp text utt2spk; do
            cat "${RECIPE_DIR}/data/afrispeech_train/${f}" \
                "${RECIPE_DIR}/data/voxpopuli_train/${f}" \
                > "${RECIPE_DIR}/data/train_combined/${f}"
        done
        # Rebuild spk2utt from utt2spk
        python - <<'PYEOF'
import pathlib, collections
d = pathlib.Path("data/train_combined")
spk2utts = collections.defaultdict(list)
for line in (d / "utt2spk").read_text().splitlines():
    utt, spk = line.split()
    spk2utts[spk].append(utt)
lines = [f"{spk} {' '.join(sorted(utts))}\n" for spk, utts in sorted(spk2utts.items())]
(d / "spk2utt").write_text("".join(lines))
PYEOF
    fi
fi

log "Data preparation complete."
log "  Train : data/train_combined"
log "  Dev   : data/afrispeech_dev  +  data/voxpopuli_dev"
log "  Test  : data/afrispeech_test"
log "  Eval  : data/librispeech_dev_clean (forgetting eval)"
