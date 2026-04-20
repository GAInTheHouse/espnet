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
#   --min_duration   F   min audio duration in seconds (default 0.5)
#   --max_duration   F   max audio duration in seconds (default 20.0)
#   --stage          N   start from this sub-stage (1-8)
#   --stop_stage     N   stop after this sub-stage

set -euo pipefail

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') (data.sh) $*"; }

# Defaults
max_voxpopuli=10000
seed=42
min_duration=0.5    # NeMo data loader min_duration=0.5s
max_duration=20.0   # NeMo data loader max_duration=20.0s
stage=1
stop_stage=100

. utils/parse_options.sh 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR="${RECIPE_DIR}/data/downloads"

# Helper: validate a kaldi dir and check it is non-empty
_validate_kaldi_dir() {
    local dir="$1"
    local label="$2"
    local n
    n=$(wc -l < "${dir}/wav.scp" 2>/dev/null || echo 0)
    if [ "${n}" -eq 0 ]; then
        log "ERROR: ${label} wav.scp is empty."
        log "       The HuggingFace download may have succeeded but produced no audio"
        log "       (e.g. the split exists but has 0 matching samples after duration"
        log "       filtering).  Check your HuggingFace credentials, dataset access,"
        log "       and --min_duration / --max_duration settings, then re-run Stage 1."
        exit 1
    fi
    utils/validate_data_dir.sh --no-feats "${dir}" \
        || { log "ERROR: ${label} failed Kaldi validation — see output above"; exit 1; }
    log "  ${label}: ${n} utterances — OK"
}

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
        --seed "${seed}" \
        --min_duration "${min_duration}" \
        --max_duration "${max_duration}"
    _validate_kaldi_dir "${RECIPE_DIR}/data/afrispeech_train" "afrispeech_train"
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
        --seed "${seed}" \
        --min_duration "${min_duration}" \
        --max_duration "${max_duration}"
    # Explicit empty-dir check: the dev split is the validation set for training.
    # A silent empty directory here causes a confusing crash many stages later.
    _validate_kaldi_dir "${RECIPE_DIR}/data/afrispeech_dev" "afrispeech_dev"
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
        --seed "${seed}" \
        --min_duration "${min_duration}" \
        --max_duration "${max_duration}"
    _validate_kaldi_dir "${RECIPE_DIR}/data/afrispeech_test" "afrispeech_test"
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
        --seed "${seed}" \
        --min_duration "${min_duration}" \
        --max_duration "${max_duration}"
    _validate_kaldi_dir "${RECIPE_DIR}/data/voxpopuli_train" "voxpopuli_train"
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
        --seed "${seed}" \
        --min_duration "${min_duration}" \
        --max_duration "${max_duration}"
    _validate_kaldi_dir "${RECIPE_DIR}/data/voxpopuli_dev" "voxpopuli_dev"
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
        --seed "${seed}" \
        --min_duration "${min_duration}" \
        --max_duration "${max_duration}"
    _validate_kaldi_dir "${RECIPE_DIR}/data/librispeech_dev_clean" "librispeech_dev_clean"
fi

# ---------------------------------------------------------------------------
# Stage 7: LibriSpeech train.clean.100 — 5000 utterances (anti-forgetting)
#
# NeMo included 5000 LibriSpeech training samples alongside AfriSpeech +
# VoxPopuli to act as an in-training regulariser for catastrophic forgetting.
# ---------------------------------------------------------------------------
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: LibriSpeech train.clean.100 (5000 samples, anti-forgetting)"
    python "${SCRIPT_DIR}/data_hf.py" \
        --dataset librispeech \
        --split "train.clean.100" \
        --output_dir "${RECIPE_DIR}/data/librispeech_train_5k" \
        --audio_dir "${DOWNLOAD_DIR}/librispeech/train_clean_100" \
        --max_samples 5000 \
        --seed "${seed}" \
        --min_duration "${min_duration}" \
        --max_duration "${max_duration}"
    _validate_kaldi_dir "${RECIPE_DIR}/data/librispeech_train_5k" "librispeech_train_5k"
fi

# ---------------------------------------------------------------------------
# Stage 8: Combine AfriSpeech + VoxPopuli + LibriSpeech (5k) into train_combined
# ---------------------------------------------------------------------------
# utils/combine_data.sh is intentionally NOT used here: it renames utterance
# IDs by appending dataset suffixes, which breaks the speaker-prefix invariant
# required by Kaldi's validate_data_dir.sh.  Direct sorted concatenation
# preserves the IDs written by data_hf.py.
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Combining AfriSpeech + VoxPopuli + LibriSpeech(5k) into data/train_combined"
    mkdir -p "${RECIPE_DIR}/data/train_combined"
    export LC_ALL=C
    for f in wav.scp text utt2spk; do
        cat "${RECIPE_DIR}/data/afrispeech_train/${f}" \
            "${RECIPE_DIR}/data/voxpopuli_train/${f}" \
            "${RECIPE_DIR}/data/librispeech_train_5k/${f}" \
            | sort -k1,1 \
            > "${RECIPE_DIR}/data/train_combined/${f}"
    done
    # Rebuild spk2utt from the merged utt2spk
    python3 - <<'PYEOF'
import pathlib, collections
d = pathlib.Path("data/train_combined")
spk2utts = collections.defaultdict(list)
for line in (d / "utt2spk").read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
        spk2utts[parts[1]].append(parts[0])
lines = [f"{spk} {' '.join(sorted(utts))}\n" for spk, utts in sorted(spk2utts.items())]
(d / "spk2utt").write_text("".join(lines))
print(f"train_combined: {sum(len(v) for v in spk2utts.values())} utterances, {len(lines)} speakers")
PYEOF
    utils/validate_data_dir.sh --no-feats "${RECIPE_DIR}/data/train_combined" \
        || { log "ERROR: train_combined failed validation — check utt2spk sort order"; exit 1; }
    log "  train_combined: OK"
fi

log "Data preparation complete."
log "  Train : data/train_combined (AfriSpeech + VoxPopuli + LibriSpeech 5k)"
log "  Dev   : data/afrispeech_dev  +  data/voxpopuli_dev"
log "  Test  : data/afrispeech_test"
log "  Eval  : data/librispeech_dev_clean (forgetting eval)"
