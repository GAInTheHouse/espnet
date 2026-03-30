#!/usr/bin/env bash
# run_librispeech_subset.sh
#
# Validation pipeline for ESPnet2 on a 100-utterance subset of LibriSpeech
# test-clean using a pretrained Conformer model from espnet_model_zoo.
#
# Usage:
#   cd egs2/librispeech/asr1
#   bash run_librispeech_subset.sh
#
# Prerequisites:
#   pip install espnet_model_zoo jiwer
#   Kaldi utils/ symlink must be present (standard for ESPnet recipes).
#
# What this script does:
#   Stage 1  — Download + symlink the pretrained model via espnet_model_zoo.
#   Stage 2  — Create a 100-utterance subset of test-clean from dump/raw/.
#   Stage 3  — Copy BPE token list and model config into the local exp dir.
#   Stage 4  — Run asr_inference on the subset (stage 12 of the full recipe).
#   Stage 5  — Score with score_sclite to produce result.wrd.txt (WER).
#
# Output:
#   exp/asr_conformer_subset/decode_test_clean_subset/result.wrd.txt
#   exp/asr_conformer_subset/decode_test_clean_subset/1best_recog/text

set -e
set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------

ESPNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RECIPE_DIR="${ESPNET_ROOT}/egs2/librispeech/asr1"

# espnet_model_zoo tag for the pretrained Conformer trained on LibriSpeech 960h
MODEL_TAG="espnet/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr25e-4"

SUBSET_SIZE=100          # number of utterances to keep
BEAM_SIZE=10             # smaller beam for quick validation
CTC_WEIGHT=0.3
LM_WEIGHT=0.0            # skip LM for speed; set to 0.6 and add --lm_* args if available
NBEST=1
NGPU=0                   # set to 1 to use GPU

EXP_DIR="${RECIPE_DIR}/exp/asr_conformer_subset"
DECODE_DIR="${EXP_DIR}/decode_test_clean_subset"
SUBSET_DATA_DIR="${RECIPE_DIR}/data/test_clean_subset100"
DUMP_RAW_DIR="${RECIPE_DIR}/dump/raw/test_clean"

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*"; }

# ---------------------------------------------------------------------------
# Stage 1: Download pretrained model
# ---------------------------------------------------------------------------
log "=== Stage 1: Downloading pretrained model ==="
mkdir -p "${EXP_DIR}"

if [ ! -f "${EXP_DIR}/config.txt" ]; then
    python3 - <<PYEOF
from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader()
info = d.download_and_unpack("${MODEL_TAG}")
import json, pathlib
pathlib.Path("${EXP_DIR}/config.txt").write_text(json.dumps(info, indent=2))
print("Downloaded model info:", info)
PYEOF
else
    log "Model already downloaded; skipping."
fi

# Parse paths out of the downloaded config
ASR_TRAIN_CONFIG=$(python3 -c "
import json
info = json.load(open('${EXP_DIR}/config.txt'))
print(info['asr_train_config'])
")
ASR_MODEL_FILE=$(python3 -c "
import json
info = json.load(open('${EXP_DIR}/config.txt'))
print(info['asr_model_file'])
")

log "  asr_train_config : ${ASR_TRAIN_CONFIG}"
log "  asr_model_file   : ${ASR_MODEL_FILE}"

# ---------------------------------------------------------------------------
# Stage 2: Create 100-utterance subset of test-clean
# ---------------------------------------------------------------------------
log "=== Stage 2: Creating ${SUBSET_SIZE}-utterance subset of test-clean ==="
mkdir -p "${SUBSET_DATA_DIR}"

# Prefer dump/raw/test_clean if it exists; otherwise fall back to data/test_clean
if [ -d "${DUMP_RAW_DIR}" ]; then
    SRC_WAV_SCP="${DUMP_RAW_DIR}/wav.scp"
else
    # Try the pre-dump data directory (may not exist if only downloading model)
    SRC_WAV_SCP="${RECIPE_DIR}/data/test_clean/wav.scp"
    if [ ! -f "${SRC_WAV_SCP}" ]; then
        log "ERROR: Neither ${DUMP_RAW_DIR} nor ${RECIPE_DIR}/data/test_clean exist."
        log "Run stages 1-4 of the full recipe first to prepare test-clean data, OR"
        log "provide a wav.scp manually at ${SRC_WAV_SCP}."
        exit 1
    fi
fi

# Take the first SUBSET_SIZE lines of wav.scp
head -n "${SUBSET_SIZE}" "${SRC_WAV_SCP}" > "${SUBSET_DATA_DIR}/wav.scp"

# Copy or create a text file with references if available
SRC_TEXT=""
for candidate in \
    "${DUMP_RAW_DIR}/text" \
    "${RECIPE_DIR}/data/test_clean/text"; do
    if [ -f "${candidate}" ]; then
        SRC_TEXT="${candidate}"
        break
    fi
done
if [ -n "${SRC_TEXT}" ]; then
    # Filter text to only the utterances in our subset wav.scp
    cut -d' ' -f1 "${SUBSET_DATA_DIR}/wav.scp" > "${SUBSET_DATA_DIR}/utt_list.txt"
    # shellcheck disable=SC2046
    grep -F -f "${SUBSET_DATA_DIR}/utt_list.txt" "${SRC_TEXT}" \
        > "${SUBSET_DATA_DIR}/text" 2>/dev/null || true
    log "  Reference text: ${SUBSET_DATA_DIR}/text"
fi

# Copy feats_type marker needed by asr_inference
echo "raw" > "${SUBSET_DATA_DIR}/feats_type"
echo "flac" > "${SUBSET_DATA_DIR}/audio_format"

log "  Subset wav.scp written: $(wc -l < "${SUBSET_DATA_DIR}/wav.scp") utterances"

# ---------------------------------------------------------------------------
# Stage 3: Prepare output directory
# ---------------------------------------------------------------------------
log "=== Stage 3: Preparing decode output directory ==="
mkdir -p "${DECODE_DIR}/logdir"

# Create a single-shard key file (all utterances in one job)
cp "${SUBSET_DATA_DIR}/wav.scp" "${DECODE_DIR}/logdir/keys.1.scp"

# ---------------------------------------------------------------------------
# Stage 4: Run asr_inference
# ---------------------------------------------------------------------------
log "=== Stage 4: Running asr_inference (beam_size=${BEAM_SIZE}, ctc_weight=${CTC_WEIGHT}) ==="

python3 -m espnet2.bin.asr_inference \
    --ngpu "${NGPU}" \
    --batch_size 1 \
    --beam_size "${BEAM_SIZE}" \
    --ctc_weight "${CTC_WEIGHT}" \
    --lm_weight "${LM_WEIGHT}" \
    --nbest "${NBEST}" \
    --data_path_and_name_and_type \
        "${SUBSET_DATA_DIR}/wav.scp,speech,sound" \
    --key_file "${DECODE_DIR}/logdir/keys.1.scp" \
    --asr_train_config "${ASR_TRAIN_CONFIG}" \
    --asr_model_file "${ASR_MODEL_FILE}" \
    --output_dir "${DECODE_DIR}/logdir/output.1" \
    2>&1 | tee "${DECODE_DIR}/logdir/asr_inference.1.log"

# Merge the single-job output into the expected directory layout
mkdir -p "${DECODE_DIR}/1best_recog"
for f in text token token_int score; do
    if [ -f "${DECODE_DIR}/logdir/output.1/1best_recog/${f}" ]; then
        cp "${DECODE_DIR}/logdir/output.1/1best_recog/${f}" \
           "${DECODE_DIR}/1best_recog/${f}"
    fi
done
log "  Hypothesis text written to: ${DECODE_DIR}/1best_recog/text"

# ---------------------------------------------------------------------------
# Stage 5: Score with score_sclite (WER)
# ---------------------------------------------------------------------------
log "=== Stage 5: Scoring ==="

if [ -f "${SUBSET_DATA_DIR}/text" ] && [ -f "${DECODE_DIR}/1best_recog/text" ]; then
    python3 - <<PYEOF
import jiwer, pathlib, sys

hyp_file = pathlib.Path("${DECODE_DIR}/1best_recog/text")
ref_file = pathlib.Path("${SUBSET_DATA_DIR}/text")

hyps = {}
for line in hyp_file.read_text().strip().splitlines():
    uid, *words = line.split()
    hyps[uid] = " ".join(words)

refs = {}
for line in ref_file.read_text().strip().splitlines():
    uid, *words = line.split()
    refs[uid] = " ".join(words)

common = sorted(set(hyps) & set(refs))
if not common:
    print("WARNING: No common utterance IDs between hypothesis and reference.")
    sys.exit(0)

all_ref = [refs[u] for u in common]
all_hyp = [hyps[u] for u in common]

wer = jiwer.wer(all_ref, all_hyp)
cer = jiwer.cer(all_ref, all_hyp)

result_lines = [
    f"WER: {wer * 100:.2f}%",
    f"CER: {cer * 100:.2f}%",
    f"Utterances scored: {len(common)} / {len(refs)} (ref), {len(hyps)} (hyp)",
    "",
    "Per-utterance results:",
]
for uid in common:
    u_wer = jiwer.wer(refs[uid], hyps[uid])
    result_lines.append(f"  {uid}  WER={u_wer*100:.1f}%  REF={refs[uid]}  HYP={hyps[uid]}")

result_text = "\n".join(result_lines)
print(result_text)

out = pathlib.Path("${DECODE_DIR}/result.txt")
out.write_text(result_text)
print(f"\nWER result written to: {out}")
PYEOF
else
    log "WARNING: Reference text not found at ${SUBSET_DATA_DIR}/text."
    log "Hypothesis text is at ${DECODE_DIR}/1best_recog/text."
    log "Run scoring manually with jiwer once you have references."
fi

log "=== Done. Decode output: ${DECODE_DIR} ==="
log "=== WER result       : ${DECODE_DIR}/result.txt ==="
