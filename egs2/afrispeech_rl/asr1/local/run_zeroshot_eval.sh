#!/usr/bin/env bash
# local/run_zeroshot_eval.sh
#
# Zero-shot evaluation using the pretrained checkpoint (no SFT/RL fine-tuning).
# Loops over afrispeech_test and afrispeech_dev, runs asr_inference, then
# calls local/eval_extended.py for extended metrics.
#
# No seed dependency — the pretrained model is fixed for all seeds.
#
# Usage (from egs2/afrispeech_rl/asr1/):
#   bash local/run_zeroshot_eval.sh
#
# Prerequisites: Stage 2 (dump/raw) and Stage 5 (exp/pretrained) must be done.

set -euo pipefail

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') (run_zeroshot_eval.sh:${BASH_LINENO[0]}) $*"; }

. ./path.sh
. ./cmd.sh

MODEL_INFO="exp/pretrained/model_info.json"
DECODE_CONFIG="conf/decode_asr.yaml"

if [ ! -f "${MODEL_INFO}" ]; then
    log "ERROR: ${MODEL_INFO} not found. Run Stage 5 first."
    exit 1
fi

asr_train_config=$(python3 -c "
import json; info = json.load(open('${MODEL_INFO}'))
print(info.get('asr_train_config',''))
")

asr_model_file=$(python3 -c "
import json; info = json.load(open('${MODEL_INFO}'))
print(info.get('asr_model_file',''))
")

if [ -z "${asr_train_config}" ] || [ -z "${asr_model_file}" ]; then
    log "ERROR: Could not read asr_train_config or asr_model_file from ${MODEL_INFO}."
    exit 1
fi

log "Pretrained config: ${asr_train_config}"
log "Pretrained model:  ${asr_model_file}"

for test_set in afrispeech_test afrispeech_dev; do
    decode_dir="exp/pretrained/decode_${test_set}"
    mkdir -p "${decode_dir}/logdir"

    wav_scp="dump/raw/${test_set}/wav.scp"
    if [ ! -f "${wav_scp}" ]; then
        log "WARNING: ${wav_scp} not found; skipping ${test_set}."
        continue
    fi

    cp "${wav_scp}" "${decode_dir}/logdir/keys.1.scp"

    log "Decoding ${test_set} with pretrained checkpoint ..."
    ${decode_cmd} "${decode_dir}/logdir/decode.log" \
        python -m espnet2.bin.asr_inference \
            --ngpu 0 \
            --batch_size 1 \
            --config "${DECODE_CONFIG}" \
            --asr_train_config "${asr_train_config}" \
            --asr_model_file "${asr_model_file}" \
            --data_path_and_name_and_type \
                "${wav_scp},speech,sound" \
            --key_file "${decode_dir}/logdir/keys.1.scp" \
            --output_dir "${decode_dir}/logdir/output.1"

    hyp_text="${decode_dir}/logdir/output.1/1best_recog/text"
    ref_text="data/${test_set}/text"

    if [ -f "${hyp_text}" ] && [ -f "${ref_text}" ]; then
        log "Running extended eval for ${test_set} ..."
        python3 local/eval_extended.py \
            --hyp_file "${hyp_text}" \
            --ref_file "${ref_text}" \
            --domain_terms_file conf/domain_terms_clinical.txt \
            --bootstrap_iters 1000 \
            --output_json "${decode_dir}/extended_metrics.json" \
            || log "WARNING: eval_extended.py failed for ${test_set}"
        log "  Metrics: ${decode_dir}/extended_metrics.json"
    else
        log "WARNING: hyp or ref not found for ${test_set}; skipping eval."
    fi
done

log "Zero-shot eval complete. Results in exp/pretrained/decode_*/"
