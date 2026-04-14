#!/usr/bin/env bash
# egs2/afrispeech_rl/asr1/run.sh
#
# ESPnet2 two-stage RL fine-tuning experiment on AfriSpeech-200 clinical
# and VoxPopuli EN, with catastrophic-forgetting evaluation on LibriSpeech.
#
# Mirrors the NeMo methodology from docs/FT Pipeline Methodology.docx:
#   Stage 1 SFT  — standard CTC/attention fine-tuning (rl_weight=0.0)
#   Stage 2 RL   — reward-augmented fine-tuning from SFT checkpoint
#
# Usage
# -----
#   cd egs2/afrispeech_rl/asr1
#   bash run.sh
#   bash run.sh --reward_mode wwer --reward_loss_type reinforce
#   bash run.sh --use_lora true --ngpu 1
#   bash run.sh --stage 6 --stop_stage 6   # re-run RL stage only
#
# Prerequisites
# -------------
#   pip install -r ../../../requirements_rl.txt
#   ESPNET_ROOT must be set or inferable from this script's location.

set -euo pipefail

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') (run.sh:${BASH_LINENO[0]}) $*"; }

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
stage=1
stop_stage=8
ngpu=1
nj=4

# Pretrained model (LibriSpeech Conformer from ESPnet model zoo)
pretrained_model="espnet/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr25e-4"

# Data directories
train_set="train_combined"
valid_set="afrispeech_dev"
test_sets="afrispeech_test"
forgetting_set="librispeech_dev_clean"

# BPE
nbpe=5000
bpe_train_text="data/afrispeech_train/text"

# Experiment directories
sft_expdir="exp/asr_sft"
rl_expdir="exp/asr_rl"

# Config files
sft_config="conf/train_asr_sft.yaml"
rl_config="conf/train_asr_rl.yaml"
decode_config="conf/decode_asr.yaml"

# RL options (override via CLI)
reward_mode="mwer"           # mwer | wwer | llm | all
reward_loss_type="penalty"   # penalty (NeMo default) | reinforce
gemini_api_key="${GEMINI_API_KEY:-}"
mock_llm=false
domain_terms=""              # space-separated list, e.g. "hypertension arrhythmia"

# LoRA (optional)
use_lora=false

# Feature settings
feats_type=raw
audio_format=wav
fs=16k

log "$0 $*"
. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

ESPNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASR_SH="${ESPNET_ROOT}/egs2/TEMPLATE/asr1/asr.sh"

if [ ! -f "${ASR_SH}" ]; then
    log "ERROR: ${ASR_SH} not found. Check ESPNET_ROOT=${ESPNET_ROOT}"
    exit 1
fi

# Build optional LoRA flag string for asr_train_rl calls
lora_opts=""
if [ "${use_lora}" = true ]; then
    lora_opts="--use_adapter true --adapter lora"
fi

# Build optional domain_terms flag
domain_terms_opts=""
if [ -n "${domain_terms}" ]; then
    # shellcheck disable=SC2086
    domain_terms_opts="--domain_terms ${domain_terms}"
fi

# Build optional Gemini key flag
gemini_opts=""
if [ -n "${gemini_api_key}" ]; then
    gemini_opts="--gemini_api_key ${gemini_api_key}"
fi
if [ "${mock_llm}" = true ]; then
    gemini_opts="${gemini_opts} --mock_llm true"
fi

# ---------------------------------------------------------------------------
# Stage 1: Data preparation (HuggingFace download → kaldi dirs)
# ---------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "=== Stage 1: Data preparation ==="
    bash local/data.sh
fi

# ---------------------------------------------------------------------------
# Stage 2: Format wav.scp / dump raw audio features
# ---------------------------------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "=== Stage 2: Feature formatting (raw) ==="
    bash "${ASR_SH}" \
        --stage 2 --stop_stage 4 \
        --skip_train true \
        --feats_type "${feats_type}" \
        --audio_format "${audio_format}" \
        --fs "${fs}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets} ${forgetting_set}" \
        --nj "${nj}"
fi

# ---------------------------------------------------------------------------
# Stage 3: BPE tokenizer training
# ---------------------------------------------------------------------------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "=== Stage 3: BPE training (${nbpe} units) ==="
    bash "${ASR_SH}" \
        --stage 5 --stop_stage 5 \
        --skip_train true \
        --bpemode unigram \
        --nbpe "${nbpe}" \
        --bpe_train_text "${bpe_train_text}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}"
fi

# ---------------------------------------------------------------------------
# Stage 4: Collect normalization statistics
# ---------------------------------------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "=== Stage 4: Collecting statistics ==="
    bash "${ASR_SH}" \
        --stage 9 --stop_stage 9 \
        --skip_train true \
        --feats_type "${feats_type}" \
        --asr_config "${sft_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --nj "${nj}"
fi

# ---------------------------------------------------------------------------
# Stage 5: Download pretrained model and resolve checkpoint path
# ---------------------------------------------------------------------------
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "=== Stage 5: Downloading pretrained model ==="
    mkdir -p exp/pretrained
    python3 - <<PYEOF
from espnet_model_zoo.downloader import ModelDownloader
import json, pathlib
d = ModelDownloader()
info = d.download_and_unpack("${pretrained_model}")
pathlib.Path("exp/pretrained/model_info.json").write_text(json.dumps(info, indent=2))
print("  asr_train_config:", info.get("asr_train_config",""))
print("  asr_model_file  :", info.get("asr_model_file",""))
PYEOF
fi

# ---------------------------------------------------------------------------
# Stage 6 (internal 6a): SFT training
# ---------------------------------------------------------------------------
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "=== Stage 6a: Supervised fine-tuning (SFT) ==="

    # Resolve pretrained checkpoint path
    pretrained_pth=$(python3 -c "
import json; info = json.load(open('exp/pretrained/model_info.json'))
print(info.get('asr_model_file',''))
")
    if [ -z "${pretrained_pth}" ]; then
        log "ERROR: Could not resolve pretrained model path. Run stage 5 first."
        exit 1
    fi
    log "  Pretrained checkpoint: ${pretrained_pth}"

    token_list=$(python3 -c "
import json; info = json.load(open('exp/pretrained/model_info.json'))
print(info.get('token_list',''))
")

    mkdir -p "${sft_expdir}"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${sft_expdir}/train.log" \
        python -m espnet2.bin.asr_train_rl \
            --ngpu "${ngpu}" \
            --config "${sft_config}" \
            --token_list "${token_list}" \
            --init_param "${pretrained_pth}" \
            --train_data_path_and_name_and_type \
                "dump/raw/${train_set}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type \
                "data/${train_set}/text,text,text" \
            --valid_data_path_and_name_and_type \
                "dump/raw/${valid_set}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type \
                "data/${valid_set}/text,text,text" \
            --output_dir "${sft_expdir}" \
            --rl_weight 0.0 \
            ${lora_opts}
    log "SFT checkpoint: ${sft_expdir}/valid.loss.best.pth"
fi

# ---------------------------------------------------------------------------
# Stage 6 (internal 6b): RL reward-augmented fine-tuning
# ---------------------------------------------------------------------------
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "=== Stage 6b: RL reward-augmented fine-tuning ==="

    sft_ckpt="${sft_expdir}/valid.loss.best.pth"
    if [ ! -f "${sft_ckpt}" ]; then
        log "ERROR: SFT checkpoint not found at ${sft_ckpt}. Run stage 6a first."
        exit 1
    fi

    token_list=$(python3 -c "
import json; info = json.load(open('exp/pretrained/model_info.json'))
print(info.get('token_list',''))
")

    mkdir -p "${rl_expdir}"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${rl_expdir}/train.log" \
        python -m espnet2.bin.asr_train_rl \
            --ngpu "${ngpu}" \
            --config "${rl_config}" \
            --token_list "${token_list}" \
            --init_param "${sft_ckpt}" \
            --train_data_path_and_name_and_type \
                "dump/raw/${train_set}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type \
                "data/${train_set}/text,text,text" \
            --valid_data_path_and_name_and_type \
                "dump/raw/${valid_set}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type \
                "data/${valid_set}/text,text,text" \
            --output_dir "${rl_expdir}" \
            --reward_mode "${reward_mode}" \
            --reward_loss_type "${reward_loss_type}" \
            ${domain_terms_opts} \
            ${gemini_opts} \
            ${lora_opts}
    log "RL checkpoint: ${rl_expdir}/valid.loss.best.pth"
fi

# ---------------------------------------------------------------------------
# Stage 7: Decode AfriSpeech test + LibriSpeech (forgetting eval)
# ---------------------------------------------------------------------------
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "=== Stage 7: Decode and score ==="

    token_list=$(python3 -c "
import json; info = json.load(open('exp/pretrained/model_info.json'))
print(info.get('token_list',''))
")

    for model_tag in sft rl; do
        expdir="exp/asr_${model_tag}"
        ckpt="${expdir}/valid.loss.best.pth"
        [ -f "${ckpt}" ] || { log "Skipping ${model_tag}: checkpoint not found."; continue; }

        for test_set in ${test_sets} ${forgetting_set}; do
            decode_dir="${expdir}/decode_${test_set}"
            mkdir -p "${decode_dir}/logdir"
            cp "dump/raw/${test_set}/wav.scp" "${decode_dir}/logdir/keys.1.scp"

            log "  Decoding ${test_set} with ${model_tag} ..."
            ${decode_cmd} "${decode_dir}/logdir/decode.log" \
                python -m espnet2.bin.asr_inference \
                    --ngpu 0 \
                    --batch_size 1 \
                    --config "${decode_config}" \
                    --asr_train_config "${expdir}/config.yaml" \
                    --asr_model_file "${ckpt}" \
                    --data_path_and_name_and_type \
                        "dump/raw/${test_set}/wav.scp,speech,sound" \
                    --key_file "${decode_dir}/logdir/keys.1.scp" \
                    --output_dir "${decode_dir}/logdir/output.1"

            # Score with jiwer
            python3 - <<PYEOF
import jiwer, pathlib, json
hyp_f = pathlib.Path("${decode_dir}/logdir/output.1/1best_recog/text")
ref_f = pathlib.Path("data/${test_set}/text")
if not hyp_f.exists() or not ref_f.exists():
    print("WARNING: hypothesis or reference not found; skipping scoring.")
    exit(0)
hyps = {l.split()[0]: " ".join(l.split()[1:]) for l in hyp_f.read_text().splitlines() if l.strip()}
refs = {l.split()[0]: " ".join(l.split()[1:]) for l in ref_f.read_text().splitlines() if l.strip()}
common = sorted(set(hyps) & set(refs))
if not common:
    print("WARNING: no common utterance IDs.")
    exit(0)
wer = jiwer.wer([refs[u] for u in common], [hyps[u] for u in common])
cer = jiwer.cer([refs[u] for u in common], [hyps[u] for u in common])
result = f"Model: ${model_tag}  Set: ${test_set}\nWER: {wer*100:.2f}%  CER: {cer*100:.2f}%  Utts: {len(common)}\n"
print(result)
pathlib.Path("${decode_dir}/result.txt").write_text(result)
PYEOF
        done
    done

    log "=== Final results ==="
    for f in exp/asr_*/decode_*/result.txt; do
        [ -f "${f}" ] && cat "${f}" && echo "---"
    done
fi

log "=== run.sh complete ==="
