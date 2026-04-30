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
#   bash run.sh --smoke_test true          # tiny train + ≤10 utts/decode + fast bootstrap
#
# Prerequisites
# -------------
#   pip install -r ../../../requirements_rl.txt
#   ESPNET_ROOT must be set or inferable from this script's location.
#   One-time recipe links (same as other egs2/*/asr1 recipes):
#     ln -sf ../../TEMPLATE/asr1/utils     utils
#     ln -sf ../../TEMPLATE/asr1/steps     steps
#     ln -sf ../../TEMPLATE/asr1/scripts scripts
#     ln -sf ../../TEMPLATE/asr1/pyscripts pyscripts

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
# pyf98/librispeech_conformer is a publicly accessible ESPnet2 Conformer
# trained on LibriSpeech 960h, BPE-5000 — same architecture as the original
# private espnet/ model.  Swap back once you have HF org access.
pretrained_model="pyf98/librispeech_conformer"

# Data directories
train_set="train_combined"   # AfriSpeech(clinical) + VoxPopuli(EN) + LibriSpeech(5k)
valid_set="afrispeech_dev"
test_sets="afrispeech_test"
forgetting_set="librispeech_dev_clean"

# BPE
nbpe=5000
bpe_train_text="data/afrispeech_train/text"

# Experiment directories
sft_expdir="exp/asr_sft"
rl_expdir="exp/asr_rl"
asr_stats_dir="exp/asr_stats_raw_bpe5000"

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

# Reproducibility
seed=42                      # NeMo seed=42

# LoRA (optional)
use_lora=false

# Smoke test mode: limits training to a tiny number of batches per epoch so
# the full pipeline (data → SFT → RL → decode → eval) completes in ~5 minutes.
# Usage: bash run.sh --smoke_test true
#        bash run.sh --smoke_test   # same as true (parse_options needs a value)
# Sets: max_epoch=1, num_iters_per_epoch=20 for both SFT and RL stages.
# Stage 8 (decode): only first smoke_decode_n utterances per test set (full sets otherwise).
smoke_test=false
smoke_decode_n=10

# Feature settings
feats_type=raw
audio_format=wav
fs=16k

log "$0 $*"
# Kaldi utils/parse_options.sh always expects --option value.  Bare --smoke_test
# would leave $2 unset and fail under set -u; default the value to true.
_rebuilt_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke_test)
            if [ $# -ge 2 ] && [[ "$2" != --* ]]; then
                _rebuilt_args+=("$1" "$2")
                shift 2
            else
                _rebuilt_args+=("$1" "true")
                shift
            fi
            ;;
        *)
            _rebuilt_args+=("$1")
            shift
            ;;
    esac
done
set -- "${_rebuilt_args[@]}"
. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

ESPNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASR_SH="${ESPNET_ROOT}/egs2/TEMPLATE/asr1/asr.sh"

if [ ! -f "${ASR_SH}" ]; then
    log "ERROR: ${ASR_SH} not found. Check ESPNET_ROOT=${ESPNET_ROOT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Device guard: fall back to CPU if the requested GPU count exceeds what
# PyTorch can actually see.  This prevents the cryptic CUDA assertion on
# machines without NVIDIA hardware (e.g. Macs, CPU-only cloud VMs).
# ---------------------------------------------------------------------------
if [ "${ngpu}" -gt 0 ]; then
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log "WARNING: ngpu=${ngpu} requested but torch.cuda.is_available()=False."
        log "         Falling back to --ngpu 0 (CPU).  Set --ngpu 0 explicitly to silence this."
        ngpu=0
    fi
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

# Build smoke-test overrides: 1 epoch, 20 batches — full pipeline in ~5 min
smoke_test_opts=""
if [ "${smoke_test}" = true ]; then
    smoke_test_opts="--max_epoch 1 --num_iters_per_epoch 20"
    log "SMOKE TEST MODE: max_epoch=1, num_iters_per_epoch=20, decode ≤${smoke_decode_n} utts/set, bootstrap_iters=32"
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
    # Run WITHOUT --skip_train so that dump/raw/<train_set> and
    # dump/raw/<valid_set> are also formatted (needed for Stage 6 training).
    bash "${ASR_SH}" \
        --stage 2 --stop_stage 4 \
        --feats_type "${feats_type}" \
        --audio_format "${audio_format}" \
        --fs "${fs}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets} ${forgetting_set}" \
        --nj "${nj}"
fi

# ---------------------------------------------------------------------------
# Stage 3: BPE tokenizer — verify pretrained BPE model is in place
# ---------------------------------------------------------------------------
# We reuse the pretrained model's tokenizer (same vocabulary as init_param),
# so no new BPE training is needed.  setup_pretrained.py (Stage 5) copies the
# model to the expected location.  If Stage 5 has already run we simply verify
# the file is present; if not, Stage 5 must run first.
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "=== Stage 3: Verifying pretrained BPE model ==="
    bpe_dst="data/token_list/bpe_unigram${nbpe}/bpe.model"
    if [ ! -f "${bpe_dst}" ]; then
        log "ERROR: ${bpe_dst} not found."
        log "       Run Stage 5 first (bash run.sh --stage 5 --stop_stage 5) to"
        log "       download the pretrained model and place the BPE tokenizer."
        exit 1
    fi
    log "  BPE model OK: ${bpe_dst}"
fi

# ---------------------------------------------------------------------------
# Stage 4: Collect normalization statistics
# ---------------------------------------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "=== Stage 4: Collecting statistics (shape files for batch sampler) ==="
    # Guard: dump/raw must exist (created by Stage 2). If missing, Stage 2 did
    # not run or failed — rerun with --stage 2 --stop_stage 2 first.
    for _set in "${train_set}" "${valid_set}"; do
        if [ ! -f "dump/raw/${_set}/wav.scp" ]; then
            log "ERROR: dump/raw/${_set}/wav.scp not found."
            log "       Stage 2 (feature formatting) must run before Stage 4."
            log "       Fix: bash run.sh --stage 2 --stop_stage 2 --ngpu ${ngpu}"
            exit 1
        fi
    done
    # asr.sh stage 10 (collect_stats) cannot parse rl_weight from the config.
    # Generate speech_shape and text_shape.bpe directly.
    python3 - <<PYEOF
import pathlib, soundfile, sentencepiece

bpe_model = "data/token_list/bpe_unigram5000/bpe.model"
sp = sentencepiece.SentencePieceProcessor()
sp.Load(bpe_model)

    stats_dir = pathlib.Path("${asr_stats_dir}")
pairs = [
    ("train", "dump/raw/${train_set}/wav.scp", "data/${train_set}/text"),
    ("valid", "dump/raw/${valid_set}/wav.scp", "data/${valid_set}/text"),
]
for split, wav_scp, text_file in pairs:
    out_dir = stats_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = []
    for line in pathlib.Path(wav_scp).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2: continue
        uid, wp = parts
        try: shapes.append(f"{uid} {soundfile.info(wp).frames}\n")
        except Exception as e: print(f"WARNING: {wp}: {e}")
    (out_dir / "speech_shape").write_text("".join(sorted(shapes)))
    tshapes = []
    for line in pathlib.Path(text_file).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) < 1: continue
        uid = parts[0]; ref = parts[1] if len(parts) > 1 else ""
        tshapes.append(f"{uid} {len(sp.EncodeAsPieces(ref)) + 1}\n")
    (out_dir / "text_shape.bpe").write_text("".join(sorted(tshapes)))
    print(f"{split}: {len(shapes)} speech, {len(tshapes)} text shapes")
PYEOF
fi

# ---------------------------------------------------------------------------
# Stage 5: Download pretrained model, extract config, patch training YAMLs
# ---------------------------------------------------------------------------
# local/setup_pretrained.py handles everything:
#   - ModelDownloader.download_and_unpack
#   - token_list  → exp/pretrained/tokens.txt
#   - bpe.model   → data/token_list/bpe_unigram{nbpe}/bpe.model
#   - patches encoder_conf / decoder_conf / normalize in both training YAMLs
#   - writes exp/pretrained/model_info.json
#   - exits non-zero if any required field is missing
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "=== Stage 5: Downloading and configuring pretrained model ==="
    python3 local/setup_pretrained.py \
        --model   "${pretrained_model}" \
        --outdir  exp/pretrained \
        --sft_config "${sft_config}" \
        --rl_config  "${rl_config}" \
        --nbpe    "${nbpe}"
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

    bpemodel=$(python3 -c "
import json; info = json.load(open('exp/pretrained/model_info.json'))
print(info.get('bpemodel',''))
")

    mkdir -p "${sft_expdir}"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${sft_expdir}/train.log" \
        python -m espnet2.bin.asr_train_rl \
            --ngpu "${ngpu}" \
            --config "${sft_config}" \
            --token_list "${token_list}" \
            --bpemodel "${bpemodel}" \
            --init_param "${pretrained_pth}" \
            --train_data_path_and_name_and_type \
                "dump/raw/${train_set}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type \
                "data/${train_set}/text,text,text" \
            --train_shape_file "${asr_stats_dir}/train/speech_shape" \
            --train_shape_file "${asr_stats_dir}/train/text_shape.bpe" \
            --valid_data_path_and_name_and_type \
                "dump/raw/${valid_set}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type \
                "data/${valid_set}/text,text,text" \
            --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${asr_stats_dir}/valid/text_shape.bpe" \
            --output_dir "${sft_expdir}" \
            --rl_weight 0.0 \
            --seed "${seed}" \
            ${lora_opts} \
            ${smoke_test_opts}
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

    bpemodel=$(python3 -c "
import json; info = json.load(open('exp/pretrained/model_info.json'))
print(info.get('bpemodel',''))
")

    mkdir -p "${rl_expdir}"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${rl_expdir}/train.log" \
        python -m espnet2.bin.asr_train_rl \
            --ngpu "${ngpu}" \
            --config "${rl_config}" \
            --token_list "${token_list}" \
            --bpemodel "${bpemodel}" \
            --init_param "${sft_ckpt}" \
            --train_data_path_and_name_and_type \
                "dump/raw/${train_set}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type \
                "data/${train_set}/text,text,text" \
            --train_shape_file "${asr_stats_dir}/train/speech_shape" \
            --train_shape_file "${asr_stats_dir}/train/text_shape.bpe" \
            --valid_data_path_and_name_and_type \
                "dump/raw/${valid_set}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type \
                "data/${valid_set}/text,text,text" \
            --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${asr_stats_dir}/valid/text_shape.bpe" \
            --output_dir "${rl_expdir}" \
            --reward_mode "${reward_mode}" \
            --reward_loss_type "${reward_loss_type}" \
            --seed "${seed}" \
            ${domain_terms_opts} \
            ${gemini_opts} \
            ${lora_opts} \
            ${smoke_test_opts}
    log "RL checkpoint: ${rl_expdir}/valid.loss.best.pth"
fi

# ---------------------------------------------------------------------------
# Stage 8: Decode AfriSpeech test + LibriSpeech (forgetting eval) + extended eval
# ---------------------------------------------------------------------------
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "=== Stage 8: Decode, score (WER/CER), and extended eval ==="

    token_list=$(python3 -c "
import json; info = json.load(open('exp/pretrained/model_info.json'))
print(info.get('token_list',''))
")

    for expdir in "${sft_expdir}" "${rl_expdir}"; do
        model_tag=$(basename "${expdir}")
        ckpt="${expdir}/valid.loss.best.pth"
        [ -f "${ckpt}" ] || { log "Skipping ${model_tag}: checkpoint not found."; continue; }

        for test_set in ${test_sets} ${forgetting_set}; do
            decode_dir="${expdir}/decode_${test_set}"
            mkdir -p "${decode_dir}/logdir"
            if [ "${smoke_test}" = true ]; then
                head -n "${smoke_decode_n}" "dump/raw/${test_set}/wav.scp" \
                    > "${decode_dir}/logdir/keys.1.scp"
            else
                cp "dump/raw/${test_set}/wav.scp" "${decode_dir}/logdir/keys.1.scp"
            fi

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

            hyp_text="${decode_dir}/logdir/output.1/1best_recog/text"
            ref_text="data/${test_set}/text"

            # --- Basic WER/CER (plain text result) ---
            python3 - <<PYEOF
import jiwer, pathlib
hyp_f = pathlib.Path("${hyp_text}")
ref_f = pathlib.Path("${ref_text}")
if not hyp_f.exists() or not ref_f.exists():
    print("WARNING: hyp or ref not found; skipping basic scoring.")
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

            # --- Extended eval (SER, EWER, domain F1, degenerate, bootstrap) ---
            if [ -f "${hyp_text}" ] && [ -f "${ref_text}" ]; then
                # For RL model on primary test set, compute bootstrap p-value vs SFT
                baseline_arg=""
                if [ "${expdir}" = "${rl_expdir}" ] && [ "${test_set}" = "${test_sets%% *}" ]; then
                    sft_hyp="${sft_expdir}/decode_${test_set}/logdir/output.1/1best_recog/text"
                    [ -f "${sft_hyp}" ] && baseline_arg="--baseline_hyp_file ${sft_hyp}"
                fi

                bootstrap_iters=1000
                if [ "${smoke_test}" = true ]; then
                    bootstrap_iters=32
                fi

                python3 local/eval_extended.py \
                    --hyp_file "${hyp_text}" \
                    --ref_file "${ref_text}" \
                    --domain_terms_file conf/domain_terms_clinical.txt \
                    --bootstrap_iters "${bootstrap_iters}" \
                    --output_json "${decode_dir}/extended_metrics.json" \
                    ${baseline_arg} \
                    || log "WARNING: eval_extended.py failed for ${model_tag} / ${test_set}"
            fi
        done
    done

    log "=== Final results ==="
    for f in exp/asr_*/decode_*/result.txt; do
        [ -f "${f}" ] && echo "--- ${f} ---" && cat "${f}"
    done

    log "=== Extended metrics (NeMo-comparable) ==="
    python3 - <<'PYEOF'
import json, pathlib, sys
files = sorted(pathlib.Path("exp").glob("asr_*/decode_*/extended_metrics.json"))
if not files:
    print("No extended_metrics.json files found yet.")
    sys.exit(0)
for f in files:
    parts = str(f).split("/")
    tag = f"{parts[1]}/{parts[2]}"
    m = json.loads(f.read_text())
    print(f"\n{tag}")
    print(f"  WER:        {m['wer_pct']:.2f}%   CER: {m['cer_pct']:.2f}%")
    print(f"  SER:        {m['ser_pct']:.2f}%")
    print(f"  EWER:       {m['ewer_pct']:.2f}%  (domain utts: {m['n_utterances_with_domain_tokens']})")
    print(f"  Domain F1:  {m['domain_f1']:.4f}  P={m['domain_precision']:.4f}  R={m['domain_recall']:.4f}")
    print(f"  Degen frac: {m['degenerate_hyp_frac']:.6f}  mean_hyp_len: {m['mean_hyp_len_chars']:.1f}")
    if m.get('bootstrap_pval_vs_baseline') is not None:
        print(f"  Bootstrap p (vs SFT): {m['bootstrap_pval_vs_baseline']:.4f}")
PYEOF
fi

log "=== run.sh complete ==="
