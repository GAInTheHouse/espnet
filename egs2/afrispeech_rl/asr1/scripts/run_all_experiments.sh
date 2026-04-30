#!/usr/bin/env bash
# scripts/run_all_experiments.sh
#
# Master orchestration script for all ESPnet RL experiments.
# Chains every bash run.sh call in the correct order with timestamped logging.
#
# Usage
# -----
#   cd egs2/afrispeech_rl/asr1
#   bash scripts/run_all_experiments.sh                   # run all steps
#   START_STEP=4 bash scripts/run_all_experiments.sh      # resume from step 4
#   STOP_STEP=12 bash scripts/run_all_experiments.sh      # run steps 1-12 only
#
# Steps 1–12:  AfriSpeech WWER/MWER, full and LoRA (seeds 42 / 33 / 0)
# Steps 13–19: VoxPopuli shape files + WWER/MWER (seeds 42 / 33 / 0)
# Steps 20–28: LLM reward mode — appended by Phase C (C.4)

set -euo pipefail

START_STEP="${START_STEP:-1}"
STOP_STEP="${STOP_STEP:-19}"

LOGDIR="logs/run_all_experiments"
mkdir -p "${LOGDIR}"

log_step() {
    local step="$1"
    local desc="$2"
    echo ""
    echo "============================================================"
    echo "  STEP ${step}: ${desc}"
    echo "  $(date '+%Y-%m-%dT%H:%M:%S')"
    echo "============================================================"
}

finish_step() {
    local step="$1"
    echo "  [DONE] Step ${step} finished at $(date '+%Y-%m-%dT%H:%M:%S')"
}

skip_if_before() {
    local step="$1"
    [ "${step}" -ge "${START_STEP}" ] && [ "${step}" -le "${STOP_STEP}" ]
}

# ============================================================
# Step 1: Seed=42 — Stage 8 val eval for existing SFT + WWER RL (~1h)
# ============================================================
STEP=1
if skip_if_before ${STEP}; then
    log_step ${STEP} "Seed=42 — Stage 8 val eval (existing SFT + WWER RL)"
    bash run.sh --stage 8 --stop_stage 8 --ngpu 1 --seed 42 \
        --sft_expdir exp/asr_sft \
        --rl_expdir  exp/asr_rl \
        --rl_config  conf/train_asr_rl.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 2: Seed=42 — Stages 7+8 MWER RL (~5h)
# ============================================================
STEP=2
if skip_if_before ${STEP}; then
    log_step ${STEP} "Seed=42 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 42 \
        --sft_expdir exp/asr_sft \
        --rl_expdir  exp/asr_rl_mwer_s42 \
        --rl_config  conf/train_asr_rl_mwer.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 3: Seed=33 — Stages 6+8 SFT + WWER RL (~14h)
# ============================================================
STEP=3
if skip_if_before ${STEP}; then
    log_step ${STEP} "Seed=33 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 33 \
        --sft_expdir exp/asr_sft_s33 \
        --rl_expdir  exp/asr_rl_s33 \
        --rl_config  conf/train_asr_rl.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 4: Seed=33 — Stages 7+8 MWER RL (~5h)
# ============================================================
STEP=4
if skip_if_before ${STEP}; then
    log_step ${STEP} "Seed=33 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 33 \
        --sft_expdir exp/asr_sft_s33 \
        --rl_expdir  exp/asr_rl_mwer_s33 \
        --rl_config  conf/train_asr_rl_mwer.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 5: Seed=0 — Stages 6+8 SFT + WWER RL (~14h)
# ============================================================
STEP=5
if skip_if_before ${STEP}; then
    log_step ${STEP} "Seed=0 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 0 \
        --sft_expdir exp/asr_sft_s0 \
        --rl_expdir  exp/asr_rl_s0 \
        --rl_config  conf/train_asr_rl.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 6: Seed=0 — Stages 7+8 MWER RL (~5h)
# ============================================================
STEP=6
if skip_if_before ${STEP}; then
    log_step ${STEP} "Seed=0 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 0 \
        --sft_expdir exp/asr_sft_s0 \
        --rl_expdir  exp/asr_rl_mwer_s0 \
        --rl_config  conf/train_asr_rl_mwer.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 7: LoRA Seed=42 — Stages 6+8 SFT + WWER RL (~14h)
# ============================================================
STEP=7
if skip_if_before ${STEP}; then
    log_step ${STEP} "LoRA Seed=42 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 42 \
        --use_lora   true \
        --sft_expdir exp/asr_lora_sft_s42 \
        --rl_expdir  exp/asr_lora_rl_s42 \
        --rl_config  conf/train_asr_rl.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 8: LoRA Seed=42 — Stages 7+8 MWER RL (~5h)
# ============================================================
STEP=8
if skip_if_before ${STEP}; then
    log_step ${STEP} "LoRA Seed=42 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 42 \
        --use_lora   true \
        --sft_expdir exp/asr_lora_sft_s42 \
        --rl_expdir  exp/asr_lora_rl_mwer_s42 \
        --rl_config  conf/train_asr_rl_mwer.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 9: LoRA Seed=33 — Stages 6+8 SFT + WWER RL (~14h)
# ============================================================
STEP=9
if skip_if_before ${STEP}; then
    log_step ${STEP} "LoRA Seed=33 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 33 \
        --use_lora   true \
        --sft_expdir exp/asr_lora_sft_s33 \
        --rl_expdir  exp/asr_lora_rl_s33 \
        --rl_config  conf/train_asr_rl.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 10: LoRA Seed=33 — Stages 7+8 MWER RL (~5h)
# ============================================================
STEP=10
if skip_if_before ${STEP}; then
    log_step ${STEP} "LoRA Seed=33 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 33 \
        --use_lora   true \
        --sft_expdir exp/asr_lora_sft_s33 \
        --rl_expdir  exp/asr_lora_rl_mwer_s33 \
        --rl_config  conf/train_asr_rl_mwer.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 11: LoRA Seed=0 — Stages 6+8 SFT + WWER RL (~14h)
# ============================================================
STEP=11
if skip_if_before ${STEP}; then
    log_step ${STEP} "LoRA Seed=0 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 0 \
        --use_lora   true \
        --sft_expdir exp/asr_lora_sft_s0 \
        --rl_expdir  exp/asr_lora_rl_s0 \
        --rl_config  conf/train_asr_rl.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 12: LoRA Seed=0 — Stages 7+8 MWER RL (~5h)
# ============================================================
STEP=12
if skip_if_before ${STEP}; then
    log_step ${STEP} "LoRA Seed=0 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 0 \
        --use_lora   true \
        --sft_expdir exp/asr_lora_sft_s0 \
        --rl_expdir  exp/asr_lora_rl_mwer_s0 \
        --rl_config  conf/train_asr_rl_mwer.yaml \
        --test_sets  "afrispeech_dev afrispeech_test librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 13: VoxPopuli — Stage 4 shape files (~10 min)
# ============================================================
STEP=13
if skip_if_before ${STEP}; then
    log_step ${STEP} "VoxPopuli — Stage 4 shape files"
    bash run.sh --stage 4 --stop_stage 4 --ngpu 1 \
        --train_set    voxpopuli_train \
        --valid_set    voxpopuli_dev \
        --asr_stats_dir exp/vox_stats_raw_bpe5000 \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 14: VoxPopuli Seed=42 — Stages 6+8 SFT + WWER RL (~9h)
# ============================================================
STEP=14
if skip_if_before ${STEP}; then
    log_step ${STEP} "VoxPopuli Seed=42 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 42 \
        --train_set     voxpopuli_train \
        --valid_set     voxpopuli_dev \
        --asr_stats_dir exp/vox_stats_raw_bpe5000 \
        --sft_expdir    exp/vox_sft_s42 \
        --rl_expdir     exp/vox_rl_s42 \
        --rl_config     conf/train_asr_rl.yaml \
        --test_sets     "voxpopuli_dev librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 15: VoxPopuli Seed=42 — Stages 7+8 MWER RL (~4h)
# ============================================================
STEP=15
if skip_if_before ${STEP}; then
    log_step ${STEP} "VoxPopuli Seed=42 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 42 \
        --train_set     voxpopuli_train \
        --valid_set     voxpopuli_dev \
        --asr_stats_dir exp/vox_stats_raw_bpe5000 \
        --sft_expdir    exp/vox_sft_s42 \
        --rl_expdir     exp/vox_rl_mwer_s42 \
        --rl_config     conf/train_asr_rl_mwer.yaml \
        --test_sets     "voxpopuli_dev librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 16: VoxPopuli Seed=33 — Stages 6+8 SFT + WWER RL (~9h)
# ============================================================
STEP=16
if skip_if_before ${STEP}; then
    log_step ${STEP} "VoxPopuli Seed=33 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 33 \
        --train_set     voxpopuli_train \
        --valid_set     voxpopuli_dev \
        --asr_stats_dir exp/vox_stats_raw_bpe5000 \
        --sft_expdir    exp/vox_sft_s33 \
        --rl_expdir     exp/vox_rl_s33 \
        --rl_config     conf/train_asr_rl.yaml \
        --test_sets     "voxpopuli_dev librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 17: VoxPopuli Seed=33 — Stages 7+8 MWER RL (~4h)
# ============================================================
STEP=17
if skip_if_before ${STEP}; then
    log_step ${STEP} "VoxPopuli Seed=33 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 33 \
        --train_set     voxpopuli_train \
        --valid_set     voxpopuli_dev \
        --asr_stats_dir exp/vox_stats_raw_bpe5000 \
        --sft_expdir    exp/vox_sft_s33 \
        --rl_expdir     exp/vox_rl_mwer_s33 \
        --rl_config     conf/train_asr_rl_mwer.yaml \
        --test_sets     "voxpopuli_dev librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 18: VoxPopuli Seed=0 — Stages 6+8 SFT + WWER RL (~9h)
# ============================================================
STEP=18
if skip_if_before ${STEP}; then
    log_step ${STEP} "VoxPopuli Seed=0 — Stages 6+8 SFT + WWER RL"
    bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 0 \
        --train_set     voxpopuli_train \
        --valid_set     voxpopuli_dev \
        --asr_stats_dir exp/vox_stats_raw_bpe5000 \
        --sft_expdir    exp/vox_sft_s0 \
        --rl_expdir     exp/vox_rl_s0 \
        --rl_config     conf/train_asr_rl.yaml \
        --test_sets     "voxpopuli_dev librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Step 19: VoxPopuli Seed=0 — Stages 7+8 MWER RL (~4h)
# ============================================================
STEP=19
if skip_if_before ${STEP}; then
    log_step ${STEP} "VoxPopuli Seed=0 — Stages 7+8 MWER RL"
    bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 0 \
        --train_set     voxpopuli_train \
        --valid_set     voxpopuli_dev \
        --asr_stats_dir exp/vox_stats_raw_bpe5000 \
        --sft_expdir    exp/vox_sft_s0 \
        --rl_expdir     exp/vox_rl_mwer_s0 \
        --rl_config     conf/train_asr_rl_mwer.yaml \
        --test_sets     "voxpopuli_dev librispeech_dev_clean" \
        2>&1 | tee "${LOGDIR}/step${STEP}.log"
    finish_step ${STEP}
fi

# ============================================================
# Steps 20–28 added by Phase C (scripts/run_all_experiments.sh C.4)
# ============================================================

echo ""
echo "============================================================"
echo "  All steps ${START_STEP}–${STOP_STEP} complete."
echo "  $(date '+%Y-%m-%dT%H:%M:%S')"
echo "============================================================"
