---
name: ESPnet RL — VM Execution Plan
overview: All GCP VM execution steps for the ESPnet2 RL experiment suite. Prerequisite — local changes plan must be completed and pushed before starting here. Covers Phase B (one-time setup, full WWER/MWER/LoRA/VoxPopuli experiment sequence via run_all_experiments.sh, artifact collection) and Phase C VM (bitsandbytes install, LLM model download, 9 LLM reward RL runs Steps 20-28). Total estimated GPU time ~207h on 1× T4 (~$230 USD).
todos:
  - id: B1-setup-scripts
    content: "VM: Run local/capture_env.sh, local/capture_arch_snapshot.py, and local/run_zeroshot_eval.sh (~1.5h total)"
    status: pending
  - id: B2-s42-gaps
    content: "VM: Run scripts/run_all_experiments.sh step 1 — seed=42 val eval (Stage 8) + MWER RL (Stage 7+8) (~6h)"
    status: pending
  - id: B3-s33-main
    content: "VM: Run scripts/run_all_experiments.sh step 2 — seed=33 SFT + WWER RL + MWER RL on combined set (~18h)"
    status: pending
  - id: B4-s0-main
    content: "VM: Run scripts/run_all_experiments.sh step 3 — seed=0 SFT + WWER RL + MWER RL on combined set (~18h)"
    status: pending
  - id: B5-lora-all-seeds
    content: "VM: Run scripts/run_all_experiments.sh steps 4-6 — LoRA SFT + WWER RL + MWER RL for seeds 42, 33, 0 (~54h)"
    status: pending
  - id: B6-vox-all-seeds
    content: "VM: Run scripts/run_all_experiments.sh steps 7-10 — VoxPopuli shape files + SFT + WWER RL + MWER RL for seeds 42, 33, 0 (~36h)"
    status: pending
  - id: B7-collect-artifacts
    content: "VM: Run local/collect_run_artifacts.py after all Phase B training to generate run_summary.json files with timing and cost (~5 min)"
    status: pending
  - id: C5-vm-deps
    content: "VM: pip install bitsandbytes accelerate in espnet_rl conda env (required for 4-bit quantization on T4)"
    status: pending
  - id: C6-vm-download
    content: "VM: python local/download_reward_llms.py — pre-downloads all 3 LLM reward models (~15 GB, ~20 min); verify HuggingFace token for gated models (MedGemma requires HF login)"
    status: pending
  - id: C7-vm-run-llm
    content: "VM: START_STEP=20 bash scripts/run_all_experiments.sh — runs all 9 LLM reward RL experiments (~45h on 1x T4)"
    status: pending
  - id: C8-final-artifacts
    content: "VM: Re-run local/collect_run_artifacts.py after Phase C to capture LLM reward run summaries, then sync exp/ tree locally"
    status: pending
isProject: false
---

# ESPnet RL — VM Execution Plan

> **Prerequisite:** [espnet_rl_local_changes.plan.md](espnet_rl_local_changes.plan.md) must be fully completed and pushed before starting here.  
> No inline code is written on the VM — only pre-committed scripts are invoked.

---

## Overview

**Total estimated GPU compute: ~207h continuous on 1× T4**  
**Estimated GCP cost: ~$230 USD** (n1-standard-16 + T4 at $1.11/hr)


| Phase                | Content                                                           | Est. GPU time |
| -------------------- | ----------------------------------------------------------------- | ------------- |
| B — Core experiments | WWER/MWER RL × 3 seeds, LoRA × 3 seeds, VoxPopuli × 3 seeds       | ~162h         |
| C — LLM reward       | MedPhi + MedGemma (AfriSpeech) + Phi3-Legal (VoxPopuli) × 3 seeds | ~45h          |


---

## Connection

SSH to the VM and activate the environment before any step below:

```bash
gcloud compute ssh finding-nemo-again \
  --zone asia-east1-c --project adaptive-ai-487419
conda activate espnet_rl
cd ~/espnet/egs2/afrispeech_rl/asr1
```

Pull the latest local changes first:

```bash
cd ~/espnet && git pull
cd egs2/afrispeech_rl/asr1
bash run.sh --smoke_test true --ngpu 1   # verify run.sh changes don't break the pipeline
```

> **Seed note:** `--seed N` controls training initialization only. Data is identical across seeds (original `data.sh` used seed=42 for subsampling; we do not re-run data prep for new seeds).

---

## Phase B: Core Experiment Sequence

### B.1 — One-time setup scripts (~1.5h)

Run these three scripts once before any training:

```bash
bash local/capture_env.sh
python local/capture_arch_snapshot.py
bash local/run_zeroshot_eval.sh
```

### B.2 — Full experiment sequence (Steps 1–19, ~162h)

Run the master script. It can be left running unattended (~6.75 days):

```bash
bash scripts/run_all_experiments.sh
```

To resume after an interruption at a specific step:

```bash
START_STEP=5 bash scripts/run_all_experiments.sh
```

For reference, the individual `run.sh` calls encoded in Steps 1–19:

**Step 1 — Seed=42: val eval for existing checkpoints (~1h)**

```bash
bash run.sh --stage 8 --stop_stage 8 --ngpu 1 \
  --sft_expdir exp/asr_sft --rl_expdir exp/asr_rl \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
```

**Step 2 — Seed=42: MWER RL (~5h)**

```bash
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 42 \
  --sft_expdir exp/asr_sft --rl_expdir exp/asr_rl_mwer_s42 \
  --rl_config conf/train_asr_rl_mwer.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
```

**Step 3 — Seed=33: SFT + WWER RL (~14h)**

```bash
bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 33 \
  --sft_expdir exp/asr_sft_s33 --rl_expdir exp/asr_rl_wwer_s33 \
  --rl_config conf/train_asr_rl.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
```

**Step 4 — Seed=33: MWER RL (~5h)**

```bash
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 33 \
  --sft_expdir exp/asr_sft_s33 --rl_expdir exp/asr_rl_mwer_s33 \
  --rl_config conf/train_asr_rl_mwer.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
```

**Step 5 — Seed=0: SFT + WWER RL (~14h)**

```bash
bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed 0 \
  --sft_expdir exp/asr_sft_s0 --rl_expdir exp/asr_rl_wwer_s0 \
  --rl_config conf/train_asr_rl.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
```

**Step 6 — Seed=0: MWER RL (~5h)**

```bash
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed 0 \
  --sft_expdir exp/asr_sft_s0 --rl_expdir exp/asr_rl_mwer_s0 \
  --rl_config conf/train_asr_rl_mwer.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
```

**Steps 7–12 — LoRA variants, seeds 42 / 33 / 0 (~18h each)**

Pattern for each seed `S` (replace `S` with 42, 33, 0):

```bash
# WWER RL LoRA
bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed S --use_lora true \
  --sft_expdir exp/asr_sft_lora_sS --rl_expdir exp/asr_rl_wwer_lora_sS \
  --rl_config conf/train_asr_rl.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
# MWER RL LoRA
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed S --use_lora true \
  --sft_expdir exp/asr_sft_lora_sS --rl_expdir exp/asr_rl_mwer_lora_sS \
  --rl_config conf/train_asr_rl_mwer.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"
```

**Step 13 — VoxPopuli shape files (~10 min)**

```bash
bash run.sh --stage 4 --stop_stage 4 --ngpu 0 \
  --train_set voxpopuli_train --valid_set voxpopuli_dev \
  --asr_stats_dir exp/vox_stats_raw_bpe5000
```

**Steps 14–19 — VoxPopuli standalone, seeds 42 / 33 / 0 (~12h each)**

Pattern for each seed `S`:

```bash
# WWER RL
bash run.sh --stage 6 --stop_stage 8 --ngpu 1 --seed S \
  --train_set voxpopuli_train --valid_set voxpopuli_dev \
  --asr_stats_dir exp/vox_stats_raw_bpe5000 \
  --sft_expdir exp/vox_sft_sS --rl_expdir exp/vox_rl_wwer_sS \
  --rl_config conf/train_asr_rl.yaml \
  --test_sets "voxpopuli_dev librispeech_dev_clean"
# MWER RL
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed S \
  --train_set voxpopuli_train --valid_set voxpopuli_dev \
  --asr_stats_dir exp/vox_stats_raw_bpe5000 \
  --sft_expdir exp/vox_sft_sS --rl_expdir exp/vox_rl_mwer_sS \
  --rl_config conf/train_asr_rl_mwer.yaml \
  --test_sets "voxpopuli_dev librispeech_dev_clean"
```

### B.3 — Collect timing and cost artifacts (~5 min, after Steps 1–19)

```bash
python local/collect_run_artifacts.py
```

Writes `run_summary.json` into every `exp/*/` directory.

---

## Phase C: LLM Reward Mode — VM Execution

Prerequisite: Phase C local changes (C.1–C.4 in the local plan) must be committed and pulled before these steps.

### C.5 — Install quantization dependencies (one-time)

```bash
conda activate espnet_rl
pip install bitsandbytes accelerate
```

Required for `BitsAndBytesConfig(load_in_4bit=True)`. Run once before any LLM reward training.

### C.6 — Pre-download LLM reward models (~20 min, ~15 GB)

```bash
cd ~/espnet/egs2/afrispeech_rl/asr1
huggingface-cli login          # needed for gated models (MedGemma at minimum)
python local/download_reward_llms.py
```

Verify all three models complete without 401/403 errors before starting training.

### C.7 — Run LLM reward experiments (~45h)

```bash
START_STEP=20 bash scripts/run_all_experiments.sh
```

Runs Steps 20–28 sequentially. Each of the 9 runs reuses an existing SFT checkpoint and only runs Stage 7 (RL) + Stage 8 (eval), ~5h per run on the T4.

To resume after interruption at a specific step:

```bash
START_STEP=24 bash scripts/run_all_experiments.sh
```

Individual commands encoded in Steps 20–28:

```bash
# Steps 20–22: AfriSpeech + MedPhi-3.8b, seeds 42 / 33 / 0
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed S \
  --sft_expdir exp/asr_sft_sS \
  --rl_expdir exp/asr_rl_llm_medphi_sS \
  --rl_config conf/train_asr_rl_llm_medphi.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"

# Steps 23–25: AfriSpeech + MedGemma-4b-it, seeds 42 / 33 / 0
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed S \
  --sft_expdir exp/asr_sft_sS \
  --rl_expdir exp/asr_rl_llm_medgemma_sS \
  --rl_config conf/train_asr_rl_llm_medgemma.yaml \
  --test_sets "afrispeech_dev afrispeech_test librispeech_dev_clean"

# Steps 26–28: VoxPopuli + Phi3-Legal, seeds 42 / 33 / 0
bash run.sh --stage 7 --stop_stage 8 --ngpu 1 --seed S \
  --train_set voxpopuli_train --valid_set voxpopuli_dev \
  --asr_stats_dir exp/vox_stats_raw_bpe5000 \
  --sft_expdir exp/vox_sft_sS \
  --rl_expdir exp/vox_rl_llm_philegal_sS \
  --rl_config conf/train_asr_rl_llm_philegal.yaml \
  --test_sets "voxpopuli_dev librispeech_dev_clean"
```

Seed=42 AfriSpeech SFT uses `exp/asr_sft` (no `_sS` suffix — existing Phase B checkpoint).

### C.8 — Final artifact collection (~5 min)

```bash
python local/collect_run_artifacts.py
```

Re-run after Phase C to capture LLM reward run summaries alongside the earlier ones. Then sync the full `exp/` tree locally for analysis.

---

## Full Experiment Matrix

### Phase B runs


| Group         | Seed | WWER RL expdir           | MWER RL expdir           | Est. GPU time   |
| ------------- | ---- | ------------------------ | ------------------------ | --------------- |
| Main combined | 42   | exp/asr_rl (existing)    | exp/asr_rl_mwer_s42      | ~6h (gaps only) |
| Main combined | 33   | exp/asr_rl_wwer_s33      | exp/asr_rl_mwer_s33      | ~18h            |
| Main combined | 0    | exp/asr_rl_wwer_s0       | exp/asr_rl_mwer_s0       | ~18h            |
| LoRA          | 42   | exp/asr_rl_wwer_lora_s42 | exp/asr_rl_mwer_lora_s42 | ~18h            |
| LoRA          | 33   | exp/asr_rl_wwer_lora_s33 | exp/asr_rl_mwer_lora_s33 | ~18h            |
| LoRA          | 0    | exp/asr_rl_wwer_lora_s0  | exp/asr_rl_mwer_lora_s0  | ~18h            |
| VoxPopuli     | 42   | exp/vox_rl_wwer_s42      | exp/vox_rl_mwer_s42      | ~12h            |
| VoxPopuli     | 33   | exp/vox_rl_wwer_s33      | exp/vox_rl_mwer_s33      | ~12h            |
| VoxPopuli     | 0    | exp/vox_rl_wwer_s0       | exp/vox_rl_mwer_s0       | ~12h            |


**One-time (no seed):** zero-shot (1h) + env capture (10 min) + arch snapshot (10 min) = ~1.5h  
**Phase B total: ~155h training + ~7h overhead ≈ 162h**

### Phase C runs


| Group          | LLM reward model     | Seeds     | Expdir pattern               | Est. GPU time |
| -------------- | -------------------- | --------- | ---------------------------- | ------------- |
| AfriSpeech LLM | MedPhi-3.8b          | 42, 33, 0 | `exp/asr_rl_llm_medphi_sS`   | ~15h          |
| AfriSpeech LLM | MedGemma-4b-it       | 42, 33, 0 | `exp/asr_rl_llm_medgemma_sS` | ~15h          |
| VoxPopuli LLM  | Phi3-Legal-Finetuned | 42, 33, 0 | `exp/vox_rl_llm_philegal_sS` | ~15h          |


**Phase C total: ~~45h additional GPU time (~~$50 USD)**  
**Combined total (B+C): ~~207h continuous on 1× T4 (~~$230 USD)**