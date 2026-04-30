---
name: ESPnet RL Full Experiment Plan — 3 Seeds + LoRA + VoxPopuli
overview: Execute all ESPnet2 RL experiment TODOs from §12 of espnet_results.md, expanded to 3 seeds (42, 33, 0), LoRA variants for all seeds on the main combined experiment, and VoxPopuli standalone for all seeds. Split into Phase A (all local prep — code changes + script writing) and Phase B (VM execution — run scripts only, no inline code on VM).
todos:
  - id: A1-fix-stage8
    content: "LOCAL: Fix run.sh Stage 8 to use ${sft_expdir}/${rl_expdir} variables — remove hardcoded exp/asr_sft and exp/asr_rl paths"
    status: pending
  - id: A2-fix-statsdir
    content: "LOCAL: Add asr_stats_dir as a top-level configurable variable in run.sh (needed for VoxPopuli standalone shape files)"
    status: pending
  - id: A3-mwer-yaml
    content: "LOCAL: Create and commit conf/train_asr_rl_mwer.yaml (copy of rl config with reward_mode changed to mwer)"
    status: pending
  - id: A4-capture-env-sh
    content: "LOCAL: Write local/capture_env.sh — captures python/torch/espnet/jiwer/datasets versions, pip freeze, git hash, nvidia-smi"
    status: pending
  - id: A5-arch-snapshot-py
    content: "LOCAL: Write local/capture_arch_snapshot.py — reads exp/pretrained/model_info.json and saves pretrained model config YAML"
    status: pending
  - id: A6-zeroshot-sh
    content: "LOCAL: Write local/run_zeroshot_eval.sh — runs asr_inference + eval_extended.py on afrispeech_test and afrispeech_dev using pretrained checkpoint"
    status: pending
  - id: A7-collect-artifacts-py
    content: "LOCAL: Write local/collect_run_artifacts.py — scans all exp/ dirs, parses train.log timestamps, computes GPU-hours and USD cost, writes run_summary.json per run"
    status: pending
  - id: A8-master-script
    content: "LOCAL: Write scripts/run_all_experiments.sh — master orchestration script that chains every bash run.sh call in the correct order with logging"
    status: pending
  - id: A9-sync-vm
    content: "LOCAL: git push all changes; then git pull on VM and run smoke test to verify run.sh changes work"
    status: pending
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
    content: "VM: Run local/collect_run_artifacts.py after all training runs to generate run_summary.json files with timing and cost (~5 min)"
    status: pending
  - id: C1-dataset-quality
    content: "ONGOING: P2-H review of AfriSpeech/VoxPopuli/LibriSpeech subset quality and label noise"
    status: pending
  - id: C2-llm-backend
    content: "LOCAL: Modify espnet2/asr/rl_espnet_model.py — add llm_reward_model arg, local HuggingFace 4-bit quantized inference path (AutoModelForCausalLM + BitsAndBytesConfig), domain-tagged scoring prompt; keep Gemini path as fallback"
    status: pending
  - id: C3-llm-configs
    content: "LOCAL: Create conf/train_asr_rl_llm_medphi.yaml, conf/train_asr_rl_llm_medgemma.yaml, conf/train_asr_rl_llm_philegal.yaml — each sets reward_mode: llm and llm_reward_model: <hf-id>"
    status: pending
  - id: C4-download-script
    content: "LOCAL: Write local/download_reward_llms.py — uses huggingface_hub.snapshot_download to pre-pull MedPhi-3.8b, google/medgemma-4b-it, Phi3-Legal-Finetuned"
    status: pending
  - id: C5-extend-master
    content: "LOCAL: Extend scripts/run_all_experiments.sh with Steps 20–28 — AfriSpeech×MedPhi (seeds 42/33/0), AfriSpeech×MedGemma (seeds 42/33/0), VoxPopuli×Phi3Legal (seeds 42/33/0)"
    status: pending
  - id: C6-vm-deps
    content: "VM: pip install bitsandbytes accelerate in espnet_rl conda env (required for 4-bit quantization on T4)"
    status: pending
  - id: C7-vm-download
    content: "VM: python local/download_reward_llms.py — pre-downloads all 3 LLM reward models (~15 GB, ~20 min); verify HuggingFace token for gated models (MedGemma requires HF login)"
    status: pending
  - id: C8-vm-run-llm
    content: "VM: START_STEP=20 bash scripts/run_all_experiments.sh — runs all 9 LLM reward RL experiments (~45h on 1x T4)"
    status: pending
isProject: false
---

# ESPnet RL Full Experiment Plan — 3 Seeds + LoRA + VoxPopuli

> **Superseded — this combined plan has been split into two standalone plans:**
>
> - [espnet_rl_local_changes.plan.md](espnet_rl_local_changes.plan.md) — all local machine work (Phase A + Phase C LOCAL)
> - [espnet_rl_vm_execution.plan.md](espnet_rl_vm_execution.plan.md) — all VM execution (Phase B + Phase C VM)
>
> Kept here for reference only.

---

## Overview

The plan is split into two phases:

- **Phase A — Local prep** (everything done on the local machine): code changes to `run.sh`, new config file, and 5 new scripts committed to the repo. Nothing is run yet.
- **Phase B — VM execution** (only pre-written scripts are invoked): SSH to the GCP VM, git pull, then run scripts in order. No inline code or one-liners typed on the VM.

**Total estimated GPU compute: ~~162h (~~6.75 days, sequential on 1× T4)**
**Estimated GCP cost: ~$180 USD** (n1-standard-16 + T4 at $1.11/hr)

---

## Phase A: Local Preparation

All work below happens on the local machine. After Phase A is complete, a single `git push` + `git pull` on the VM is the only sync step needed before running experiments.

### A.1 — Code change: fix Stage 8 expdirs in `run.sh`

**File:** [egs2/afrispeech_rl/asr1/run.sh](egs2/afrispeech_rl/asr1/run.sh) (~line 411)

Stage 8 currently hardcodes the experiment directories, ignoring `--sft_expdir` / `--rl_expdir` CLI flags. This must be fixed before any multi-seed or LoRA run can use stage 8.

```bash
# BEFORE (remove these two lines)
for model_tag in sft rl; do
    expdir="exp/asr_${model_tag}"

# AFTER
for expdir in "${sft_expdir}" "${rl_expdir}"; do
```

Also update the bootstrap `baseline_arg` block inside stage 8 to derive the SFT hypothesis path from `${sft_expdir}` instead of the hardcoded `exp/asr_sft` string.

### A.2 — Code change: make `asr_stats_dir` configurable in `run.sh`

**File:** [egs2/afrispeech_rl/asr1/run.sh](egs2/afrispeech_rl/asr1/run.sh) (top-level variables block)

Add `asr_stats_dir` as a top-level variable (default unchanged for existing runs) so VoxPopuli standalone can use a separate shape-file directory:

```bash
asr_stats_dir="exp/asr_stats_raw_bpe5000"   # add near other variable declarations
```

Replace all three hardcoded occurrences of `"exp/asr_stats_raw_bpe5000"` in stages 4, 6, and 7 with `"${asr_stats_dir}"`.

### A.3 — New config file: `conf/train_asr_rl_mwer.yaml`

**File:** [egs2/afrispeech_rl/asr1/conf/train_asr_rl_mwer.yaml](egs2/afrispeech_rl/asr1/conf/train_asr_rl_mwer.yaml)

Copy of `conf/train_asr_rl.yaml` with one line changed:

```yaml
reward_mode: mwer   # was: reward_mode: wwer
```

Commit this file so no `sed` or `cp` is needed on the VM.

### A.4 — New script: `local/capture_env.sh`

**File:** [egs2/afrispeech_rl/asr1/local/capture_env.sh](egs2/afrispeech_rl/asr1/local/capture_env.sh)

Shell script that, when run from `egs2/afrispeech_rl/asr1/` inside the `espnet_rl` conda environment, captures all package versions and hardware state into `espnet_rl_pip_freeze.txt` and `espnet_rl_env_summary.txt`:

- Python version and platform string
- PyTorch version, CUDA version, `cuda_available`
- ESPnet version (from `espnet.__version`__)
- jiwer, datasets versions
- Full `pip freeze` output
- Git commit hash of the ESPnet repo
- `nvidia-smi` output

Output files are written to `espnet-docs/env/` (create this dir in the script).

### A.5 — New script: `local/capture_arch_snapshot.py`

**File:** [egs2/afrispeech_rl/asr1/local/capture_arch_snapshot.py](egs2/afrispeech_rl/asr1/local/capture_arch_snapshot.py)

Python script that:

1. Reads `exp/pretrained/model_info.json`
2. Opens the `asr_train_config` YAML path stored in that JSON
3. Writes the first 60 lines to `espnet-docs/espnet_pretrained_config_snapshot.yaml`
4. Prints a confirmation message

Run from `egs2/afrispeech_rl/asr1/` after the pretrained model has been set up (Stage 5 already done).

### A.6 — New script: `local/run_zeroshot_eval.sh`

**File:** [egs2/afrispeech_rl/asr1/local/run_zeroshot_eval.sh](egs2/afrispeech_rl/asr1/local/run_zeroshot_eval.sh)

Shell script that:

1. Reads `exp/pretrained/model_info.json` to extract `asr_train_config` and `asr_model_file` paths
2. Loops over `afrispeech_test` and `afrispeech_dev`
3. For each set: creates decode dir, runs `python -m espnet2.bin.asr_inference` with the pretrained checkpoint (no SFT/RL), then runs `python local/eval_extended.py`
4. Writes `exp/pretrained/decode_<set>/extended_metrics.json`

No seed dependency — the pretrained model is fixed for all seeds.

### A.7 — New script: `local/collect_run_artifacts.py`

**File:** [egs2/afrispeech_rl/asr1/local/collect_run_artifacts.py](egs2/afrispeech_rl/asr1/local/collect_run_artifacts.py)

Python script that:

1. Glob-scans all `exp/*/train.log` files
2. Parses the `date '+%Y-%m-%dT%H:%M:%S'` prefix timestamps to compute per-epoch and total elapsed training time
3. Computes GPU-hours from elapsed seconds
4. Estimates USD cost at $1.11/hr (n1-standard-16 + T4, 2026 on-demand rate)
5. Writes `run_summary.json` alongside each `exp/*/train.log`:

```json
{
  "expdir": "exp/asr_rl_mwer_s42",
  "epochs": 2,
  "total_train_time_s": 12747,
  "gpu_hours": 3.54,
  "instance_type": "n1-standard-16 + 1x T4",
  "estimated_cost_usd": 3.93,
  "seed": 42
}
```

Run once after all training is complete.

### A.8 — New script: `scripts/run_all_experiments.sh`

**File:** [egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh](egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh)

Master orchestration script that chains every `bash run.sh ...` call in the correct order. Key design:

- `set -euo pipefail` so any failure stops the chain
- Each step is wrapped in a `log_step "description"` helper that prints a timestamped header and footer
- Steps are numbered (1–10) and can be fast-forwarded via a `START_STEP` env var: `START_STEP=4 bash scripts/run_all_experiments.sh`
- All `bash run.sh` flags are spelled out explicitly (no defaults assumed)

**Steps encoded in the script:**

```
Step 1:  Seed=42 — Stage 8 val eval for existing SFT + WWER RL (~1h)
Step 2:  Seed=42 — Stages 7+8 MWER RL (~5h)
Step 3:  Seed=33 — Stages 6+8 SFT + WWER RL (~14h)
Step 4:  Seed=33 — Stages 7+8 MWER RL (~5h)
Step 5:  Seed=0  — Stages 6+8 SFT + WWER RL (~14h)
Step 6:  Seed=0  — Stages 7+8 MWER RL (~5h)
Step 7:  LoRA Seed=42 — Stages 6+8 SFT + WWER RL (~14h)
Step 8:  LoRA Seed=42 — Stages 7+8 MWER RL (~5h)
Step 9:  LoRA Seed=33 — Stages 6+8 SFT + WWER RL (~14h)
Step 10: LoRA Seed=33 — Stages 7+8 MWER RL (~5h)
Step 11: LoRA Seed=0  — Stages 6+8 SFT + WWER RL (~14h)
Step 12: LoRA Seed=0  — Stages 7+8 MWER RL (~5h)
Step 13: VoxPopuli — Stage 4 shape files (~10 min)
Step 14: VoxPopuli Seed=42 — Stages 6+8 SFT + WWER RL (~9h)
Step 15: VoxPopuli Seed=42 — Stages 7+8 MWER RL (~4h)
Step 16: VoxPopuli Seed=33 — Stages 6+8 SFT + WWER RL (~9h)
Step 17: VoxPopuli Seed=33 — Stages 7+8 MWER RL (~4h)
Step 18: VoxPopuli Seed=0  — Stages 6+8 SFT + WWER RL (~9h)
Step 19: VoxPopuli Seed=0  — Stages 7+8 MWER RL (~4h)
```

Each step's `bash run.sh` invocation is identical to the commands listed in §Phase B below.

### A.9 — Sync to VM

After all local changes are committed:

```bash
git add \
  egs2/afrispeech_rl/asr1/run.sh \
  egs2/afrispeech_rl/asr1/conf/train_asr_rl_mwer.yaml \
  egs2/afrispeech_rl/asr1/local/capture_env.sh \
  egs2/afrispeech_rl/asr1/local/capture_arch_snapshot.py \
  egs2/afrispeech_rl/asr1/local/run_zeroshot_eval.sh \
  egs2/afrispeech_rl/asr1/local/collect_run_artifacts.py \
  egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh
git commit -m "feat: multi-seed experiment scripts, run.sh expdir+statsdir fixes, MWER config"
git push
```

Then on the VM:

```bash
cd ~/espnet && git pull
cd egs2/afrispeech_rl/asr1
bash run.sh --smoke_test true --ngpu 1   # verify run.sh changes don't break the pipeline
```

---

## Phase B: VM Execution

SSH to the VM for all steps below:

```bash
gcloud compute ssh finding-nemo-again \
  --zone asia-east1-c --project adaptive-ai-487419
conda activate espnet_rl
cd ~/espnet/egs2/afrispeech_rl/asr1
```

> **Seed note:** `--seed N` controls training initialization only. Data is identical across seeds (original `data.sh` used seed=42 for subsampling; we do not re-run data prep for new seeds).

### B.1 — One-time setup scripts (~1.5h)

Run these three scripts once before any training:

```bash
bash local/capture_env.sh
python local/capture_arch_snapshot.py
bash local/run_zeroshot_eval.sh
```

### B.2 — Full experiment sequence

Run the master script. It can be left running unattended (~6.75 days):

```bash
bash scripts/run_all_experiments.sh
```

To resume after an interruption at a specific step number:

```bash
START_STEP=5 bash scripts/run_all_experiments.sh
```

For reference, the individual `run.sh` calls encoded in the master script are:

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

### B.3 — Collect timing and cost artifacts (~5 min, after all training)

```bash
python local/collect_run_artifacts.py
```

Writes `run_summary.json` into every `exp/*/` directory. Copy the full `exp/` tree locally for analysis.

---

## Full Experiment Matrix


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
**Total: ~155h training + ~7h overhead ≈ 162h continuous on 1× T4**

---

## Phase C: LLM Reward Mode

Domain-specific LLMs loaded locally from HuggingFace replace the deferred Gemini API approach. The model used depends on the dataset domain:


| Dataset    | Domain             | LLM reward models                                           |
| ---------- | ------------------ | ----------------------------------------------------------- |
| AfriSpeech | Clinical / medical | MedPhi-3.8b **and** MedGemma-4b-it (two separate ablations) |
| VoxPopuli  | Parliamentary      | Phi3-Legal-Finetuned                                        |


**Phase C adds 9 new RL runs** (3 model variants × 3 seeds each), all using Stage 7 only since SFT checkpoints already exist from Phase B.  
**Estimated additional GPU time: ~45h**

---

### C.1 LOCAL — Modify LLM reward backend in `rl_espnet_model.py`

**File:** `espnet2/asr/rl_espnet_model.py`

The existing LLM reward code calls the Gemini API via `GEMINI_API_KEY`. Add a local HuggingFace inference path alongside it:

1. Add `llm_reward_model: str = ""` to the model's `__init`__ signature (populated from the YAML config)
2. When `llm_reward_model` is non-empty, load with 4-bit quantization to fit on T4 VRAM:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_reward_model)
self.llm_model = AutoModelForCausalLM.from_pretrained(
    llm_reward_model, quantization_config=bnb_cfg, device_map="auto"
)
```

1. The reward scoring prompt should include a domain tag derived from the model name (e.g. `medical` for MedPhi/MedGemma, `legal` for Phi3-Legal) so the LLM receives appropriate context
2. Keep the existing Gemini path as a fallback (env var check) — no existing code breaks

---

### C.2 LOCAL — Three new YAML config files

**Files in:** [egs2/afrispeech_rl/asr1/conf/](egs2/afrispeech_rl/asr1/conf/)

Each is a copy of `conf/train_asr_rl.yaml` with two lines changed:

```yaml
reward_mode: llm
llm_reward_model: <hf-model-id>
```


| Config filename                  | Dataset    | HuggingFace model ID                                                   |
| -------------------------------- | ---------- | ---------------------------------------------------------------------- |
| `train_asr_rl_llm_medphi.yaml`   | AfriSpeech | `allenai/MedPhi-3.8b` *(verify exact HF repo slug before committing)*  |
| `train_asr_rl_llm_medgemma.yaml` | AfriSpeech | `google/medgemma-4b-it`                                                |
| `train_asr_rl_llm_philegal.yaml` | VoxPopuli  | `<Phi3-Legal HF repo>` *(verify exact HF repo slug before committing)* |


> Note: `google/medgemma-4b-it` is confirmed. Verify the exact HuggingFace repo slugs for MedPhi-3.8b and Phi3-Legal-Finetuned before writing the YAMLs.

---

### C.3 LOCAL — New script: `local/download_reward_llms.py`

**File:** [egs2/afrispeech_rl/asr1/local/download_reward_llms.py](egs2/afrispeech_rl/asr1/local/download_reward_llms.py)

```python
from huggingface_hub import snapshot_download

REWARD_MODELS = [
    "allenai/MedPhi-3.8b",      # AfriSpeech clinical
    "google/medgemma-4b-it",     # AfriSpeech clinical (ablation)
    "<Phi3-Legal-HF-id>",        # VoxPopuli parliamentary
]
for repo_id in REWARD_MODELS:
    print(f"Downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id)
    print(f"  Done: {repo_id}")
```

Downloads ~~5 GB per model (~~15 GB total). MedGemma requires a HuggingFace token (`huggingface-cli login`) — verify gating status for the other two models as well.

---

### C.4 LOCAL — Extend `scripts/run_all_experiments.sh` (Steps 20–28)

**File:** [egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh](egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh)

Append 9 steps to the existing master script. Pattern for each seed `S`:

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

All 9 steps use Stage 7 only (SFT checkpoints reused from Phase B). Seed=42 SFT for AfriSpeech uses `exp/asr_sft` (no `_sS` suffix, existing checkpoint).

---

### C.5 VM — Install quantization dependencies

```bash
conda activate espnet_rl
pip install bitsandbytes accelerate
```

Required for `BitsAndBytesConfig(load_in_4bit=True)`. One-time install before any LLM reward training.

---

### C.6 VM — Pre-download LLM reward models (~20 min, ~15 GB)

```bash
cd ~/espnet/egs2/afrispeech_rl/asr1
huggingface-cli login          # needed for gated models (MedGemma at minimum)
python local/download_reward_llms.py
```

Verify all three models complete without 401/403 errors before starting training.

---

### C.7 VM — Run LLM reward experiments (~45h)

```bash
START_STEP=20 bash scripts/run_all_experiments.sh
```

Runs Steps 20–28 sequentially. Each of the 9 runs reuses an existing SFT checkpoint and only runs Stage 7 (RL) + Stage 8 (eval), ~5h per run on the T4.

To resume after interruption at a specific step:

```bash
START_STEP=24 bash scripts/run_all_experiments.sh
```

---

## Updated Experiment Matrix (Phase C additions)


| Group          | LLM reward model     | Seeds     | Expdir pattern               | Est. GPU time |
| -------------- | -------------------- | --------- | ---------------------------- | ------------- |
| AfriSpeech LLM | MedPhi-3.8b          | 42, 33, 0 | `exp/asr_rl_llm_medphi_sS`   | ~15h          |
| AfriSpeech LLM | MedGemma-4b-it       | 42, 33, 0 | `exp/asr_rl_llm_medgemma_sS` | ~15h          |
| VoxPopuli LLM  | Phi3-Legal-Finetuned | 42, 33, 0 | `exp/vox_rl_llm_philegal_sS` | ~15h          |


**Phase C total: ~~45h additional GPU time (~~$50 USD at $1.11/hr)**  
**Combined total (Phases A+B+C): ~207h continuous on 1× T4**

---

## Ongoing / Deferred

- **P2-H — Dataset quality:** Assess AfriSpeech label noise, VoxPopuli `take()` reproducibility, LibriSpeech 5k shuffle seed consistency before any full re-run

