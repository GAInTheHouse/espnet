---
name: ESPnet RL — Local Changes Plan
overview: All local machine work for the ESPnet2 RL experiment suite. Write and commit every code change and script before touching the VM. Covers Phase A (run.sh fixes, MWER config, 5 helper scripts, master orchestration script, git sync) and Phase C LOCAL (LLM reward backend, 3 domain configs, model download script, master script extension). No training is run here — only code is written and pushed.
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
    content: "LOCAL: Write scripts/run_all_experiments.sh — master orchestration script (Steps 1–19) that chains every bash run.sh call in the correct order with logging"
    status: pending
  - id: A9-sync-vm
    content: "LOCAL: git push all changes; then confirm git pull on VM and run smoke test to verify run.sh changes work"
    status: pending
  - id: C1-llm-backend
    content: "LOCAL: Modify espnet2/asr/rl_espnet_model.py — add llm_reward_model arg, local HuggingFace 4-bit quantized inference path (AutoModelForCausalLM + BitsAndBytesConfig), domain-tagged scoring prompt; keep Gemini path as fallback"
    status: pending
  - id: C2-llm-configs
    content: "LOCAL: Create conf/train_asr_rl_llm_medphi.yaml, conf/train_asr_rl_llm_medgemma.yaml, conf/train_asr_rl_llm_philegal.yaml — each sets reward_mode: llm and llm_reward_model: <hf-id>"
    status: pending
  - id: C3-download-script
    content: "LOCAL: Write local/download_reward_llms.py — uses huggingface_hub.snapshot_download to pre-pull MedPhi-3.8b, google/medgemma-4b-it, Phi3-Legal-Finetuned"
    status: pending
  - id: C4-extend-master
    content: "LOCAL: Extend scripts/run_all_experiments.sh with Steps 20–28 — AfriSpeech×MedPhi (seeds 42/33/0), AfriSpeech×MedGemma (seeds 42/33/0), VoxPopuli×Phi3Legal (seeds 42/33/0)"
    status: pending
  - id: D1-dataset-quality
    content: "ONGOING: P2-H review of AfriSpeech/VoxPopuli/LibriSpeech subset quality and label noise"
    status: pending
isProject: false
---

# ESPnet RL — Local Changes Plan

> **Companion plan:** See [espnet_rl_vm_execution.plan.md](espnet_rl_vm_execution.plan.md) for all VM execution steps.  
> This plan must be completed and pushed before starting VM execution.

---

## Overview

Everything here runs on the local machine. The end state is a single `git push` that delivers all code the VM needs. No experiments are run locally.

**Phase A** handles the core infrastructure: `run.sh` fixes, MWER config, 5 helper scripts, and the master orchestration script covering the full WWER/MWER/LoRA/VoxPopuli experiment matrix (Steps 1–19).

**Phase C LOCAL** adds the LLM reward mode: backend code changes, 3 domain-specific YAML configs, a model download helper, and the master script extension (Steps 20–28).

---

## Phase A: Core Infrastructure

All work is in [egs2/afrispeech_rl/asr1/](egs2/afrispeech_rl/asr1/) unless otherwise noted.

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

Add `asr_stats_dir` as a top-level variable (default unchanged for existing runs) so VoxPopuli standalone can point to a separate shape-file directory:

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

Shell script that, when run on the VM from `egs2/afrispeech_rl/asr1/` inside the `espnet_rl` conda environment, captures all package versions and hardware state into `espnet_rl_pip_freeze.txt` and `espnet_rl_env_summary.txt`:

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

Run once after all training is complete (triggered from the VM plan).

### A.8 — New script: `scripts/run_all_experiments.sh` (Steps 1–19)

**File:** [egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh](egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh)

Master orchestration script that chains every `bash run.sh ...` call in the correct order. Key design:

- `set -euo pipefail` so any failure stops the chain
- Each step is wrapped in a `log_step "description"` helper that prints a timestamped header and footer
- Steps are numbered and can be fast-forwarded via `START_STEP` env var: `START_STEP=4 bash scripts/run_all_experiments.sh`
- All `bash run.sh` flags are spelled out explicitly (no defaults assumed)

**Steps 1–19 encoded in the initial script:**

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

Steps 20–28 are added in Phase C (C.4 below).

### A.9 — Sync to VM

After all Phase A changes (A.1–A.8) are committed:

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

After Phase C local work (C.1–C.4) is also committed, do a second push:

```bash
git add \
  espnet2/asr/rl_espnet_model.py \
  egs2/afrispeech_rl/asr1/conf/train_asr_rl_llm_medphi.yaml \
  egs2/afrispeech_rl/asr1/conf/train_asr_rl_llm_medgemma.yaml \
  egs2/afrispeech_rl/asr1/conf/train_asr_rl_llm_philegal.yaml \
  egs2/afrispeech_rl/asr1/local/download_reward_llms.py \
  egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh
git commit -m "feat: LLM reward mode — HuggingFace local inference, domain configs, Steps 20-28"
git push
```

---

## Phase C: LLM Reward Mode — Local Work

Domain-specific LLMs loaded locally from HuggingFace replace the deferred Gemini API approach. The model used depends on the dataset domain:


| Dataset    | Domain             | LLM reward models                                           |
| ---------- | ------------------ | ----------------------------------------------------------- |
| AfriSpeech | Clinical / medical | MedPhi-3.8b **and** MedGemma-4b-it (two separate ablations) |
| VoxPopuli  | Parliamentary      | Phi3-Legal-Finetuned                                        |


### C.1 — Modify LLM reward backend in `rl_espnet_model.py`

**File:** `espnet2/asr/rl_espnet_model.py`

The existing LLM reward code calls the Gemini API via `GEMINI_API_KEY`. Add a local HuggingFace inference path alongside it:

1. Add `llm_reward_model: str = ""` to the model's `__init__` signature (populated from the YAML config)
2. When `llm_reward_model` is non-empty, load with 4-bit quantization to fit on T4 VRAM:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_reward_model)
self.llm_model = AutoModelForCausalLM.from_pretrained(
    llm_reward_model, quantization_config=bnb_cfg, device_map="auto"
)
```

1. The reward scoring prompt should include a domain tag derived from the model name (`medical` for MedPhi/MedGemma, `legal` for Phi3-Legal)
2. Keep the existing Gemini path as a fallback (env var check) — no existing code breaks

### C.2 — Three new YAML config files

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


> `google/medgemma-4b-it` is confirmed. Verify the exact HuggingFace repo slugs for MedPhi-3.8b and Phi3-Legal-Finetuned before writing the YAMLs.

### C.3 — New script: `local/download_reward_llms.py`

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

Downloads ~~5 GB per model (~~15 GB total). MedGemma requires a HuggingFace token — verify gating status for the other two models as well.

### C.4 — Extend `scripts/run_all_experiments.sh` (Steps 20–28)

**File:** [egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh](egs2/afrispeech_rl/asr1/scripts/run_all_experiments.sh)

Append 9 steps to the master script. Pattern for each seed `S`:

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

All 9 LLM steps use Stage 7 only (SFT checkpoints reused from earlier runs). Seed=42 AfriSpeech SFT uses `exp/asr_sft` (no suffix — existing checkpoint).

---

## Ongoing / Deferred

- **P2-H — Dataset quality:** Assess AfriSpeech label noise, VoxPopuli `take()` reproducibility, LibriSpeech 5k shuffle seed consistency before any full re-run

