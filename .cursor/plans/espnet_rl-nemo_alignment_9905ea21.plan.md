---
name: ESPnet RL-NeMo Alignment
overview: Align the ESPnet2 RL extension with the NeMo reward-augmented fine-tuning methodology, add multi-mode rewards and NeMo-style loss, prepare AfriSpeech-200 / VoxPopuli data, and provide a complete GCP VM setup and end-to-end run pipeline.
todos:
  - id: gcp-setup
    content: Write gcp_scripts/setup_gcp_vm.sh — CUDA/audio system deps, ESPnet source install, Python RL deps
    status: completed
  - id: data-prep-python
    content: Write egs2/afrispeech_rl/asr1/local/data_hf.py — HuggingFace download + kaldi-format conversion for AfriSpeech-200 clinical, VoxPopuli EN, LibriSpeech dev-clean
    status: completed
  - id: data-prep-shell
    content: Write egs2/afrispeech_rl/asr1/local/data.sh — calls data_hf.py, validates kaldi dirs
    status: completed
  - id: recipe-skeleton
    content: Write egs2/afrispeech_rl/asr1/{run.sh, db.sh, path.sh, cmd.sh} — recipe skeleton
    status: completed
  - id: recipe-configs
    content: Write conf/train_asr_sft.yaml (stage-1) and conf/train_asr_rl.yaml (stage-2) and conf/decode_asr.yaml
    status: completed
  - id: rl-model-reward-modes
    content: Add reward_mode, reward_loss_type, wwer, llm/Gemini+mock, domain terms, caching to rl_espnet_model.py
    status: completed
  - id: rl-model-forward
    content: "Update forward(): compute_reward flag + reward dispatch + loss formula selection + torch.no_grad() guard for penalty mode + FP32 cast for REINFORCE seq_log_probs"
    status: completed
  - id: rl-trainer-options
    content: Add all new options to RLTrainerOptions and add_arguments() in rl_trainer.py
    status: completed
  - id: rl-trainer-injection
    content: Update train_one_epoch() to inject compute_reward, reward_mode, reward_loss_type and other new options into batch dict
    status: completed
  - id: requirements
    content: Write requirements_rl.txt with pinned RL-specific Python deps
    status: completed
  - id: gcp-run-script
    content: Write gcp_scripts/run_espnet_rl_experiment.sh — end-to-end GCP orchestration (env check, data, SFT, RL, eval)
    status: completed
  - id: lora-integration
    content: Add --use_lora flag to run.sh, LoRA adapter_conf block to SFT and RL YAMLs, loralib to requirements_rl.txt
    status: completed
isProject: false
---

# ESPnet RL Extension — NeMo Methodology Alignment (GCP-Ready)

## Overview of all deliverables

```
espnet/
├── gcp_scripts/
│   ├── setup_gcp_vm.sh              # NEW: one-shot VM setup
│   └── run_espnet_rl_experiment.sh  # NEW: end-to-end experiment driver
├── requirements_rl.txt              # NEW: pinned RL deps (jiwer, google-generativeai, datasets, soundfile, loralib)
├── egs2/afrispeech_rl/asr1/
│   ├── run.sh                       # NEW: recipe (stages 1-8)
│   ├── db.sh / path.sh / cmd.sh     # NEW: recipe boilerplate
│   ├── local/
│   │   ├── data.sh                  # NEW: calls data_hf.py
│   │   └── data_hf.py              # NEW: HF → kaldi conversion
│   └── conf/
│       ├── train_asr_sft.yaml       # NEW: SFT stage config
│       ├── train_asr_rl.yaml        # NEW: RL stage config
│       └── decode_asr.yaml          # NEW: decode config
├── espnet2/asr/rl_espnet_model.py   # MODIFY: reward modes, loss formula, caching
└── espnet2/train/rl_trainer.py      # MODIFY: new options, step-interval injection
```

---

## A. GCP VM Setup

### `gcp_scripts/setup_gcp_vm.sh`

Target VM: `a2-highgpu-1g` (A100 40 GB) or `n1-standard-8 + V100` — CUDA 12.x, Ubuntu 22.04.

Steps:

1. System packages: `ffmpeg libsndfile1 sox bc git wget`
2. Verify CUDA: `nvcc --version` — exit if not found
3. Clone ESPnet from source and run `./tools/installers/install_espnet.sh`
4. `pip install -e ".[all]"` inside the ESPnet root
5. `pip install -r requirements_rl.txt`
6. `pip install espnet_model_zoo` (for pretrained model download)
7. Write `~/.espnet_env` with export lines for `ESPNET_ROOT`, `PATH`, `PYTHONPATH`

### `requirements_rl.txt`

```
jiwer>=3.0.0
google-generativeai>=0.5.0
datasets>=2.14.0
soundfile>=0.12.0
librosa>=0.10.0
huggingface_hub>=0.20.0
```

---

## B. Data Preparation

### Datasets (mirrors NeMo methodology)

- **AfriSpeech-200 clinical** (`tobiolatunji/afrispeech-200`, filter `domain=="clinical"`) — primary adaptation target
- **VoxPopuli EN** (`facebook/voxpopuli`, `en`, 10 000 utterances sampled with fixed seed) — secondary domain target
- **LibriSpeech dev-clean** (`openslr/librispeech_asr`) — catastrophic-forgetting evaluation only

### `egs2/afrispeech_rl/asr1/local/data_hf.py`

Python script that accepts `--dataset`, `--split`, `--output_dir`, `--max_samples`, `--seed` flags.

For each dataset:

- Load via `datasets.load_dataset(...)` (streaming where possible)
- Filter by domain (AfriSpeech)
- Write `{output_dir}/wav.scp` (utt_id → absolute `.wav` path), `text`, `utt2spk`, `spk2utt`
- Save audio files under `data/downloads/{dataset}/{split}/` as 16 kHz mono WAV

Output kaldi dirs created:


| Dir                          | Source                    | Role                       |
| ---------------------------- | ------------------------- | -------------------------- |
| `data/afrispeech_train`      | AfriSpeech clinical train | SFT + RL train             |
| `data/afrispeech_dev`        | AfriSpeech clinical dev   | Validation                 |
| `data/afrispeech_test`       | AfriSpeech clinical test  | Evaluation                 |
| `data/voxpopuli_train`       | VoxPopuli EN 10k          | Secondary train (combined) |
| `data/voxpopuli_dev`         | VoxPopuli EN dev          | Combined validation        |
| `data/librispeech_dev_clean` | LibriSpeech dev-clean     | Forgetting eval            |


### `egs2/afrispeech_rl/asr1/local/data.sh`

Bash wrapper:

1. Calls `python local/data_hf.py` for each dataset/split
2. Validates kaldi dirs with `utils/validate_data_dir.sh`
3. Combines AfriSpeech + VoxPopuli train dirs: `utils/combine_data.sh data/train_combined data/afrispeech_train data/voxpopuli_train`

---

## C. ESPnet Recipe

### `egs2/afrispeech_rl/asr1/run.sh` — 8-stage recipe

```
Stage 1  data prep (local/data.sh)
Stage 2  format wav.scp (via asr.sh --stop_stage 4)
Stage 3  BPE training (via asr.sh --stage 5 --stop_stage 5)
Stage 4  collect stats (via asr.sh --stage 9 --stop_stage 9)
Stage 5  SFT training   ← direct python call, rl_weight=0.0
Stage 6  RL training    ← direct python call, init from SFT ckpt
Stage 7  decode (via asr.sh --stage 11)
Stage 8  forgetting eval on librispeech_dev_clean
```

Stages 5–6 are called directly (not via `asr.sh`) because they use `RLTrainer` / `RLESPnetModel` which are not yet integrated into `asr.sh`'s model class dispatching.

Key variables in `run.sh`:

```bash
pretrained_model="espnet/librispeech_asr_train_asr_conformer_raw_bpe_..."
sft_config=conf/train_asr_sft.yaml
rl_config=conf/train_asr_rl.yaml
sft_expdir=exp/asr_sft
rl_expdir=exp/asr_rl
reward_mode=mwer         # override via --reward_mode
reward_loss_type=penalty # override via --reward_loss_type
gemini_api_key=""        # set via --gemini_api_key or env var GEMINI_API_KEY
```

### `conf/train_asr_sft.yaml` (Stage 1 — SFT)

- `model_class: espnet2.asr.rl_espnet_model.RLESPnetModel`
- `encoder: conformer`, `decoder: transformer`
- `init_param: [<pretrained_conformer.pth>]`
- `freeze_param: [encoder.encoders.0, ..., encoder.encoders.5]` (bottom 6 layers)
- `optim: adamw`, `lr: 1.0e-4`, `weight_decay: 1.0e-3`
- `scheduler: warmuplr`, `warmup_steps: 2000`
- `max_epoch: 5` (matches `SFT_EPOCHS=5` in NeMo plan)
- `rl_weight: 0.0` — no RL signal in SFT stage

### `conf/train_asr_rl.yaml` (Stage 2 — RL)

- Same architecture as SFT config
- `rl_weight: 0.05` (matches NeMo `REWARD_WEIGHT=0.05`)
- `reward_mode: mwer` (set in run.sh, override with `--reward_mode`)
- `reward_loss_type: penalty` (NeMo default; switch to `reinforce` for REINFORCE experiments)
- `reward_step_interval: 4` (matches NeMo `REWARD_STEP_INTERVAL=4`)
- `max_encoder_len_for_reward: 1500`
- `optim: adamw`, `lr: 1.0e-5` (matches NeMo `LEARNING_RATE_RL=1e-5`)
- `max_epoch: 3` (matches NeMo `RL_EPOCHS`)

---

## D. RL Extension Code Changes

### `espnet2/asr/rl_espnet_model.py` — modifications

**Constructor** — new keyword args (safe defaults preserve backward compatibility):

- `reward_mode: str = "mwer"` — `mwer | wwer | llm | all`
- `reward_loss_type: str = "reinforce"` — `reinforce` (default) or `penalty` (NeMo)
- `domain_terms: List[str] = []` — clinical vocabulary for wwer
- `domain_term_weight: float = 3.0` — cost multiplier for domain terms
- `max_encoder_len_for_reward: int = 1500` — frame threshold for reward skip
- `gemini_api_key: Optional[str] = None`
- `mock_llm: bool = False`

`**_cached_reward`** — `torch.Tensor` of shape `(1,)` initialized to `0.5`; updated on every compute step; reused on skip steps.

`**forward()` new args:**

```python
compute_reward: bool = True,   # injected by RLTrainer
reward_mode: str = "mwer",
reward_loss_type: str = "reinforce",
```

**Guard logic (reward caching + long-utterance skip):**

```python
if not compute_reward or encoder_out.shape[1] > self.max_encoder_len_for_reward:
    rewards = self._cached_reward.expand(batch_size)
else:
    rewards = self._dispatch_reward(hypotheses, references, device)
    self._cached_reward = rewards.mean().detach().unsqueeze(0)
```

**GPU fix 1 — `torch.no_grad()` guard for `penalty` mode:**

In `penalty` mode `seq_log_probs` are never used in the loss, but without this guard PyTorch retains the full CTC activation graph `(B, T, V)` unnecessarily. In `reinforce` mode the context must stay active so gradients flow through `seq_log_probs`.

```python
import contextlib
_no_grad = torch.no_grad() if reward_loss_type == "penalty" else contextlib.nullcontext()
with _no_grad:
    ctc_log_probs = self.ctc.log_softmax(encoder_out)
    token_seqs, seq_log_probs = _ctc_greedy_decode(
        ctc_log_probs, encoder_out_lens, self.blank_id
    )
```

**GPU fix 2 — FP32 cast for REINFORCE `seq_log_probs`:**

The entire `model(**batch)` call runs inside ESPnet's `autocast` context so `seq_log_probs` is FP16. It is a sum of T frame-level log-probs (e.g. T=300, avg ≈ −3 nats → sum ≈ −900). FP16 has ~1-bit precision at this magnitude, causing gradient underflow or NaN. Cast to FP32 before the loss.

**Loss formula dispatch:**

```python
if reward_loss_type == "penalty":
    penalty = 1.0 - rewards.mean()
    total_loss = ce_loss + rl_weight * penalty
    stats["penalty"] = penalty.detach()
else:  # reinforce — FP32 cast guards against AMP underflow
    pg_loss = -(rewards.detach() * seq_log_probs.float()).mean()
    total_loss = (1 - rl_weight) * ce_loss + rl_weight * pg_loss
    stats["pg_loss"] = pg_loss.detach()
stats["reward_mean"] = rewards.mean().detach()
```

**New private helpers:**

- `_compute_mwer(hyps, refs, device)` — existing logic, renamed
- `_compute_wwer(hyps, refs, device)` — `jiwer.process_words` alignment; words in `domain_terms` get edit cost × `domain_term_weight`; return `1 - weighted_rate` clamped to [0,1]
- `_compute_llm_reward(hyps, refs, device)` — calls `google.generativeai.GenerativeModel("gemini-1.5-flash")`; prompt asks for quality score 0–1; mock path: `mwer + clip(N(0,0.05),−0.2,0.2)`; fallback to mwer on any API error
- `_dispatch_reward(hyps, refs, device)` — routes by `reward_mode`; `all` = element-wise mean of mwer, wwer, llm tensors

### `espnet2/train/rl_trainer.py` — modifications

`**RLTrainerOptions`** — add:

```python
reward_mode: str = "mwer"
reward_loss_type: str = "reinforce"
reward_step_interval: int = 4
max_encoder_len_for_reward: int = 1500
domain_terms: List[str] = dataclasses.field(default_factory=list)
domain_term_weight: float = 3.0
gemini_api_key: Optional[str] = None
mock_llm: bool = False
```

`**add_arguments()**` — register all above as CLI args with matching types/defaults/help strings.

`**train_one_epoch()**` — augment batch dict:

```python
batch["rl_weight"] = options.rl_weight
batch["compute_reward"] = (iiter % options.reward_step_interval == 0)
batch["reward_mode"] = options.reward_mode
batch["reward_loss_type"] = options.reward_loss_type
batch["max_encoder_len_for_reward"] = options.max_encoder_len_for_reward
batch["domain_terms"] = options.domain_terms
batch["domain_term_weight"] = options.domain_term_weight
batch["gemini_api_key"] = options.gemini_api_key
batch["mock_llm"] = options.mock_llm
```

---

## E. GCP End-to-End Run Script

### `gcp_scripts/run_espnet_rl_experiment.sh`

Accepts flags: `--reward_mode`, `--reward_loss_type`, `--gemini_api_key`, `--ngpu`, `--stage`, `--stop_stage`.

Steps:

1. Source `~/.espnet_env`; assert `ESPNET_ROOT` set
2. Validate GPU is available (`nvidia-smi`)
3. `cd $ESPNET_ROOT/egs2/afrispeech_rl/asr1`
4. Run recipe stages 1–8 via `bash run.sh --ngpu $ngpu --reward_mode $reward_mode ...`
5. Print final WER for AfriSpeech test and LibriSpeech dev-clean (forgetting eval) from `exp/*/decode*/result.txt`

---

## F. LoRA Integration (Option 2 — recipe wiring)

ESPnet2 already has LoRA support via `loralib` and `espnet2/layers/create_adapter_fn.py`. The existing `AbsTask.build_model()` calls `create_adapter(model, ...)` post-build when `--use_adapter true` is set. No core architecture files need to change. The only gap is that the default `target_modules=["query"]` does not match Conformer's attribute names.

### `requirements_rl.txt` — add:

```
loralib>=0.1.2
```

### `conf/train_asr_sft.yaml` and `conf/train_asr_rl.yaml` — add LoRA block:

```yaml
# LoRA — activated only when run.sh is called with --use_lora true
# When use_lora=false (default), use_adapter remains false and these keys are ignored.
use_adapter: false          # overridden to true by run.sh --use_lora true
adapter: lora
adapter_conf:
  rank: 8
  alpha: 16
  target_modules:           # Conformer linear layer names (suffix-matched)
    - linear_q
    - linear_k
    - linear_v
    - linear_out
    - w_1
    - w_2
```

At rank 8 with `output_size=256` and 12 encoder blocks this yields ~~1.2 M trainable parameters (~~1.3% of the full model), consistent with NeMo's LoRA range (~1.7% at rank 32).

### `run.sh` — add flag and override:

```bash
use_lora=false   # set --use_lora true to activate

# Before asr_train calls for both SFT and RL stages:
lora_opts=""
if [ "${use_lora}" = true ]; then
    lora_opts="--use_adapter true --adapter lora"
fi

python -m espnet2.bin.asr_train ${lora_opts} ...
```

### Trainer checkpoint handling

`espnet2/train/trainer.py` already detects LoRA and saves with `lora.lora_state_dict()` when `save_strategy=adapter_only`. No trainer changes needed.

### Targets covered / not covered


| Layer type                                   | Covered by Option 2                                                     |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| Attention Q/K/V/out (`linear_q/k/v/out`)     | Yes                                                                     |
| Rel-pos projection (`linear_pos`)            | Yes (add to `target_modules` if needed)                                 |
| FFN up/down (`w_1`, `w_2`)                   | Yes                                                                     |
| Macaron FFN (`feed_forward_macaron.w_1/w_2`) | Yes                                                                     |
| Conv module (`nn.Conv1d`)                    | No — requires extending `create_adapter_fn.py` (Option 3, out of scope) |


---

## What diverges from NeMo and why

- **Loss formula**: REINFORCE remains the default (`--reward_loss_type reinforce`) since it provides true policy gradients. NeMo's penalty approach is selectable via `--reward_loss_type penalty`. Both modes are implemented for the paper comparison.
- **Data loader**: ESPnet uses kaldi-style wav.scp/text instead of HuggingFace JSON manifests. The `data_hf.py` script bridges this by downloading via HF datasets and writing kaldi-compatible files.
- **LoRA/PEFT**: ESPnet2 has existing `loralib`-based LoRA support; Option 2 wires it into the recipe via `--use_lora` flag and correct `target_modules` for Conformer linears — matching NeMo's optional `--use_lora` flag. Conv1d targets are not covered (out of scope; NeMo's LoRA also targets only linear projections).
- **Two-stage training (SFT → RL)**: Implemented via `--rl_weight 0.0` for SFT and `--init_param <sft_checkpoint>` for RL stage — no changes to the trainer loop needed.
- **Trainer framework**: ESPnet uses its own trainer loop; NeMo uses PyTorch Lightning. The reward-step-interval and long-utterance guards are implemented equivalently within ESPnet's `train_one_epoch`.

