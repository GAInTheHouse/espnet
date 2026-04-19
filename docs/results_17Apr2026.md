# Results summary — 17 April 2026

This note compiles **AfriSpeech clinical** (and related) metrics from two `vm_results` folders. Tables below use HTML with **inline** wrap-friendly styles (works in Cursor preview and many viewers; pipe tables stay on one long line).


| Role                           | `vm_results` folder                                             | `run_id`                                   | Notes                                                                                                                                                                |
| ------------------------------ | --------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SFT + full pipeline config** | `vm_results/sft_working_afrispeech_clinical_seed42_1776207077/` | `afrispeech_clinical_seed42_1776207077`    | Complete `*_results.json`: `config`, zero-shot, SFT, Libri after SFT, **collapsed RL** (invalid), test splits, bootstrap `p`-value.                                  |
| **Healthy RL-only artifact**   | `vm_results/afrispeech_clinical_seed42_rl_1776462369/`          | `afrispeech_clinical_seed42_rl_1776462369` | JSON has **RL test metrics only** (no embedded `config`, no SFT block). Assumes **same training recipe** as config in `1776207077` unless the VM job overrode flags. |


**Paper-facing caveat:** ideal reporting uses **one** `*_results.json` from a single end-to-end run (SFT → RL → eval). Here, **SFT numbers come from `1776207077`** and **AfriSpeech RL numbers come from `1776462369`**. Treat LibriSpeech-after-RL, `test_rl`, and bootstrap p-value in `1776207077` as tied to the **failed** RL in that file, not to `1776462369`.

---

## 1) Hyperparameters and run settings (from `1776207077` `config`)


| Field                                | Value                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| Base model                           | `stt_en_conformer_ctc_medium`                                                         |
| Dataset                              | `afrispeech_clinical`                                                                 |
| Train / val / test caps              | `TRAIN_SAMPLES`, `VAL_SAMPLES`, `TEST_SAMPLES` = `null` (no artificial cap in config) |
| VoxPopuli train subset cap           | `10000`                                                                               |
| LibriSpeech train cap                | `5000`                                                                                |
| Batch size                           | `16`                                                                                  |
| LR SFT / LR RL                       | `1e-4` / `1e-5`                                                                       |
| SFT epochs / RL epochs               | `5` / `2`                                                                             |
| Reward mode / weight / step interval | `wwer` / `0.02` / every `4` steps                                                     |
| Max audio seconds for reward         | `25.0`                                                                                |
| Sample rate                          | `16000`                                                                               |
| RL objective / grad clip / precision | `reweight_ctc` / `1.0` / `FORCE_FP32: true`                                           |
| Seed / LoRA / smoke test             | `42` / `false` / `false`                                                              |
| Normalize text (eval / SER)          | `NORMALIZE_TEXT: false`                                                               |
| Tokenizer UNK guard                  | `true`                                                                                |
| Debug reward / sample dump           | `false` / `true` (every `200` steps, `10` samples)                                    |
| Eval toggles                         | Zero-shot val, Libri forgetting, final test eval: all `true`                          |
| Bootstrap iters (paired WER `p`)     | `1000`                                                                                |
| Gemini / LLM reward                  | `GEMINI_MODEL: gemini-1.5-flash`, `USE_MOCK_LLM: true`                                |
| Domain term weight (WWER)            | `3.0`                                                                                 |
| Clinical domain terms                | 37-token list in JSON (e.g. `patient`, `hypertension`, `malaria`, …)                  |
| Parliamentary domain terms           | 13-token list in JSON (e.g. `parliament`, `directive`, …)                             |


**Data loader (training manifests)** — from `nemo/gcp_scripts/nemo_afrispeech_training.py` → `build_data_config`: `max_duration` **20.0** s, `min_duration` **0.5** s, `trim_silence` **false**, `shuffle` **true**, `num_workers` **4** when not smoke test.

---

## 2) AfriSpeech clinical **validation** metrics (`n_utterances` = **1813**)

Transposed so columns stay narrow; values wrap inside cells.


| Metric               | Zero-shot (base, val) | After SFT `1776207077` | After RL `1776462369` | After RL `1776207077` (collapsed) |
| -------------------- | --------------------- | ---------------------- | --------------------- | --------------------------------- |
| WER (%)              | 57.88                 | **45.95**              | **45.92**             | 100.0                             |
| CER (%)              | 25.87                 | **14.19**              | **14.23**             | 100.0                             |
| SER (%)              | 100.0                 | 100.0                  | 100.0                 | 100.0                             |
| EWER (%)             | 19.97                 | 20.27                  | 18.92                 | 100.0                             |
| Domain P / R / F1    | 0.884 / 0.857 / 0.858 | 0.881 / 0.867 / 0.861  | 0.899 / 0.885 / 0.879 | 0 / 0 / 0                         |
| Empty hyp frac       | —                     | 0.0                    | 0.0                   | 0.0                               |
| Degenerate hyp frac  | —                     | **0.00055**            | **0.0**               | **1.0**                           |
| Mean hyp len (chars) | —                     | **93.84**              | **93.97**             | 1.0                               |
| Train time (s)       | —                     | **12243.1**            | **4502.3**            | 5007.1                            |


**RL reward summary (`1776462369`):** `reward_mean` ≈ **0.609**, `reward_std` ≈ **0.076** (batch-level rewards over training; full series in JSON `reward_trajectory`).

---

## 3) Other splits in `1776207077` only (same collapsed RL caveat for RL rows)


| Eval                | Split      | WER (%) | CER (%) | `n_utterances` |
| ------------------- | ---------- | ------- | ------- | -------------- |
| LibriSpeech         | After SFT  | 10.44   | 4.09    | 2694           |
| LibriSpeech         | After RL   | 100.0   | 100.0   | 2694           |
| AfriSpeech clinical | `test_sft` | 50.68   | 17.13   | 3508           |
| AfriSpeech clinical | `test_rl`  | 100.0   | 100.0   | 3508           |



| Stat                                                 | Value                                                                        |
| ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| `paired_bootstrap_pval_sft_vs_rl_wer` (`1776207077`) | `0.51` — **not meaningful** because RL hypotheses in that run are collapsed. |


---

## 4) Training curves (Lightning CSVs)

### SFT (`1776207077` — `*_sft_epoch_metrics.csv`)

End of epoch 4 (train_end): `val_loss` ≈ **30.85**, `val_wer` ≈ **0.255** (NeMo/Lightning **fraction**, not percent).

### RL — healthy run (`1776462369` — `*_rl_epoch_metrics.csv`)


| Epoch | train_end `val_loss` | train_end `val_wer` (fraction) |
| ----- | -------------------- | ------------------------------ |
| 0     | 30.74                | 0.252                          |
| 1     | 30.81                | 0.252                          |


### RL — collapsed run (`1776207077` — `*_rl_epoch_metrics.csv`)


| Epoch | train_end `val_loss` | train_end `val_wer` (fraction) |
| ----- | -------------------- | ------------------------------ |
| 0     | **nan**              | 0.939                          |
| 1     | **nan**              | 0.939                          |


---

## 5) How these numbers were computed (brief)

All reported WER/CER/SER/EWER/domain metrics come from `evaluate_manifest_bundle` in `nemo/gcp_scripts/nemo_afrispeech_training.py`:

1. **Transcription:** `model.transcribe(audio_paths, batch_size=CFG.BATCH_SIZE)` with model in **eval** + `torch.no_grad()`.
2. **WER / CER:** `jiwer` via helpers `compute_wer_jiwer` / `compute_cer_jiwer`; stored as **percentage** (`× 100`).
3. **SER:** `sentence_error_rate` — fraction of utterances where `**_normalize_text(ref) != _normalize_text(hyp)`**, with `_normalize_text` = lowercase, strip, collapse whitespace to single spaces. **100% SER** means **no** exact full-string match on that normalization (common under case/punctuation/tokenization drift even when WER is moderate).
4. **EWER:** `entity_wer_from_text` — per utterance, keep only **domain-vocabulary** words that appear in the reference; compute WER on that substring; average over utterances that have ≥1 such reference token; report **mean × 100**. Utterances with no domain tokens in the ref are skipped.
5. **Domain precision / recall / F1:** `aggregate_f1` of per-utterance token sets from `domain_term_precision_recall_f1` (precision uses domain tokens present in hypothesis vs reference domain tokens in ref).
6. **Diagnostics:** `_empty_hyp_frac`, `_degenerate_hyp_frac`, `_mean_hyp_len_chars` — script-side sanity stats on hypothesis strings.

---

## 6) Brief analysis / observations

1. **SFT vs healthy RL (AfriSpeech val):** WER **45.95% → 45.92%** (~~0.03 pp), CER **14.19% → 14.23%** (~~0.04 pp). This is effectively **flat** on aggregate WER/CER; small gains may appear in **EWER** (20.27% → 18.92%) and domain F1 (0.861 → 0.879), which weight clinical vocabulary more explicitly than plain WER.
2. **SER = 100%** for zero-shot, SFT, and healthy RL is **consistent with strict exact-match SER** under light normalization — it does **not** by itself indicate collapse when WER/CER are healthy.
3. **Run `1776207077` RL is invalid** for reporting (`wer=cer=100`, `_degenerate_hyp_frac=1`, `val_loss=nan` in RL epoch CSV). The **LibriSpeech-after-RL** and `**test_rl`** rows in that JSON reflect the same broken checkpoint.
4. **Run `1776462369` RL is valid** for reporting on AfriSpeech val: finite metrics, **zero** degenerate fraction, normal mean hypothesis length, non-NaN RL epoch metrics in its CSV.
5. **Provenance for the paper:** either (a) re-run a **single** pipeline that writes one `*_results.json` including SFT + **fixed** RL + Libri + test + bootstrap, or (b) keep this split documentation and **do not** mix `1776207077`’s `librispeech_after_rl` / `test_rl` with `1776462369`’s AfriSpeech RL.

---

## 7) On-disk artifacts (local)

- SFT / full pipeline: `vm_results/sft_working_afrispeech_clinical_seed42_1776207077/`
- RL (healthy): `vm_results/afrispeech_clinical_seed42_rl_1776462369/`

Primary JSON paths:

- `vm_results/sft_working_afrispeech_clinical_seed42_1776207077/afrispeech_clinical_seed42_1776207077_results.json`
- `vm_results/afrispeech_clinical_seed42_rl_1776462369/afrispeech_clinical_seed42_rl_1776462369_results.json`

