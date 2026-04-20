#!/usr/bin/env python3
"""Download and configure a pretrained ESPnet2 model for fine-tuning.

This script is called by run.sh Stage 5. It:
  1. Downloads the model via espnet_model_zoo.
  2. Extracts token_list → exp/pretrained/tokens.txt
  3. Copies bpe.model → data/token_list/bpe_unigram{N}/bpe.model
  4. Reads encoder_conf / decoder_conf / normalize / normalize_conf from the
     pretrained training config and patches conf/train_asr_sft.yaml and
     conf/train_asr_rl.yaml in-place so their architecture matches exactly.
  5. Writes all resolved paths to exp/pretrained/model_info.json.
  6. Validates that all required keys are non-empty and exits non-zero if not.

Usage (from the recipe root):
    python local/setup_pretrained.py \\
        --model  pyf98/librispeech_conformer \\
        --outdir exp/pretrained \\
        --sft_config  conf/train_asr_sft.yaml \\
        --rl_config   conf/train_asr_rl.yaml \\
        --nbpe   5000
"""

import argparse
import json
import logging
import pathlib
import shutil
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Keys in encoder_conf / decoder_conf that affect tensor shapes and MUST match
# the pretrained checkpoint exactly.
_SHAPE_KEYS_ENCODER = {"output_size", "attention_heads", "linear_units", "num_blocks"}
_SHAPE_KEYS_DECODER = {"attention_heads", "linear_units", "num_blocks"}


def _load_yaml(path: pathlib.Path) -> dict:
    import yaml  # pyyaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(data: dict, path: pathlib.Path) -> None:
    import yaml

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _patch_config(config_path: pathlib.Path, pretrained_cfg: dict, stats_file: str) -> None:
    """Overwrite architecture-critical fields in a training YAML to match the
    pretrained model.  Only shape-critical keys are touched; training
    hyper-parameters (lr, epochs, etc.) are left untouched.
    """
    our_cfg = _load_yaml(config_path)
    changed = []

    # --- normalize ---
    pt_norm = pretrained_cfg.get("normalize")
    pt_norm_conf = pretrained_cfg.get("normalize_conf") or {}
    if pt_norm:
        if our_cfg.get("normalize") != pt_norm:
            our_cfg["normalize"] = pt_norm
            changed.append(f"normalize={pt_norm}")
        # Always write the resolved stats_file (absolute path on this machine)
        our_cfg.setdefault("normalize_conf", {})
        if stats_file:
            our_cfg["normalize_conf"]["stats_file"] = stats_file
            changed.append(f"normalize_conf.stats_file={stats_file}")
    else:
        # Pretrained has no normalize layer — remove ours if present
        if "normalize" in our_cfg:
            del our_cfg["normalize"]
            our_cfg.pop("normalize_conf", None)
            changed.append("normalize=<removed>")

    # --- encoder_conf ---
    pt_enc = pretrained_cfg.get("encoder_conf") or {}
    our_enc = our_cfg.setdefault("encoder_conf", {})
    for key in _SHAPE_KEYS_ENCODER:
        if key in pt_enc and our_enc.get(key) != pt_enc[key]:
            our_enc[key] = pt_enc[key]
            changed.append(f"encoder_conf.{key}={pt_enc[key]}")

    # --- decoder_conf ---
    pt_dec = pretrained_cfg.get("decoder_conf") or {}
    our_dec = our_cfg.setdefault("decoder_conf", {})
    for key in _SHAPE_KEYS_DECODER:
        if key in pt_dec and our_dec.get(key) != pt_dec[key]:
            our_dec[key] = pt_dec[key]
            changed.append(f"decoder_conf.{key}={pt_dec[key]}")

    if changed:
        _dump_yaml(our_cfg, config_path)
        log.info("  patched %s: %s", config_path.name, ", ".join(changed))
    else:
        log.info("  %s already matches pretrained architecture — no changes", config_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model tag or local path")
    parser.add_argument("--outdir", default="exp/pretrained", help="Output directory for model_info.json and tokens.txt")
    parser.add_argument("--sft_config", default="conf/train_asr_sft.yaml")
    parser.add_argument("--rl_config", default="conf/train_asr_rl.yaml")
    parser.add_argument("--nbpe", type=int, default=5000, help="BPE vocabulary size")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download model
    # ------------------------------------------------------------------
    log.info("Downloading model: %s", args.model)
    from espnet_model_zoo.downloader import ModelDownloader

    d = ModelDownloader()
    info = d.download_and_unpack(args.model)
    log.info("  asr_train_config : %s", info.get("asr_train_config", ""))
    log.info("  asr_model_file   : %s", info.get("asr_model_file", ""))

    # ------------------------------------------------------------------
    # 2. Load pretrained training config
    # ------------------------------------------------------------------
    pretrained_config_path = pathlib.Path(info["asr_train_config"])
    pretrained_cfg = _load_yaml(pretrained_config_path)

    # ------------------------------------------------------------------
    # 3. Extract token_list → exp/pretrained/tokens.txt
    # ------------------------------------------------------------------
    tokens = pretrained_cfg.get("token_list", [])
    if isinstance(tokens, list) and tokens:
        tok_file = outdir / "tokens.txt"
        tok_file.write_text("\n".join(tokens) + "\n")
        info["token_list"] = str(tok_file.resolve())
        log.info("  token_list       : %s (%d tokens)", tok_file, len(tokens))
    else:
        log.warning("  token_list not found in pretrained config — training will likely fail")

    # ------------------------------------------------------------------
    # 4. Copy bpe.model → data/token_list/bpe_unigram{N}/bpe.model
    # ------------------------------------------------------------------
    # Locate the bpe.model referenced in the pretrained config
    pt_bpemodel = pretrained_cfg.get("bpemodel", "")
    if not pt_bpemodel or not pathlib.Path(pt_bpemodel).exists():
        # Fall back to searching the snapshot directory
        snapshot_dir = pretrained_config_path.parent.parent.parent  # …/snapshots/<hash>
        candidates = list(snapshot_dir.rglob("bpe.model"))
        if candidates:
            pt_bpemodel = str(candidates[0])
            log.info("  bpe.model found at: %s", pt_bpemodel)
        else:
            log.error("  bpe.model not found — cannot set up tokenizer")
            sys.exit(1)

    dst_bpe_dir = pathlib.Path(f"data/token_list/bpe_unigram{args.nbpe}")
    dst_bpe_dir.mkdir(parents=True, exist_ok=True)
    dst_bpe = dst_bpe_dir / "bpe.model"
    shutil.copy2(pt_bpemodel, dst_bpe)
    info["bpemodel"] = str(dst_bpe.resolve())
    log.info("  bpemodel         : %s", dst_bpe)

    # ------------------------------------------------------------------
    # 5. Resolve normalize stats_file (absolute path for this machine)
    # ------------------------------------------------------------------
    stats_file = ""
    pt_stats = (pretrained_cfg.get("normalize_conf") or {}).get("stats_file", "")
    if pt_stats and pathlib.Path(pt_stats).exists():
        stats_file = str(pathlib.Path(pt_stats).resolve())
    else:
        # Search for feats_stats.npz in the snapshot directory
        snapshot_dir = pretrained_config_path.parent.parent.parent
        candidates = list(snapshot_dir.rglob("feats_stats.npz"))
        if candidates:
            stats_file = str(candidates[0].resolve())
            log.info("  feats_stats.npz found at: %s", stats_file)
        else:
            log.warning("  feats_stats.npz not found — normalize layer may fail")
    info["normalize_stats_file"] = stats_file

    # ------------------------------------------------------------------
    # 6. Patch training configs to match pretrained architecture
    # ------------------------------------------------------------------
    log.info("Patching training configs to match pretrained architecture …")
    for cfg_path_str in (args.sft_config, args.rl_config):
        cfg_path = pathlib.Path(cfg_path_str)
        if cfg_path.exists():
            _patch_config(cfg_path, pretrained_cfg, stats_file)
        else:
            log.warning("  config not found, skipping patch: %s", cfg_path_str)

    # ------------------------------------------------------------------
    # 7. Write model_info.json
    # ------------------------------------------------------------------
    info_path = outdir / "model_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    log.info("model_info.json written to %s", info_path)

    # ------------------------------------------------------------------
    # 8. Validate required keys
    # ------------------------------------------------------------------
    required = {"asr_model_file": info.get("asr_model_file", ""),
                "token_list": info.get("token_list", ""),
                "bpemodel": info.get("bpemodel", "")}
    missing = [k for k, v in required.items() if not v]
    if missing:
        log.error("setup_pretrained: missing required fields after setup: %s", missing)
        sys.exit(1)

    log.info("setup_pretrained: all required fields present — Stage 5 OK")


if __name__ == "__main__":
    main()
