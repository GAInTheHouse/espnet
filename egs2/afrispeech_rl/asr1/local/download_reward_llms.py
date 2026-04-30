#!/usr/bin/env python3
"""local/download_reward_llms.py

Pre-downloads all LLM reward models from HuggingFace to the local cache so
that training nodes can load them without internet access.

Models (~5 GB each, ~15 GB total):
  - microsoft/MediPhi          — AfriSpeech clinical (medical domain)
  - google/medgemma-4b-it      — AfriSpeech clinical (medical domain, ablation)
  - sairamn/Phi3-Legal-Finetuned — VoxPopuli parliamentary (legal domain)

Gating notes:
  - google/medgemma-4b-it requires accepting the model licence on HuggingFace.
    Log in with `huggingface-cli login` before running this script.
  - microsoft/MediPhi and sairamn/Phi3-Legal-Finetuned appear ungated as of
    2026-04, but verify at https://huggingface.co/<repo> before running.

Usage:
    python3 local/download_reward_llms.py
    HF_TOKEN=<token> python3 local/download_reward_llms.py  # explicit token
"""

import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print(
        "ERROR: huggingface_hub is not installed.\n"
        "       Install with: pip install huggingface-hub",
        file=sys.stderr,
    )
    sys.exit(1)

REWARD_MODELS = [
    "microsoft/MediPhi",           # AfriSpeech clinical — medical domain
    "google/medgemma-4b-it",       # AfriSpeech clinical (ablation) — requires HF login
    "sairamn/Phi3-Legal-Finetuned", # VoxPopuli parliamentary — legal domain
]

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def main() -> None:
    print(f"Downloading {len(REWARD_MODELS)} reward models ...")
    if HF_TOKEN:
        print("  Using HF_TOKEN from environment.")
    else:
        print(
            "  No HF_TOKEN set. google/medgemma-4b-it may fail if not logged in.\n"
            "  Run `huggingface-cli login` or set HF_TOKEN=<token>."
        )

    failed = []
    for repo_id in REWARD_MODELS:
        print(f"\n{'='*60}")
        print(f"  Downloading: {repo_id}")
        print(f"{'='*60}")
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                token=HF_TOKEN,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            print(f"  Cached at: {local_dir}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed.append(repo_id)

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(REWARD_MODELS)}):")
        for r in failed:
            print(f"  - {r}")
        sys.exit(1)
    else:
        print(f"All {len(REWARD_MODELS)} models downloaded successfully.")


if __name__ == "__main__":
    main()
