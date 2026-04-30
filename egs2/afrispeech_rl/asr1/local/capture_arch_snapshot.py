#!/usr/bin/env python3
"""local/capture_arch_snapshot.py

Reads exp/pretrained/model_info.json, opens the asr_train_config YAML stored
in that JSON, and writes the first 60 lines to
espnet-docs/espnet_pretrained_config_snapshot.yaml.

Run from egs2/afrispeech_rl/asr1/ after Stage 5 has completed:
    python3 local/capture_arch_snapshot.py
"""

import json
import pathlib
import sys

MODEL_INFO = pathlib.Path("exp/pretrained/model_info.json")
OUT_FILE = pathlib.Path("espnet-docs/espnet_pretrained_config_snapshot.yaml")
SNAPSHOT_LINES = 60


def main() -> None:
    if not MODEL_INFO.exists():
        print(f"ERROR: {MODEL_INFO} not found. Run Stage 5 first.", file=sys.stderr)
        sys.exit(1)

    info = json.loads(MODEL_INFO.read_text())
    config_path = info.get("asr_train_config", "")

    if not config_path:
        print(
            "ERROR: 'asr_train_config' key missing from model_info.json.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        print(
            f"ERROR: asr_train_config path does not exist: {config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    lines = config_path.read_text().splitlines()
    snapshot = "\n".join(lines[:SNAPSHOT_LINES]) + "\n"

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(snapshot)

    print(f"Wrote first {min(SNAPSHOT_LINES, len(lines))} lines of:")
    print(f"  {config_path}")
    print(f"to:")
    print(f"  {OUT_FILE}")


if __name__ == "__main__":
    main()
