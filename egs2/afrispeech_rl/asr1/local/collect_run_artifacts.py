#!/usr/bin/env python3
"""local/collect_run_artifacts.py

Scans all exp/*/train.log files, parses ISO-8601 timestamps from each line,
computes per-epoch and total elapsed training time, estimates GPU-hours and
USD cost, then writes run_summary.json alongside each train.log.

Run once after all training is complete (or at any checkpoint):
    python3 local/collect_run_artifacts.py

Cost model: n1-standard-16 + 1x NVIDIA T4 on GCP, 2026 on-demand rate.
"""

import glob
import json
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import Optional

COST_PER_HOUR_USD = 1.11
INSTANCE_TYPE = "n1-standard-16 + 1x T4"

# Matches lines that start with the date prefix written by log() in run.sh:
# 2024-05-01T12:34:56
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def _parse_seed(expdir: pathlib.Path) -> Optional[int]:
    """Extract seed from directory name, e.g. exp/asr_rl_mwer_s42 → 42."""
    m = re.search(r"_s(\d+)$", expdir.name)
    return int(m.group(1)) if m else None


def _parse_log(log_path: pathlib.Path) -> dict:
    """Return timing statistics parsed from a train.log."""
    timestamps = []
    with log_path.open(errors="replace") as fh:
        for line in fh:
            m = TS_RE.match(line)
            if m:
                try:
                    ts = datetime.fromisoformat(m.group(1)).replace(
                        tzinfo=timezone.utc
                    )
                    timestamps.append(ts)
                except ValueError:
                    pass

    if len(timestamps) < 2:
        return {"epochs": 0, "total_train_time_s": 0}

    total_s = (timestamps[-1] - timestamps[0]).total_seconds()

    # Count epoch boundaries: lines containing "epoch results"
    epoch_count = 0
    with log_path.open(errors="replace") as fh:
        for line in fh:
            if "epoch results" in line.lower() or re.search(
                r"\bepoch\s+\d+\b.*\bfinish", line.lower()
            ):
                epoch_count += 1

    # Fallback: count "saving" events as a proxy for epoch count
    if epoch_count == 0:
        with log_path.open(errors="replace") as fh:
            for line in fh:
                if "saving checkpoint" in line.lower():
                    epoch_count += 1

    return {
        "epochs": max(epoch_count, 1),
        "total_train_time_s": round(total_s, 1),
    }


def main() -> None:
    log_files = sorted(glob.glob("exp/*/train.log"))
    if not log_files:
        print("No exp/*/train.log files found.", file=sys.stderr)
        sys.exit(0)

    summaries = []
    for log_str in log_files:
        log_path = pathlib.Path(log_str)
        expdir = log_path.parent

        timing = _parse_log(log_path)
        total_s = timing["total_train_time_s"]
        gpu_hours = round(total_s / 3600, 2)
        cost_usd = round(gpu_hours * COST_PER_HOUR_USD, 2)

        summary = {
            "expdir": str(expdir),
            "epochs": timing["epochs"],
            "total_train_time_s": total_s,
            "gpu_hours": gpu_hours,
            "instance_type": INSTANCE_TYPE,
            "estimated_cost_usd": cost_usd,
            "seed": _parse_seed(expdir),
        }

        out_path = expdir / "run_summary.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(
            f"{expdir.name:40s}  {timing['epochs']} epochs  "
            f"{gpu_hours:.2f} GPU-hrs  ${cost_usd:.2f}"
        )

    total_cost = sum(s["estimated_cost_usd"] for s in summaries)
    total_hrs = sum(s["gpu_hours"] for s in summaries)
    print(f"\nTotal: {total_hrs:.2f} GPU-hrs  ${total_cost:.2f}  ({len(summaries)} runs)")


if __name__ == "__main__":
    main()
