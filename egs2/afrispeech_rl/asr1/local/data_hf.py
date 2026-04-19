#!/usr/bin/env python3
"""Download datasets from HuggingFace and write ESPnet kaldi-style manifests.

Supported datasets
------------------
afrispeech      tobiolatunji/afrispeech-200  (filter domain=="clinical")
voxpopuli       facebook/voxpopuli           (language="en", cap --max_samples)
librispeech     openslr/librispeech_asr      (split="validation.clean")

Output per dataset/split
------------------------
<output_dir>/
    wav.scp       utt_id  /abs/path/to/audio.wav
    text          utt_id  TRANSCRIPTION
    utt2spk       utt_id  speaker_id
    spk2utt       speaker_id  utt_id1 utt_id2 ...

Usage
-----
python local/data_hf.py \\
    --dataset afrispeech --split train \\
    --output_dir data/afrispeech_train \\
    --audio_dir data/downloads/afrispeech/train \\
    --max_samples -1 --seed 42

python local/data_hf.py \\
    --dataset voxpopuli --split train \\
    --output_dir data/voxpopuli_train \\
    --audio_dir data/downloads/voxpopuli/train \\
    --max_samples 10000 --seed 42

python local/data_hf.py \\
    --dataset librispeech --split validation.clean \\
    --output_dir data/librispeech_dev_clean \\
    --audio_dir data/downloads/librispeech/dev_clean
"""

import argparse
import collections
import logging
import os
import pathlib
import random
import sys
from typing import Iterator, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _save_wav(audio_array, sampling_rate: int, out_path: pathlib.Path) -> None:
    """Write a numpy or list audio array to a 16 kHz mono WAV file."""
    import numpy as np
    import soundfile as sf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(audio_array, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)

    if sampling_rate != 16000:
        import librosa

        arr = librosa.resample(arr, orig_sr=sampling_rate, target_sr=16000)

    sf.write(str(out_path), arr, 16000, subtype="PCM_16")


def _duration_ok(
    audio_array,
    sampling_rate: int,
    min_duration: float,
    max_duration: float,
) -> bool:
    """Return True if audio duration is within [min_duration, max_duration] seconds.

    Mirrors NeMo data loader's min_duration=0.5 / max_duration=20.0 filter.
    """
    import numpy as np

    n_samples = len(np.asarray(audio_array))
    duration = n_samples / max(sampling_rate, 1)
    return min_duration <= duration <= max_duration


# ---------------------------------------------------------------------------
# Dataset-specific iterators  →  yield (utt_id, spk_id, wav_path, text)
# ---------------------------------------------------------------------------


def _iter_afrispeech(
    split: str,
    audio_dir: pathlib.Path,
    max_samples: int,
    seed: int,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
) -> Iterator[Tuple[str, str, pathlib.Path, str]]:
    """Iterate AfriSpeech-200, filtered to domain=='clinical'.

    Applies duration filter: 0.5s ≤ duration ≤ 20.0s (NeMo defaults).
    """
    import datasets as hf_datasets

    log.info("Loading tobiolatunji/afrispeech-200 split=%s ...", split)
    ds = hf_datasets.load_dataset(
        "tobiolatunji/afrispeech-200",
        split=split,
        trust_remote_code=True,
    )

    # Filter to clinical domain
    ds = ds.filter(lambda ex: ex.get("domain", "").lower() == "clinical")
    log.info("  After clinical filter: %d utterances", len(ds))

    indices = list(range(len(ds)))
    if max_samples > 0 and len(indices) > max_samples:
        rng = random.Random(seed)
        indices = rng.sample(indices, max_samples)
        log.info("  Subsampled to %d utterances (seed=%d)", max_samples, seed)

    skipped = 0
    for idx in indices:
        ex = ds[idx]
        audio = ex["audio"]
        if not _duration_ok(audio["array"], audio["sampling_rate"], min_duration, max_duration):
            skipped += 1
            continue

        spk = str(ex.get("accent", ex.get("speaker_id", f"spk{idx:06d}"))).replace(
            " ", "_"
        )
        utt_id = f"afrispeech_{split}_{idx:07d}"

        wav_path = audio_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            _save_wav(audio["array"], audio["sampling_rate"], wav_path)

        transcript = str(ex.get("transcript", ex.get("text", ""))).strip().upper()
        yield utt_id, spk, wav_path.resolve(), transcript

    if skipped:
        log.info("  Duration filter skipped %d utterances (outside %.1fs–%.1fs)", skipped, min_duration, max_duration)


def _iter_voxpopuli(
    split: str,
    audio_dir: pathlib.Path,
    max_samples: int,
    seed: int,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
) -> Iterator[Tuple[str, str, pathlib.Path, str]]:
    """Iterate VoxPopuli EN.  Applies duration filter (NeMo: 0.5s–20.0s)."""
    import datasets as hf_datasets

    log.info("Loading facebook/voxpopuli (en) split=%s ...", split)
    ds = hf_datasets.load_dataset(
        "facebook/voxpopuli",
        "en",
        split=split,
        streaming=False,
        trust_remote_code=True,
    )

    indices = list(range(len(ds)))
    if max_samples > 0 and len(indices) > max_samples:
        rng = random.Random(seed)
        indices = rng.sample(indices, max_samples)
        log.info("  Subsampled to %d utterances (seed=%d)", max_samples, seed)

    skipped = 0
    for rank, idx in enumerate(indices):
        ex = ds[idx]
        audio = ex["audio"]
        if not _duration_ok(audio["array"], audio["sampling_rate"], min_duration, max_duration):
            skipped += 1
            continue

        spk = str(ex.get("speaker_id", f"spk{idx:06d}"))
        utt_id = f"voxpopuli_{split}_{rank:07d}"

        wav_path = audio_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            _save_wav(audio["array"], audio["sampling_rate"], wav_path)

        transcript = str(
            ex.get("normalized_text", ex.get("raw_text", ""))
        ).strip().upper()
        yield utt_id, spk, wav_path.resolve(), transcript

    if skipped:
        log.info("  Duration filter skipped %d utterances (outside %.1fs–%.1fs)", skipped, min_duration, max_duration)


def _iter_librispeech(
    split: str,
    audio_dir: pathlib.Path,
    max_samples: int,
    seed: int,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
) -> Iterator[Tuple[str, str, pathlib.Path, str]]:
    """Iterate LibriSpeech via HuggingFace openslr/librispeech_asr.

    Applies duration filter (NeMo: 0.5s–20.0s).
    """
    import datasets as hf_datasets

    log.info("Loading openslr/librispeech_asr split=%s ...", split)
    ds = hf_datasets.load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split=split,
        trust_remote_code=True,
    )

    indices = list(range(len(ds)))
    if max_samples > 0 and len(indices) > max_samples:
        rng = random.Random(seed)
        indices = rng.sample(indices, max_samples)
        log.info("  Subsampled to %d utterances (seed=%d)", max_samples, seed)

    skipped = 0
    for idx in indices:
        ex = ds[idx]
        audio = ex["audio"]
        if not _duration_ok(audio["array"], audio["sampling_rate"], min_duration, max_duration):
            skipped += 1
            continue

        spk = str(ex.get("speaker_id", f"spk{idx:06d}"))
        chapter = str(ex.get("chapter_id", "0"))
        utt_id = f"{spk}-{chapter}-{idx:07d}"

        wav_path = audio_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            _save_wav(audio["array"], audio["sampling_rate"], wav_path)

        transcript = str(ex.get("text", "")).strip().upper()
        yield utt_id, spk, wav_path.resolve(), transcript

    if skipped:
        log.info("  Duration filter skipped %d utterances (outside %.1fs–%.1fs)", skipped, min_duration, max_duration)


# ---------------------------------------------------------------------------
# Kaldi directory writer
# ---------------------------------------------------------------------------


def write_kaldi_dir(
    iterator: Iterator[Tuple[str, str, pathlib.Path, str]],
    output_dir: pathlib.Path,
) -> int:
    """Consume iterator and write wav.scp / text / utt2spk / spk2utt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_lines = []
    text_lines = []
    utt2spk_lines = []
    spk2utts: dict = collections.defaultdict(list)

    count = 0
    for utt_id, spk, wav_path, transcript in iterator:
        if not transcript:
            log.debug("Skipping %s: empty transcript", utt_id)
            continue
        wav_lines.append(f"{utt_id} {wav_path}\n")
        text_lines.append(f"{utt_id} {transcript}\n")
        utt2spk_lines.append(f"{utt_id} {spk}\n")
        spk2utts[spk].append(utt_id)
        count += 1
        if count % 1000 == 0:
            log.info("  Processed %d utterances ...", count)

    (output_dir / "wav.scp").write_text("".join(wav_lines))
    (output_dir / "text").write_text("".join(text_lines))
    (output_dir / "utt2spk").write_text("".join(utt2spk_lines))

    spk2utt_lines = [
        f"{spk} {' '.join(sorted(utts))}\n"
        for spk, utts in sorted(spk2utts.items())
    ]
    (output_dir / "spk2utt").write_text("".join(spk2utt_lines))

    log.info("Wrote %d utterances to %s", count, output_dir)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS = ("afrispeech", "voxpopuli", "librispeech")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download HuggingFace dataset and write ESPnet kaldi-format manifests."
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="Dataset identifier.",
    )
    p.add_argument(
        "--split",
        required=True,
        help="HuggingFace split name (e.g. 'train', 'validation', 'validation.clean').",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=pathlib.Path,
        help="Output kaldi data directory.",
    )
    p.add_argument(
        "--audio_dir",
        required=True,
        type=pathlib.Path,
        help="Directory to cache downloaded audio WAV files.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max utterances to use (-1 = all).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling.",
    )
    p.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds (default 0.5). "
        "Mirrors NeMo data loader min_duration=0.5.",
    )
    p.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="Maximum audio duration in seconds (default 20.0). "
        "Mirrors NeMo data loader max_duration=20.0.",
    )
    return p


def main(argv=None) -> None:
    args = get_parser().parse_args(argv)

    audio_dir: pathlib.Path = args.audio_dir.resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    dur_kwargs = dict(min_duration=args.min_duration, max_duration=args.max_duration)

    if args.dataset == "afrispeech":
        it = _iter_afrispeech(args.split, audio_dir, args.max_samples, args.seed, **dur_kwargs)
    elif args.dataset == "voxpopuli":
        it = _iter_voxpopuli(args.split, audio_dir, args.max_samples, args.seed, **dur_kwargs)
    elif args.dataset == "librispeech":
        it = _iter_librispeech(args.split, audio_dir, args.max_samples, args.seed, **dur_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    n = write_kaldi_dir(it, args.output_dir.resolve())
    if n == 0:
        log.error("No utterances written — check dataset name / split / domain filter.")
        sys.exit(1)


if __name__ == "__main__":
    main()
