#!/usr/bin/env python3
"""Extended evaluation matching NeMo's evaluate_manifest_bundle metrics.

Computes the full set of metrics reported in the NeMo results
(docs/results_17Apr2026.md) that are not produced by the basic jiwer WER
scoring in run.sh stage 8:

  WER / CER       Standard word / character error rate (via jiwer)
  SER             Sentence Error Rate — fraction where normalised ref ≠ hyp
  EWER            Entity (domain) WER — WER restricted to domain-vocab words
  Domain P/R/F1   Token-set precision, recall, F1 over domain vocabulary
  Degenerate frac Fraction of hypotheses with ≤ 1 character (collapse signal)
  Mean hyp len    Mean hypothesis character length
  Bootstrap p     Paired bootstrap WER p-value between two model outputs

Usage
-----
  # Single-model extended metrics
  python local/eval_extended.py \\
      --hyp_file exp/asr_rl/decode_afrispeech_test/1best_recog/text \\
      --ref_file data/afrispeech_test/text \\
      --domain_terms_file conf/domain_terms_clinical.txt \\
      --output_json exp/asr_rl/decode_afrispeech_test/extended_metrics.json

  # Paired bootstrap p-value (SFT vs RL)
  python local/eval_extended.py \\
      --hyp_file exp/asr_rl/decode_afrispeech_test/1best_recog/text \\
      --ref_file data/afrispeech_test/text \\
      --baseline_hyp_file exp/asr_sft/decode_afrispeech_test/1best_recog/text \\
      --domain_terms_file conf/domain_terms_clinical.txt \\
      --bootstrap_iters 1000 \\
      --output_json exp/asr_rl/decode_afrispeech_test/extended_metrics.json

Input file format (ESPnet kaldi-style):
  <utt_id> <text...>
"""

import argparse
import json
import logging
import pathlib
import random
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import jiwer as _jiwer

    _HAS_JIWER = True
except ImportError:
    _jiwer = None
    _HAS_JIWER = False
    log.error("jiwer is required: pip install jiwer")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_kaldi_text(path: pathlib.Path) -> Dict[str, str]:
    """Load a kaldi-style 'utt_id text...' file into a dict."""
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        uid = parts[0]
        text = parts[1] if len(parts) > 1 else ""
        result[uid] = text
    return result


def _load_domain_terms(path: pathlib.Path) -> frozenset:
    """Load domain terms file; one term per line, ignore comment lines."""
    terms = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.add(line.lower())
    return frozenset(terms)


# ---------------------------------------------------------------------------
# Normalisation (mirrors NeMo's _normalize_text)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Per-metric implementations
# ---------------------------------------------------------------------------


def compute_ser(
    refs: List[str], hyps: List[str]
) -> Tuple[float, List[bool]]:
    """Sentence Error Rate: fraction where normalised ref ≠ hyp.

    Returns (ser_percent, per_utt_errors).
    """
    errors = [_normalize(r) != _normalize(h) for r, h in zip(refs, hyps)]
    ser = 100.0 * sum(errors) / max(len(errors), 1)
    return ser, errors


def compute_ewer(
    refs: List[str], hyps: List[str], domain_set: frozenset
) -> Tuple[float, int]:
    """Entity (domain) WER.

    Per utterance, keep only words present in the domain vocabulary from the
    reference; compute WER on that subset; average over utterances with ≥1
    domain token in the reference.

    Mirrors NeMo's entity_wer_from_text logic.

    Returns (ewer_percent, n_utterances_with_domain_tokens).
    """
    if not _HAS_JIWER:
        raise RuntimeError("jiwer required for EWER")

    wers: List[float] = []
    for ref, hyp in zip(refs, hyps):
        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()
        # Keep only reference words that are domain terms
        domain_ref_words = [w for w in ref_words if w in domain_set]
        if not domain_ref_words:
            continue
        # Filter hypothesis to same domain-term positions using ref alignment
        # Simpler: filter hypothesis words to domain vocab (same logic as NeMo)
        domain_hyp_words = [w for w in hyp_words if w in domain_set]
        domain_ref_str = " ".join(domain_ref_words)
        domain_hyp_str = " ".join(domain_hyp_words) if domain_hyp_words else "<empty>"
        try:
            wer = _jiwer.wer(domain_ref_str, domain_hyp_str)
        except Exception as exc:
            log.warning("EWER jiwer error (%s); skipping utterance", exc)
            continue
        wers.append(wer)

    if not wers:
        return 0.0, 0
    return 100.0 * sum(wers) / len(wers), len(wers)


def compute_domain_f1(
    refs: List[str], hyps: List[str], domain_set: frozenset
) -> Tuple[float, float, float]:
    """Aggregate domain-term precision, recall, and F1.

    Per utterance: ref_domain_tokens = set(domain terms in ref),
    hyp_domain_tokens = set(domain terms in hyp).
    Precision = |intersection| / |hyp_domain| (0 if hyp has no domain terms).
    Recall    = |intersection| / |ref_domain|  (skip if ref has no domain terms).
    Aggregate: macro-average of per-utterance P/R/F1 over utterances with ≥1
    domain term in the reference.

    Mirrors NeMo's aggregate_f1 / domain_term_precision_recall_f1 logic.

    Returns (precision, recall, f1) as fractions in [0, 1].
    """
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for ref, hyp in zip(refs, hyps):
        ref_domain = set(w.lower() for w in ref.split() if w.lower() in domain_set)
        hyp_domain = set(w.lower() for w in hyp.split() if w.lower() in domain_set)
        if not ref_domain:
            continue
        tp = len(ref_domain & hyp_domain)
        p = tp / len(hyp_domain) if hyp_domain else 0.0
        r = tp / len(ref_domain)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    if not precisions:
        return 0.0, 0.0, 0.0
    n = len(precisions)
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def compute_degenerate_frac(hyps: List[str]) -> float:
    """Fraction of hypotheses with ≤ 1 character (training collapse signal)."""
    degenerate = sum(1 for h in hyps if len(h.strip()) <= 1)
    return degenerate / max(len(hyps), 1)


def compute_mean_hyp_len(hyps: List[str]) -> float:
    """Mean hypothesis character length."""
    if not hyps:
        return 0.0
    return sum(len(h) for h in hyps) / len(hyps)


def paired_bootstrap_wer_pvalue(
    refs: List[str],
    hyps_a: List[str],
    hyps_b: List[str],
    n_iters: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test for WER improvement from model A to model B.

    H0: WER(A) == WER(B).
    Returns p-value: fraction of bootstrap resamples where |WER_b* - WER_a*|
    exceeds the observed difference |WER_b - WER_a| (two-tailed).

    Mirrors NeMo's bootstrap_wer_pvalue with paired resampling.
    """
    if not _HAS_JIWER:
        raise RuntimeError("jiwer required for bootstrap p-value")

    n = len(refs)
    rng = random.Random(seed)

    # Per-utterance WER (0 or 1 for exact wrong, fractional for partial)
    def _per_utt_wer(r: str, h: str) -> float:
        if not r.strip():
            return 0.0
        try:
            return _jiwer.wer(r, h if h.strip() else "<empty>")
        except Exception:
            return 1.0

    wers_a = [_per_utt_wer(r, h) for r, h in zip(refs, hyps_a)]
    wers_b = [_per_utt_wer(r, h) for r, h in zip(refs, hyps_b)]

    observed_diff = abs(sum(wers_b) / n - sum(wers_a) / n)

    exceed = 0
    for _ in range(n_iters):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_a = sum(wers_a[i] for i in indices) / n
        boot_b = sum(wers_b[i] for i in indices) / n
        if abs(boot_b - boot_a) >= observed_diff:
            exceed += 1

    return exceed / n_iters


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------


def evaluate(
    hyp_file: pathlib.Path,
    ref_file: pathlib.Path,
    domain_terms_file: Optional[pathlib.Path],
    baseline_hyp_file: Optional[pathlib.Path],
    bootstrap_iters: int,
    output_json: pathlib.Path,
) -> dict:
    if not _HAS_JIWER:
        raise RuntimeError("jiwer not installed: pip install jiwer")

    hyps_dict = _load_kaldi_text(hyp_file)
    refs_dict = _load_kaldi_text(ref_file)
    common = sorted(set(hyps_dict) & set(refs_dict))
    if not common:
        raise ValueError(f"No common utterance IDs between {hyp_file} and {ref_file}")
    log.info("Scoring %d utterances", len(common))

    refs = [refs_dict[u] for u in common]
    hyps = [hyps_dict[u] for u in common]

    # --- WER / CER ---
    wer = _jiwer.wer(refs, hyps) * 100.0
    cer = _jiwer.cer(refs, hyps) * 100.0

    # --- SER ---
    ser, _ = compute_ser(refs, hyps)

    # --- Domain metrics ---
    domain_set: frozenset = frozenset()
    if domain_terms_file and domain_terms_file.exists():
        domain_set = _load_domain_terms(domain_terms_file)
        log.info("Domain vocabulary: %d terms", len(domain_set))
    else:
        log.warning("No domain terms file; EWER and domain F1 will be 0.")

    ewer, n_domain_utts = compute_ewer(refs, hyps, domain_set)
    dom_p, dom_r, dom_f1 = compute_domain_f1(refs, hyps, domain_set)

    # --- Diagnostics ---
    deg_frac = compute_degenerate_frac(hyps)
    mean_len = compute_mean_hyp_len(hyps)

    # --- Bootstrap p-value ---
    bootstrap_pval: Optional[float] = None
    if baseline_hyp_file and baseline_hyp_file.exists():
        baseline_dict = _load_kaldi_text(baseline_hyp_file)
        baseline_hyps = [baseline_dict.get(u, "") for u in common]
        log.info("Running %d bootstrap iterations for p-value ...", bootstrap_iters)
        bootstrap_pval = paired_bootstrap_wer_pvalue(
            refs, baseline_hyps, hyps, n_iters=bootstrap_iters
        )
        baseline_wer = _jiwer.wer(refs, baseline_hyps) * 100.0
        log.info(
            "Bootstrap: baseline WER=%.2f%%, system WER=%.2f%%, p=%.4f",
            baseline_wer,
            wer,
            bootstrap_pval,
        )

    results = {
        "n_utterances": len(common),
        "wer_pct": round(wer, 4),
        "cer_pct": round(cer, 4),
        "ser_pct": round(ser, 4),
        "ewer_pct": round(ewer, 4),
        "n_utterances_with_domain_tokens": n_domain_utts,
        "domain_precision": round(dom_p, 6),
        "domain_recall": round(dom_r, 6),
        "domain_f1": round(dom_f1, 6),
        "degenerate_hyp_frac": round(deg_frac, 6),
        "mean_hyp_len_chars": round(mean_len, 3),
        "bootstrap_pval_vs_baseline": bootstrap_pval,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2))

    log.info("Results:")
    for k, v in results.items():
        log.info("  %-40s %s", k, v)
    log.info("Written to %s", output_json)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extended NeMo-aligned evaluation for ESPnet ASR hypotheses."
    )
    p.add_argument(
        "--hyp_file",
        required=True,
        type=pathlib.Path,
        help="ESPnet hypothesis file (kaldi format: utt_id text...).",
    )
    p.add_argument(
        "--ref_file",
        required=True,
        type=pathlib.Path,
        help="Reference text file (kaldi format: utt_id text...).",
    )
    p.add_argument(
        "--domain_terms_file",
        type=pathlib.Path,
        default=None,
        help="Path to domain terms file (one term per line). "
        "Default: conf/domain_terms_clinical.txt relative to cwd.",
    )
    p.add_argument(
        "--baseline_hyp_file",
        type=pathlib.Path,
        default=None,
        help="Baseline hypothesis file for paired bootstrap p-value "
        "(e.g. SFT output when scoring RL). Skipped if not provided.",
    )
    p.add_argument(
        "--bootstrap_iters",
        type=int,
        default=1000,
        help="Number of bootstrap resampling iterations (default 1000).",
    )
    p.add_argument(
        "--output_json",
        required=True,
        type=pathlib.Path,
        help="Path to write JSON results.",
    )
    return p


def main(argv=None) -> None:
    args = get_parser().parse_args(argv)

    domain_terms_file = args.domain_terms_file
    if domain_terms_file is None:
        # Default relative to recipe root
        default = pathlib.Path("conf/domain_terms_clinical.txt")
        if default.exists():
            domain_terms_file = default

    evaluate(
        hyp_file=args.hyp_file,
        ref_file=args.ref_file,
        domain_terms_file=domain_terms_file,
        baseline_hyp_file=args.baseline_hyp_file,
        bootstrap_iters=args.bootstrap_iters,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
