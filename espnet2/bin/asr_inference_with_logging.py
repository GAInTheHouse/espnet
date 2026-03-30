#!/usr/bin/env python3
"""ASR inference with per-utterance JSONL prediction logging.

This is a drop-in extension of ``espnet2.bin.asr_inference`` that adds:

1. ``--prediction_log_path`` — path to a JSONL file where each decoded
   utterance is appended as one JSON line with the following fields::

       {
         "utt_id":      "1272-128104-0000",
         "hypothesis":  "the dog sat on the mat",
         "reference":   "the cat sat on the mat",    # if text key-file is provided
         "wer":         0.1667,                       # per-utterance WER via jiwer
         "beam_score":  -3.412,                       # total beam score (log-prob)
         "beam_scores": {                             # per-scorer breakdown
           "decoder":    -2.81,
           "ctc":        -0.60,
           "length_bonus": 5.0
         },
         "token":       ["the", "▁dog", "▁sat", ...],
         "token_int":   [17, 542, 312, ...]
       }

2. WER is computed with ``jiwer.wer()`` when a reference is available.
   If jiwer is not installed, the ``wer`` field is omitted.

3. A reference text file can be supplied via ``--ref_text_path`` (one
   ``utt_id  text`` per line, Kaldi format) to populate the ``reference``
   and ``wer`` fields.

All original ``asr_inference`` arguments and behaviour are preserved.
The extra logging is purely additive.

Example
-------
python -m espnet2.bin.asr_inference_with_logging \\
    --ngpu 0 \\
    --beam_size 10 \\
    --ctc_weight 0.3 \\
    --data_path_and_name_and_type data/test_clean_subset/wav.scp,speech,sound \\
    --asr_train_config exp/asr/config.yaml \\
    --asr_model_file  exp/asr/valid.acc.best.pth \\
    --output_dir      exp/decode/test_clean \\
    --ref_text_path   data/test_clean/text \\
    --prediction_log_path exp/decode/test_clean/predictions.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Re-export everything from the original module so we stay compatible.
from espnet2.bin.asr_inference import (  # noqa: F401
    Speech2Text,
    get_parser as _orig_get_parser,
    inference as _orig_inference,
)
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none

logger = logging.getLogger(__name__)

try:
    import jiwer as _jiwer

    _HAS_JIWER = True
except ImportError:
    _jiwer = None
    _HAS_JIWER = False
    logger.warning(
        "jiwer is not installed; WER will not be computed in the prediction log. "
        "Install with: pip install jiwer"
    )


# ---------------------------------------------------------------------------
# Reference text loader
# ---------------------------------------------------------------------------


def _load_ref_text(path: Optional[str]) -> Dict[str, str]:
    """Load Kaldi-format text file into {utt_id: transcript} dict."""
    if path is None:
        return {}
    refs: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]
            elif len(parts) == 1:
                refs[parts[0]] = ""
    logger.info(f"Loaded {len(refs)} references from {path}")
    return refs


# ---------------------------------------------------------------------------
# Extended inference function
# ---------------------------------------------------------------------------


def inference_with_logging(
    # --- original asr_inference args (subset; full list forwarded via **kwargs) ---
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    normalize_length: bool,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    transducer_conf: Optional[dict],
    streaming: bool,
    enh_s2t_task: bool,
    quantize_asr_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    hugging_face_decoder: bool,
    hugging_face_decoder_conf: Dict[str, Any],
    time_sync: bool,
    multi_asr: bool,
    lang_prompt_token: Optional[str],
    nlp_prompt_token: Optional[str],
    prompt_token_file: Optional[str],
    partial_ar: bool,
    threshold_probability: float,
    max_seq_len: int,
    max_mask_parallel: int,
    # --- new args ---
    prediction_log_path: Optional[str],
    ref_text_path: Optional[str],
):
    """Run ASR inference and write per-utterance JSONL logs.

    All arguments up to ``partial_ar`` / ``max_mask_parallel`` are identical
    to :func:`espnet2.bin.asr_inference.inference`.  Two new keyword args are
    added:

    Args:
        prediction_log_path: Path for the output JSONL file.
            Each decoded utterance appends one JSON line.
            Set to None to disable logging.
        ref_text_path: Kaldi-format text file with reference transcripts.
            Used to populate the ``reference`` and ``wer`` fields.
    """
    import torch
    from espnet2.torch_utils.set_all_random_seed import set_all_random_seed

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")

    device = "cuda" if ngpu >= 1 else "cpu"
    set_all_random_seed(seed)

    # 1. Build Speech2Text (same as base inference)
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        transducer_conf=transducer_conf,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        normalize_length=normalize_length,
        streaming=streaming,
        enh_s2t_task=enh_s2t_task,
        multi_asr=multi_asr,
        quantize_asr_model=quantize_asr_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        hugging_face_decoder=hugging_face_decoder,
        hugging_face_decoder_conf=hugging_face_decoder_conf,
        time_sync=time_sync,
        prompt_token_file=prompt_token_file,
        lang_prompt_token=lang_prompt_token,
        nlp_prompt_token=nlp_prompt_token,
        partial_ar=partial_ar,
        threshold_probability=threshold_probability,
        max_seq_len=max_seq_len,
        max_mask_parallel=max_mask_parallel,
    )

    # 2. Build data iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 3. Load references (optional)
    refs = _load_ref_text(ref_text_path)

    # 4. Open prediction log file
    log_file = None
    if prediction_log_path is not None:
        Path(prediction_log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(prediction_log_path, "w", encoding="utf-8")
        logger.info(f"Prediction log: {prediction_log_path}")

    # 5. Decoding loop
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            key = keys[0]

            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]] * nbest

            # Unwrap interCTC tuple (normal ASR path)
            encoder_interctc_res = None
            if isinstance(results, tuple):
                results, encoder_interctc_res = results

            for n, (text, token, token_int, hyp) in zip(
                range(1, nbest + 1), results
            ):
                ibest_writer = writer[f"{n}best_recog"]
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)
                if text is not None:
                    ibest_writer["text"][key] = text

            # ----------------------------------------------------------------
            # Prediction logging (1-best only)
            # ----------------------------------------------------------------
            if log_file is not None and results:
                text_1best, token_1best, token_int_1best, hyp_1best = results[0]
                hypothesis_str = text_1best if text_1best is not None else ""

                reference_str = refs.get(key, "")

                # Per-scorer beam score breakdown
                beam_scores_dict: Dict[str, float] = {}
                if hasattr(hyp_1best, "scores") and hyp_1best.scores:
                    beam_scores_dict = {
                        k: float(v) for k, v in hyp_1best.scores.items()
                    }

                # WER (requires jiwer and a non-empty reference)
                wer_value: Optional[float] = None
                if _HAS_JIWER and reference_str.strip() and hypothesis_str is not None:
                    try:
                        wer_value = float(
                            _jiwer.wer(
                                reference_str,
                                hypothesis_str if hypothesis_str.strip() else "<empty>",
                            )
                        )
                    except Exception as exc:
                        logger.warning(f"jiwer.wer() failed for {key}: {exc}")

                log_entry: Dict[str, Any] = {
                    "utt_id": key,
                    "hypothesis": hypothesis_str,
                    "reference": reference_str,
                    "beam_score": float(hyp_1best.score),
                    "beam_scores": beam_scores_dict,
                    "token": token_1best,
                    "token_int": token_int_1best,
                }
                if wer_value is not None:
                    log_entry["wer"] = wer_value

                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                log_file.flush()

                logger.info(
                    f"[LOG] {key}  score={float(hyp_1best.score):.3f}"
                    f"  wer={wer_value * 100:.1f}%"
                    if wer_value is not None
                    else f"[LOG] {key}  score={float(hyp_1best.score):.3f}"
                )

            # Write interCTC predictions if present
            if encoder_interctc_res is not None:
                ibest_writer = writer["1best_recog"]
                for idx, interctc_text in encoder_interctc_res.items():
                    ibest_writer[f"encoder_interctc_layer{idx}.txt"][key] = (
                        " ".join(interctc_text)
                    )

    if log_file is not None:
        log_file.close()
        logger.info(f"Prediction log written to {prediction_log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    """Return argument parser, extending the original with logging arguments."""
    parser = _orig_get_parser()

    group = parser.add_argument_group("Prediction logging")
    group.add_argument(
        "--prediction_log_path",
        type=str_or_none,
        default=None,
        help=(
            "Path to the JSONL output file for per-utterance prediction logging. "
            "Each line is a JSON object with fields: "
            "utt_id, hypothesis, reference, wer, beam_score, beam_scores, token, token_int. "
            "If not set, no log is written."
        ),
    )
    group.add_argument(
        "--ref_text_path",
        type=str_or_none,
        default=None,
        help=(
            "Path to a Kaldi-format text file (utt_id  transcript per line) "
            "used to populate the 'reference' and 'wer' fields in the log. "
            "If not set, these fields are empty / omitted."
        ),
    )
    return parser


def main(cmd=None):
    from espnet2.legacy.utils.cli_utils import get_commandline_args

    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)

    # Pull out the two new args before forwarding to inference_with_logging
    prediction_log_path = kwargs.pop("prediction_log_path", None)
    ref_text_path = kwargs.pop("ref_text_path", None)

    inference_with_logging(
        **kwargs,
        prediction_log_path=prediction_log_path,
        ref_text_path=ref_text_path,
    )


if __name__ == "__main__":
    main()
