"""REINFORCE / reward-augmented fine-tuning trainer extending ESPnet2's Trainer.

Design overview
---------------
``RLTrainer`` overrides ``train_one_epoch()`` to inject RL configuration into
each batch dict before the model forward call.  ``RLESPnetModel`` reads these
keys from its ``forward(**kwargs)`` signature and computes the blended loss
internally, keeping DDP all-reduce semantics intact.

Two loss formulae are supported (selected via ``--reward_loss_type``):

penalty (NeMo-aligned, default for the AfriSpeech RL experiment)::

    total_loss = ce_loss + rl_weight * (1 - mean(reward))

reinforce (true REINFORCE policy gradient)::

    total_loss = (1 - rl_weight) * ce_loss + rl_weight * pg_loss
    pg_loss    = -mean(reward * seq_log_prob.float())

Reward modes (``--reward_mode``):
    mwer    WER-based reward via jiwer (default)
    wwer    Domain-weighted WER (uses --domain_terms / --domain_term_weight)
    llm     Gemini-1.5-flash quality score; mock fallback = mwer + N(0,0.05)
    all     Element-wise mean of mwer, wwer, and llm

GPU efficiency improvements vs. the original single-mode trainer:
    1. ``compute_reward = (iiter % reward_step_interval == 0)`` — reward is
       computed only every N steps; the model reuses a cached neutral value
       on skip steps, matching NeMo's REWARD_STEP_INTERVAL=4.
    2. In ``penalty`` mode, ``RLESPnetModel`` wraps the greedy decode in
       ``torch.no_grad()`` so no CTC activation graph is retained.
    3. In ``reinforce`` mode, ``seq_log_probs.float()`` prevents FP16 underflow
       for long utterances under AMP.

Usage
-----
In your ASR task class, set::

    trainer = RLTrainer

and use ``RLESPnetModel`` (espnet2.asr.rl_espnet_model) as the model class,
or run via ``espnet2.bin.asr_train_rl`` (uses ``RLASRTask``).
"""

import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn
import torch.optim
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.build_dataclass import build_dataclass

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast

    _autocast_args: dict = {}
    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        _autocast_args = dict(dtype=torch.bfloat16)
else:

    @contextmanager
    def autocast(enabled=True, **kwargs):
        yield

    GradScaler = None
    _autocast_args = {}

try:
    import jiwer as _jiwer

    _HAS_JIWER = True
except ImportError:
    _jiwer = None
    _HAS_JIWER = False
    logging.warning(
        "jiwer is not installed. WER-based rewards will be unavailable. "
        "Install with: pip install jiwer"
    )

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


# ---------------------------------------------------------------------------
# Trainer options
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RLTrainerOptions(TrainerOptions):
    """TrainerOptions extended with RL / reward-augmented fine-tuning parameters.

    All fields map 1-to-1 to CLI arguments registered in
    ``RLTrainer.add_arguments()``.
    """

    rl_weight: float = 0.1
    """Blend weight for the RL loss term.  0.0 = pure CE (SFT stage)."""

    reward_mode: str = "mwer"
    """Reward function: ``mwer`` | ``wwer`` | ``llm`` | ``all``."""

    reward_loss_type: str = "reinforce"
    """Loss formula: ``reinforce`` (REINFORCE PG) | ``penalty`` (NeMo-style)."""

    reward_step_interval: int = 4
    """Compute reward every N optimizer steps; reuse cached value otherwise.
    Matches NeMo REWARD_STEP_INTERVAL=4."""

    max_encoder_len_for_reward: int = 1500
    """Skip reward computation for utterances whose encoder output exceeds
    this frame count (GPU memory guard).  Reuses cached reward."""

    domain_terms: List[str] = dataclasses.field(default_factory=list)
    """Vocabulary list for ``wwer`` domain weighting."""

    domain_term_weight: float = 3.0
    """Edit-cost multiplier for domain terms in ``wwer`` mode."""

    gemini_api_key: str = ""
    """Gemini API key for ``llm`` reward mode.  Falls back to mock if empty."""

    mock_llm: bool = False
    """Use mock LLM (mwer + Gaussian noise) even if ``gemini_api_key`` is set."""

    reward_sample_dump_interval: int = 200
    """Log up to 10 sample (utt_id, ref, hyp, reward) tuples every N optimizer
    steps to the standard Python logger at INFO level.
    Mirrors NeMo's reward sample dump for human verification that rewards are
    sensible and to detect collapse early.  Set to 0 to disable."""


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------


class RLTrainer(Trainer):
    """Reward-augmented fine-tuning trainer for ESPnet2 ASR models.

    Inherits from ``Trainer`` and overrides ``train_one_epoch`` to inject RL
    configuration into each batch dict before the model forward call.  The
    model (``RLESPnetModel``) picks up these keys and computes the blended
    loss internally, keeping DDP all-reduce semantics intact.
    """

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> RLTrainerOptions:
        return build_dataclass(RLTrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register RL-specific CLI arguments."""
        super().add_arguments(parser)
        group = parser.add_argument_group("RLTrainer")
        group.add_argument(
            "--rl_weight",
            type=float,
            default=0.1,
            help=(
                "Blend weight for the RL loss. "
                "penalty mode: total = ce + w*(1-mean_reward). "
                "reinforce mode: total = (1-w)*ce + w*pg_loss. "
                "Set to 0.0 to run pure supervised training (SFT stage)."
            ),
        )
        group.add_argument(
            "--reward_mode",
            type=str,
            default="mwer",
            choices=["mwer", "wwer", "llm", "all"],
            help=(
                "Reward function to use: "
                "mwer=WER-based, wwer=domain-weighted WER, "
                "llm=Gemini quality score, all=average of all three."
            ),
        )
        group.add_argument(
            "--reward_loss_type",
            type=str,
            default="reinforce",
            choices=["reinforce", "penalty", "reweight_ctc"],
            help=(
                "Loss formula: "
                "'reinforce' = true REINFORCE PG loss; "
                "'penalty' = NeMo-style auxiliary penalty (no policy gradient); "
                "'reweight_ctc' = per-utterance CTC loss weighted by (1-reward_i), "
                "the objective used in the actual NeMo run."
            ),
        )
        group.add_argument(
            "--reward_step_interval",
            type=int,
            default=4,
            help=(
                "Compute reward every N optimizer steps. "
                "Cached neutral reward (0.5) is reused on skip steps. "
                "Matches NeMo REWARD_STEP_INTERVAL=4."
            ),
        )
        group.add_argument(
            "--max_encoder_len_for_reward",
            type=int,
            default=1500,
            help=(
                "Skip reward computation when encoder output exceeds this "
                "frame count (long-utterance GPU memory guard)."
            ),
        )
        group.add_argument(
            "--domain_terms",
            nargs="*",
            default=[],
            metavar="TERM",
            help=(
                "Domain vocabulary for wwer reward weighting. "
                "Example: --domain_terms hypertension arrhythmia tachycardia"
            ),
        )
        group.add_argument(
            "--domain_term_weight",
            type=float,
            default=3.0,
            help="Edit-cost multiplier for domain terms in wwer mode (default 3.0).",
        )
        group.add_argument(
            "--gemini_api_key",
            type=str,
            default="",
            help=(
                "Google Gemini API key for llm reward mode. "
                "If empty, mock fallback (mwer + Gaussian noise) is used."
            ),
        )
        group.add_argument(
            "--mock_llm",
            default=False,
            action="store_true",
            help="Force mock LLM path even when --gemini_api_key is provided.",
        )
        group.add_argument(
            "--reward_sample_dump_interval",
            type=int,
            default=200,
            help=(
                "Log up to 10 sample (utt_id, ref, hyp, reward) tuples to the "
                "logger every N optimizer steps (INFO level). "
                "Mirrors NeMo reward sample dump for verification and collapse "
                "detection.  Set to 0 to disable."
            ),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @classmethod
    @typechecked
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[object]],
        scaler: Optional[object],
        reporter: SubReporter,
        summary_writer,
        options: RLTrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        """Run one epoch of reward-augmented ASR training.

        Mirrors ``Trainer.train_one_epoch`` exactly, with RL config injected
        into each batch dict before the model call.  ``RLESPnetModel`` reads
        these keys and computes the blended loss, keeping DDP semantics intact.
        """
        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        distributed = distributed_option.distributed

        # RL options
        rl_weight = options.rl_weight
        reward_mode = options.reward_mode
        reward_loss_type = options.reward_loss_type
        reward_step_interval = options.reward_step_interval
        max_encoder_len_for_reward = options.max_encoder_len_for_reward
        domain_terms = options.domain_terms or []
        domain_term_weight = options.domain_term_weight
        gemini_api_key = options.gemini_api_key or ""
        mock_llm = options.mock_llm
        reward_sample_dump_interval = options.reward_sample_dump_interval

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True

        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            # -----------------------------------------------------------
            # [RL INJECTION] Augment batch dict with all RL config.
            # RLESPnetModel reads these from its forward() signature.
            # Non-tensor values pass through to_device() unchanged.
            # -----------------------------------------------------------
            compute_reward = iiter % reward_step_interval == 0
            # Reward sample dump: on dump steps, ask the model to log samples
            log_reward_samples = (
                reward_sample_dump_interval > 0
                and compute_reward
                and iiter % reward_sample_dump_interval == 0
            )
            batch["rl_weight"] = rl_weight
            batch["compute_reward"] = compute_reward
            batch["reward_mode"] = reward_mode
            batch["reward_loss_type"] = reward_loss_type
            batch["max_encoder_len_for_reward"] = max_encoder_len_for_reward
            batch["domain_terms"] = domain_terms
            batch["domain_term_weight"] = domain_term_weight
            batch["gemini_api_key"] = gemini_api_key
            batch["mock_llm"] = mock_llm
            batch["log_reward_samples"] = log_reward_samples

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            with autocast(scaler is not None, **_autocast_args):
                with reporter.measure_time("forward_time"):
                    retval = model(**batch)

                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if isinstance(optim_idx, torch.Tensor):
                                optim_idx = (
                                    optim_idx[0].item()
                                    if optim_idx.dim() == 1
                                    else optim_idx.item()
                                )
                            else:
                                raise RuntimeError(
                                    f"optim_idx must be int or 1-d Tensor, "
                                    f"got {type(optim_idx)}"
                                )
                    else:
                        loss, stats, weight = retval
                        optim_idx = None

                    retval = None

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    loss = (loss * weight.type(loss.dtype)).sum()
                    stats, weight = recursive_average(stats, weight, distributed)
                    loss /= weight
                if distributed:
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad

            reporter.register(stats, weight)

            with reporter.measure_time("backward_time"):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            del loss

            if iiter % accum_grad == 0:
                if scaler is not None:
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        "[RLTrainer] grad_norm is %s; skipping update.", grad_norm
                    )
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()
                else:
                    reporter.register(
                        {
                            "grad_norm": grad_norm,
                            "clip": torch.where(
                                grad_norm > grad_clip,
                                grad_norm.new_tensor(100),
                                grad_norm.new_tensor(0),
                            ),
                            "loss_scale": scaler.get_scale() if scaler else 1.0,
                        }
                    )
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()

                for iopt, optimizer in enumerate(optimizers):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    optimizer.zero_grad()

                reporter.register(
                    dict(
                        {
                            f"optim{i}_lr{j}": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    )
                )
                start_time = time.perf_counter()

            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        return all_steps_are_invalid
