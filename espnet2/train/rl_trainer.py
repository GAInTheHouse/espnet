"""REINFORCE policy-gradient trainer extending ESPnet2's base Trainer.

Design overview
---------------
The standard ESPnet2 Trainer computes a supervised cross-entropy (CE) loss
inside the model's forward() and then calls backward().  RLTrainer introduces
a second signal — a WER-based REINFORCE reward — and blends it with the CE
loss before the backward pass:

    total_loss = (1 - rl_weight) * ce_loss + rl_weight * pg_loss

    pg_loss = -E[reward * log_prob_of_greedy_hypothesis]

The injection happens by augmenting each batch dict with an ``rl_weight``
key before calling the model.  ``RLESPnetModel`` reads this key from
``**kwargs`` in its ``forward()`` and runs both CE and RL branches,
returning the blended loss in the standard (loss, stats, weight) tuple.
This design keeps DDP gradient synchronisation intact: every parameter
update flows through the same single DDP-wrapped forward call.

Usage
-----
In your ASR task class, replace::

    trainer = Trainer

with::

    trainer = RLTrainer

and use ``RLESPnetModel`` (from ``espnet2.asr.rl_espnet_model``) as the
model class.
"""

import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    """TrainerOptions extended with RL-specific hyper-parameters."""

    rl_weight: float = 0.1


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------


class RLTrainer(Trainer):
    """REINFORCE policy-gradient trainer for ESPnet2 ASR models.

    Inherits from ``Trainer`` and overrides ``train_one_epoch`` to blend a
    WER-based REINFORCE signal with the supervised CE loss.

    The RL weight is passed to the model through the batch dict so that
    DDP gradient synchronisation is preserved across all ranks.
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
                "Mixing weight for the REINFORCE policy-gradient loss. "
                "total_loss = (1 - rl_weight) * ce_loss + rl_weight * pg_loss. "
                "Set to 0.0 to fall back to purely supervised training."
            ),
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @classmethod
    def compute_reward(
        cls,
        hypotheses: List[str],
        references: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute per-utterance WER-based REINFORCE reward.

        reward_i = clip(1 - WER(reference_i, hypothesis_i), 0, 1)

        A perfect hypothesis yields reward 1.0; a hypothesis whose WER
        exceeds 100 % (insertions can push WER > 1) is clipped to 0.0.

        Args:
            hypotheses: Decoded hypothesis strings, one per utterance.
            references:  Ground-truth transcript strings, one per utterance.
            device:      Device for the returned tensor.

        Returns:
            Tensor of shape (B,) with values in [0, 1].

        Raises:
            RuntimeError: When ``jiwer`` is not installed.
        """
        if not _HAS_JIWER:
            raise RuntimeError(
                "jiwer is required for WER-based rewards. "
                "Install with: pip install jiwer"
            )

        rewards: List[float] = []
        for hyp, ref in zip(hypotheses, references):
            if not ref.strip():
                # Empty reference — reward is undefined; treat as 0.
                rewards.append(0.0)
                continue
            try:
                wer_score = _jiwer.wer(ref, hyp if hyp.strip() else "<empty>")
            except Exception as exc:
                logging.warning(f"jiwer.wer() failed ({exc}); defaulting reward to 0.")
                wer_score = 1.0
            rewards.append(max(0.0, 1.0 - wer_score))

        return torch.tensor(rewards, dtype=torch.float32, device=device)

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
        """Run one epoch of REINFORCE-blended ASR training.

        This method mirrors the structure of ``Trainer.train_one_epoch``
        exactly, with a single augmentation: ``rl_weight`` is injected into
        each batch dict before the model forward call.  The model
        (``RLESPnetModel``) picks up this key and computes both CE and PG
        losses internally, keeping DDP all-reduce semantics intact.

        The additional stats emitted by the model (``reward``, ``pg_loss``,
        ``ce_loss``) are transparently forwarded through ESPnet2's existing
        Reporter infrastructure.
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
        rl_weight = getattr(options, "rl_weight", 0.1)

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True

        # Distributed early-stop flag (same as base Trainer).
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

            # -------------------------------------------------------
            # [RL INJECTION] Pass rl_weight through the batch dict so
            # the model can blend CE and PG losses in forward().
            # -------------------------------------------------------
            batch["rl_weight"] = rl_weight

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            with autocast(scaler is not None, **_autocast_args):
                with reporter.measure_time("forward_time"):
                    retval = model(**batch)

                    # Standard ESPnet2 return-value contract:
                    #   dict with keys 'loss', 'stats', 'weight', optionally 'optim_idx'
                    #   -OR- a plain (loss, stats, weight) tuple.
                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if isinstance(optim_idx, torch.Tensor):
                                if optim_idx.dim() == 1:
                                    optim_idx = optim_idx[0].item()
                                else:
                                    optim_idx = optim_idx.item()
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
                        f"[RLTrainer] grad_norm is {grad_norm}; skipping update."
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
