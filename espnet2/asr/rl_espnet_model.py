"""RL-compatible ESPnet2 ASR model.

Extends ``ESPnetASRModel`` with two capabilities:

1. **Blended forward pass** (``forward()``)
   When ``rl_weight > 0`` is passed in the batch dict, the model encodes
   speech once, computes the standard CE loss (CTC + attention-decoder),
   and additionally computes a REINFORCE policy-gradient loss using greedy
   CTC decoding.  The blended loss is returned in the standard
   ``(loss, stats, weight)`` tuple so that the rest of the ESPnet2
   training infrastructure (Trainer, Reporter, DDP) requires zero changes.

2. **Explicit RL rollout** (``forward_rl()``)
   Returns ``(sequence_log_probs, hypothesis_texts, encoder_out)`` for use
   by alternative RL trainers (e.g. PPO, where rollouts and gradient steps
   are separated in time).  This call is gradient-enabled: the returned
   ``sequence_log_probs`` carry gradients through the CTC projection.

Architecture notes
------------------
- Encoder is shared between CE and RL branches — speech is encoded exactly
  once per forward call.
- RL branch uses **greedy CTC decoding** (``argmax`` frame-by-frame) for
  efficiency.  Beam search would produce better hypotheses for reward
  computation but is too slow for per-batch rollouts.
- The per-utterance "sequence log-prob" is computed as the sum of frame-
  level CTC log-probs along the greedy path.  Formally this is the log-
  probability of the Viterbi (most-likely) CTC alignment, not the full
  marginalised CTC probability.  For REINFORCE this gives a valid (if
  slightly biased) policy gradient estimator.
- ``interctc`` intermediate outputs are not used in the RL branch to keep
  the implementation self-contained.  Add them if needed.

Loss formula
------------
    pg_loss   = -mean_over_batch(reward_i * seq_log_prob_i)
    total_loss = (1 - rl_weight) * ce_loss + rl_weight * pg_loss

Reward
------
    reward_i = clip(1 - WER(reference_i, hypothesis_i), 0, 1)

WER is computed with ``jiwer`` (must be installed separately).  If jiwer is
not available, the RL branch is disabled and a warning is emitted.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.device_funcs import force_gatherable

try:
    import jiwer as _jiwer

    _HAS_JIWER = True
except ImportError:
    _jiwer = None
    _HAS_JIWER = False
    logging.warning(
        "jiwer is not installed. RL reward computation will be disabled. "
        "Install with: pip install jiwer"
    )


# ---------------------------------------------------------------------------
# Standalone helpers (module-level so they are picklable across workers)
# ---------------------------------------------------------------------------


def _ctc_greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int,
) -> Tuple[List[List[int]], torch.Tensor]:
    """Greedy CTC decode from frame-level log-probabilities.

    Performs argmax at each frame, then removes blank tokens and collapses
    consecutive duplicate non-blank tokens.

    Args:
        log_probs: (B, T, V) — CTC log-softmax outputs.
        lengths:   (B,)      — valid frame lengths per utterance.
        blank_id:  Index of the CTC blank token.

    Returns:
        token_seqs:  List of decoded token-ID lists (variable length, no blanks).
        seq_log_prob: (B,) sum of per-frame argmax log-probs (differentiable).
    """
    # Sum of log-probs along the greedy path — differentiable w.r.t. encoder.
    greedy_log_probs, greedy_ids = log_probs.max(dim=-1)  # (B, T) each
    seq_log_prob = torch.stack(
        [greedy_log_probs[i, : lengths[i]].sum() for i in range(log_probs.size(0))]
    )  # (B,)

    # Decode token sequences (CPU, no grad needed).
    greedy_ids_cpu = greedy_ids.detach().cpu().tolist()
    lengths_cpu = lengths.cpu().tolist()
    token_seqs: List[List[int]] = []
    for i, (ids, L) in enumerate(zip(greedy_ids_cpu, lengths_cpu)):
        prev = None
        decoded: List[int] = []
        for t in range(int(L)):
            tok = ids[t]
            if tok != blank_id and tok != prev:
                decoded.append(tok)
            prev = tok
        token_seqs.append(decoded)

    return token_seqs, seq_log_prob


def _compute_wer_reward(
    hypotheses: List[str],
    references: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Per-utterance WER-based reward tensor.

    reward_i = clip(1 - WER(ref_i, hyp_i), 0, 1)

    Returns a (B,) float tensor on ``device``.
    Requires ``jiwer`` to be installed.
    """
    if not _HAS_JIWER:
        raise RuntimeError(
            "jiwer is required for WER rewards. Install: pip install jiwer"
        )
    rewards: List[float] = []
    for hyp, ref in zip(hypotheses, references):
        if not ref.strip():
            rewards.append(0.0)
            continue
        try:
            wer = _jiwer.wer(ref, hyp if hyp.strip() else "<empty>")
        except Exception as exc:
            logging.warning(f"jiwer.wer() error ({exc}); reward defaulting to 0.")
            wer = 1.0
        rewards.append(max(0.0, 1.0 - wer))
    return torch.tensor(rewards, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class RLESPnetModel(ESPnetASRModel):
    """ESPnet ASR model with REINFORCE policy-gradient support.

    Drop-in replacement for ``ESPnetASRModel``.  All constructor arguments
    are identical; no extra hyper-parameters are needed at construction time.
    The ``rl_weight`` is passed dynamically through the batch dict during
    training.
    """

    # ------------------------------------------------------------------
    # Public forward — called by Trainer / DDP
    # ------------------------------------------------------------------

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        rl_weight: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encode speech and compute CE + (optionally) PG loss.

        Args:
            speech:         (B, T_in, ...)   — raw waveform or features.
            speech_lengths: (B,)
            text:           (B, T_ref)        — reference token IDs.
            text_lengths:   (B,)
            rl_weight:      Mixing coefficient for the PG loss [0, 1].
                            Injected into the batch dict by ``RLTrainer``.
            **kwargs:       Forwarded to the parent (e.g. ``utt_id``).

        Returns:
            Standard ESPnet2 triple: (loss, stats, batch_size_weight).
        """
        assert text_lengths.dim() == 1
        batch_size = speech.shape[0]

        text = text.clone()
        text[text == -1] = self.ignore_id
        text = text[:, : text_lengths.max()]

        # ----------------------------------------------------------------
        # 1. Encode once (shared between CE and RL branches)
        # ----------------------------------------------------------------
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        # Discard intermediate CTC outputs if present (not used in RL branch).
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # ----------------------------------------------------------------
        # 2. Supervised CE loss — reuses parent private helpers
        # ----------------------------------------------------------------
        stats: Dict[str, Optional[torch.Tensor]] = {}

        loss_ctc = loss_att = None

        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        if self.ctc_weight != 1.0 and self.decoder is not None:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        if self.ctc_weight == 0.0:
            ce_loss = loss_att
        elif self.ctc_weight == 1.0:
            ce_loss = loss_ctc
        else:
            ce_loss = self.ctc_weight * loss_ctc + (1.0 - self.ctc_weight) * loss_att

        # ----------------------------------------------------------------
        # 3. REINFORCE policy-gradient branch (skipped when rl_weight == 0)
        # ----------------------------------------------------------------
        if rl_weight > 0.0 and self.ctc is not None:
            try:
                pg_loss, reward_mean = self._compute_pg_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
                total_loss = (1.0 - rl_weight) * ce_loss + rl_weight * pg_loss
                stats["ce_loss"] = ce_loss.detach()
                stats["pg_loss"] = pg_loss.detach()
                stats["reward"] = reward_mean
            except Exception as exc:
                # Graceful fallback: if reward computation fails (e.g. jiwer
                # missing), log a warning and continue with CE only.
                logging.warning(
                    f"RL branch failed ({exc}); falling back to CE-only loss."
                )
                total_loss = ce_loss
        else:
            total_loss = ce_loss

        stats["loss"] = total_loss.detach()

        loss, stats, weight = force_gatherable(
            (total_loss, stats, batch_size), total_loss.device
        )
        return loss, stats, weight

    # ------------------------------------------------------------------
    # Explicit RL rollout (for PPO / offline use)
    # ------------------------------------------------------------------

    def forward_rl(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Run a greedy CTC rollout and return quantities needed for RL.

        Unlike ``forward()``, this method does **not** compute any loss.  It
        is intended for PPO-style trainers where rollout collection and the
        gradient step are separated.

        Under DDP, this method must be called via ``model.module.forward_rl``
        (bypassing DDP's ``__call__``), and gradients must be gathered
        manually.

        Args:
            speech:         (B, T_in, ...) — raw waveform or features.
            speech_lengths: (B,)
            text:           (B, T_ref)     — reference token IDs (for reward).
            text_lengths:   (B,)
            **kwargs:       Ignored (accepts ``utt_id`` etc. from batch).

        Returns:
            seq_log_probs: (B,) — sum of CTC log-probs along the greedy path.
                           Carries gradients; use as ``log π(a|s)`` in PG.
            hypotheses:    List[str] of decoded hypothesis strings.
            encoder_out:   (B, T_enc, D) — encoder output tensor.
        """
        assert self.ctc is not None, (
            "forward_rl() requires a CTC head (ctc_weight > 0). "
            "Set ctc_weight > 0 in model_conf."
        )

        text = text.clone()
        text[text == -1] = self.ignore_id

        # Encode
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # CTC log-softmax — (B, T, V), gradient flows through here.
        ctc_log_probs = self.ctc.log_softmax(encoder_out)

        # Greedy decode
        token_seqs, seq_log_probs = _ctc_greedy_decode(
            ctc_log_probs, encoder_out_lens, self.blank_id
        )

        # Convert token IDs to strings
        hypotheses = self._token_ids_to_text(token_seqs)

        return seq_log_probs, hypotheses, encoder_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_pg_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the REINFORCE policy-gradient loss.

        Args:
            encoder_out:      (B, T, D) — encoder output.
            encoder_out_lens: (B,)
            text:             (B, T_ref) — reference token IDs.
            text_lengths:     (B,)

        Returns:
            pg_loss:     Scalar PG loss tensor (differentiable).
            reward_mean: Scalar mean reward for logging (detached).
        """
        # Greedy CTC decode → sequence log-probs (differentiable) + hyps.
        ctc_log_probs = self.ctc.log_softmax(encoder_out)  # (B, T, V)
        token_seqs, seq_log_probs = _ctc_greedy_decode(
            ctc_log_probs, encoder_out_lens, self.blank_id
        )
        hypotheses = self._token_ids_to_text(token_seqs)

        # Decode ground-truth references from padded tensor.
        references = self._decode_references(text, text_lengths)

        # WER-based per-utterance reward (non-differentiable scalar).
        rewards = _compute_wer_reward(hypotheses, references, encoder_out.device)

        # REINFORCE: pg_loss = -E[reward * log_prob]
        # rewards is detached (no grad); seq_log_probs carries gradient.
        pg_loss = -(rewards.detach() * seq_log_probs).mean()

        return pg_loss, rewards.mean().detach()

    def _decode_references(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> List[str]:
        """Convert padded reference token-ID tensor to text strings.

        Args:
            text:         (B, T_ref) — token IDs; ignore_id padding.
            text_lengths: (B,)

        Returns:
            List of space-joined token strings, one per utterance.
        """
        refs: List[str] = []
        for i in range(text.size(0)):
            ids = text[i, : text_lengths[i]].tolist()
            tokens = []
            for tok_id in ids:
                if tok_id in (self.sos, self.eos) or tok_id < 0:
                    continue
                if tok_id < len(self.token_list):
                    tokens.append(self.token_list[tok_id])
            refs.append(" ".join(tokens))
        return refs

    def _token_ids_to_text(self, token_seqs: List[List[int]]) -> List[str]:
        """Convert lists of token IDs (already blank/repeat-collapsed) to strings.

        Args:
            token_seqs: List of decoded token-ID lists (no blanks, no repeats).

        Returns:
            List of space-joined token strings.
        """
        texts: List[str] = []
        for ids in token_seqs:
            tokens = [
                self.token_list[i]
                for i in ids
                if 0 <= i < len(self.token_list)
                and i not in (self.blank_id, self.sos, self.eos)
            ]
            texts.append(" ".join(tokens))
        return texts

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}
