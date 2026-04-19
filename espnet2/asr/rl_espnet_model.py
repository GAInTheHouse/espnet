"""RL-compatible ESPnet2 ASR model.

Extends ``ESPnetASRModel`` with two capabilities:

1. **Blended forward pass** (``forward()``)
   When ``rl_weight > 0`` is passed via the batch dict, the model encodes
   speech once, computes the standard CE loss (CTC + attention-decoder),
   and optionally computes a reward-based loss term.  The blended loss is
   returned in the standard ``(loss, stats, weight)`` tuple so that the rest
   of ESPnet2's training infrastructure requires zero changes.

2. **Explicit RL rollout** (``forward_rl()``)
   Returns ``(sequence_log_probs, hypothesis_texts, encoder_out)`` for use
   by alternative RL trainers (e.g. PPO with separated rollout/update phases).

Loss formulae
-------------
penalty (NeMo-style, ``reward_loss_type="penalty"``)::

    penalty    = 1 - mean(reward)
    total_loss = ce_loss + rl_weight * penalty

reinforce (REINFORCE policy gradient, ``reward_loss_type="reinforce"``)::

    pg_loss    = -mean(reward * seq_log_prob.float())   # FP32 cast: AMP safety
    total_loss = (1 - rl_weight) * ce_loss + rl_weight * pg_loss

reweight_ctc (NeMo actual run objective, ``reward_loss_type="reweight_ctc"``)::

    penalty_i          = 1 - reward_i          # per-utterance (high WER → high weight)
    reweighted_ctc     = mean_i[penalty_i * ctc_loss_per_utt_i]   # reduction='none'
    total_loss         = reweighted_ctc + (1 - ctc_weight) * loss_att

    This is the objective used in the actual NeMo run (reward_weight=0.02, wwer mode).
    Unlike ``penalty`` which adds a scalar correction term, ``reweight_ctc`` selectively
    back-propagates harder through high-WER utterances.  Unlike ``reinforce`` it does not
    require log-prob gradients.  Greedy decode still runs under ``torch.no_grad()``.

Reward modes (``reward_mode``)
------------------------------
mwer   Standard WER-based reward:  reward = clip(1 - WER(ref, hyp), 0, 1).
wwer   Domain-weighted WER:  domain terms incur ``domain_term_weight`` × normal cost.
llm    Gemini-1.5-flash quality score [0, 1]; mock fallback = mwer + N(0, 0.05).
all    Element-wise mean of mwer, wwer, and llm reward tensors.

GPU notes
---------
- ``penalty`` / ``reweight_ctc`` modes: greedy decode runs under
  ``torch.no_grad()`` because ``seq_log_probs`` are never used in the loss,
  preventing unnecessary activation retention for the ``(B, T, V)`` CTC graph.
  For ``reweight_ctc``, the per-utterance CTC loss is then recomputed
  separately with gradients via ``F.ctc_loss(reduction='none')``.
- ``reinforce`` mode: ``seq_log_probs.float()`` casts to FP32 before the PG
  loss to prevent underflow when running under AMP (FP16 has ~1-bit precision
  for sums of ~900 log-prob units, causing silent NaN gradients).
- Reward computation is skipped (cached neutral 0.5 reused) when
  ``compute_reward=False`` (per-step interval from trainer) or when
  ``encoder_out.shape[1] > max_encoder_len_for_reward`` (long-utterance guard).
"""

import contextlib
import logging
import random as _random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.device_funcs import force_gatherable

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

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

try:
    import google.generativeai as _genai

    _HAS_GENAI = True
except ImportError:
    _genai = None
    _HAS_GENAI = False


# ---------------------------------------------------------------------------
# Module-level helpers (picklable across DataLoader workers)
# ---------------------------------------------------------------------------


def _ctc_greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int,
) -> Tuple[List[List[int]], torch.Tensor]:
    """Greedy CTC decode from frame-level log-probabilities.

    Args:
        log_probs: (B, T, V) CTC log-softmax outputs.
        lengths:   (B,) valid frame lengths per utterance.
        blank_id:  Index of the CTC blank token.

    Returns:
        token_seqs:   List of decoded token-ID lists (blank/repeat collapsed).
        seq_log_prob: (B,) sum of per-frame argmax log-probs (differentiable).
    """
    greedy_log_probs, greedy_ids = log_probs.max(dim=-1)  # (B, T) each
    seq_log_prob = torch.stack(
        [greedy_log_probs[i, : lengths[i]].sum() for i in range(log_probs.size(0))]
    )  # (B,)

    # Token sequence decoding (CPU, no grad needed)
    greedy_ids_cpu = greedy_ids.detach().cpu().tolist()
    lengths_cpu = lengths.cpu().tolist()
    token_seqs: List[List[int]] = []
    for ids, L in zip(greedy_ids_cpu, lengths_cpu):
        prev = None
        decoded: List[int] = []
        for t in range(int(L)):
            tok = ids[t]
            if tok != blank_id and tok != prev:
                decoded.append(tok)
            prev = tok
        token_seqs.append(decoded)

    return token_seqs, seq_log_prob


def _compute_mwer(
    hypotheses: List[str],
    references: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Standard WER-based reward: reward_i = clip(1 - WER(ref_i, hyp_i), 0, 1)."""
    if not _HAS_JIWER:
        raise RuntimeError("jiwer required for mwer reward. pip install jiwer")
    rewards: List[float] = []
    for hyp, ref in zip(hypotheses, references):
        if not ref.strip():
            rewards.append(0.0)
            continue
        try:
            wer = _jiwer.wer(ref, hyp if hyp.strip() else "<empty>")
        except Exception as exc:
            logging.warning("jiwer.wer() error (%s); reward=0.", exc)
            wer = 1.0
        rewards.append(max(0.0, 1.0 - wer))
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def _compute_wwer(
    hypotheses: List[str],
    references: List[str],
    device: torch.device,
    domain_set: frozenset,
    domain_term_weight: float,
) -> torch.Tensor:
    """Domain-weighted WER reward.

    Domain terms in the reference that are substituted or deleted incur cost
    ``domain_term_weight`` instead of 1.0.  Insertions always cost 1.0.

    reward_i = clip(1 - weighted_error_rate_i, 0, 1)
    """
    if not _HAS_JIWER:
        raise RuntimeError("jiwer required for wwer reward. pip install jiwer")
    rewards: List[float] = []
    for hyp, ref in zip(hypotheses, references):
        if not ref.strip():
            rewards.append(0.0)
            continue
        try:
            ref_words = ref.lower().split()
            hyp_str = hyp if hyp.strip() else "<empty>"
            out = _jiwer.process_words(ref, hyp_str)

            # Total weighted reference length
            total_ref_weight = sum(
                domain_term_weight if w in domain_set else 1.0 for w in ref_words
            )
            if total_ref_weight == 0:
                rewards.append(0.0)
                continue

            # Walk alignment chunks and accumulate weighted errors
            weighted_errors = 0.0
            alignments = out.alignments[0] if out.alignments else []
            for chunk in alignments:
                if chunk.type in ("substitute", "delete"):
                    chunk_ref = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
                    for w in chunk_ref:
                        weighted_errors += (
                            domain_term_weight if w in domain_set else 1.0
                        )
                elif chunk.type == "insert":
                    weighted_errors += chunk.hyp_end_idx - chunk.hyp_start_idx

            rate = min(1.0, weighted_errors / total_ref_weight)
            rewards.append(max(0.0, 1.0 - rate))
        except Exception as exc:
            logging.warning("wwer failed (%s); falling back to mwer.", exc)
            try:
                wer = _jiwer.wer(ref, hyp if hyp.strip() else "<empty>")
                rewards.append(max(0.0, 1.0 - wer))
            except Exception:
                rewards.append(0.0)
    return torch.tensor(rewards, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class RLESPnetModel(ESPnetASRModel):
    """ESPnet ASR model with multi-mode reward-augmented fine-tuning support.

    Drop-in replacement for ``ESPnetASRModel``.  All parent constructor
    arguments are supported; the RL-specific arguments below are new.

    Args:
        reward_mode: Reward function to use — ``mwer``, ``wwer``, ``llm``,
            or ``all`` (element-wise mean of mwer, wwer, llm).
        reward_loss_type: Loss formula — ``reinforce`` (REINFORCE PG; default)
            or ``penalty`` (NeMo-style auxiliary penalty; no policy gradient).
        domain_terms: Vocabulary list for ``wwer`` weighting.
        domain_term_weight: Edit-cost multiplier for domain terms (default 3.0).
        max_encoder_len_for_reward: Skip reward computation for utterances
            whose encoder output exceeds this frame count (GPU memory guard).
        gemini_api_key: Gemini API key for the ``llm`` reward mode.
            Falls back to mock (mwer + noise) if None or if API call fails.
        mock_llm: Force mock LLM path even when ``gemini_api_key`` is set.
            Useful for smoke-testing without billing.
        All other args are forwarded verbatim to ``ESPnetASRModel.__init__``.
    """

    def __init__(
        self,
        *args,
        reward_mode: str = "mwer",
        reward_loss_type: str = "reinforce",
        domain_terms: Optional[List[str]] = None,
        domain_term_weight: float = 3.0,
        max_encoder_len_for_reward: int = 1500,
        gemini_api_key: Optional[str] = None,
        mock_llm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reward_mode = reward_mode
        self.reward_loss_type = reward_loss_type
        self.domain_terms: List[str] = list(domain_terms) if domain_terms else []
        self._domain_set: frozenset = frozenset(t.lower() for t in self.domain_terms)
        self.domain_term_weight = domain_term_weight
        self.max_encoder_len_for_reward = max_encoder_len_for_reward
        self.gemini_api_key = gemini_api_key or ""
        self.mock_llm = mock_llm

        # Cached neutral reward (0.5), updated each time reward is computed.
        # Registered as a buffer so it moves with the model to the right device.
        self.register_buffer("_cached_reward", torch.tensor([0.5]))

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
        compute_reward: bool = True,
        reward_mode: Optional[str] = None,
        reward_loss_type: Optional[str] = None,
        max_encoder_len_for_reward: Optional[int] = None,
        domain_terms: Optional[List[str]] = None,
        domain_term_weight: Optional[float] = None,
        gemini_api_key: Optional[str] = None,
        mock_llm: Optional[bool] = None,
        log_reward_samples: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encode speech and compute CE + (optionally) reward-augmented loss.

        Per-step values injected by ``RLTrainer`` via the batch dict override
        the instance defaults set at construction time.

        Args:
            speech:                   (B, T_in, ...) raw waveform or features.
            speech_lengths:           (B,)
            text:                     (B, T_ref) reference token IDs.
            text_lengths:             (B,)
            rl_weight:                Blend weight [0, 1].  0 = CE only.
            compute_reward:           False = skip reward computation this step
                                      and reuse cached reward (interval guard).
            reward_mode:              Override instance ``reward_mode``.
            reward_loss_type:         Override instance ``reward_loss_type``.
            max_encoder_len_for_reward: Override instance value.
            domain_terms:             Override instance domain term list.
            domain_term_weight:       Override instance domain term weight.
            gemini_api_key:           Override instance API key.
            mock_llm:                 Override instance mock flag.
            log_reward_samples:       If True, log up to 10 (utt_id, ref, hyp,
                                      reward) sample tuples at INFO level.
                                      Injected by RLTrainer on dump steps.
            **kwargs:                 Forwarded to parent (utt_id, etc.).

        Returns:
            Standard ESPnet2 triple: (loss, stats, batch_size_weight).
        """
        # Resolve per-step overrides (None → use instance default)
        reward_mode = reward_mode if reward_mode is not None else self.reward_mode
        reward_loss_type = (
            reward_loss_type
            if reward_loss_type is not None
            else self.reward_loss_type
        )
        max_enc_len = (
            max_encoder_len_for_reward
            if max_encoder_len_for_reward is not None
            else self.max_encoder_len_for_reward
        )
        d_terms: List[str] = list(domain_terms) if domain_terms is not None else self.domain_terms
        d_weight = domain_term_weight if domain_term_weight is not None else self.domain_term_weight
        api_key = gemini_api_key if gemini_api_key is not None else self.gemini_api_key
        use_mock = mock_llm if mock_llm is not None else self.mock_llm
        domain_set = frozenset(t.lower() for t in d_terms)

        assert text_lengths.dim() == 1
        batch_size = speech.shape[0]

        text = text.clone()
        text[text == -1] = self.ignore_id
        text = text[:, : text_lengths.max()]

        # ------------------------------------------------------------
        # 1. Encode once (shared between CE and RL branches)
        # ------------------------------------------------------------
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # ------------------------------------------------------------
        # 2. Supervised CE loss
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # 3. Reward-augmented RL branch (skipped when rl_weight == 0)
        # ------------------------------------------------------------
        if rl_weight > 0.0 and self.ctc is not None:
            try:
                total_loss = self._compute_rl_loss(
                    ce_loss=ce_loss,
                    loss_att=loss_att,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    text=text,
                    text_lengths=text_lengths,
                    rl_weight=rl_weight,
                    compute_reward=compute_reward,
                    reward_mode=reward_mode,
                    reward_loss_type=reward_loss_type,
                    max_enc_len=max_enc_len,
                    domain_set=domain_set,
                    d_weight=d_weight,
                    api_key=api_key,
                    use_mock=use_mock,
                    log_reward_samples=log_reward_samples,
                    utt_ids=kwargs.get("utt_id", []),
                    stats=stats,
                )
            except Exception as exc:
                logging.warning(
                    "RL branch failed (%s); falling back to CE-only loss.", exc
                )
                total_loss = ce_loss
        else:
            total_loss = ce_loss

        stats["loss"] = total_loss.detach()
        stats["ce_loss"] = ce_loss.detach()

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
        """Greedy CTC rollout — returns quantities needed for RL.

        Under DDP, call via ``model.module.forward_rl`` to bypass DDP's
        all-reduce hook and gather gradients manually.

        Returns:
            seq_log_probs: (B,) sum of CTC log-probs along greedy path.
                           Gradient-enabled; use as ``log π(a|s)`` in PG.
            hypotheses:    Decoded hypothesis strings.
            encoder_out:   (B, T_enc, D) encoder output.
        """
        assert self.ctc is not None, (
            "forward_rl() requires a CTC head.  Set ctc_weight > 0."
        )
        text = text.clone()
        text[text == -1] = self.ignore_id

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        token_seqs, seq_log_probs = _ctc_greedy_decode(
            ctc_log_probs, encoder_out_lens, self.blank_id
        )
        hypotheses = self._token_ids_to_text(token_seqs)
        return seq_log_probs, hypotheses, encoder_out

    # ------------------------------------------------------------------
    # Private RL helpers
    # ------------------------------------------------------------------

    def _compute_rl_loss(
        self,
        ce_loss: torch.Tensor,
        loss_att: Optional[torch.Tensor],
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        rl_weight: float,
        compute_reward: bool,
        reward_mode: str,
        reward_loss_type: str,
        max_enc_len: int,
        domain_set: frozenset,
        d_weight: float,
        api_key: str,
        use_mock: bool,
        stats: dict,
        log_reward_samples: bool = False,
        utt_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute blended CE + RL loss and populate ``stats``.

        Args:
            loss_att: Attention-decoder loss (batch mean); used by
                ``reweight_ctc`` mode which bypasses the blended ``ce_loss``
                and recomputes the CTC component per-utterance.
        """
        device = encoder_out.device
        batch_size = encoder_out.size(0)

        # --- Greedy decode for reward (no_grad for penalty and reweight_ctc) ---
        # Both penalty and reweight_ctc never use seq_log_probs in the loss, so
        # we avoid retaining the full (B, T, V) CTC activation graph.
        _decode_no_grad = reward_loss_type in ("penalty", "reweight_ctc")
        _decode_ctx = (
            torch.no_grad() if _decode_no_grad else contextlib.nullcontext()
        )
        with _decode_ctx:
            ctc_log_probs = self.ctc.log_softmax(encoder_out)
            token_seqs, seq_log_probs = _ctc_greedy_decode(
                ctc_log_probs, encoder_out_lens, self.blank_id
            )

        hypotheses = self._token_ids_to_text(token_seqs)
        references = self._decode_references(text, text_lengths)

        # --- Reward caching (interval + long-utterance guards) ---
        if not compute_reward or encoder_out.size(1) > max_enc_len:
            rewards = self._cached_reward.expand(batch_size).to(device)
        else:
            rewards = self._dispatch_reward(
                hypotheses=hypotheses,
                references=references,
                device=device,
                reward_mode=reward_mode,
                domain_set=domain_set,
                d_weight=d_weight,
                api_key=api_key,
                use_mock=use_mock,
            )
            self._cached_reward = rewards.mean().detach().unsqueeze(0)

            # --- Diagnostic sample dump (NeMo reward_sample_dump_interval) ---
            if log_reward_samples:
                n_show = min(10, len(hypotheses))
                ids = (
                    list(utt_ids)[:n_show]
                    if utt_ids
                    else [f"batch[{i}]" for i in range(n_show)]
                )
                for i in range(n_show):
                    logging.info(
                        "[RL reward sample] utt=%s  reward=%.4f  "
                        "ref=%r  hyp=%r",
                        ids[i],
                        rewards[i].item(),
                        references[i][:80],
                        hypotheses[i][:80],
                    )

        stats["reward_mean"] = rewards.mean().detach()
        stats["reward_std"] = rewards.std().detach()

        # --- Loss formula ---
        if reward_loss_type == "penalty":
            # NeMo-style auxiliary penalty: scalar correction, no PG
            penalty = 1.0 - rewards.mean()
            total_loss = ce_loss + rl_weight * penalty
            stats["penalty"] = penalty.detach()

        elif reward_loss_type == "reweight_ctc":
            # Per-utterance CTC reweighting — the objective used in the actual
            # NeMo run (reward_weight=0.02, wwer mode).
            # Gradient flows through ctc_per_utt; greedy decode is already done.
            # FP32 cast guards against AMP underflow in the CTC log-softmax.
            ctc_logits = self.ctc.ctc_lo(encoder_out).log_softmax(-1).float()
            ctc_per_utt = F.ctc_loss(
                ctc_logits.transpose(0, 1),       # (T, B, V)
                text[:, : text_lengths.max()],
                encoder_out_lens.clamp(min=1),
                text_lengths.clamp(min=1),
                blank=self.blank_id,
                reduction="none",
                zero_infinity=True,
            )  # (B,)
            penalty_per_utt = (1.0 - rewards.detach()).clamp(0.0, 1.0)
            reweighted_ctc = (penalty_per_utt * ctc_per_utt).mean()
            att_component = (
                loss_att
                if loss_att is not None
                else torch.zeros(1, device=device)
            )
            total_loss = reweighted_ctc + (1.0 - self.ctc_weight) * att_component
            stats["reweighted_ctc"] = reweighted_ctc.detach()

        else:
            # REINFORCE — GPU fix 2: FP32 cast prevents AMP underflow
            # seq_log_probs are sums over T frames; FP16 loses precision at ~±900
            pg_loss = -(rewards.detach() * seq_log_probs.float()).mean()
            total_loss = (1.0 - rl_weight) * ce_loss + rl_weight * pg_loss
            stats["pg_loss"] = pg_loss.detach()

        return total_loss

    def _dispatch_reward(
        self,
        hypotheses: List[str],
        references: List[str],
        device: torch.device,
        reward_mode: str,
        domain_set: frozenset,
        d_weight: float,
        api_key: str,
        use_mock: bool,
    ) -> torch.Tensor:
        """Route to the correct reward function based on ``reward_mode``."""
        if reward_mode == "mwer":
            return _compute_mwer(hypotheses, references, device)

        if reward_mode == "wwer":
            return _compute_wwer(
                hypotheses, references, device, domain_set, d_weight
            )

        if reward_mode == "llm":
            return self._compute_llm_reward(
                hypotheses, references, device, api_key, use_mock
            )

        if reward_mode == "all":
            r_mwer = _compute_mwer(hypotheses, references, device)
            r_wwer = _compute_wwer(
                hypotheses, references, device, domain_set, d_weight
            )
            r_llm = self._compute_llm_reward(
                hypotheses, references, device, api_key, use_mock
            )
            return (r_mwer + r_wwer + r_llm) / 3.0

        logging.warning("Unknown reward_mode=%s; falling back to mwer.", reward_mode)
        return _compute_mwer(hypotheses, references, device)

    def _compute_llm_reward(
        self,
        hypotheses: List[str],
        references: List[str],
        device: torch.device,
        api_key: str,
        use_mock: bool,
    ) -> torch.Tensor:
        """Gemini-1.5-flash quality score [0, 1] per utterance.

        Falls back to mwer + Gaussian noise (mock path) when:
        - ``use_mock=True`` (smoke-test mode)
        - ``api_key`` is empty or not provided
        - ``google-generativeai`` is not installed
        - The Gemini API call raises any exception
        """
        if not _HAS_JIWER:
            raise RuntimeError("jiwer required even for llm reward (mock fallback).")

        rewards: List[float] = []

        can_use_live = _HAS_GENAI and bool(api_key) and not use_mock

        for hyp, ref in zip(hypotheses, references):
            if not ref.strip():
                rewards.append(0.0)
                continue

            score: Optional[float] = None

            if can_use_live:
                try:
                    _genai.configure(api_key=api_key)
                    model = _genai.GenerativeModel("gemini-1.5-flash")
                    prompt = (
                        "You are evaluating an automatic speech recognition (ASR) hypothesis.\n"
                        f'Reference: "{ref}"\n'
                        f'Hypothesis: "{hyp if hyp.strip() else "<empty>"}"\n'
                        "Rate how closely the hypothesis matches the reference on a scale "
                        "from 0.0 (completely wrong) to 1.0 (perfect match).\n"
                        "Consider word accuracy, domain-specific term correctness, and "
                        "overall meaning preservation.\n"
                        "Reply with a single decimal number between 0.0 and 1.0 only."
                    )
                    response = model.generate_content(prompt)
                    score = float(response.text.strip())
                    score = max(0.0, min(1.0, score))
                except Exception as exc:
                    logging.warning(
                        "Gemini API call failed (%s); using mock fallback.", exc
                    )
                    score = None

            if score is None:
                # Mock path: mwer + small Gaussian noise, clamped to [0, 1]
                try:
                    wer = _jiwer.wer(ref, hyp if hyp.strip() else "<empty>")
                    base = max(0.0, 1.0 - wer)
                except Exception:
                    base = 0.5
                noise = _random.gauss(0.0, 0.05)
                score = max(0.0, min(1.0, base + noise))

            rewards.append(score)

        return torch.tensor(rewards, dtype=torch.float32, device=device)

    def _decode_references(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> List[str]:
        """Convert padded reference token-ID tensor to text strings."""
        refs: List[str] = []
        for i in range(text.size(0)):
            ids = text[i, : text_lengths[i]].tolist()
            tokens = [
                self.token_list[tok_id]
                for tok_id in ids
                if tok_id not in (self.sos, self.eos)
                and tok_id >= 0
                and tok_id < len(self.token_list)
            ]
            refs.append(" ".join(tokens))
        return refs

    def _token_ids_to_text(self, token_seqs: List[List[int]]) -> List[str]:
        """Convert blank/repeat-collapsed token-ID lists to strings."""
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
