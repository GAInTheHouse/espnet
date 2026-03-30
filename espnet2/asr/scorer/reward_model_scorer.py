"""Reward model scorer for beam search injection.

This module provides ``RewardModelScorer``, a :class:`ScorerInterface`
implementation that wraps an arbitrary reward function and injects its signal
as an additive bonus into every beam hypothesis at each decoding step.

Architecture fit
----------------
ESPnet2's beam search accumulates a weighted sum of scorer log-probs at each
token step::

    weighted_score[v] = Σ_k  weight_k * scorer_k.score(y, state, x)[v]

``RewardModelScorer`` participates in this sum as a **full scorer** (it
returns a score for every vocabulary token).  The reward function is called
with the current partial hypothesis (decoded to text) and returns a scalar
bonus that is broadcast uniformly across all next tokens.  This is equivalent
to a hypothesis-level bonus rather than a token-level preference, which is
the correct semantics for sentence-level reward models (e.g. a LM perplexity
or WER estimator).

Injection via YAML config
-------------------------
Add the scorer to your inference config::

    # conf/decode_asr_with_reward.yaml
    beam_size: 10
    ctc_weight: 0.3
    # scorer weights — the reward_model key must match the name used when
    # registering the scorer in Speech2Text.__init__
    beam_search_conf:
      weights:
        decoder: 0.7
        ctc: 0.3
        length_bonus: 0.0
        reward_model: 0.1

And in the inference script::

    from espnet2.asr.scorer.reward_model_scorer import (
        RewardModelScorer, mock_no_oov_reward_fn
    )
    scorers["reward_model"] = RewardModelScorer(
        reward_fn=mock_no_oov_reward_fn,
        token_list=token_list,
        tokenizer=tokenizer,
        vocab_size=len(token_list),
    )
    weights["reward_model"] = 0.1
    # Then pass scorers and weights to BeamSearch as usual.

Scorer contract
---------------
``score(y, state, x)`` is called by ``BeamSearch.score_full()`` once per
hypothesis per beam step.  The arguments are:

* ``y`` — 1-D int64 tensor of prefix token IDs (including ``<sos>``).
* ``state`` — scorer state from the previous call (passed through unchanged
  here since the reward model is stateless).
* ``x`` — 2-D encoder feature tensor ``(T, D)`` (not used by this scorer).

The method returns ``(scores, state)`` where ``scores`` is a 1-D float32
tensor of shape ``(n_vocab,)`` containing the bonus for each possible next
token.  Because the reward is hypothesis-level, all tokens receive the same
bonus.
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

import torch

from espnet2.legacy.nets.scorer_interface import ScorerInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock reward function (for testing / demonstration)
# ---------------------------------------------------------------------------


def mock_no_oov_reward_fn(partial_text: str) -> float:
    """Mock reward: +0.1 if the partial hypothesis contains no OOV tokens.

    An "OOV token" is defined here as any sub-word piece starting with
    ``<unk>`` (the literal string, as typically appears when a token is
    unknown to the BPE vocabulary).  This simulates a language model reward
    that penalises unlikely word forms.

    In a real system this could be replaced by:
    - A neural LM perplexity score.
    - A semantic similarity score from a sentence encoder.
    - A WER estimate from a small reference model.

    Args:
        partial_text: The decoded partial hypothesis as a plain string.

    Returns:
        float: Reward bonus in an arbitrary range (typically [-1, 1] or [0, 1]).
    """
    if "<unk>" in partial_text.lower():
        return 0.0
    return 0.1


# ---------------------------------------------------------------------------
# RewardModelScorer
# ---------------------------------------------------------------------------


class RewardModelScorer(ScorerInterface):
    """Beam scorer that injects a reward function signal as a uniform bonus.

    The reward function is called once per hypothesis per beam step with the
    partial hypothesis decoded to text.  Its scalar return value is broadcast
    uniformly across all vocabulary positions so that the bonus applies
    regardless of which next token is chosen — the reward model evaluates
    the current prefix, not the next token.

    This is appropriate for:
    - Sentence-level LM rewards (perplexity of prefix).
    - Domain classifier scores (is this prefix domain-relevant?).
    - Fluency estimators.

    For **token-level** rewards (e.g. a neural LM that predicts P(next_token |
    context)), override ``score()`` to return a non-uniform score vector
    instead.

    Args:
        reward_fn:  Callable ``(partial_text: str) -> float``.  Must be fast
            (called once per hypothesis per step during beam search).
        token_list: List of token strings (index = token ID).  Used to
            detokenize prefix IDs to text.
        tokenizer:  Optional ESPnet tokenizer instance with a ``tokens2text``
            method for converting token strings to words.  If None, tokens are
            joined with spaces.
        vocab_size: Vocabulary size (sets the width of the returned score
            tensor).
        blank_id:   ID of the CTC blank token (excluded from detokenisation).
        sos_id:     Start-of-sequence token ID (excluded from detokenisation).
        eos_id:     End-of-sequence token ID (excluded from detokenisation).
    """

    def __init__(
        self,
        reward_fn: Callable[[str], float],
        token_list: List[str],
        tokenizer=None,
        vocab_size: Optional[int] = None,
        blank_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        self.reward_fn = reward_fn
        self.token_list = token_list
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size if vocab_size is not None else len(token_list)
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id

    # ------------------------------------------------------------------
    # ScorerInterface implementation
    # ------------------------------------------------------------------

    def init_state(self, x: torch.Tensor) -> Any:
        """Return initial scorer state (stateless; always None)."""
        return None

    def select_state(self, state: Any, i: int, new_id: int = None) -> Any:
        """Select state for hypothesis pruning (stateless; pass-through)."""
        return None

    def score(
        self,
        y: torch.Tensor,
        state: Any,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Score next token given prefix ``y`` using the reward function.

        Args:
            y:     1-D int64 prefix token tensor (including <sos>).
            state: Previous scorer state (None for this stateless scorer).
            x:     Encoder output (T, D). Not used.

        Returns:
            Tuple of:
                scores: (n_vocab,) float32 tensor — uniform bonus across all
                        next tokens equal to ``reward_fn(detokenized_prefix)``.
                state:  None (stateless scorer).
        """
        partial_text = self._detokenize(y)
        try:
            bonus = float(self.reward_fn(partial_text))
        except Exception as exc:
            logger.warning(f"reward_fn raised {exc}; defaulting to 0.")
            bonus = 0.0

        scores = torch.full(
            (self.vocab_size,),
            fill_value=bonus,
            dtype=x.dtype,
            device=x.device,
        )
        return scores, None

    def final_score(self, state: Any) -> float:
        """Final EOS bonus (not used; reward applied at each step already)."""
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detokenize(self, y: torch.Tensor) -> str:
        """Convert a prefix token-ID tensor to a plain text string.

        Skips ``<sos>``, ``<eos>``, and blank tokens.

        Args:
            y: 1-D int64 token IDs tensor.

        Returns:
            Decoded text string.
        """
        skip_ids = {self.blank_id, self.sos_id, self.eos_id}
        ids = [
            int(t) for t in y.tolist()
            if int(t) not in skip_ids and int(t) < len(self.token_list)
        ]
        tokens = [self.token_list[i] for i in ids]

        if self.tokenizer is not None:
            try:
                return self.tokenizer.tokens2text(tokens)
            except Exception:
                pass

        # Fallback: join sub-word pieces heuristically.
        # BPE pieces starting with "▁" (sentencepiece word boundary) or "##"
        # (WordPiece) are handled so the joined string looks like natural text.
        text_parts: List[str] = []
        for tok in tokens:
            if tok.startswith("▁"):
                text_parts.append(" " + tok[1:])
            elif tok.startswith("##"):
                text_parts.append(tok[2:])
            else:
                text_parts.append(tok)
        return "".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Example: how to inject RewardModelScorer into BeamSearch
# ---------------------------------------------------------------------------

_INJECTION_EXAMPLE = """
# ── In your inference script (or a subclass of Speech2Text) ──────────────

from espnet2.asr.scorer.reward_model_scorer import (
    RewardModelScorer,
    mock_no_oov_reward_fn,
)
from espnet2.legacy.nets.beam_search import BeamSearch

# After building the standard scorers dict in Speech2Text.__init__:
scorers["reward_model"] = RewardModelScorer(
    reward_fn=mock_no_oov_reward_fn,
    token_list=token_list,       # asr_model.token_list
    tokenizer=tokenizer,         # speech2text.tokenizer
    vocab_size=len(token_list),
    blank_id=asr_model.blank_id,
    sos_id=asr_model.sos,
    eos_id=asr_model.eos,
)

weights = dict(
    decoder=1.0 - ctc_weight,   # e.g. 0.7
    ctc=ctc_weight,              # e.g. 0.3
    lm=lm_weight,                # e.g. 0.0
    ngram=ngram_weight,          # e.g. 0.0
    length_bonus=penalty,        # e.g. 0.0
    reward_model=0.1,            # ← new RL reward weight
)

beam_search = BeamSearch(
    beam_size=beam_size,
    weights=weights,
    scorers=scorers,
    sos=asr_model.sos,
    eos=asr_model.eos,
    vocab_size=len(token_list),
    token_list=token_list,
    pre_beam_score_key=None if ctc_weight == 1.0 else "full",
)

# ── Equivalent YAML config (decode_asr_with_reward.yaml) ─────────────────
# beam_size: 10
# ctc_weight: 0.3
# lm_weight: 0.0
# penalty: 0.0       # weight for length_bonus scorer
#
# # There is no native YAML key for custom scorers in the base inference
# # script.  You must subclass Speech2Text or modify asr_inference.py to
# # inject the scorer before BeamSearch is constructed.
# # The weight mapping corresponds 1:1 to the 'weights' dict above.
#
# beam_search_conf:
#   scorers:
#     decoder: full
#     ctc: 0.3
#     reward_model: 0.1   # weight for injected reward scorer
"""
