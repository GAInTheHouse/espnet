#!/usr/bin/env python3
"""Entry point for RL-augmented ASR training (RLESPnetModel + RLTrainer).

Usage
-----
    python -m espnet2.bin.asr_train_rl --config conf/train_asr_rl.yaml ...

All arguments accepted by ``espnet2.bin.asr_train`` are supported, plus
RL-specific arguments registered by ``RLTrainer.add_arguments()``:

    --rl_weight             float   Blend weight for RL loss (default 0.1).
    --reward_mode           str     mwer | wwer | llm | all (default mwer).
    --reward_loss_type      str     reinforce | penalty (default reinforce).
    --reward_step_interval  int     Compute reward every N steps (default 4).
    --max_encoder_len_for_reward int  Skip reward for long utterances (default 1500).
    --domain_terms          str...  Domain vocabulary for wwer weighting.
    --domain_term_weight    float   Cost multiplier for domain terms (default 3.0).
    --gemini_api_key        str     Gemini API key for llm reward mode.
    --mock_llm              flag    Use mock LLM (mwer + noise) even if key is set.
"""

import sys


def main(cmd=None):
    from espnet2.tasks.asr_rl import RLASRTask

    RLASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
