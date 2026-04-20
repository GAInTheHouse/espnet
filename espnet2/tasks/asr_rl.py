"""ASR task for RL-augmented fine-tuning (RLESPnetModel + RLTrainer).

This is a minimal subclass of ASRTask that:

1. Swaps the model class from ESPnetASRModel to RLESPnetModel at build time.
   The swap is performed by temporarily replacing the module-level name in
   espnet2.tasks.asr so that the parent build_model() logic is reused
   without duplication.

2. Sets ``trainer = RLTrainer`` so that train_one_epoch() injects RL config
   into each batch dict and the model computes the blended CE + RL loss.

Usage (via asr_train_rl.py)
---------------------------
    python -m espnet2.bin.asr_train_rl \\
        --config conf/train_asr_rl.yaml \\
        --train_data_path_and_name_and_type ... \\
        --valid_data_path_and_name_and_type ... \\
        --output_dir exp/asr_rl

RL-specific CLI arguments (registered by RLTrainer.add_arguments):
    --rl_weight             float, default 0.1
    --reward_mode           str,   default "mwer"
    --reward_loss_type      str,   default "reinforce"
    --reward_step_interval  int,   default 4
    --max_encoder_len_for_reward int, default 1500
    --domain_terms          list[str], default []
    --domain_term_weight    float, default 3.0
    --gemini_api_key        str,   default None
    --mock_llm              flag,  default False
"""

import espnet2.tasks.asr as _asr_module
from espnet2.tasks.asr import ASRTask
from espnet2.train.rl_trainer import RLTrainer


class RLASRTask(ASRTask):
    """ASRTask variant that uses RLESPnetModel and RLTrainer."""

    trainer = RLTrainer

    @classmethod
    def build_model(cls, args):
        """Build RLESPnetModel instead of ESPnetASRModel.

        Two patches are required in concert:
        1. ``model_choices`` registry — controls which class is actually
           instantiated inside build_model().
        2. Module namespace ``ESPnetASRModel`` — controls the type that
           typeguard resolves when it checks the ``-> ESPnetASRModel``
           return annotation on the parent's @typechecked build_model.
        Patching only one of the two causes a TypeCheckError (the returned
        object's type won't match what typeguard resolved for the annotation).
        """
        from espnet2.asr.rl_espnet_model import RLESPnetModel

        _orig_name = getattr(_asr_module, "ESPnetASRModel", None)
        _orig_cls = _asr_module.model_choices.classes.get("espnet")

        _asr_module.ESPnetASRModel = RLESPnetModel
        _asr_module.model_choices.classes["espnet"] = RLESPnetModel
        try:
            model = super().build_model(args)
        finally:
            if _orig_name is not None:
                _asr_module.ESPnetASRModel = _orig_name
            if _orig_cls is not None:
                _asr_module.model_choices.classes["espnet"] = _orig_cls

        return model
