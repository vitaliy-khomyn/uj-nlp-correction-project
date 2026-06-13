"""Centralized configuration for hyperparameters and project settings."""
from dataclasses import dataclass, field
import torch

# apply monkey-patches for compatibility before import of other modules
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", None)

try:
    import torch.distributed.fsdp
    if not hasattr(torch.distributed.fsdp, "FSDPModule"):
        setattr(torch.distributed.fsdp, "FSDPModule", type("FSDPModule", (), {}))
except ImportError:
    pass

try:
    import huggingface_hub.dataclasses
    huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
except ImportError:
    pass

try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except (ImportError, AttributeError):
    pass

try:
    import transformers.utils
    transformers.utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except (ImportError, AttributeError):
    pass


@dataclass
class ProjectConfig:
    """Project-wide hyperparameters and setting directories."""

    # data synthesis settings
    max_pairs: int = 50000
    max_injections_per_word: int = 750

    # model settings
    model_name: str = "allegro/plt5-small"
    max_seq_length: int = 96
    random_seed: int = 42

    # active pipelines configuration:
    # "1": small model ablation suite (10k dataset, plt5-small, output to results/small/)
    # "2": base model 10k (10k dataset, plt5-base, output to results/base/10k/)
    # "3": base model 50k (50k dataset, plt5-base, output to results/base/50k/)
    # "4": base model 50k + Herbert Reranking (50k dataset, plt5-base + Herbert reranking, output to results/base/50k_reranked/)
    active_pipelines: list[str] = field(default_factory=lambda: ["2", "3", "4"])

    # reranking configuration
    use_reranking: bool = False
    reranking_model_name: str = "dkleczek/bert-base-polish-cased-v1"
    reranking_threshold: float = 0.0

    # training hyperparameters
    train_batch_size: int = 8
    eval_batch_size: int = 32
    grad_accum_steps: int = 4
    learning_rate: float = 5e-4
    sft_num_epochs: int = 5
    num_epochs: int = 5
    resume_training: bool = True
    logging_steps: int = 10

    debug_mode: bool = False
    debug_train_samples: int = 500
    debug_val_samples: int = 100
    debug_test_samples: int = 100
    debug_human_samples: int = 50
    debug_num_epochs: int = 2

    # evaluation / inference
    infer_batch_size: int = 32
    num_beams: int = 3

    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 64
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
    )

    # ORPO configuration
    orpo_beta: float = 0.05

    # trainer settings
    early_stopping_patience: int = 2
    task_prefix: str = "Skoryguj: "

    # experiment variants
    variant_b_ratios: list[float] = field(default_factory=lambda: [0.10, 0.30])


config = ProjectConfig()
