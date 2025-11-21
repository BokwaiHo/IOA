"""
Configuration modules for IOA Framework

This package contains all configuration dataclasses and hyperparameters
as specified in the paper:
- Section 3.2: Identifier thresholds (τ_gap, τ_high, τ_low, τ_dep, α)
- Section 3.3: Organizer thresholds (τ_ZPD, τ_mastery)
- Section 4.1 & Appendix E: Training hyperparameters
"""

from .config import (
    IOAConfig,
    IdentifierConfig,
    OrganizerConfig,
    AdapterConfig,
    TrainingConfig,
    EvaluationConfig,
    ModelConfig,
    DataConfig,
    get_default_config,
    get_config_for_model_size
)

__all__ = [
    # Main config
    "IOAConfig",
    # Sub-configs
    "IdentifierConfig",
    "OrganizerConfig",
    "AdapterConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ModelConfig",
    "DataConfig",
    # Factory functions
    "get_default_config",
    "get_config_for_model_size"
]