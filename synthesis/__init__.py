"""
Synthesis modules for IOA Framework

This package contains prompting templates and data synthesis utilities.
"""

from .prompts import (
    SYSTEM_PROMPT_SYNTHESIS,
    SYSTEM_PROMPT_REMEDIAL,
    SYSTEM_PROMPT_BRIDGING,
    get_synthesis_user_prompt,
    get_remedial_prompt,
    get_bridging_prompt,
    get_difficulty_constraints,
    get_math_synthesis_template,
    get_code_synthesis_template,
    create_few_shot_examples,
    format_adapter_flags
)

from .synthesizer import (
    DataSynthesizer,
    create_synthesizer
)

__all__ = [
    # Prompts
    "SYSTEM_PROMPT_SYNTHESIS",
    "SYSTEM_PROMPT_REMEDIAL",
    "SYSTEM_PROMPT_BRIDGING",
    "get_synthesis_user_prompt",
    "get_remedial_prompt",
    "get_bridging_prompt",
    "get_difficulty_constraints",
    "get_math_synthesis_template",
    "get_code_synthesis_template",
    "create_few_shot_examples",
    "format_adapter_flags",
    # Synthesizer
    "DataSynthesizer",
    "create_synthesizer"
]