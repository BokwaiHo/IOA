"""
Evaluation modules for IOA Framework

This package contains evaluation metrics and benchmarking utilities
as described in Section 4.1 and Appendix D:
- ROUGE-L for instruction following (DollyEval, VicunaEval)
- Pass@k for reasoning (GSM8K, MATH, AIME2024)
- Pass@k for code generation (HumanEval, MBPP, LiveCodeBench)
- Accuracy for academic QA (GPQA-Diamond)
"""

from .evaluator import (
    Evaluator,
    evaluate_distillation_quality,
    compare_with_baselines
)

__all__ = [
    # Main evaluator class
    "Evaluator",
    # Utility functions
    "evaluate_distillation_quality",
    "compare_with_baselines"
]