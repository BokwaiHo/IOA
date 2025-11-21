"""
Training modules for IOA Framework

This package contains the training logic for knowledge distillation
as described in Algorithm 1 and Appendix E:
- Stage-wise training following the curriculum
- Mastery-based progression (Eq. 9)
- Remedial data generation loops
- Support for both full fine-tuning and LoRA
"""

from .trainer import (
    IOATrainer,
    TrainingState,
    train_curriculum
)

__all__ = [
    # Main trainer class
    "IOATrainer",
    # Training state tracking
    "TrainingState",
    # Curriculum training function
    "train_curriculum"
]