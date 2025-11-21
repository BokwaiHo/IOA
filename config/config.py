"""
IOA Framework Configuration

This module contains all hyperparameters and configuration settings
as specified in the paper "Pedagogically-Inspired Data Synthesis for
Language Model Knowledge Distillation".

References:
- Section 3.2: Identifier thresholds (τ_gap, τ_high, τ_low, τ_dep, α)
- Section 3.3: Organizer thresholds (τ_ZPD, τ_mastery)
- Section 4.1 & Appendix E: Training hyperparameters
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class IdentifierConfig:
    """Configuration for the Knowledge Identifier module (Section 3.2)"""
    
    # Performance gap threshold for identifying deficient knowledge modules
    # Knowledge modules are classified as deficient when Δ(k) > τ_gap (Eq. 2)
    tau_gap: float = 0.3
    
    # Thresholds for determining if student has mastered prerequisite knowledge
    # Used in dependency strength calculation (Eq. 3)
    tau_high: float = 0.9  # High mastery threshold
    tau_low: float = 0.7   # Low mastery threshold
    
    # Dependency strength threshold for graph construction
    # Dependencies with strength > τ_dep are included in the graph
    tau_dep: float = 0.3
    
    # Weight for performance gap vs. structural importance in severity score (Eq. 4)
    # Higher α prioritizes absolute performance gaps
    alpha: float = 0.7
    
    # Epsilon for numerical stability in dependency calculation
    epsilon: float = 1e-8
    
    # Percentage of deficient modules to target (dynamic, typically 20-30%)
    target_module_percentage: float = 0.25


@dataclass
class OrganizerConfig:
    """Configuration for the Knowledge Organizer module (Section 3.3)"""
    
    # Zone of Proximal Development threshold
    # Controls difficulty increment between consecutive stages (Eq. 8)
    # τ_ZPD = 0.15 ensures difficulty increases remain within student's learning capacity
    tau_zpd: float = 0.15
    
    # Mastery threshold for progressive learning (Eq. 9)
    # Student must achieve τ_mastery of teacher's performance before advancing
    tau_mastery: float = 0.9
    
    # Maximum number of remedial iterations per stage
    max_remedial_iterations: int = 5


@dataclass
class AdapterConfig:
    """Configuration for the Knowledge Adapter module (Section 3.4)"""
    
    # Number of synthetic samples to generate per seed data item
    # J_i in Equation 1
    num_samples_per_seed: int = 10
    
    # Maximum tokens for generated content
    max_generation_tokens: int = 2048
    
    # Temperature for generation
    generation_temperature: float = 0.7
    
    # Whether to enable verification filtering
    enable_verification: bool = True
    
    # Adaptation dimensions to apply
    enable_concretization: bool = True
    enable_decomposition: bool = True
    enable_cognitive_load_management: bool = True
    enable_format_optimization: bool = True
    enable_linguistic_simplification: bool = True


@dataclass
class TrainingConfig:
    """Training configuration (Appendix E)"""
    
    # Optimizer settings
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Learning rate schedule
    lr_schedule: str = "cosine"
    warmup_ratio: float = 0.03
    
    # Learning rates
    # 2e-5 for full-parameter fine-tuning (3B models)
    # 1e-4 for LoRA-based tuning (7/8/14B models)
    learning_rate_full: float = 2e-5
    learning_rate_lora: float = 1e-4
    
    # Batch size
    global_batch_size: int = 128
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Context lengths
    max_seq_length_reasoning: int = 4096
    max_seq_length_instruction: int = 2048
    
    # Training epochs
    max_epochs: int = 3
    
    # LoRA configuration (for larger models)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    
    # Evaluation metrics
    use_rouge_l: bool = True  # For instruction following
    use_pass_at_k: bool = True  # For reasoning tasks
    pass_k: int = 1
    
    # Number of evaluation samples
    num_eval_samples: int = 100
    
    # Generation settings for evaluation
    eval_temperature: float = 0.0
    eval_max_tokens: int = 1024


@dataclass
class ModelConfig:
    """Model configuration"""
    
    # Teacher model
    teacher_model_name: str = "deepseek-ai/DeepSeek-R1"
    teacher_api_base: Optional[str] = None
    teacher_api_key: Optional[str] = None
    
    # Student model
    student_model_name: str = "Qwen/Qwen2.5-3B"
    
    # Use LoRA for models > 3B parameters
    use_lora: bool = False
    
    # Device settings
    device: str = "cuda"
    fp16: bool = True
    bf16: bool = False


@dataclass
class DataConfig:
    """Data configuration"""
    
    # Seed data paths
    seed_data_dir: str = "./data/seed"
    output_dir: str = "./outputs"
    
    # Data splits
    train_val_split: float = 0.8
    
    # Domain categories (as in Appendix B)
    domains: List[str] = field(
        default_factory=lambda: [
            "instruction_following",
            "math_problem_solving", 
            "code_generation",
            "academic_knowledge_reasoning"
        ]
    )
    
    # Seed data sizes per domain (approximate, as in Appendix B)
    seed_sizes: Dict[str, int] = field(
        default_factory=lambda: {
            "instruction_following": 800,
            "math_problem_solving": 900,
            "code_generation": 700,
            "academic_knowledge_reasoning": 600
        }
    )


@dataclass
class IOAConfig:
    """Main configuration class combining all sub-configurations"""
    
    identifier: IdentifierConfig = field(default_factory=IdentifierConfig)
    organizer: OrganizerConfig = field(default_factory=OrganizerConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment settings
    experiment_name: str = "ioa_distillation"
    seed: int = 42
    num_runs: int = 5  # For statistical significance
    
    # Logging
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "ioa-distillation"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Identifier validations
        assert 0 < self.identifier.tau_gap < 1, "τ_gap must be in (0, 1)"
        assert 0 < self.identifier.tau_high <= 1, "τ_high must be in (0, 1]"
        assert 0 < self.identifier.tau_low < self.identifier.tau_high, \
            "τ_low must be in (0, τ_high)"
        assert 0 < self.identifier.tau_dep < 1, "τ_dep must be in (0, 1)"
        assert 0 <= self.identifier.alpha <= 1, "α must be in [0, 1]"
        
        # Organizer validations
        assert 0 < self.organizer.tau_zpd < 1, "τ_ZPD must be in (0, 1)"
        assert 0 < self.organizer.tau_mastery <= 1, "τ_mastery must be in (0, 1]"
        
        # Adapter validations
        assert self.adapter.num_samples_per_seed > 0, "J_i must be positive"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IOAConfig":
        """Create configuration from dictionary"""
        identifier_cfg = IdentifierConfig(**config_dict.get("identifier", {}))
        organizer_cfg = OrganizerConfig(**config_dict.get("organizer", {}))
        adapter_cfg = AdapterConfig(**config_dict.get("adapter", {}))
        training_cfg = TrainingConfig(**config_dict.get("training", {}))
        evaluation_cfg = EvaluationConfig(**config_dict.get("evaluation", {}))
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        data_cfg = DataConfig(**config_dict.get("data", {}))
        
        return cls(
            identifier=identifier_cfg,
            organizer=organizer_cfg,
            adapter=adapter_cfg,
            training=training_cfg,
            evaluation=evaluation_cfg,
            model=model_cfg,
            data=data_cfg,
            experiment_name=config_dict.get("experiment_name", "ioa_distillation"),
            seed=config_dict.get("seed", 42),
            num_runs=config_dict.get("num_runs", 5),
            log_level=config_dict.get("log_level", "INFO"),
            use_wandb=config_dict.get("use_wandb", False),
            wandb_project=config_dict.get("wandb_project", "ioa-distillation")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)


def get_default_config() -> IOAConfig:
    """Get default configuration with paper's standard settings"""
    return IOAConfig()


def get_config_for_model_size(model_size: str) -> IOAConfig:
    """
    Get configuration adjusted for different model sizes.
    
    Args:
        model_size: One of "3B", "7B", "8B", "14B"
    
    Returns:
        Configuration adjusted for the model size
    """
    config = IOAConfig()
    
    if model_size in ["7B", "8B", "14B"]:
        # Use LoRA for larger models
        config.model.use_lora = True
        config.training.learning_rate_full = config.training.learning_rate_lora
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("IOA Configuration loaded successfully!")
    print(f"τ_gap: {config.identifier.tau_gap}")
    print(f"τ_ZPD: {config.organizer.tau_zpd}")
    print(f"τ_mastery: {config.organizer.tau_mastery}")
    print(f"J_i (samples per seed): {config.adapter.num_samples_per_seed}")