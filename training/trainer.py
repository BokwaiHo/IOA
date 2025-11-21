"""
Trainer Module for Student Model Fine-tuning

This module implements the training logic for distilling knowledge
into student models as described in Algorithm 1 and Appendix E.

Key features:
- Stage-wise training following the curriculum
- Mastery-based progression (Eq. 9)
- Remedial data generation when mastery not achieved
- Support for both full fine-tuning and LoRA
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..config.config import TrainingConfig, IOAConfig
from ..data.data_utils import SyntheticDataItem, format_for_training
from ..modules.organizer import Curriculum, CurriculumStage

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks training state across stages"""
    
    current_stage: int = 0
    total_stages: int = 0
    global_step: int = 0
    total_tokens_trained: int = 0
    
    # Per-stage metrics
    stage_losses: Dict[str, List[float]] = None
    stage_mastery_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_losses is None:
            self.stage_losses = {}
        if self.stage_mastery_scores is None:
            self.stage_mastery_scores = {}


class IOATrainer:
    """
    Trainer for IOA knowledge distillation.
    
    Implements Algorithm 1 from Section 3.5:
    - Stage-wise training with curriculum
    - Mastery checking and remedial loops
    - Progressive learning
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Any,
        tokenizer: Any,
        output_dir: str = "./outputs"
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            model: Student model to train
            tokenizer: Tokenizer for the model
            output_dir: Directory for saving outputs
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.state = TrainingState()
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler"""
        if self.model is None:
            return
        
        # Use different learning rate based on model size
        lr = self.config.learning_rate_full
        
        # Check if using LoRA
        if hasattr(self.model, 'peft_config'):
            lr = self.config.learning_rate_lora
        
        try:
            from torch.optim import AdamW
            
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
            
            logger.info(f"Optimizer setup with lr={lr}")
            
        except Exception as e:
            logger.warning(f"Could not setup optimizer: {e}")
    
    def train_on_stage(
        self,
        stage: CurriculumStage,
        synthetic_data: List[SyntheticDataItem],
        eval_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Train the model on a single curriculum stage.
        
        Args:
            stage: Curriculum stage to train on
            synthetic_data: Synthetic training data for this stage
            eval_data: Optional evaluation data
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training on stage {stage.stage_id} with {len(synthetic_data)} examples")
        
        if not synthetic_data:
            logger.warning(f"No training data for stage {stage.stage_id}")
            return {"loss": 0.0, "samples": 0}
        
        # Prepare training data
        training_examples = format_for_training(synthetic_data)
        
        # Create dataloader
        dataset = self._create_dataset(training_examples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.state.global_step += 1
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Logging
                if batch_idx % self.config.logging_steps == 0:
                    logger.debug(f"Stage {stage.stage_id} - Epoch {epoch+1}, "
                               f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            epoch_loss /= len(dataloader) if dataloader else 1
            total_loss += epoch_loss
            
            logger.info(f"Stage {stage.stage_id} - Epoch {epoch+1} complete, "
                       f"Loss: {epoch_loss:.4f}")
        
        avg_loss = total_loss / self.config.max_epochs
        
        # Store metrics
        if stage.stage_id not in self.state.stage_losses:
            self.state.stage_losses[stage.stage_id] = []
        self.state.stage_losses[stage.stage_id].append(avg_loss)
        
        metrics = {
            "loss": avg_loss,
            "samples": len(synthetic_data),
            "batches": num_batches,
            "global_step": self.state.global_step
        }
        
        logger.info(f"Stage {stage.stage_id} training complete. Avg loss: {avg_loss:.4f}")
        
        return metrics
    
    def _create_dataset(self, examples: List[Dict]) -> Any:
        """Create a PyTorch dataset from examples"""
        try:
            from datasets import Dataset
            
            # Tokenize
            def tokenize_function(example):
                # Format messages into text
                text = ""
                for msg in example["messages"]:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>\n{content}\n"
                
                # Tokenize
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.config.max_seq_length_reasoning,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # Squeeze batch dimension
                return {k: v.squeeze(0) for k, v in tokens.items()}
            
            dataset = Dataset.from_list(examples)
            dataset = dataset.map(
                tokenize_function,
                remove_columns=["messages"]
            )
            dataset.set_format("torch")
            
            return dataset
            
        except ImportError:
            logger.warning("datasets library not available, using simple dataset")
            return examples
    
    def evaluate_mastery(
        self,
        stage: CurriculumStage,
        eval_data: List[Dict],
        teacher_scores: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluate mastery on a stage.
        
        Implements mastery checking from Eq. 9:
        min_{k∈s_i} P_S(k)/P_T(k) ≥ τ_mastery
        
        Args:
            stage: Curriculum stage
            eval_data: Evaluation data
            teacher_scores: Teacher performance scores
        
        Returns:
            Tuple of (mastery_achieved, module_scores)
        """
        logger.info(f"Evaluating mastery for stage {stage.stage_id}")
        
        if not eval_data:
            logger.warning("No evaluation data provided")
            return True, {}
        
        self.model.eval()
        module_scores = {}
        
        # Group eval data by module
        module_data = {}
        for item in eval_data:
            module = item.get("module", "unknown")
            if module not in module_data:
                module_data[module] = []
            module_data[module].append(item)
        
        # Evaluate each module
        for module, items in module_data.items():
            correct = 0
            total = len(items)
            
            for item in items:
                # Generate response
                response = self._generate_response(item["input"])
                
                # Check correctness (simplified)
                expected = item.get("output", "")
                if self._check_answer(response, expected):
                    correct += 1
            
            score = correct / total if total > 0 else 0.0
            module_scores[module] = score
        
        # Check mastery condition
        mastery_achieved = True
        min_ratio = 1.0
        
        for module, student_score in module_scores.items():
            teacher_score = teacher_scores.get(module, 1.0)
            
            if teacher_score > 0:
                ratio = student_score / teacher_score
            else:
                ratio = 1.0
            
            min_ratio = min(min_ratio, ratio)
            
            if ratio < 0.9:  # τ_mastery = 0.9
                mastery_achieved = False
        
        # Store mastery score
        self.state.stage_mastery_scores[stage.stage_id] = min_ratio
        
        logger.info(f"Stage {stage.stage_id} mastery: {mastery_achieved} "
                   f"(min_ratio={min_ratio:.3f})")
        
        return mastery_achieved, module_scores
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from student model"""
        if self.model is None:
            return ""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def _check_answer(self, response: str, expected: str) -> bool:
        """Check if response matches expected answer"""
        from ..data.data_utils import compute_exact_match, compute_rouge_l
        
        if compute_exact_match(response, expected):
            return True
        
        rouge = compute_rouge_l(response, expected)
        return rouge >= 0.5
    
    def save_checkpoint(self, stage_id: str) -> None:
        """Save a checkpoint after completing a stage"""
        checkpoint_dir = self.output_dir / f"checkpoint-{stage_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model is not None:
            self.model.save_pretrained(checkpoint_dir)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                "current_stage": self.state.current_stage,
                "total_stages": self.state.total_stages,
                "global_step": self.state.global_step,
                "stage_losses": self.state.stage_losses,
                "stage_mastery_scores": self.state.stage_mastery_scores
            }, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load a checkpoint"""
        checkpoint_path = Path(checkpoint_dir)
        
        # Load training state
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state_dict = json.load(f)
            
            self.state.current_stage = state_dict.get("current_stage", 0)
            self.state.total_stages = state_dict.get("total_stages", 0)
            self.state.global_step = state_dict.get("global_step", 0)
            self.state.stage_losses = state_dict.get("stage_losses", {})
            self.state.stage_mastery_scores = state_dict.get("stage_mastery_scores", {})
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
    
    def get_training_summary(self) -> str:
        """Get a summary of training progress"""
        summary = [
            "Training Summary:",
            f"  Stages completed: {self.state.current_stage}/{self.state.total_stages}",
            f"  Global steps: {self.state.global_step}",
            ""
        ]
        
        for stage_id, scores in self.state.stage_mastery_scores.items():
            losses = self.state.stage_losses.get(stage_id, [])
            avg_loss = sum(losses) / len(losses) if losses else 0
            summary.append(f"  {stage_id}: mastery={scores:.3f}, loss={avg_loss:.4f}")
        
        return "\n".join(summary)


def train_curriculum(
    trainer: IOATrainer,
    curriculum: Curriculum,
    synthetic_data: Dict[str, List[SyntheticDataItem]],
    eval_data: Dict[str, List[Dict]],
    teacher_scores: Dict[str, float],
    adapter: Any = None
) -> Dict[str, Any]:
    """
    Train the student model following the curriculum.
    
    This implements Algorithm 1 from Section 3.5.
    
    Args:
        trainer: IOATrainer instance
        curriculum: Organized curriculum
        synthetic_data: Dict mapping stage_id to synthetic items
        eval_data: Dict mapping stage_id to eval items
        teacher_scores: Teacher performance scores per module
        adapter: Optional adapter for remedial data generation
    
    Returns:
        Dictionary with final training results
    """
    logger.info(f"Starting curriculum training with {len(curriculum.stages)} stages")
    
    trainer.state.total_stages = len(curriculum.stages)
    results = {"stages": {}}
    
    for i, stage in enumerate(curriculum.stages):
        trainer.state.current_stage = i
        stage_id = stage.stage_id
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Stage {i+1}/{len(curriculum.stages)}: {stage_id}")
        logger.info(f"Modules: {stage.modules}")
        logger.info(f"{'='*50}")
        
        # Get data for this stage
        stage_data = synthetic_data.get(stage_id, [])
        stage_eval = eval_data.get(stage_id, [])
        
        # Train on stage
        train_metrics = trainer.train_on_stage(stage, stage_data, stage_eval)
        
        # Evaluate mastery
        mastery_achieved, module_scores = trainer.evaluate_mastery(
            stage, stage_eval, teacher_scores
        )
        
        # Remedial loop
        remedial_count = 0
        max_remedial = 5
        
        while not mastery_achieved and remedial_count < max_remedial:
            logger.info(f"Mastery not achieved, generating remedial data "
                       f"(iteration {remedial_count + 1})")
            
            if adapter is not None:
                # Get weak modules
                weak_modules = [
                    m for m, s in module_scores.items()
                    if s < 0.9 * teacher_scores.get(m, 1.0)
                ]
                
                # Generate remedial data
                remedial_data = adapter.generate_remedial_data(
                    stage_id=stage_id,
                    knowledge_modules=stage.modules,
                    weak_subskills=weak_modules,
                    seed_items=[],
                    num_examples=10
                )
                
                # Train on remedial data
                if remedial_data:
                    trainer.train_on_stage(stage, remedial_data)
                
                # Re-evaluate
                mastery_achieved, module_scores = trainer.evaluate_mastery(
                    stage, stage_eval, teacher_scores
                )
            
            remedial_count += 1
        
        stage.mastery_achieved = mastery_achieved
        stage.remedial_count = remedial_count
        
        # Save checkpoint
        trainer.save_checkpoint(stage_id)
        
        # Store results
        results["stages"][stage_id] = {
            "train_metrics": train_metrics,
            "mastery_achieved": mastery_achieved,
            "module_scores": module_scores,
            "remedial_iterations": remedial_count
        }
    
    logger.info("\n" + trainer.get_training_summary())
    
    return results


if __name__ == "__main__":
    # Test the trainer
    from ..config.config import TrainingConfig
    
    config = TrainingConfig()
    print(f"Training config loaded:")
    print(f"  Learning rate: {config.learning_rate_full}")
    print(f"  Batch size: {config.global_batch_size}")
    print(f"  Max epochs: {config.max_epochs}")