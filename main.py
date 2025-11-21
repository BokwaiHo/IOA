"""
Main Entry Point for IOA Knowledge Distillation Framework

This script implements Algorithm 1 from Section 3.5:
Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation

Usage:
    python main.py --config config.yaml
    python main.py --domain math_problem_solving --student Qwen/Qwen2.5-3B
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import IOAConfig, get_default_config
from utils.llm_client import LLMClient, create_teacher_client
from utils.graph_utils import KnowledgeDependencyGraph
from data.seed_data import load_seed_data, SeedDataset, get_synthesis_seeds
from data.data_utils import save_synthetic_data, load_synthetic_data
from modules.identifier import KnowledgeIdentifier
from modules.organizer import KnowledgeOrganizer, Curriculum
from modules.adapter import KnowledgeAdapter
from training.trainer import IOATrainer, train_curriculum
from evaluation.evaluator import Evaluator, evaluate_distillation_quality

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ioa_distillation.log')
    ]
)
logger = logging.getLogger(__name__)


class IOAPipeline:
    """
    Main pipeline for IOA knowledge distillation.
    
    Implements the complete workflow from Algorithm 1:
    1. Knowledge Identification (Identifier)
    2. Curriculum Organization (Organizer)
    3. Mastery-Based Progressive Learning with Adaptation (Adapter + Training)
    """
    
    def __init__(
        self,
        config: IOAConfig,
        output_dir: str = "./outputs"
    ):
        """
        Initialize the IOA pipeline.
        
        Args:
            config: IOA configuration
            output_dir: Directory for outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.teacher_client: Optional[LLMClient] = None
        self.seed_dataset: Optional[SeedDataset] = None
        self.identifier: Optional[KnowledgeIdentifier] = None
        self.organizer: Optional[KnowledgeOrganizer] = None
        self.adapter: Optional[KnowledgeAdapter] = None
        self.trainer: Optional[IOATrainer] = None
        self.evaluator: Optional[Evaluator] = None
        
        # Models
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        
        # Results storage
        self.target_modules = []
        self.dependency_graph: Optional[KnowledgeDependencyGraph] = None
        self.curriculum: Optional[Curriculum] = None
        self.synthetic_data: Dict[str, Any] = {}
        
        logger.info(f"IOA Pipeline initialized with config: {config.experiment_name}")
    
    def setup(self) -> None:
        """Setup all components"""
        logger.info("Setting up IOA pipeline components...")
        
        # 1. Setup teacher client
        self._setup_teacher_client()
        
        # 2. Load seed data
        self._load_seed_data()
        
        # 3. Load models
        self._load_models()
        
        # 4. Initialize modules
        self._initialize_modules()
        
        logger.info("Pipeline setup complete")
    
    def _setup_teacher_client(self) -> None:
        """Setup teacher LLM client"""
        teacher_config = self.config.model
        
        # Determine teacher type from model name
        if "deepseek" in teacher_config.teacher_model_name.lower():
            teacher_type = "deepseek"
        elif "gpt" in teacher_config.teacher_model_name.lower() or \
             "o1" in teacher_config.teacher_model_name.lower():
            teacher_type = "openai"
        else:
            teacher_type = "local"
        
        self.teacher_client = create_teacher_client(
            teacher_type=teacher_type,
            api_key=teacher_config.teacher_api_key,
            api_base=teacher_config.teacher_api_base,
            model_name=teacher_config.teacher_model_name
        )
        
        logger.info(f"Teacher client setup: {teacher_type}")
    
    def _load_seed_data(self) -> None:
        """Load seed dataset"""
        self.seed_dataset = load_seed_data(
            data_dir=self.config.data.seed_data_dir,
            train_val_split=self.config.data.train_val_split,
            seed=self.config.seed
        )
        
        logger.info(f"Seed data loaded: {len(self.seed_dataset)} items")
    
    def _load_models(self) -> None:
        """Load teacher and student models"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load student model
            student_name = self.config.model.student_model_name
            logger.info(f"Loading student model: {student_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                student_name,
                trust_remote_code=True
            )
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_name,
                torch_dtype=torch.float16 if self.config.model.fp16 else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Apply LoRA if configured
            if self.config.model.use_lora:
                self._apply_lora()
            
            logger.info(f"Student model loaded: {student_name}")
            
        except ImportError:
            logger.warning("Transformers not installed, models not loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def _apply_lora(self) -> None:
        """Apply LoRA to the student model"""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=self.config.training.lora_r,
                lora_alpha=self.config.training.lora_alpha,
                lora_dropout=self.config.training.lora_dropout,
                target_modules=self.config.training.lora_target_modules,
                task_type="CAUSAL_LM"
            )
            
            self.student_model = get_peft_model(self.student_model, lora_config)
            logger.info("LoRA applied to student model")
            
        except ImportError:
            logger.warning("PEFT not installed, skipping LoRA")
    
    def _initialize_modules(self) -> None:
        """Initialize Identifier, Organizer, Adapter, Trainer, Evaluator"""
        # Identifier
        self.identifier = KnowledgeIdentifier(
            config=self.config.identifier,
            teacher_client=self.teacher_client,
            seed_dataset=self.seed_dataset
        )
        
        # Adapter
        self.adapter = KnowledgeAdapter(
            config=self.config.adapter,
            teacher_client=self.teacher_client
        )
        
        # Trainer
        self.trainer = IOATrainer(
            config=self.config.training,
            model=self.student_model,
            tokenizer=self.tokenizer,
            output_dir=str(self.output_dir)
        )
        
        # Evaluator
        self.evaluator = Evaluator(
            model=self.student_model,
            tokenizer=self.tokenizer
        )
        
        logger.info("All modules initialized")
    
    def run_identification(self, domain: str) -> None:
        """
        Run the Identifier module.
        
        Step 1-3 of Algorithm 1:
        - Decompose knowledge domain
        - Evaluate performance gaps
        - Build dependency graph
        - Select target modules
        
        Args:
            domain: Target capability domain
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 1: KNOWLEDGE IDENTIFICATION")
        logger.info(f"{'='*60}")
        
        self.target_modules, self.dependency_graph = self.identifier.identify(
            domain=domain,
            teacher_model=self.teacher_model,
            student_model=self.student_model
        )
        
        # Save results
        results = {
            "target_modules": self.target_modules,
            "num_total_modules": len(self.identifier.knowledge_modules),
            "num_deficient": len(self.identifier.deficient_modules),
            "performance_gaps": self.identifier.performance_gaps
        }
        
        with open(self.output_dir / "identification_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Identified {len(self.target_modules)} target modules")
    
    def run_organization(self, domain: str) -> None:
        """
        Run the Organizer module.
        
        Step 4-5 of Algorithm 1:
        - Construct curriculum sequence
        - Apply ZPD constraints
        
        Args:
            domain: Target capability domain
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 2: CURRICULUM ORGANIZATION")
        logger.info(f"{'='*60}")
        
        # Initialize organizer with dependency graph
        self.organizer = KnowledgeOrganizer(
            config=self.config.organizer,
            dependency_graph=self.dependency_graph
        )
        
        self.curriculum = self.organizer.organize(
            target_modules=self.target_modules,
            domain=domain
        )
        
        # Log curriculum summary
        logger.info(self.organizer.get_curriculum_summary())
        
        # Save curriculum
        curriculum_data = {
            "domain": domain,
            "num_stages": len(self.curriculum.stages),
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "modules": s.modules,
                    "prerequisites": list(s.prerequisites),
                    "difficulty": s.avg_difficulty
                }
                for s in self.curriculum.stages
            ]
        }
        
        with open(self.output_dir / "curriculum.json", 'w') as f:
            json.dump(curriculum_data, f, indent=2)
    
    def run_adaptation_and_synthesis(self) -> None:
        """
        Run the Adapter module to generate synthetic data.
        
        Step 8-9 of Algorithm 1:
        - Adapt knowledge representation
        - Generate synthetic training data
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 3: KNOWLEDGE ADAPTATION & SYNTHESIS")
        logger.info(f"{'='*60}")
        
        self.synthetic_data = {}
        
        for stage in self.curriculum.stages:
            logger.info(f"\nGenerating data for stage {stage.stage_id}")
            
            # Get seed items for this stage
            seed_items = []
            for module in stage.modules:
                items = get_synthesis_seeds(self.seed_dataset, module)
                seed_items.extend(items)
            
            # Generate synthetic data
            synthetic_items = self.adapter.adapt_for_stage(
                stage_id=stage.stage_id,
                knowledge_modules=stage.modules,
                prerequisites=list(stage.prerequisites),
                seed_items=seed_items,
                domain=self.curriculum.domain
            )
            
            self.synthetic_data[stage.stage_id] = synthetic_items
            
            # Save stage data
            stage_file = self.output_dir / f"synthetic_{stage.stage_id}.jsonl"
            save_synthetic_data(synthetic_items, str(stage_file))
            
            logger.info(f"Generated {len(synthetic_items)} items for {stage.stage_id}")
        
        # Log synthesis stats
        stats = self.adapter.get_stats()
        logger.info(f"\nSynthesis Statistics:")
        logger.info(f"  Total generated: {stats['total_generated']}")
        logger.info(f"  Validation passed: {stats['validation_passed']}")
        logger.info(f"  Final count: {stats['final_count']}")
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run the progressive training loop.
        
        Step 7-14 of Algorithm 1:
        - Train on each stage
        - Check mastery
        - Generate remedial data if needed
        - Advance to next stage
        
        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 4: PROGRESSIVE TRAINING")
        logger.info(f"{'='*60}")
        
        # Prepare evaluation data (using validation split of seed data)
        eval_data = {}
        for stage in self.curriculum.stages:
            stage_eval = []
            for module in stage.modules:
                items = self.seed_dataset.get_val_items_by_module(module)
                for item in items:
                    stage_eval.append({
                        "input": item.input_text,
                        "output": item.output_text,
                        "module": module
                    })
            eval_data[stage.stage_id] = stage_eval
        
        # Get teacher scores (placeholder - would need actual evaluation)
        teacher_scores = {}
        for module in self.target_modules:
            node = self.identifier.get_module_by_id(module)
            if node:
                teacher_scores[module] = node.teacher_score
            else:
                teacher_scores[module] = 1.0
        
        # Run curriculum training
        results = train_curriculum(
            trainer=self.trainer,
            curriculum=self.curriculum,
            synthetic_data=self.synthetic_data,
            eval_data=eval_data,
            teacher_scores=teacher_scores,
            adapter=self.adapter
        )
        
        # Save training results
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def run_evaluation(
        self,
        benchmarks: Optional[Dict[str, list]] = None
    ) -> Dict[str, Any]:
        """
        Run final evaluation on benchmarks.
        
        Args:
            benchmarks: Optional benchmark data
        
        Returns:
            Evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 5: FINAL EVALUATION")
        logger.info(f"{'='*60}")
        
        if benchmarks is None:
            # Create placeholder benchmarks from seed data
            benchmarks = {
                "instruction_following": [],
                "reasoning": [],
                "code_generation": []
            }
            
            for item in self.seed_dataset.get_val_items():
                entry = {"input": item.input_text, "output": item.output_text}
                if item.domain == "instruction_following":
                    benchmarks["instruction_following"].append(entry)
                elif item.domain == "math_problem_solving":
                    benchmarks["reasoning"].append(entry)
                elif item.domain == "code_generation":
                    benchmarks["code_generation"].append(entry)
        
        # Run evaluation
        results = self.evaluator.evaluate_all_benchmarks(benchmarks)
        
        # Log results
        logger.info(self.evaluator.get_results_summary())
        
        # Save results
        self.evaluator.save_results(str(self.output_dir / "evaluation_results.json"))
        
        return results
    
    def run(self, domain: str) -> Dict[str, Any]:
        """
        Run the complete IOA pipeline.
        
        This is the main entry point that executes Algorithm 1.
        
        Args:
            domain: Target capability domain
        
        Returns:
            Complete results dictionary
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"# IOA Knowledge Distillation Pipeline")
        logger.info(f"# Domain: {domain}")
        logger.info(f"# Experiment: {self.config.experiment_name}")
        logger.info(f"{'#'*60}\n")
        
        start_time = datetime.now()
        
        # Setup
        self.setup()
        
        # Phase 1: Identification
        self.run_identification(domain)
        
        # Phase 2: Organization
        self.run_organization(domain)
        
        # Phase 3: Adaptation & Synthesis
        self.run_adaptation_and_synthesis()
        
        # Phase 4: Training
        training_results = self.run_training()
        
        # Phase 5: Evaluation
        eval_results = self.run_evaluation()
        
        # Compile final results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        final_results = {
            "experiment": self.config.experiment_name,
            "domain": domain,
            "duration_seconds": duration,
            "num_target_modules": len(self.target_modules),
            "num_curriculum_stages": len(self.curriculum.stages) if self.curriculum else 0,
            "total_synthetic_samples": sum(
                len(items) for items in self.synthetic_data.values()
            ),
            "training_results": training_results,
            "evaluation_results": eval_results
        }
        
        # Save final results
        with open(self.output_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info("IOA PIPELINE COMPLETE")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*60}\n")
        
        return final_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IOA Knowledge Distillation Framework"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default="math_problem_solving",
        choices=[
            "math_problem_solving",
            "code_generation",
            "instruction_following",
            "academic_knowledge_reasoning"
        ],
        help="Target capability domain"
    )
    
    parser.add_argument(
        "--student",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Student model name"
    )
    
    parser.add_argument(
        "--teacher",
        type=str,
        default="deepseek-ai/DeepSeek-R1",
        help="Teacher model name"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        # Load from file
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                import yaml
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        config = IOAConfig.from_dict(config_dict)
    else:
        # Use default config with command line overrides
        config = get_default_config()
        config.model.student_model_name = args.student
        config.model.teacher_model_name = args.teacher
        config.seed = args.seed
        
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.experiment_name = f"ioa_{args.domain}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Run pipeline
    pipeline = IOAPipeline(
        config=config,
        output_dir=str(output_dir)
    )
    
    results = pipeline.run(domain=args.domain)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Domain: {args.domain}")
    print(f"Student: {config.model.student_model_name}")
    print(f"Teacher: {config.model.teacher_model_name}")
    print(f"Duration: {results['duration_seconds']:.2f}s")
    print(f"Target modules: {results['num_target_modules']}")
    print(f"Curriculum stages: {results['num_curriculum_stages']}")
    print(f"Synthetic samples: {results['total_synthetic_samples']}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()