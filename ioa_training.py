"""
Training utilities and complete example for IOA framework
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IOADataset(Dataset):
    """Dataset for IOA synthesized examples"""
    
    def __init__(self, synthesized_examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = synthesized_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the example into training text
        if isinstance(example, dict):
            # Handle SynthesizedExample converted to dict
            problem = example.get('problem', '')
            solution = example.get('solution', {})
            
            if isinstance(solution, dict):
                steps = solution.get('steps', [])
                final_answer = solution.get('final_answer', '')
                
                # Create instruction-following format
                text = f"Problem: {problem}\n\nSolution:\n"
                for step in steps:
                    text += f"{step}\n"
                text += f"\nFinal Answer: {final_answer}"
            else:
                text = f"Problem: {problem}\nSolution: {solution}"
        else:
            # Handle direct text
            text = str(example)
            
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class StudentModelTrainer:
    """Trainer for student models using synthesized data"""
    
    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
    def prepare_dataset(self, synthesized_examples: List[Dict], max_length: int = 512):
        """Prepare dataset from synthesized examples"""
        # Convert SynthesizedExample objects to dictionaries if needed
        examples_dict = []
        for example in synthesized_examples:
            if hasattr(example, '__dict__'):
                examples_dict.append(example.__dict__)
            else:
                examples_dict.append(example)
                
        return IOADataset(examples_dict, self.tokenizer, max_length)
        
    def train(self, train_dataset: IOADataset, 
             output_dir: str = "./finetuned_model",
             num_epochs: int = 3,
             learning_rate: float = 2e-5,
             batch_size: int = 4,
             save_steps: int = 500):
        """Fine-tune student model"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=save_steps,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info(f"Starting training with {len(train_dataset)} examples")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        
    def evaluate_model(self, test_examples: List[Dict], max_samples: int = 100):
        """Simple evaluation of the trained model"""
        self.model.eval()
        correct = 0
        total = 0
        
        for i, example in enumerate(test_examples[:max_samples]):
            if isinstance(example, dict):
                problem = example.get('problem', '')
                expected = example.get('solution', {}).get('final_answer', '')
            else:
                continue
                
            # Generate response
            input_text = f"Problem: {problem}\n\nSolution:"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(input_text, "").strip()
            
            # Simple correctness check
            if expected.lower() in response.lower():
                correct += 1
            total += 1
            
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Evaluation accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy

def create_complete_example():
    """Complete example of using the IOA framework"""
    
    # Import the main framework classes
    from ioa_framework import IOAFramework, ProbeTask
    from ioa_config import DatasetBuilder, IOAConfig
    
    logger.info("Starting IOA Framework Complete Example")
    
    # Step 1: Setup configuration and data
    config = IOAConfig()
    builder = DatasetBuilder()
    
    # Create seed data
    logger.info("Creating seed data...")
    math_seed_data = builder.create_math_seed_data()
    probe_tasks_data = builder.create_probe_tasks("mathematics")
    
    # Convert to ProbeTask objects
    probe_tasks = [
        ProbeTask(
            module_id=task["module_id"],
            question=task["question"],
            answer=task["answer"],
            task_type=task["task_type"]
        ) for task in probe_tasks_data
    ]
    
    # Step 2: Initialize framework
    logger.info("Initializing IOA framework...")
    
    # Note: Replace with actual OpenAI API key
    framework = IOAFramework(openai_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"))
    
    # Step 3: Initialize student model trainer
    logger.info("Initializing student model...")
    
    # Use a small model for testing (replace with desired model)
    student_trainer = StudentModelTrainer("microsoft/DialoGPT-small")
    
    # Step 4: Run knowledge distillation
    logger.info("Running knowledge distillation...")
    
    try:
        results = framework.distill_knowledge(
            teacher_model_name="gpt-3.5-turbo",  # More accessible than gpt-4
            student_model=student_trainer.model,
            target_domain="mathematics",
            seed_data=math_seed_data,
            probe_tasks=probe_tasks
        )
        
        logger.info("Knowledge distillation completed successfully!")
        
        # Step 5: Train student model with synthesized data
        if results['synthesized_data']:
            logger.info("Training student model with synthesized data...")
            
            train_dataset = student_trainer.prepare_dataset(results['synthesized_data'])
            student_trainer.train(
                train_dataset=train_dataset,
                output_dir="./math_student_model",
                num_epochs=2,
                batch_size=2  # Small batch for testing
            )
            
            # Step 6: Evaluate trained model
            logger.info("Evaluating trained model...")
            accuracy = student_trainer.evaluate_model(results['synthesized_data'])
            
            # Step 7: Save results
            results_summary = {
                "curriculum_stages": len(results['curriculum']),
                "total_synthesized_examples": len(results['synthesized_data']),
                "knowledge_modules": [module.__dict__ if hasattr(module, '__dict__') else str(module) 
                                    for module in results['knowledge_modules']],
                "performance_gaps": results['performance_gaps'],
                "final_accuracy": accuracy
            }
            
            with open("ioa_results.json", 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
                
            logger.info("Results saved to ioa_results.json")
            
        else:
            logger.warning("No synthesized data generated. Check your OpenAI API key and configuration.")
            
    except Exception as e:
        logger.error(f"Error during distillation: {e}")
        logger.info("Running with mock data for demonstration...")
        
        # Fallback: Create mock synthesized data for testing
        mock_synthesized_data = [
            {
                "problem": "Solve: 2x + 3 = 7",
                "solution": {
                    "steps": [
                        "Step 1: Subtract 3 from both sides: 2x = 4",
                        "Step 2: Divide both sides by 2: x = 2",
                        "Step 3: Verify: 2(2) + 3 = 7 ✓"
                    ],
                    "final_answer": "x = 2"
                },
                "module": "linear_equations"
            },
            {
                "problem": "Find the area of a rectangle with length 5 and width 3",
                "solution": {
                    "steps": [
                        "Step 1: Use the formula Area = length × width",
                        "Step 2: Substitute values: Area = 5 × 3",
                        "Step 3: Calculate: Area = 15"
                    ],
                    "final_answer": "15 square units"
                },
                "module": "arithmetic"
            }
        ]
        
        logger.info("Training with mock data...")
        train_dataset = student_trainer.prepare_dataset(mock_synthesized_data)
        student_trainer.train(
            train_dataset=train_dataset,
            output_dir="./mock_student_model",
            num_epochs=1,
            batch_size=1
        )
        
        accuracy = student_trainer.evaluate_model(mock_synthesized_data)
        logger.info(f"Mock training completed with accuracy: {accuracy:.2%}")

class ModelEvaluator:
    """Evaluate student models on various benchmarks"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
    def evaluate_on_benchmark(self, benchmark_data: List[Dict], 
                            benchmark_name: str = "custom") -> Dict[str, float]:
        """Evaluate model on benchmark data"""
        
        logger.info(f"Evaluating on {benchmark_name} benchmark ({len(benchmark_data)} examples)")
        
        results = {
            "total_examples": len(benchmark_data),
            "correct": 0,
            "accuracy": 0.0
        }
        
        for example in tqdm(benchmark_data, desc=f"Evaluating {benchmark_name}"):
            problem = example.get("problem", example.get("question", ""))
            expected = example.get("answer", example.get("solution", ""))
            
            # Generate prediction
            prediction = self._generate_response(problem)
            
            # Check correctness (simplified)
            if self._is_correct(prediction, expected):
                results["correct"] += 1
                
        results["accuracy"] = results["correct"] / results["total_examples"]
        
        logger.info(f"{benchmark_name} Results: {results['accuracy']:.2%} ({results['correct']}/{results['total_examples']})")
        
        return results
        
    def _generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(f"Problem: {prompt}\n\nSolution:", 
                              return_tensors="pt", max_length=256, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        response = response.replace(prompt_text, "").strip()
        
        return response
        
    def _is_correct(self, prediction: str, expected: str) -> bool:
        """Simple correctness check"""
        pred_clean = prediction.lower().strip()
        exp_clean = expected.lower().strip()
        
        # Extract numbers from both strings for math problems
        import re
        pred_nums = re.findall(r'-?\d+\.?\d*', pred_clean)
        exp_nums = re.findall(r'-?\d+\.?\d*', exp_clean)
        
        # Check if expected answer appears in prediction
        return exp_clean in pred_clean or (pred_nums and exp_nums and pred_nums[-1] == exp_nums[-1])

if __name__ == "__main__":
    # Set environment variable for testing
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Run complete example
    create_complete_example()
    
    # Additional evaluation example
    logger.info("\n" + "="*50)
    logger.info("Additional evaluation example:")
    
    try:
        # This would run if a model was actually trained
        evaluator = ModelEvaluator("./math_student_model")
        
        # Create some test data
        test_data = [
            {"problem": "What is 5 + 3?", "answer": "8"},
            {"problem": "Solve: x + 2 = 5", "answer": "x = 3"},
        ]
        
        results = evaluator.evaluate_on_benchmark(test_data, "Math Test")
        logger.info(f"Evaluation completed: {results}")
        
    except Exception as e:
        logger.info(f"Evaluation skipped: {e}")
    
    logger.info("IOA Framework example completed!")