"""
Configuration and data utilities for IOA framework
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class IOAConfig:
    """Configuration for IOA framework"""
    # Identifier parameters
    tau_gap: float = 0.3
    tau_high: float = 0.9
    tau_low: float = 0.7
    tau_dep: float = 0.3
    
    # Organizer parameters
    tau_zpd: float = 0.15
    tau_mastery: float = 0.9
    
    # Adapter parameters
    num_examples_per_stage: int = 10
    max_retries: int = 3
    temperature: float = 0.7
    
    # Training parameters
    max_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 8
    
    # Model settings
    teacher_model: str = "gpt-4"
    max_context_length: int = 4096
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'IOAConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
        
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'tau_gap': self.tau_gap,
            'tau_high': self.tau_high,
            'tau_low': self.tau_low,
            'tau_dep': self.tau_dep,
            'tau_zpd': self.tau_zpd,
            'tau_mastery': self.tau_mastery,
            'num_examples_per_stage': self.num_examples_per_stage,
            'max_retries': self.max_retries,
            'temperature': self.temperature,
            'max_epochs': self.max_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'teacher_model': self.teacher_model,
            'max_context_length': self.max_context_length
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

class DatasetBuilder:
    """Build seed datasets and probe tasks for different domains"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def create_math_seed_data(self) -> List[Dict[str, Any]]:
        """Create seed data for mathematics domain"""
        seed_data = [
            {
                "problem": "You have 15 apples and give away 7. How many apples do you have left?",
                "answer": "15 - 7 = 8 apples",
                "module": "arithmetic",
                "difficulty": "introductory"
            },
            {
                "problem": "Solve for x: 3x + 5 = 14",
                "answer": "3x = 9, so x = 3",
                "module": "linear_equations",
                "difficulty": "intermediate"
            },
            {
                "problem": "Find the derivative of f(x) = x² + 3x - 2",
                "answer": "f'(x) = 2x + 3",
                "module": "calculus_derivatives",
                "difficulty": "advanced"
            },
            {
                "problem": "Factor the quadratic: x² - 5x + 6",
                "answer": "(x - 2)(x - 3)",
                "module": "quadratic_equations",
                "difficulty": "intermediate"
            }
        ]
        
        # Save to file
        with open(self.data_dir / "math_seed_data.json", 'w') as f:
            json.dump(seed_data, f, indent=2)
            
        return seed_data
        
    def create_programming_seed_data(self) -> List[Dict[str, Any]]:
        """Create seed data for programming domain"""
        seed_data = [
            {
                "problem": "Write a function to find the maximum number in a list",
                "answer": "def find_max(nums): return max(nums)",
                "module": "python_basics",
                "difficulty": "introductory"
            },
            {
                "problem": "Implement a function to reverse a string",
                "answer": "def reverse_string(s): return s[::-1]",
                "module": "string_processing", 
                "difficulty": "introductory"
            },
            {
                "problem": "Write a binary search function",
                "answer": """def binary_search(arr, target):
                        left, right = 0, len(arr) - 1
                        while left <= right:
                            mid = (left + right) // 2
                            if arr[mid] == target:
                                return mid
                            elif arr[mid] < target:
                                left = mid + 1
                            else:
                                right = mid - 1
                        return -1""",
                "module": "algorithms",
                "difficulty": "advanced"
            }
        ]
        
        with open(self.data_dir / "programming_seed_data.json", 'w') as f:
            json.dump(seed_data, f, indent=2)
            
        return seed_data
        
    def create_probe_tasks(self, domain: str) -> List[Dict[str, Any]]:
        """Create probe tasks for evaluation"""
        if domain == "mathematics":
            probe_tasks = [
                {
                    "module_id": "arithmetic",
                    "question": "Calculate: 23 + 17 × 3",
                    "answer": "74",
                    "task_type": "math"
                },
                {
                    "module_id": "linear_equations", 
                    "question": "Solve: 2x - 3 = 7",
                    "answer": "x = 5",
                    "task_type": "math"
                },
                {
                    "module_id": "quadratic_equations",
                    "question": "Find roots of x² - 4x + 3 = 0",
                    "answer": "x = 1, 3",
                    "task_type": "math"
                },
                {
                    "module_id": "calculus_derivatives",
                    "question": "Find derivative of 2x³ - x²",
                    "answer": "6x² - 2x", 
                    "task_type": "math"
                }
            ]
        elif domain == "programming":
            probe_tasks = [
                {
                    "module_id": "python_basics",
                    "question": "What does range(5) produce?",
                    "answer": "[0, 1, 2, 3, 4]",
                    "task_type": "code"
                },
                {
                    "module_id": "string_processing",
                    "question": "How to convert string to uppercase?",
                    "answer": "str.upper()",
                    "task_type": "code"
                },
                {
                    "module_id": "algorithms",
                    "question": "What is time complexity of binary search?",
                    "answer": "O(log n)",
                    "task_type": "code"
                }
            ]
        else:
            probe_tasks = []
            
        # Save probe tasks
        with open(self.data_dir / f"{domain}_probe_tasks.json", 'w') as f:
            json.dump(probe_tasks, f, indent=2)
            
        return probe_tasks
        
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)

class PromptTemplates:
    """Templates for different adaptation strategies"""
    
    @staticmethod
    def get_concretization_prompt(concept: str) -> str:
        """Abstract concept concretization prompt"""
        return f"""Explain the abstract concept of {concept} using a concrete analogy or real-world example.
        Start with the analogy, then gradually transition to the symbolic or mathematical expression.
        Keep the language simple and accessible."""
        
    @staticmethod
    def get_decomposition_prompt() -> str:
        """Complex reasoning decomposition prompt"""
        return """Break down the problem-solving process into small, clear steps:
        1. Extract relevant information
        2. Identify relationships and patterns
        3. Formulate equations or logical structure
        4. Solve step by step
        5. Verify the solution
        
        Provide each step explicitly with clear reasoning."""
        
    @staticmethod
    def get_cognitive_load_prompt() -> str:
        """Cognitive load management prompt"""
        return """Simplify the problem to reduce cognitive load:
        - Start with the simplest version of this type of problem
        - Use small numbers and simple cases
        - Break complex problems into manageable segments
        - Provide intermediate verification steps"""
        
    @staticmethod
    def get_format_optimization_prompt() -> str:
        """Representation format optimization prompt"""
        return """Present the solution using a consistent, structured format:
        
        Step 1: [Identify/Extract] - 
        Step 2: [Analyze/Process] - 
        Step 3: [Apply/Calculate] - 
        Step 4: [Verify/Check] - 
        
        Use this template consistently while varying the content."""
        
    @staticmethod
    def get_linguistic_simplification_prompt() -> str:
        """Linguistic complexity reduction prompt"""
        return """Rewrite explanations using simpler language:
        - Use short, direct sentences
        - Replace advanced terms with simpler synonyms
        - Use clear connectors: "first", "next", "then", "therefore"
        - Avoid complex sentence structures
        - Ensure mathematical reasoning remains correct"""

class EvaluationMetrics:
    """Evaluation metrics for the framework"""
    
    @staticmethod
    def rouge_l(prediction: str, reference: str) -> float:
        """Simplified ROUGE-L calculation"""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
            
        # Find LCS length
        lcs_length = EvaluationMetrics._lcs_length(pred_tokens, ref_tokens)
        
        if len(ref_tokens) == 0:
            return 0.0
            
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * precision * recall / (precision + recall)
        return f1
        
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
        
    @staticmethod
    def pass_at_k(predictions: List[str], reference: str, k: int = 1) -> float:
        """Calculate pass@k metric"""
        if k > len(predictions):
            k = len(predictions)
            
        correct_count = 0
        for pred in predictions[:k]:
            if EvaluationMetrics._is_correct(pred, reference):
                correct_count += 1
                
        return correct_count / k if k > 0 else 0.0
        
    @staticmethod
    def _is_correct(prediction: str, reference: str) -> bool:
        """Simple correctness check (can be enhanced for specific domains)"""
        pred_clean = prediction.strip().lower()
        ref_clean = reference.strip().lower()
        return pred_clean == ref_clean or ref_clean in pred_clean

# Example configuration file creation
def create_default_config():
    """Create default configuration file"""
    config = IOAConfig()
    config.to_yaml("config.yaml")
    
    print("Default configuration created at config.yaml")
    print("Edit the configuration file to customize parameters")

if __name__ == "__main__":
    # Create sample data and configuration
    builder = DatasetBuilder()
    
    # Create sample datasets
    math_data = builder.create_math_seed_data()
    prog_data = builder.create_programming_seed_data()
    
    # Create probe tasks  
    math_probes = builder.create_probe_tasks("mathematics")
    prog_probes = builder.create_probe_tasks("programming")
    
    # Create default config
    create_default_config()
    
    print("Sample data and configuration files created successfully!")
    print("Files created:")
    print("- data/math_seed_data.json")
    print("- data/programming_seed_data.json") 
    print("- data/mathematics_probe_tasks.json")
    print("- data/programming_probe_tasks.json")
    print("- config.yaml")