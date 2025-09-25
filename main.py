"""
Main execution script for IOA Framework
Usage: python main.py [mode] [options]
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from ioa_framework import IOAFramework
from ioa_config import DatasetBuilder, IOAConfig
from ioa_training import StudentModelTrainer, create_complete_example
from ioa_utils import IOALogger, ExperimentTracker, setup_environment

def run_tests():
    """Run test suite"""
    os.system("python ioa_tests.py")

def run_demo():
    """Run demonstration with mock data"""
    print("Running IOA Framework Demo")
    print("=" * 40)
    
    setup_environment()
    logger = IOALogger()
    
    try:
        # Create sample data
        builder = DatasetBuilder()
        math_data = builder.create_math_seed_data()
        logger.info(f"Created {len(math_data)} math examples")
        
        # Run mock training
        trainer = StudentModelTrainer("microsoft/DialoGPT-small")
        logger.info("Initialized student model trainer")
        
        # Mock synthesized data
        mock_data = [
            {
                "problem": "What is 2 + 3?",
                "solution": {
                    "steps": ["Step 1: Add 2 + 3", "Step 2: The result is 5"],
                    "final_answer": "5"
                },
                "module": "arithmetic"
            },
            {
                "problem": "Solve x + 2 = 5",
                "solution": {
                    "steps": ["Step 1: Subtract 2 from both sides", "Step 2: x = 3"],
                    "final_answer": "x = 3"
                },
                "module": "linear_equations"
            }
        ]
        
        # Create training dataset
        train_dataset = trainer.prepare_dataset(mock_data)
        logger.info(f"Prepared training dataset with {len(train_dataset)} examples")
        
        # Simulate training (shortened for demo)
        logger.info("Simulating model training...")
        # trainer.train(train_dataset, num_epochs=1, batch_size=1)
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        
def run_full_experiment():
    """Run full experiment with real API calls"""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return
        
    print("Running Full IOA Framework Experiment")
    print("=" * 45)
    
    setup_environment()
    create_complete_example()

def main():
    parser = argparse.ArgumentParser(description="IOA Framework - Pedagogically-Inspired Knowledge Distillation")
    parser.add_argument("mode", choices=["test", "demo", "full"], 
                       help="Mode to run: test (run tests), demo (mock demo), full (complete experiment)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    print("IOA Framework")
    print("=" * 30)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Log Level: {args.log_level}")
    print()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.mode == "test":
        run_tests()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "full":
        run_full_experiment()
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())