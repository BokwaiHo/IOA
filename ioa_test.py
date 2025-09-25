
"""
Test suite for IOA framework
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ioa_framework import IOAIdentifier, IOAOrganizer, IOAAdapter, IOAFramework, KnowledgeModule, ProbeTask
from ioa_config import DatasetBuilder, IOAConfig, EvaluationMetrics, PromptTemplates
from ioa_utils import IOALogger, ExperimentTracker, CacheManager, DataValidator, ConfigValidator

class TestIOAIdentifier(unittest.TestCase):
    """Test IOAIdentifier functionality"""
    
    def setUp(self):
        self.identifier = IOAIdentifier()
        
    def test_knowledge_decomposition(self):
        """Test knowledge decomposition"""
        modules = self.identifier.decompose_knowledge("mathematics")
        self.assertGreater(len(modules), 0)
        self.assertIsInstance(modules[0], KnowledgeModule)
        
        # Check prerequisite structure
        arithmetic_module = next((m for m in modules if m.id == "arithmetic"), None)
        self.assertIsNotNone(arithmetic_module)
        self.assertEqual(len(arithmetic_module.prerequisites), 0)
        
    def test_dependency_graph_construction(self):
        """Test dependency graph construction"""
        modules = self.identifier.decompose_knowledge("mathematics")
        graph = self.identifier.construct_dependency_graph(modules)
        
        self.assertGreater(graph.number_of_nodes(), 0)
        # Check that nodes have correct attributes
        for node in graph.nodes():
            self.assertIn(node, [m.id for m in modules])

class TestIOAOrganizer(unittest.TestCase):
    """Test IOAOrganizer functionality"""
    
    def setUp(self):
        self.organizer = IOAOrganizer()
        self.identifier = IOAIdentifier()
        
    def test_curriculum_construction(self):
        """Test curriculum sequence construction"""
        modules = self.identifier.decompose_knowledge("mathematics")
        graph = self.identifier.construct_dependency_graph(modules)
        target_modules = [m.id for m in modules[:3]]
        
        curriculum = self.organizer.construct_curriculum_sequence(target_modules, graph)
        
        self.assertGreater(len(curriculum), 0)
        self.assertIsInstance(curriculum[0], list)
        
        # Check that all modules are included
        all_modules_in_curriculum = set()
        for stage in curriculum:
            all_modules_in_curriculum.update(stage)
            
        self.assertTrue(set(target_modules).issubset(all_modules_in_curriculum))

class TestDatasetBuilder(unittest.TestCase):
    """Test DatasetBuilder functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.builder = DatasetBuilder(self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_seed_data_creation(self):
        """Test seed data creation"""
        math_data = self.builder.create_math_seed_data()
        self.assertGreater(len(math_data), 0)
        
        # Check required fields
        for item in math_data:
            self.assertIn("problem", item)
            self.assertIn("answer", item)
            self.assertIn("module", item)
            
        # Check file creation
        self.assertTrue((Path(self.temp_dir) / "math_seed_data.json").exists())
        
    def test_probe_tasks_creation(self):
        """Test probe task creation"""
        probe_tasks = self.builder.create_probe_tasks("mathematics")
        self.assertGreater(len(probe_tasks), 0)
        
        # Check required fields
        for task in probe_tasks:
            self.assertIn("module_id", task)
            self.assertIn("question", task)
            self.assertIn("answer", task)
            self.assertIn("task_type", task)

class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics"""
    
    def test_rouge_l(self):
        """Test ROUGE-L calculation"""
        pred = "the cat sat on the mat"
        ref = "cat sat on mat"
        
        score = EvaluationMetrics.rouge_l(pred, ref)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
        
        # Perfect match
        perfect_score = EvaluationMetrics.rouge_l("hello world", "hello world")
        self.assertEqual(perfect_score, 1.0)
        
        # No match
        no_match_score = EvaluationMetrics.rouge_l("abc", "def")
        self.assertEqual(no_match_score, 0.0)
        
    def test_pass_at_k(self):
        """Test Pass@K calculation"""
        predictions = ["correct answer", "wrong answer", "correct answer"]
        reference = "correct answer"
        
        pass_1 = EvaluationMetrics.pass_at_k(predictions, reference, k=1)
        pass_2 = EvaluationMetrics.pass_at_k(predictions, reference, k=2)
        
        self.assertGreaterEqual(pass_2, pass_1)
        self.assertLessEqual(pass_1, 1.0)
        self.assertLessEqual(pass_2, 1.0)

class TestDataValidator(unittest.TestCase):
    """Test data validation"""
    
    def test_solution_format_validation(self):
        """Test solution format validation"""
        valid_solution = {
            "steps": ["Step 1: Do this", "Step 2: Do that"],
            "final_answer": "42",
            "verification": "Correct"
        }
        
        is_valid, errors = DataValidator.validate_solution_format(valid_solution)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid solution
        invalid_solution = {
            "steps": [],  # Empty steps
            "final_answer": "42"
            # Missing verification
        }
        
        is_valid, errors = DataValidator.validate_solution_format(invalid_solution)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
    def test_content_quality_check(self):
        """Test content quality checking"""
        good_text = "This is a well-formed sentence with numbers like 42 and math symbols like x = 5."
        metrics = DataValidator.check_content_quality(good_text)
        
        self.assertTrue(metrics["valid"])
        self.assertGreater(metrics["word_count"], 0)
        self.assertTrue(metrics["has_numbers"])
        self.assertTrue(metrics["has_math_symbols"])
        
        # Test empty text
        empty_metrics = DataValidator.check_content_quality("")
        self.assertFalse(empty_metrics["valid"])

class TestConfigValidator(unittest.TestCase):
    """Test configuration validation"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        valid_config = {
            "tau_gap": 0.3,
            "tau_high": 0.9,
            "tau_low": 0.7,
            "tau_dep": 0.3,
            "tau_zpd": 0.15,
            "tau_mastery": 0.9,
            "num_examples_per_stage": 10
        }
        
        is_valid, errors = ConfigValidator.validate_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid config
        invalid_config = {
            "tau_gap": 1.5,  # Invalid range
            "tau_mastery": 0.3,  # Too low
            "num_examples_per_stage": -1  # Negative
        }
        
        is_valid, errors = ConfigValidator.validate_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_experiment_tracker(self):
        """Test experiment tracking"""
        tracker = ExperimentTracker(self.temp_dir)
        
        exp_id = tracker.start_experiment("test_exp", {"param": "value"})
        self.assertIsNotNone(exp_id)
        
        tracker.log_metric("accuracy", 0.85)
        tracker.save_artifact({"test": "data"}, "test_artifact")
        
        final_id = tracker.finish_experiment()
        self.assertEqual(exp_id, final_id)
        
        # Check files were created
        exp_path = Path(self.temp_dir) / exp_id
        self.assertTrue(exp_path.exists())
        self.assertTrue((exp_path / "summary.json").exists())
        self.assertTrue((exp_path / "config.json").exists())
        
    def test_cache_manager(self):
        """Test cache manager"""
        cache = CacheManager(self.temp_dir)
        
        # Test basic caching
        cache.set("test_key", {"data": "value"})
        retrieved = cache.get("test_key")
        self.assertEqual(retrieved, {"data": "value"})
        
        # Test cache miss
        miss = cache.get("nonexistent_key")
        self.assertIsNone(miss)

def run_integration_test():
    """Run integration test of the full pipeline"""
    print("Running integration test...")
    
    try:
        # Setup
        builder = DatasetBuilder()
        seed_data = builder.create_math_seed_data()
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
        
        # Test identifier
        identifier = IOAIdentifier()
        modules = identifier.decompose_knowledge("mathematics")
        graph = identifier.construct_dependency_graph(modules)
        
        print(f"✓ Created {len(modules)} knowledge modules")
        print(f"✓ Constructed dependency graph with {graph.number_of_nodes()} nodes")
        
        # Test organizer
        organizer = IOAOrganizer()
        target_modules = [m.id for m in modules[:3]]
        curriculum = organizer.construct_curriculum_sequence(target_modules, graph)
        
        print(f"✓ Created curriculum with {len(curriculum)} stages")
        
        # Test data validation
        for item in seed_data:
            if "solution" in item and isinstance(item["solution"], dict):
                is_valid, errors = DataValidator.validate_solution_format(item["solution"])
                if not is_valid:
                    print(f"! Validation warning: {errors}")
                    
        print("✓ Data validation completed")
        
        # Test evaluation metrics
        test_pred = "The answer is 42"
        test_ref = "42"
        rouge_score = EvaluationMetrics.rouge_l(test_pred, test_ref)
        print(f"✓ ROUGE-L test score: {rouge_score:.3f}")
        
        print("✓ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Main execution function for testing and demo"""
    print("IOA Framework Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("\n1. Running unit tests...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✓ All unit tests passed!")
    else:
        print("✗ Some unit tests failed")
        return False
        
    # Run integration test
    print("\n2. Running integration test...")
    integration_success = run_integration_test()
    
    if integration_success:
        print("\n✓ All tests completed successfully!")
        print("\nTo run the full IOA framework:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("2. Run: python ioa_training.py")
        return True
    else:
        print("\n✗ Integration test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)