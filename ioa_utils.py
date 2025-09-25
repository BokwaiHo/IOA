"""
Utility functions, logging, and helper classes for IOA framework
"""

import logging
import json
import pickle
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import torch
from contextlib import contextmanager

class IOALogger:
    """Centralized logging for IOA framework"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("IOA")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = self.log_dir / f"ioa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def info(self, message: str):
        self.logger.info(message)
        
    def debug(self, message: str):
        self.logger.debug(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)

class ExperimentTracker:
    """Track experiments and results"""
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_experiment = None
        
    def start_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """Start a new experiment"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{name}_{timestamp}"
        
        experiment_path = self.experiment_dir / experiment_id
        experiment_path.mkdir(exist_ok=True)
        
        # Save config
        with open(experiment_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        self.current_experiment = {
            "id": experiment_id,
            "path": experiment_path,
            "start_time": time.time(),
            "config": config,
            "metrics": {},
            "artifacts": []
        }
        
        return experiment_id
        
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric"""
        if self.current_experiment is None:
            return
            
        if name not in self.current_experiment["metrics"]:
            self.current_experiment["metrics"][name] = []
            
        self.current_experiment["metrics"][name].append({
            "value": value,
            "step": step,
            "timestamp": time.time()
        })
        
    def save_artifact(self, artifact: Any, name: str, artifact_type: str = "json"):
        """Save experiment artifact"""
        if self.current_experiment is None:
            return
            
        artifact_path = self.current_experiment["path"] / f"{name}.{artifact_type}"
        
        if artifact_type == "json":
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=2, default=str)
        elif artifact_type == "pickle":
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
        elif artifact_type == "txt":
            with open(artifact_path, 'w') as f:
                f.write(str(artifact))
                
        self.current_experiment["artifacts"].append({
            "name": name,
            "type": artifact_type,
            "path": str(artifact_path)
        })
        
    def finish_experiment(self):
        """Finish current experiment"""
        if self.current_experiment is None:
            return
            
        self.current_experiment["end_time"] = time.time()
        self.current_experiment["duration"] = (
            self.current_experiment["end_time"] - self.current_experiment["start_time"]
        )
        
        # Save experiment summary
        with open(self.current_experiment["path"] / "summary.json", 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)
            
        experiment_id = self.current_experiment["id"]
        self.current_experiment = None
        return experiment_id

class CacheManager:
    """Cache manager for expensive operations"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached item"""
        cache_file = self.cache_dir / f"{key}.pickle"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
        
    def set(self, key: str, value: Any):
        """Set cached item"""
        cache_file = self.cache_dir / f"{key}.pickle"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
            
    def cached_function(self, func):
        """Decorator for caching function results"""
        def wrapper(*args, **kwargs):
            cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
            result = self.get(cache_key)
            if result is not None:
                return result
                
            result = func(*args, **kwargs)
            self.set(cache_key, result)
            return result
        return wrapper

@contextmanager
def timer(name: str, logger: IOALogger = None):
    """Context manager for timing operations"""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        message = f"{name} completed in {duration:.2f} seconds"
        if logger:
            logger.info(message)
        else:
            print(message)

class ModelUtils:
    """Utilities for model operations"""
    
    @staticmethod
    def count_parameters(model) -> int:
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    @staticmethod
    def get_model_size(model) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
        
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
            }
        return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}

class DataValidator:
    """Validate synthesized data quality"""
    
    @staticmethod
    def validate_json_schema(data: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validate JSON data against required schema"""
        errors = []
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                
        return len(errors) == 0, errors
        
    @staticmethod
    def validate_solution_format(solution: Dict) -> Tuple[bool, List[str]]:
        """Validate solution format"""
        errors = []
        required_fields = ["steps", "final_answer", "verification"]
        
        valid, field_errors = DataValidator.validate_json_schema(solution, required_fields)
        errors.extend(field_errors)
        
        # Validate steps
        if "steps" in solution:
            steps = solution["steps"]
            if not isinstance(steps, list):
                errors.append("Steps must be a list")
            elif len(steps) == 0:
                errors.append("Steps list cannot be empty")
            else:
                for i, step in enumerate(steps):
                    if not isinstance(step, str) or not step.strip():
                        errors.append(f"Step {i+1} must be a non-empty string")
                        
        return len(errors) == 0, errors
        
    @staticmethod
    def check_content_quality(text: str) -> Dict[str, Any]:
        """Check basic content quality metrics"""
        if not text:
            return {"valid": False, "reason": "Empty text"}
            
        # Basic checks
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        quality_metrics = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": word_count / max(sentence_count, 1),
            "has_numbers": any(char.isdigit() for char in text),
            "has_math_symbols": any(char in "+-*/=()[]" for char in text),
            "valid": True
        }
        
        # Quality thresholds
        if word_count < 3:
            quality_metrics["valid"] = False
            quality_metrics["reason"] = "Text too short"
        elif word_count > 1000:
            quality_metrics["valid"] = False  
            quality_metrics["reason"] = "Text too long"
            
        return quality_metrics

class ConfigValidator:
    """Validate configuration settings"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate IOA configuration"""
        errors = []
        
        # Check required fields
        required_fields = [
            "tau_gap", "tau_high", "tau_low", "tau_dep",
            "tau_zpd", "tau_mastery", "num_examples_per_stage"
        ]
        
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required config field: {field}")
                
        # Check value ranges
        if "tau_gap" in config and not (0 < config["tau_gap"] < 1):
            errors.append("tau_gap must be between 0 and 1")
            
        if "tau_mastery" in config and not (0.5 <= config["tau_mastery"] <= 1.0):
            errors.append("tau_mastery should be between 0.5 and 1.0")
            
        if "num_examples_per_stage" in config and config["num_examples_per_stage"] < 1:
            errors.append("num_examples_per_stage must be positive")
            
        return len(errors) == 0, errors

def setup_environment():
    """Setup environment variables and directories"""
    # Set tokenizers parallelism
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Create necessary directories
    dirs = ["logs", "experiments", "cache", "data", "models"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        
    print("Environment setup completed")

def load_json_safe(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def save_json_safe(data: Any, file_path: str) -> bool:
    """Safely save JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False

def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_system_info() -> Dict[str, Any]:
    """Get system information for logging"""
    info = {
        "python_version": os.sys.version,
        "torch_version": torch.__version__ if 'torch' in globals() else "not installed",
        "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
        "timestamp": datetime.now().isoformat()
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        
    return info

if __name__ == "__main__":
    # Test utilities
    print("Testing IOA utilities...")
    
    # Setup environment
    setup_environment()
    
    # Test logger
    logger = IOALogger()
    logger.info("Logger test successful")
    
    # Test experiment tracker
    tracker = ExperimentTracker()
    exp_id = tracker.start_experiment("test", {"param1": 1, "param2": "test"})
    tracker.log_metric("accuracy", 0.85)
    tracker.save_artifact({"test": "data"}, "test_artifact")
    tracker.finish_experiment()
    
    print(f"Experiment {exp_id} completed")
    
    # Test cache manager
    cache = CacheManager()
    
    @cache.cached_function
    def expensive_function(x):
        time.sleep(0.1)  # Simulate expensive operation
        return x * 2
        
    # First call (should be slow)
    with timer("First call"):
        result1 = expensive_function(5)
        
    # Second call (should be fast due to caching)
    with timer("Second call"):
        result2 = expensive_function(5)
        
    assert result1 == result2 == 10
    print("Cache test successful")
    
    # Test data validator
    test_solution = {
        "steps": ["Step 1: Do this", "Step 2: Do that"],
        "final_answer": "42",
        "verification": "Check: correct"
    }
    
    valid, errors = DataValidator.validate_solution_format(test_solution)
    print(f"Solution validation: {valid}, errors: {errors}")
    
    print("All utility tests passed!")