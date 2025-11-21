"""
Evaluator Module for Model Performance Assessment

This module implements evaluation metrics and benchmarks as described
in Section 4.1 and Appendix D.

Evaluation types:
- Instruction Following: ROUGE-L (DollyEval, VicunaEval)
- Reasoning: Pass@k (GSM8K, MATH, AIME2024)
- Code Generation: Pass@k (HumanEval, MBPP, LiveCodeBench)
- Academic QA: Accuracy (GPQA-Diamond)
"""

import json
import logging
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from ..data.data_utils import (
    compute_rouge_l,
    compute_exact_match,
    extract_answer
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates model performance on various benchmarks.
    """
    
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cuda"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Store evaluation results
        self.results: Dict[str, Any] = {}
    
    def evaluate_rouge_l(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute average ROUGE-L score.
        
        Used for instruction following benchmarks (DollyEval, VicunaEval).
        
        Args:
            predictions: Model predictions
            references: Ground truth references
        
        Returns:
            Average ROUGE-L F1 score
        """
        if not predictions or not references:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = compute_rouge_l(pred, ref)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        
        logger.info(f"ROUGE-L: {avg_score:.4f} ({len(scores)} samples)")
        
        return avg_score
    
    def evaluate_pass_at_k(
        self,
        predictions: List[List[str]],
        references: List[str],
        k: int = 1
    ) -> float:
        """
        Compute Pass@k metric.
        
        Pass@k is the probability that at least one of k samples is correct.
        Used for reasoning and code generation benchmarks.
        
        Args:
            predictions: List of k predictions per sample
            references: Ground truth references
            k: Number of attempts
        
        Returns:
            Pass@k score
        """
        if not predictions or not references:
            return 0.0
        
        passed = 0
        
        for preds, ref in zip(predictions, references):
            # Check if any prediction matches
            for pred in preds[:k]:
                if self._check_correctness(pred, ref):
                    passed += 1
                    break
        
        pass_rate = passed / len(references)
        
        logger.info(f"Pass@{k}: {pass_rate:.4f} ({passed}/{len(references)})")
        
        return pass_rate
    
    def _check_correctness(
        self,
        prediction: str,
        reference: str,
        task_type: str = "general"
    ) -> bool:
        """
        Check if prediction is correct.
        
        Args:
            prediction: Model prediction
            reference: Ground truth
            task_type: Type of task (math, code, general)
        
        Returns:
            True if correct
        """
        # Extract answer for math problems
        if task_type == "math":
            pred_answer = extract_answer(prediction)
            ref_answer = extract_answer(reference)
            return compute_exact_match(pred_answer, ref_answer)
        
        # Execute code for code problems
        if task_type == "code":
            return self._evaluate_code(prediction, reference)
        
        # General exact match
        if compute_exact_match(prediction, reference):
            return True
        
        # Fallback to ROUGE-L threshold
        rouge = compute_rouge_l(prediction, reference)
        return rouge >= 0.7
    
    def _evaluate_code(
        self,
        code: str,
        test_cases: str,
        timeout: int = 5
    ) -> bool:
        """
        Execute code and check against test cases.
        
        Args:
            code: Generated code
            test_cases: Test cases to run
            timeout: Execution timeout in seconds
        
        Returns:
            True if all tests pass
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                # Write code and tests
                f.write(code)
                f.write("\n\n")
                f.write(test_cases)
                temp_path = f.name
            
            # Execute
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            
            # Clean up
            Path(temp_path).unlink()
            
            return result.returncode == 0
            
        except Exception as e:
            logger.debug(f"Code execution failed: {e}")
            return False
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        data: List[Dict[str, str]],
        num_samples: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate model on a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            data: Evaluation data with 'input' and 'output' keys
            num_samples: Number of samples to generate per input
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {benchmark_name} with {len(data)} examples")
        
        if self.model is None:
            logger.warning("No model provided for evaluation")
            return {"score": 0.0}
        
        predictions = []
        references = []
        
        for item in data:
            input_text = item.get("input", "")
            reference = item.get("output", "")
            
            # Generate predictions
            if num_samples == 1:
                pred = self._generate(input_text)
                predictions.append([pred])
            else:
                preds = [
                    self._generate(input_text, temperature=0.7)
                    for _ in range(num_samples)
                ]
                predictions.append(preds)
            
            references.append(reference)
        
        # Compute metrics based on benchmark type
        if benchmark_name in ["DollyEval", "VicunaEval"]:
            # Instruction following - use ROUGE-L
            flat_preds = [p[0] for p in predictions]
            score = self.evaluate_rouge_l(flat_preds, references)
            metrics = {"rouge_l": score}
            
        elif benchmark_name in ["GSM8K", "MATH", "AIME2024"]:
            # Math reasoning - use Pass@k
            score = self.evaluate_pass_at_k(predictions, references, k=1)
            metrics = {"pass_at_1": score, "task_type": "math"}
            
        elif benchmark_name in ["HumanEval", "MBPP", "LiveCodeBench"]:
            # Code generation - use Pass@k
            score = self.evaluate_pass_at_k(predictions, references, k=1)
            metrics = {"pass_at_1": score, "task_type": "code"}
            
        elif benchmark_name in ["GPQA-D", "GPQA-Diamond"]:
            # Academic QA - use accuracy
            correct = sum(
                1 for p, r in zip(predictions, references)
                if self._check_correctness(p[0], r)
            )
            score = correct / len(references) if references else 0.0
            metrics = {"accuracy": score}
            
        else:
            # Default to Pass@1
            score = self.evaluate_pass_at_k(predictions, references, k=1)
            metrics = {"pass_at_1": score}
        
        # Store results
        self.results[benchmark_name] = metrics
        
        return metrics
    
    def _generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> str:
        """Generate response from model"""
        if self.model is None or self.tokenizer is None:
            return ""
        
        try:
            import torch
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
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
    
    def evaluate_all_benchmarks(
        self,
        benchmarks: Dict[str, List[Dict]],
        num_samples: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on multiple benchmarks.
        
        Args:
            benchmarks: Dict mapping benchmark name to evaluation data
            num_samples: Number of samples per input
        
        Returns:
            Dictionary with all benchmark results
        """
        all_results = {}
        
        for benchmark_name, data in benchmarks.items():
            metrics = self.evaluate_benchmark(
                benchmark_name=benchmark_name,
                data=data,
                num_samples=num_samples
            )
            all_results[benchmark_name] = metrics
        
        # Compute aggregate metrics
        all_results["aggregate"] = self._compute_aggregate_metrics(all_results)
        
        return all_results
    
    def _compute_aggregate_metrics(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute aggregate metrics across benchmarks"""
        instruction_scores = []
        reasoning_scores = []
        code_scores = []
        
        for benchmark, metrics in results.items():
            if benchmark in ["DollyEval", "VicunaEval"]:
                instruction_scores.append(metrics.get("rouge_l", 0))
            elif benchmark in ["GSM8K", "MATH", "AIME2024"]:
                reasoning_scores.append(metrics.get("pass_at_1", 0))
            elif benchmark in ["HumanEval", "MBPP", "LiveCodeBench"]:
                code_scores.append(metrics.get("pass_at_1", 0))
        
        aggregate = {}
        
        if instruction_scores:
            aggregate["avg_instruction"] = sum(instruction_scores) / len(instruction_scores)
        if reasoning_scores:
            aggregate["avg_reasoning"] = sum(reasoning_scores) / len(reasoning_scores)
        if code_scores:
            aggregate["avg_code"] = sum(code_scores) / len(code_scores)
        
        # Overall average
        all_scores = instruction_scores + reasoning_scores + code_scores
        if all_scores:
            aggregate["overall"] = sum(all_scores) / len(all_scores)
        
        return aggregate
    
    def get_results_summary(self) -> str:
        """Get a summary of evaluation results"""
        summary = ["Evaluation Results:"]
        summary.append("-" * 50)
        
        for benchmark, metrics in self.results.items():
            if benchmark == "aggregate":
                continue
            
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
                if isinstance(v, (int, float))
            )
            summary.append(f"  {benchmark}: {metric_str}")
        
        # Aggregate
        if "aggregate" in self.results:
            summary.append("")
            summary.append("Aggregate:")
            for k, v in self.results["aggregate"].items():
                summary.append(f"  {k}: {v:.4f}")
        
        return "\n".join(summary)
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load evaluation results from file"""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")


def evaluate_distillation_quality(
    teacher_results: Dict[str, float],
    student_results: Dict[str, float]
) -> Dict[str, float]:
    """
    Evaluate distillation quality by comparing teacher and student.
    
    Args:
        teacher_results: Teacher model benchmark scores
        student_results: Student model benchmark scores
    
    Returns:
        Dictionary with retention ratios per benchmark
    """
    retention = {}
    
    for benchmark, teacher_score in teacher_results.items():
        student_score = student_results.get(benchmark, 0.0)
        
        if teacher_score > 0:
            ratio = student_score / teacher_score
        else:
            ratio = 1.0 if student_score == 0 else 0.0
        
        retention[benchmark] = ratio
    
    # Average retention
    if retention:
        retention["average"] = sum(retention.values()) / len(retention)
    
    logger.info(f"Average distillation retention: {retention.get('average', 0):.2%}")
    
    return retention


def compare_with_baselines(
    our_results: Dict[str, float],
    baseline_results: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare results with baseline methods.
    
    Args:
        our_results: Our method's results
        baseline_results: Dict mapping method name to results
    
    Returns:
        Comparison statistics
    """
    comparison = {}
    
    for benchmark, our_score in our_results.items():
        comparison[benchmark] = {"ours": our_score}
        
        for method, results in baseline_results.items():
            baseline_score = results.get(benchmark, 0.0)
            comparison[benchmark][method] = baseline_score
            
            # Compute improvement
            if baseline_score > 0:
                improvement = (our_score - baseline_score) / baseline_score * 100
                comparison[benchmark][f"{method}_improvement"] = improvement
    
    return comparison


if __name__ == "__main__":
    # Test the evaluator
    evaluator = Evaluator()
    
    # Test ROUGE-L
    predictions = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a programming language"
    ]
    references = [
        "The fast brown fox leaps over the lazy dog",
        "Python is a popular programming language"
    ]
    
    rouge = evaluator.evaluate_rouge_l(predictions, references)
    print(f"ROUGE-L: {rouge:.4f}")
    
    # Test Pass@1
    predictions_multi = [
        ["x = 4", "x = 3"],
        ["def f(): return 1", "def f(): pass"]
    ]
    references_multi = ["x = 4", "def f(): return 1"]
    
    pass_at_1 = evaluator.evaluate_pass_at_k(predictions_multi, references_multi, k=1)
    print(f"Pass@1: {pass_at_1:.4f}")