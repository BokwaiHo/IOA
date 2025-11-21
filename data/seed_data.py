"""
Seed Data Loading and Management

This module handles seed data for the IOA framework as described in Appendix B.

The seed dataset D_seed consists of ~3000 items across four domains:
- Instruction Following: ~800 items
- Math Problem Solving: ~900 items
- Code Generation: ~700 items
- Academic Knowledge Reasoning: ~600 items

The dataset is split into train/validation (8:2 ratio), with validation
reserved for probe tasks in knowledge deficiency diagnosis.
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SeedDataItem:
    """Represents a single seed data item"""
    
    # Unique identifier
    item_id: str
    
    # Domain category
    domain: str
    
    # Knowledge modules involved
    knowledge_modules: List[str] = field(default_factory=list)
    
    # Input/prompt
    input_text: str = ""
    
    # Expected output (for evaluation)
    output_text: str = ""
    
    # Difficulty level
    difficulty: str = "introductory"  # introductory, intermediate, advanced
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedDataset:
    """Container for seed data with train/val split"""
    
    # All data items
    items: List[SeedDataItem] = field(default_factory=list)
    
    # Train/val split indices
    train_indices: List[int] = field(default_factory=list)
    val_indices: List[int] = field(default_factory=list)
    
    # Domain index
    domain_index: Dict[str, List[int]] = field(default_factory=dict)
    
    # Knowledge module index
    module_index: Dict[str, List[int]] = field(default_factory=dict)
    
    def get_train_items(self) -> List[SeedDataItem]:
        """Get training set items"""
        return [self.items[i] for i in self.train_indices]
    
    def get_val_items(self) -> List[SeedDataItem]:
        """Get validation set items"""
        return [self.items[i] for i in self.val_indices]
    
    def get_items_by_domain(self, domain: str) -> List[SeedDataItem]:
        """Get all items for a specific domain"""
        indices = self.domain_index.get(domain, [])
        return [self.items[i] for i in indices]
    
    def get_items_by_module(self, module: str) -> List[SeedDataItem]:
        """Get all items for a specific knowledge module"""
        indices = self.module_index.get(module, [])
        return [self.items[i] for i in indices]
    
    def get_val_items_by_module(self, module: str) -> List[SeedDataItem]:
        """Get validation items for a specific knowledge module (for probe tasks)"""
        module_indices = set(self.module_index.get(module, []))
        val_set = set(self.val_indices)
        indices = module_indices.intersection(val_set)
        return [self.items[i] for i in indices]
    
    def get_train_items_by_module(self, module: str) -> List[SeedDataItem]:
        """Get training items for a specific knowledge module (for synthesis)"""
        module_indices = set(self.module_index.get(module, []))
        train_set = set(self.train_indices)
        indices = module_indices.intersection(train_set)
        return [self.items[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.items)


class SeedDataLoader:
    """
    Loader for seed data from various sources.
    
    Supports loading from:
    - JSON files
    - JSONL files
    - Directory of domain-specific files
    """
    
    def __init__(
        self,
        data_dir: str,
        train_val_split: float = 0.8,
        seed: int = 42
    ):
        """
        Initialize the seed data loader.
        
        Args:
            data_dir: Directory containing seed data files
            train_val_split: Ratio of training data (default 0.8 per Appendix B)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.train_val_split = train_val_split
        self.seed = seed
        random.seed(seed)
    
    def load(self) -> SeedDataset:
        """
        Load seed data from the data directory.
        
        Returns:
            SeedDataset with train/val split
        """
        items = []
        
        # Check for different file formats
        if (self.data_dir / "seed_data.json").exists():
            items = self._load_json(self.data_dir / "seed_data.json")
        elif (self.data_dir / "seed_data.jsonl").exists():
            items = self._load_jsonl(self.data_dir / "seed_data.jsonl")
        else:
            # Load from domain-specific directories
            items = self._load_from_domains()
        
        if not items:
            logger.warning(f"No seed data found in {self.data_dir}")
            items = self._create_placeholder_data()
        
        logger.info(f"Loaded {len(items)} seed data items")
        
        # Create dataset with indices
        dataset = self._create_dataset(items)
        
        return dataset
    
    def _load_json(self, filepath: Path) -> List[SeedDataItem]:
        """Load from a single JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        for i, item_data in enumerate(data):
            item = self._parse_item(item_data, i)
            items.append(item)
        
        return items
    
    def _load_jsonl(self, filepath: Path) -> List[SeedDataItem]:
        """Load from a JSONL file"""
        items = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    item_data = json.loads(line)
                    item = self._parse_item(item_data, i)
                    items.append(item)
        
        return items
    
    def _load_from_domains(self) -> List[SeedDataItem]:
        """Load from domain-specific subdirectories"""
        items = []
        item_id = 0
        
        domains = [
            "instruction_following",
            "math_problem_solving",
            "code_generation",
            "academic_knowledge_reasoning"
        ]
        
        for domain in domains:
            domain_dir = self.data_dir / domain
            if domain_dir.exists():
                for filepath in domain_dir.glob("*.json"):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item_data in data:
                            item_data["domain"] = domain
                            item = self._parse_item(item_data, item_id)
                            items.append(item)
                            item_id += 1
                    else:
                        data["domain"] = domain
                        item = self._parse_item(data, item_id)
                        items.append(item)
                        item_id += 1
        
        return items
    
    def _parse_item(self, data: Dict[str, Any], index: int) -> SeedDataItem:
        """Parse a dictionary into a SeedDataItem"""
        return SeedDataItem(
            item_id=data.get("id", f"seed_{index:05d}"),
            domain=data.get("domain", "unknown"),
            knowledge_modules=data.get("knowledge_modules", data.get("modules", [])),
            input_text=data.get("input", data.get("prompt", data.get("question", ""))),
            output_text=data.get("output", data.get("response", data.get("answer", ""))),
            difficulty=data.get("difficulty", "introductory"),
            metadata=data.get("metadata", {})
        )
    
    def _create_dataset(self, items: List[SeedDataItem]) -> SeedDataset:
        """Create a dataset with train/val split and indices"""
        dataset = SeedDataset(items=items)
        
        # Create indices
        indices = list(range(len(items)))
        random.shuffle(indices)
        
        # Split
        split_point = int(len(indices) * self.train_val_split)
        dataset.train_indices = indices[:split_point]
        dataset.val_indices = indices[split_point:]
        
        # Build domain index
        for i, item in enumerate(items):
            if item.domain not in dataset.domain_index:
                dataset.domain_index[item.domain] = []
            dataset.domain_index[item.domain].append(i)
        
        # Build module index
        for i, item in enumerate(items):
            for module in item.knowledge_modules:
                if module not in dataset.module_index:
                    dataset.module_index[module] = []
                dataset.module_index[module].append(i)
        
        logger.info(f"Dataset split: {len(dataset.train_indices)} train, "
                   f"{len(dataset.val_indices)} val")
        logger.info(f"Domains: {list(dataset.domain_index.keys())}")
        logger.info(f"Knowledge modules: {len(dataset.module_index)}")
        
        return dataset
    
    def _create_placeholder_data(self) -> List[SeedDataItem]:
        """
        Create placeholder seed data for testing.
        
        This generates minimal example data matching the paper's domain structure.
        In practice, this should be replaced with real seed data.
        """
        logger.warning("Creating placeholder seed data for testing")
        
        items = []
        item_id = 0
        
        # Instruction following examples
        instruction_modules = [
            "instruction/question-answering",
            "instruction/summarization",
            "instruction/creative-writing",
            "instruction/brainstorming"
        ]
        
        for module in instruction_modules:
            for difficulty in ["introductory", "intermediate", "advanced"]:
                items.append(SeedDataItem(
                    item_id=f"seed_{item_id:05d}",
                    domain="instruction_following",
                    knowledge_modules=[module],
                    input_text=f"Example {module} task at {difficulty} level",
                    output_text=f"Example response for {module}",
                    difficulty=difficulty
                ))
                item_id += 1
        
        # Math problem solving examples
        math_modules = [
            "math/arithmetic",
            "math/algebra/linear-equations",
            "math/algebra/quadratic-equations",
            "math/geometry/basic",
            "math/calculus/derivatives",
            "math/calculus/integrals"
        ]
        
        for module in math_modules:
            for difficulty in ["introductory", "intermediate", "advanced"]:
                items.append(SeedDataItem(
                    item_id=f"seed_{item_id:05d}",
                    domain="math_problem_solving",
                    knowledge_modules=[module],
                    input_text=f"Solve this {module} problem at {difficulty} level",
                    output_text=f"Solution for {module} problem",
                    difficulty=difficulty
                ))
                item_id += 1
        
        # Code generation examples
        code_modules = [
            "code/python/basics",
            "code/python/data-structures",
            "code/python/algorithms",
            "code/python/string-processing"
        ]
        
        for module in code_modules:
            for difficulty in ["introductory", "intermediate", "advanced"]:
                items.append(SeedDataItem(
                    item_id=f"seed_{item_id:05d}",
                    domain="code_generation",
                    knowledge_modules=[module],
                    input_text=f"Write a {module} function at {difficulty} level",
                    output_text=f"def example_function(): pass",
                    difficulty=difficulty
                ))
                item_id += 1
        
        # Academic knowledge reasoning examples
        academic_modules = [
            "academic/physics/mechanics",
            "academic/physics/electricity",
            "academic/chemistry/basics",
            "academic/biology/cells"
        ]
        
        for module in academic_modules:
            for difficulty in ["introductory", "intermediate", "advanced"]:
                items.append(SeedDataItem(
                    item_id=f"seed_{item_id:05d}",
                    domain="academic_knowledge_reasoning",
                    knowledge_modules=[module],
                    input_text=f"Explain this {module} concept at {difficulty} level",
                    output_text=f"Explanation of {module} concept",
                    difficulty=difficulty
                ))
                item_id += 1
        
        return items


def load_seed_data(
    data_dir: str,
    train_val_split: float = 0.8,
    seed: int = 42
) -> SeedDataset:
    """
    Convenience function to load seed data.
    
    Args:
        data_dir: Directory containing seed data
        train_val_split: Training data ratio (default 0.8)
        seed: Random seed
    
    Returns:
        Loaded SeedDataset
    """
    loader = SeedDataLoader(
        data_dir=data_dir,
        train_val_split=train_val_split,
        seed=seed
    )
    return loader.load()


def get_probe_tasks(
    dataset: SeedDataset,
    knowledge_module: str
) -> List[Dict[str, str]]:
    """
    Get probe tasks for evaluating performance on a knowledge module.
    
    This extracts validation items for a specific module,
    used in Equation 2 for computing performance gap.
    
    Args:
        dataset: The seed dataset
        knowledge_module: Module to get probe tasks for
    
    Returns:
        List of probe task dictionaries with 'input' and 'output' keys
    """
    items = dataset.get_val_items_by_module(knowledge_module)
    
    return [
        {
            "input": item.input_text,
            "output": item.output_text,
            "item_id": item.item_id
        }
        for item in items
    ]


def get_synthesis_seeds(
    dataset: SeedDataset,
    knowledge_module: str
) -> List[SeedDataItem]:
    """
    Get seed items for synthetic data generation.
    
    This extracts training items for a specific module,
    used as D_seed in Equation 1 for synthesis.
    
    Args:
        dataset: The seed dataset
        knowledge_module: Module to get seeds for
    
    Returns:
        List of seed data items for synthesis
    """
    return dataset.get_train_items_by_module(knowledge_module)


if __name__ == "__main__":
    # Test seed data loading
    loader = SeedDataLoader(
        data_dir="./data/seed",
        train_val_split=0.8,
        seed=42
    )
    
    dataset = loader.load()
    
    print(f"Total items: {len(dataset)}")
    print(f"Train items: {len(dataset.train_indices)}")
    print(f"Val items: {len(dataset.val_indices)}")
    print(f"Domains: {list(dataset.domain_index.keys())}")
    print(f"Modules: {list(dataset.module_index.keys())[:5]}...")