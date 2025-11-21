"""
Data Synthesizer Module

This module provides a high-level interface for synthetic data generation,
coordinating the Adapter module's knowledge representation adaptation with
the prompts module to generate pedagogically-adapted training data.

The synthesizer implements the data generation workflow from Algorithm 1,
specifically lines 9 and 12 which call AdaptKnowledgeRepresentation and
GenerateRemedialData respectively.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..config.config import AdapterConfig, IOAConfig
from ..utils.llm_client import LLMClient
from ..data.seed_data import SeedDataset, SeedDataItem
from ..data.data_utils import (
    SyntheticDataItem,
    save_synthetic_data,
    load_synthetic_data,
    validate_synthetic_item,
    parse_llm_json_response,
    filter_by_verification,
    deduplicate_items
)
from ..modules.organizer import Curriculum, CurriculumStage
from .prompts import (
    SYSTEM_PROMPT_SYNTHESIS,
    SYSTEM_PROMPT_REMEDIAL,
    SYSTEM_PROMPT_BRIDGING,
    get_synthesis_user_prompt,
    get_remedial_prompt,
    get_bridging_prompt,
    get_difficulty_constraints,
    create_few_shot_examples
)

logger = logging.getLogger(__name__)


class DataSynthesizer:
    """
    High-level data synthesizer for IOA knowledge distillation.
    
    This class coordinates the synthesis pipeline, generating pedagogically
    adapted synthetic data based on the curriculum stages from the Organizer.
    """
    
    def __init__(
        self,
        config: AdapterConfig,
        teacher_client: LLMClient,
        seed_dataset: SeedDataset,
        output_dir: str = "./outputs/synthetic"
    ):
        """
        Initialize the data synthesizer.
        
        Args:
            config: Adapter configuration
            teacher_client: LLM client for teacher model
            seed_dataset: Seed dataset for synthesis
            output_dir: Directory for saving synthetic data
        """
        self.config = config
        self.teacher_client = teacher_client
        self.seed_dataset = seed_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_seeds_processed": 0,
            "total_items_generated": 0,
            "items_passed_validation": 0,
            "items_passed_verification": 0,
            "final_items": 0,
            "stages_completed": 0
        }
        
        # Cache for generated data
        self.generated_data: Dict[str, List[SyntheticDataItem]] = {}
        
        logger.info(f"DataSynthesizer initialized with output_dir={output_dir}")
    
    def synthesize_for_curriculum(
        self,
        curriculum: Curriculum,
        domain: str
    ) -> Dict[str, List[SyntheticDataItem]]:
        """
        Generate synthetic data for an entire curriculum.
        
        This implements the data synthesis loop from Algorithm 1 (lines 7-14).
        
        Args:
            curriculum: Organized curriculum from Organizer
            domain: Target domain
        
        Returns:
            Dictionary mapping stage_id to list of synthetic items
        """
        logger.info(f"Starting synthesis for curriculum with {len(curriculum.stages)} stages")
        
        all_synthetic_data = {}
        
        for stage in curriculum.stages:
            logger.info(f"\n{'='*50}")
            logger.info(f"Synthesizing data for stage {stage.stage_id}")
            logger.info(f"Modules: {stage.modules}")
            logger.info(f"{'='*50}")
            
            # Get seed items for this stage
            seed_items = self._get_seeds_for_stage(stage, domain)
            
            if not seed_items:
                logger.warning(f"No seed items found for stage {stage.stage_id}")
                all_synthetic_data[stage.stage_id] = []
                continue
            
            # Determine difficulty based on stage index
            difficulty = self._get_stage_difficulty(stage)
            
            # Generate synthetic data
            synthetic_items = self.synthesize_for_stage(
                stage_id=stage.stage_id,
                knowledge_modules=stage.modules,
                prerequisites=list(stage.prerequisites),
                seed_items=seed_items,
                domain=domain,
                difficulty=difficulty
            )
            
            all_synthetic_data[stage.stage_id] = synthetic_items
            self.stats["stages_completed"] += 1
            
            logger.info(f"Stage {stage.stage_id}: generated {len(synthetic_items)} items")
        
        # Save all data
        self._save_all_data(all_synthetic_data, domain)
        
        # Store in cache
        self.generated_data = all_synthetic_data
        
        logger.info(f"\nSynthesis complete. Total items: {self.stats['final_items']}")
        
        return all_synthetic_data
    
    def synthesize_for_stage(
        self,
        stage_id: str,
        knowledge_modules: List[str],
        prerequisites: List[str],
        seed_items: List[SeedDataItem],
        domain: str,
        difficulty: str = "intermediate",
        baseline_ratio: float = 0.5
    ) -> List[SyntheticDataItem]:
        """
        Generate synthetic data for a single curriculum stage.
        
        Args:
            stage_id: Stage identifier
            knowledge_modules: Target modules
            prerequisites: Prerequisite modules
            seed_items: Seed data items
            domain: Target domain
            difficulty: Difficulty level
            baseline_ratio: Student's baseline performance
        
        Returns:
            List of synthetic data items
        """
        all_items = []
        
        # Get difficulty constraints
        constraints = get_difficulty_constraints(difficulty)
        
        # Process each seed
        for i, seed in enumerate(seed_items):
            logger.debug(f"Processing seed {i+1}/{len(seed_items)}: {seed.item_id}")
            
            items = self._generate_from_seed(
                seed=seed,
                stage_id=stage_id,
                knowledge_modules=knowledge_modules,
                prerequisites=prerequisites,
                domain=domain,
                constraints=constraints,
                baseline_ratio=baseline_ratio
            )
            
            all_items.extend(items)
            self.stats["total_seeds_processed"] += 1
        
        # Post-processing pipeline
        all_items = self._postprocess_items(all_items)
        
        self.stats["final_items"] += len(all_items)
        
        return all_items
    
    def _generate_from_seed(
        self,
        seed: SeedDataItem,
        stage_id: str,
        knowledge_modules: List[str],
        prerequisites: List[str],
        domain: str,
        constraints: Dict[str, str],
        baseline_ratio: float
    ) -> List[SyntheticDataItem]:
        """
        Generate synthetic items from a single seed.
        
        Args:
            seed: Seed data item
            stage_id: Stage ID
            knowledge_modules: Target modules
            prerequisites: Prerequisites
            domain: Domain name
            constraints: Difficulty constraints
            baseline_ratio: Student baseline
        
        Returns:
            List of synthetic items
        """
        # Build user prompt
        user_prompt = get_synthesis_user_prompt(
            domain=domain,
            stage_id=stage_id,
            knowledge_modules=knowledge_modules,
            prerequisites=prerequisites,
            num_examples=self.config.num_samples_per_seed,
            size_cap=constraints["size_cap"],
            complexity_cap=constraints["complexity_cap"],
            baseline_ratio=baseline_ratio
        )
        
        # Add seed context
        user_prompt += f"""

Seed Example Context:
- Input: {seed.input_text[:500]}...
- Domain: {seed.domain}
- Knowledge Module: {seed.knowledge_module if hasattr(seed, 'knowledge_module') else 'general'}

Generate {self.config.num_samples_per_seed} new examples inspired by this seed but with varied content."""

        # Add few-shot examples
        few_shot = create_few_shot_examples(domain, num_examples=1)
        if few_shot:
            user_prompt += f"\n\nReference Format:\n{few_shot}"
        
        # Generate via teacher
        try:
            response = self.teacher_client.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT_SYNTHESIS,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.generation_temperature
            )
            
            # Parse response
            items_data = parse_llm_json_response(response)
            self.stats["total_items_generated"] += len(items_data)
            
            # Convert to SyntheticDataItem objects
            items = []
            for item_data in items_data:
                is_valid, error = validate_synthetic_item(item_data)
                if is_valid:
                    self.stats["items_passed_validation"] += 1
                    item = SyntheticDataItem.from_dict(item_data)
                    item.stage_id = stage_id
                    item.seed_id = seed.item_id
                    items.append(item)
                else:
                    logger.debug(f"Validation failed: {error}")
            
            return items
            
        except Exception as e:
            logger.error(f"Generation failed for seed {seed.item_id}: {e}")
            return []
    
    def generate_remedial_data(
        self,
        stage: CurriculumStage,
        weak_modules: List[str],
        domain: str,
        num_examples: int = 10
    ) -> List[SyntheticDataItem]:
        """
        Generate remedial data for modules that haven't reached mastery.
        
        This implements line 12 of Algorithm 1:
        S ← FineTune(S, GenerateRemedialData(T, s_i, D_seed))
        
        Args:
            stage: Current curriculum stage
            weak_modules: Modules needing remediation
            domain: Target domain
            num_examples: Number of remedial examples
        
        Returns:
            List of remedial synthetic items
        """
        logger.info(f"Generating remedial data for {len(weak_modules)} weak modules")
        
        # Get seed items for weak modules
        seed_items = []
        for module in weak_modules:
            module_seeds = self.seed_dataset.get_train_items_by_module(module)
            seed_items.extend(module_seeds[:3])  # Limit seeds per module
        
        if not seed_items:
            # Fallback to stage modules
            seed_items = self._get_seeds_for_stage(stage, domain)[:5]
        
        # Create remedial prompt
        user_prompt = get_remedial_prompt(
            stage_id=stage.stage_id,
            knowledge_modules=stage.modules,
            weak_subskills=weak_modules,
            num_examples=num_examples
        )
        
        # Add simplified context
        if seed_items:
            user_prompt += f"\n\nSimplify based on examples like: {seed_items[0].input_text[:200]}"
        
        # Generate
        try:
            response = self.teacher_client.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT_REMEDIAL,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.generation_temperature
            )
            
            items_data = parse_llm_json_response(response)
            
            items = []
            for item_data in items_data:
                is_valid, _ = validate_synthetic_item(item_data)
                if is_valid:
                    item = SyntheticDataItem.from_dict(item_data)
                    item.stage_id = stage.stage_id
                    item.is_remedial = True
                    items.append(item)
            
            logger.info(f"Generated {len(items)} remedial items")
            return items
            
        except Exception as e:
            logger.error(f"Remedial generation failed: {e}")
            return []
    
    def generate_bridging_data(
        self,
        stage: CurriculumStage,
        domain: str,
        num_examples: int = 5
    ) -> List[SyntheticDataItem]:
        """
        Generate bridging data with slightly increased complexity.
        
        Used after mastery is achieved to prepare for next stage.
        
        Args:
            stage: Current curriculum stage
            domain: Target domain
            num_examples: Number of bridging examples
        
        Returns:
            List of bridging synthetic items
        """
        logger.info(f"Generating bridging data for stage {stage.stage_id}")
        
        # Create bridging prompt
        user_prompt = get_bridging_prompt(
            stage_id=stage.stage_id,
            knowledge_modules=stage.modules,
            num_examples=num_examples
        )
        
        # Generate
        try:
            response = self.teacher_client.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT_BRIDGING,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.generation_temperature
            )
            
            items_data = parse_llm_json_response(response)
            
            items = []
            for item_data in items_data:
                is_valid, _ = validate_synthetic_item(item_data)
                if is_valid:
                    item = SyntheticDataItem.from_dict(item_data)
                    item.stage_id = stage.stage_id
                    item.is_bridging = True
                    items.append(item)
            
            logger.info(f"Generated {len(items)} bridging items")
            return items
            
        except Exception as e:
            logger.error(f"Bridging generation failed: {e}")
            return []
    
    def _get_seeds_for_stage(
        self,
        stage: CurriculumStage,
        domain: str
    ) -> List[SeedDataItem]:
        """Get seed items relevant to a curriculum stage"""
        seed_items = []
        
        # Get seeds by module
        for module in stage.modules:
            module_seeds = self.seed_dataset.get_train_items_by_module(module)
            seed_items.extend(module_seeds)
        
        # If no module-specific seeds, get by domain
        if not seed_items:
            domain_seeds = self.seed_dataset.get_train_items_by_domain(domain)
            # Sample a subset
            max_seeds = 20
            seed_items = domain_seeds[:max_seeds]
        
        return seed_items
    
    def _get_stage_difficulty(self, stage: CurriculumStage) -> str:
        """Determine difficulty level based on stage"""
        total_stages = 10  # Approximate
        ratio = stage.index / max(total_stages, 1)
        
        if ratio < 0.33:
            return "introductory"
        elif ratio < 0.67:
            return "intermediate"
        else:
            return "advanced"
    
    def _postprocess_items(
        self,
        items: List[SyntheticDataItem]
    ) -> List[SyntheticDataItem]:
        """
        Post-process generated items.
        
        Applies:
        1. Verification filtering
        2. Deduplication
        3. Quality checks
        
        Args:
            items: Raw generated items
        
        Returns:
            Filtered and cleaned items
        """
        if not items:
            return items
        
        original_count = len(items)
        
        # Filter by verification
        if self.config.enable_verification:
            items = filter_by_verification(items)
            self.stats["items_passed_verification"] += len(items)
            logger.debug(f"After verification: {len(items)}/{original_count}")
        
        # Deduplicate
        items = deduplicate_items(items)
        logger.debug(f"After deduplication: {len(items)}")
        
        return items
    
    def _save_all_data(
        self,
        data: Dict[str, List[SyntheticDataItem]],
        domain: str
    ) -> None:
        """Save all synthetic data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save per-stage data
        for stage_id, items in data.items():
            if items:
                filename = f"{domain}_{stage_id}_{timestamp}.json"
                filepath = self.output_dir / filename
                save_synthetic_data(items, str(filepath))
        
        # Save combined data
        all_items = []
        for items in data.values():
            all_items.extend(items)
        
        if all_items:
            combined_path = self.output_dir / f"{domain}_combined_{timestamp}.json"
            save_synthetic_data(all_items, str(combined_path))
            logger.info(f"Saved {len(all_items)} items to {combined_path}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get synthesis statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            "total_seeds_processed": 0,
            "total_items_generated": 0,
            "items_passed_validation": 0,
            "items_passed_verification": 0,
            "final_items": 0,
            "stages_completed": 0
        }
    
    def get_summary(self) -> str:
        """Get a summary of synthesis results"""
        lines = [
            "Synthesis Summary:",
            f"  Seeds processed: {self.stats['total_seeds_processed']}",
            f"  Items generated: {self.stats['total_items_generated']}",
            f"  Passed validation: {self.stats['items_passed_validation']}",
            f"  Passed verification: {self.stats['items_passed_verification']}",
            f"  Final items: {self.stats['final_items']}",
            f"  Stages completed: {self.stats['stages_completed']}"
        ]
        return "\n".join(lines)


def create_synthesizer(
    config: IOAConfig,
    teacher_client: LLMClient,
    seed_dataset: SeedDataset
) -> DataSynthesizer:
    """
    Factory function to create a DataSynthesizer.
    
    Args:
        config: IOA configuration
        teacher_client: Teacher LLM client
        seed_dataset: Seed dataset
    
    Returns:
        Configured DataSynthesizer instance
    """
    output_dir = Path(config.data.output_dir) / "synthetic"
    
    return DataSynthesizer(
        config=config.adapter,
        teacher_client=teacher_client,
        seed_dataset=seed_dataset,
        output_dir=str(output_dir)
    )


if __name__ == "__main__":
    # Test the synthesizer
    from ..config.config import get_default_config
    
    config = get_default_config()
    print("DataSynthesizer module loaded successfully")
    print(f"  Samples per seed: {config.adapter.num_samples_per_seed}")
    print(f"  Max generation tokens: {config.adapter.max_generation_tokens}")