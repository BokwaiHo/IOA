"""
Knowledge Adapter Module

This module implements the Adapter component of the IOA framework
as described in Section 3.4 of the paper.

The Adapter answers "How to teach" by transforming knowledge representations
to match the cognitive capacity of student models through five strategies:

1. Abstract Concept Concretization
2. Complex Reasoning Decomposition
3. Cognitive Load Management
4. Representation Format Optimization
5. Linguistic Complexity Reduction
"""

import logging
from typing import List, Dict, Any, Optional

from ..config.config import AdapterConfig
from ..utils.llm_client import LLMClient
from ..data.seed_data import SeedDataItem
from ..data.data_utils import (
    SyntheticDataItem,
    validate_synthetic_item,
    parse_llm_json_response,
    filter_by_verification,
    deduplicate_items
)
from ..synthesis.prompts import (
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


class KnowledgeAdapter:
    """
    Adapts knowledge representations to student model's cognitive capacity.
    
    This module transforms the structured curriculum from the Organizer into
    cognitively appropriate representations through systematic content
    modification.
    """
    
    def __init__(
        self,
        config: AdapterConfig,
        teacher_client: LLMClient
    ):
        """
        Initialize the Knowledge Adapter.
        
        Args:
            config: Adapter configuration
            teacher_client: LLM client for teacher model
        """
        self.config = config
        self.teacher_client = teacher_client
        
        # Track synthesis statistics
        self.stats = {
            "total_generated": 0,
            "validation_passed": 0,
            "verification_passed": 0,
            "final_count": 0
        }
    
    def adapt_for_stage(
        self,
        stage_id: str,
        knowledge_modules: List[str],
        prerequisites: List[str],
        seed_items: List[SeedDataItem],
        domain: str = "",
        difficulty: str = "intermediate",
        baseline_ratio: float = 0.5
    ) -> List[SyntheticDataItem]:
        """
        Generate adapted synthetic data for a curriculum stage.
        
        Args:
            stage_id: Current curriculum stage ID
            knowledge_modules: Target modules for this stage
            prerequisites: Prerequisite modules
            seed_items: Seed data items for synthesis
            domain: Target domain
            difficulty: Difficulty level
            baseline_ratio: Student's baseline performance ratio
        
        Returns:
            List of adapted synthetic data items
        """
        logger.info(f"Adapting data for stage {stage_id} with {len(seed_items)} seeds")
        
        all_items = []
        
        # Generate synthetic data for each seed
        for seed in seed_items:
            items = self._synthesize_from_seed(
                seed=seed,
                stage_id=stage_id,
                knowledge_modules=knowledge_modules,
                prerequisites=prerequisites,
                domain=domain,
                difficulty=difficulty,
                baseline_ratio=baseline_ratio
            )
            all_items.extend(items)
        
        # Post-processing
        logger.info(f"Generated {len(all_items)} raw items")
        
        # Filter by verification
        if self.config.enable_verification:
            all_items = filter_by_verification(all_items)
            logger.info(f"After verification filter: {len(all_items)} items")
        
        # Deduplicate
        all_items = deduplicate_items(all_items)
        logger.info(f"After deduplication: {len(all_items)} items")
        
        self.stats["final_count"] += len(all_items)
        
        return all_items
    
    def _synthesize_from_seed(
        self,
        seed: SeedDataItem,
        stage_id: str,
        knowledge_modules: List[str],
        prerequisites: List[str],
        domain: str,
        difficulty: str,
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
            difficulty: Difficulty level
            baseline_ratio: Student baseline ratio
        
        Returns:
            List of synthetic items
        """
        # Get difficulty constraints
        constraints = get_difficulty_constraints(difficulty)
        
        # Create synthesis prompt
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
        user_prompt += f"\n\nSeed Example Context:\nInput: {seed.input_text}\nExpected Output Type: {seed.domain}"
        
        # Optionally add few-shot examples
        few_shot = create_few_shot_examples(domain, num_examples=1)
        if few_shot:
            user_prompt += f"\n\nReference Examples:\n{few_shot}"
        
        # Generate via teacher model
        try:
            response = self.teacher_client.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT_SYNTHESIS,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.generation_temperature
            )
            
            # Parse response
            items_data = parse_llm_json_response(response)
            self.stats["total_generated"] += len(items_data)
            
            # Convert to SyntheticDataItem objects
            items = []
            for item_data in items_data:
                # Validate
                is_valid, error = validate_synthetic_item(item_data)
                if is_valid:
                    self.stats["validation_passed"] += 1
                    item = SyntheticDataItem.from_dict(item_data)
                    items.append(item)
                else:
                    logger.debug(f"Validation failed: {error}")
            
            return items
            
        except Exception as e:
            logger.error(f"Synthesis failed for seed {seed.item_id}: {e}")
            return []
    
    def generate_remedial_data(
        self,
        stage_id: str,
        knowledge_modules: List[str],
        weak_subskills: List[str],
        seed_items: List[SeedDataItem],
        num_examples: int = 5
    ) -> List[SyntheticDataItem]:
        """
        Generate remedial data for weak subskills.
        
        As mentioned in Section 3.3: "Otherwise, the remedial data will be
        synthesized to continue learning knowledge in this stage until mastery."
        
        Args:
            stage_id: Current stage ID
            knowledge_modules: Modules in the stage
            weak_subskills: Specific weak areas
            seed_items: Seed data items
            num_examples: Number of remedial examples
        
        Returns:
            List of remedial synthetic items
        """
        logger.info(f"Generating remedial data for {len(weak_subskills)} weak subskills")
        
        # Create remedial prompt
        user_prompt = get_remedial_prompt(
            stage_id=stage_id,
            knowledge_modules=knowledge_modules,
            weak_subskills=weak_subskills,
            num_examples=num_examples
        )
        
        # Add simplified seed context if available
        if seed_items:
            user_prompt += f"\n\nSimplify based on this type: {seed_items[0].domain}"
        
        # Generate via teacher model
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
                is_valid, error = validate_synthetic_item(item_data)
                if is_valid:
                    item = SyntheticDataItem.from_dict(item_data)
                    items.append(item)
            
            logger.info(f"Generated {len(items)} remedial items")
            return items
            
        except Exception as e:
            logger.error(f"Remedial generation failed: {e}")
            return []
    
    def generate_bridging_data(
        self,
        stage_id: str,
        knowledge_modules: List[str],
        seed_items: List[SeedDataItem],
        num_examples: int = 5
    ) -> List[SyntheticDataItem]:
        """
        Generate bridging data with slightly increased complexity.
        
        Used when student achieves mastery and is ready for harder material.
        
        Args:
            stage_id: Current stage ID
            knowledge_modules: Modules in the stage
            seed_items: Seed data items
            num_examples: Number of bridging examples
        
        Returns:
            List of bridging synthetic items
        """
        logger.info(f"Generating bridging data for stage {stage_id}")
        
        # Create bridging prompt
        user_prompt = get_bridging_prompt(
            stage_id=stage_id,
            knowledge_modules=knowledge_modules,
            num_examples=num_examples
        )
        
        # Generate via teacher model
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
                is_valid, error = validate_synthetic_item(item_data)
                if is_valid:
                    item = SyntheticDataItem.from_dict(item_data)
                    items.append(item)
            
            logger.info(f"Generated {len(items)} bridging items")
            return items
            
        except Exception as e:
            logger.error(f"Bridging generation failed: {e}")
            return []
    
    def apply_concretization(
        self,
        concept: str,
        content: str
    ) -> str:
        """
        Apply Abstract Concept Concretization.
        
        Transform abstract concepts into concrete, intuitive representations
        using analogical reasoning.
        
        Args:
            concept: Abstract concept to concretize
            content: Original content
        
        Returns:
            Concretized content
        """
        if not self.config.enable_concretization:
            return content
        
        prompt = f"""Transform this abstract concept into a concrete explanation:

Concept: {concept}
Original Content: {content}

Requirements:
1. Start with a real-world analogy or everyday example
2. Connect the analogy to the formal concept
3. Then introduce the mathematical/formal notation
4. Keep the core information intact"""
        
        try:
            response = self.teacher_client.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5
            )
            return response
        except Exception as e:
            logger.warning(f"Concretization failed: {e}")
            return content
    
    def apply_decomposition(self, content: str) -> str:
        """
        Apply Complex Reasoning Decomposition.
        
        Break down multi-step reasoning into atomic cognitive operations.
        
        Args:
            content: Original reasoning content
        
        Returns:
            Decomposed content with explicit steps
        """
        if not self.config.enable_decomposition:
            return content
        
        prompt = f"""Decompose this reasoning into explicit small steps:

Content: {content}

Requirements:
1. Break into atomic sub-steps
2. Each step should do ONE thing
3. Use clear transitions (First, Next, Therefore)
4. Include intermediate results
5. Add verification at the end"""
        
        try:
            response = self.teacher_client.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5
            )
            return response
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
            return content
    
    def apply_simplification(self, content: str) -> str:
        """
        Apply Linguistic Complexity Reduction.
        
        Simplify vocabulary, syntax, and discourse structure.
        
        Args:
            content: Original content
        
        Returns:
            Simplified content
        """
        if not self.config.enable_linguistic_simplification:
            return content
        
        prompt = f"""Simplify this content linguistically:

Content: {content}

Requirements:
1. Use simple words (replace jargon)
2. Use short, direct sentences
3. Add clear connectors (first, next, therefore)
4. Keep the meaning intact
5. Make it accessible to a beginner"""
        
        try:
            response = self.teacher_client.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5
            )
            return response
        except Exception as e:
            logger.warning(f"Simplification failed: {e}")
            return content
    
    def get_stats(self) -> Dict[str, int]:
        """Get synthesis statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset synthesis statistics"""
        self.stats = {
            "total_generated": 0,
            "validation_passed": 0,
            "verification_passed": 0,
            "final_count": 0
        }


def adapt_knowledge_for_curriculum(
    adapter: KnowledgeAdapter,
    curriculum_stages: List[Dict[str, Any]],
    seed_dataset: Any,
    domain: str
) -> Dict[str, List[SyntheticDataItem]]:
    """
    Adapt knowledge for all curriculum stages.
    
    Args:
        adapter: KnowledgeAdapter instance
        curriculum_stages: List of stage information dicts
        seed_dataset: Seed dataset
        domain: Target domain
    
    Returns:
        Dictionary mapping stage_id to list of synthetic items
    """
    all_synthetic_data = {}
    
    for stage_info in curriculum_stages:
        stage_id = stage_info["stage_id"]
        modules = stage_info["modules"]
        prereqs = stage_info.get("prerequisites", [])
        difficulty = stage_info.get("difficulty", "intermediate")
        
        # Get seed items for this stage's modules
        seed_items = []
        for module in modules:
            module_seeds = seed_dataset.get_train_items_by_module(module)
            seed_items.extend(module_seeds)
        
        # Generate synthetic data
        synthetic_items = adapter.adapt_for_stage(
            stage_id=stage_id,
            knowledge_modules=modules,
            prerequisites=prereqs,
            seed_items=seed_items,
            domain=domain,
            difficulty=difficulty
        )
        
        all_synthetic_data[stage_id] = synthetic_items
        logger.info(f"Stage {stage_id}: {len(synthetic_items)} synthetic items")
    
    return all_synthetic_data


if __name__ == "__main__":
    # Test the adapter
    from ..config.config import AdapterConfig
    
    config = AdapterConfig()
    print(f"Adapter config loaded:")
    print(f"  Samples per seed: {config.num_samples_per_seed}")
    print(f"  Verification enabled: {config.enable_verification}")