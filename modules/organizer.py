"""
Knowledge Organizer Module

This module implements the Organizer component of the IOA framework
as described in Section 3.3 of the paper.

The Organizer answers "When to teach" by:
1. Curriculum Sequence Construction - topological ordering (Eq. 7)
2. Difficulty Increment Control - ZPD-based pacing (Eq. 8)
3. Mastery-Based Progressive Learning - advancement gates (Eq. 9)

Key principles:
- Bloom's Mastery Learning: advance only after demonstrating mastery
- Vygotsky's Zone of Proximal Development: controlled difficulty increments
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

from ..config.config import OrganizerConfig
from ..utils.graph_utils import (
    KnowledgeNode,
    KnowledgeDependencyGraph,
    build_curriculum_stages,
    compute_stage_difficulty,
    check_zpd_constraint
)

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Represents a single stage in the curriculum"""
    
    # Stage identifier
    stage_id: str
    
    # Stage index in sequence
    index: int
    
    # Knowledge modules in this stage
    modules: List[str] = field(default_factory=list)
    
    # Prerequisites (all modules from previous stages)
    prerequisites: Set[str] = field(default_factory=set)
    
    # Difficulty metrics
    avg_difficulty: float = 0.0
    difficulty_increment: float = 0.0
    
    # Mastery tracking
    mastery_achieved: bool = False
    mastery_scores: Dict[str, float] = field(default_factory=dict)
    
    # Remedial iterations
    remedial_count: int = 0


@dataclass
class Curriculum:
    """Complete curriculum with all stages"""
    
    # Ordered list of stages
    stages: List[CurriculumStage] = field(default_factory=list)
    
    # Current stage index
    current_stage_idx: int = 0
    
    # Domain being taught
    domain: str = ""
    
    # Configuration parameters
    tau_zpd: float = 0.15
    tau_mastery: float = 0.9
    
    def get_current_stage(self) -> Optional[CurriculumStage]:
        """Get the current stage"""
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    def advance_stage(self) -> bool:
        """Advance to next stage if possible"""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if curriculum is complete"""
        return self.current_stage_idx >= len(self.stages) - 1 and \
               self.stages[-1].mastery_achieved if self.stages else True
    
    def get_all_modules(self) -> List[str]:
        """Get all modules in the curriculum"""
        modules = []
        for stage in self.stages:
            modules.extend(stage.modules)
        return modules


class KnowledgeOrganizer:
    """
    Organizes knowledge into progressive curriculum stages with
    mastery-based advancement.
    """
    
    def __init__(
        self,
        config: OrganizerConfig,
        dependency_graph: KnowledgeDependencyGraph
    ):
        """
        Initialize the Knowledge Organizer.
        
        Args:
            config: Organizer configuration with ZPD and mastery thresholds
            dependency_graph: Knowledge dependency graph from Identifier
        """
        self.config = config
        self.graph = dependency_graph
        
        # Store constructed curriculum
        self.curriculum: Optional[Curriculum] = None
    
    def construct_curriculum_sequence(
        self,
        target_modules: List[str]
    ) -> List[List[str]]:
        """
        Construct learning sequence respecting dependencies.
        
        Implements Equation 7:
        s_i = {k ∈ K_target : ∀k' ∈ Prerequisites(k), k' ∈ ∪_{j<i} s_j}
        
        This ensures all prerequisites are mastered before introducing
        dependent knowledge.
        
        Args:
            target_modules: List of target module IDs from Identifier
        
        Returns:
            List of stages, where each stage is a list of module IDs
        """
        logger.info(f"Constructing curriculum for {len(target_modules)} target modules")
        
        # Use graph-based curriculum construction
        stages = build_curriculum_stages(self.graph, target_modules)
        
        logger.info(f"Created {len(stages)} curriculum stages")
        for i, stage in enumerate(stages):
            logger.debug(f"  Stage {i+1}: {stage}")
        
        return stages
    
    def apply_zpd_constraints(
        self,
        stages: List[List[str]]
    ) -> List[List[str]]:
        """
        Apply Zone of Proximal Development constraints to stages.
        
        Implements Equation 8:
        (1/|s_{i+1}|)Σ P_S(k) - (1/|s_i|)Σ P_S(k) ≤ τ_ZPD · (1/|s_i|)Σ P_S(k)
        
        This ensures difficulty increases remain within the student's
        learning capacity.
        
        Args:
            stages: Initial curriculum stages
        
        Returns:
            Adjusted stages satisfying ZPD constraints
        """
        logger.info(f"Applying ZPD constraints (τ_ZPD={self.config.tau_zpd})")
        
        adjusted_stages = []
        
        for i, stage in enumerate(stages):
            if i == 0:
                # First stage has no constraint
                adjusted_stages.append(stage)
                continue
            
            # Check ZPD constraint
            if not check_zpd_constraint(
                adjusted_stages[-1],
                stage,
                self.graph,
                self.config.tau_zpd
            ):
                # Split stage to reduce difficulty increment
                logger.debug(f"Stage {i+1} violates ZPD constraint, splitting...")
                
                split_stages = self._split_stage_by_difficulty(stage)
                adjusted_stages.extend(split_stages)
            else:
                adjusted_stages.append(stage)
        
        logger.info(f"Adjusted to {len(adjusted_stages)} stages after ZPD constraints")
        
        return adjusted_stages
    
    def _split_stage_by_difficulty(
        self,
        stage: List[str],
        max_difficulty_range: float = 0.3
    ) -> List[List[str]]:
        """
        Split a stage into smaller stages by difficulty.
        
        Args:
            stage: Stage to split
            max_difficulty_range: Maximum difficulty range per split stage
        
        Returns:
            List of split stages
        """
        if len(stage) <= 1:
            return [stage]
        
        # Sort modules by difficulty (student score as proxy)
        def get_difficulty(module_id: str) -> float:
            if module_id in self.graph.nodes:
                return self.graph.nodes[module_id].student_score
            return 0.0
        
        sorted_modules = sorted(stage, key=get_difficulty)
        
        # Group into stages with similar difficulty
        split_stages = []
        current_group = [sorted_modules[0]]
        current_min = get_difficulty(sorted_modules[0])
        
        for module_id in sorted_modules[1:]:
            difficulty = get_difficulty(module_id)
            
            if difficulty - current_min > max_difficulty_range:
                # Start new group
                split_stages.append(current_group)
                current_group = [module_id]
                current_min = difficulty
            else:
                current_group.append(module_id)
        
        if current_group:
            split_stages.append(current_group)
        
        return split_stages
    
    def create_curriculum(
        self,
        target_modules: List[str],
        domain: str = ""
    ) -> Curriculum:
        """
        Create a complete curriculum with stages.
        
        Args:
            target_modules: List of target module IDs
            domain: Domain name for the curriculum
        
        Returns:
            Curriculum object with all stages
        """
        # Construct initial sequence
        stages = self.construct_curriculum_sequence(target_modules)
        
        # Apply ZPD constraints
        stages = self.apply_zpd_constraints(stages)
        
        # Create curriculum object
        curriculum = Curriculum(
            domain=domain,
            tau_zpd=self.config.tau_zpd,
            tau_mastery=self.config.tau_mastery
        )
        
        # Create stage objects
        completed_modules: Set[str] = set()
        
        for i, stage_modules in enumerate(stages):
            stage = CurriculumStage(
                stage_id=f"{domain}-S{i+1}",
                index=i,
                modules=stage_modules,
                prerequisites=completed_modules.copy(),
                avg_difficulty=compute_stage_difficulty(stage_modules, self.graph)
            )
            
            # Compute difficulty increment
            if i > 0 and curriculum.stages:
                prev_difficulty = curriculum.stages[-1].avg_difficulty
                stage.difficulty_increment = stage.avg_difficulty - prev_difficulty
            
            curriculum.stages.append(stage)
            completed_modules.update(stage_modules)
        
        self.curriculum = curriculum
        
        logger.info(f"Created curriculum with {len(curriculum.stages)} stages")
        
        return curriculum
    
    def check_mastery(
        self,
        stage: CurriculumStage,
        student_scores: Dict[str, float],
        teacher_scores: Dict[str, float]
    ) -> bool:
        """
        Check if mastery requirement is met for a stage.
        
        Implements Equation 9:
        Progress(s_i → s_{i+1}) = True if min_{k∈s_i} P_S(k)/P_T(k) ≥ τ_mastery
        
        Args:
            stage: Current curriculum stage
            student_scores: Student performance on stage modules
            teacher_scores: Teacher performance on stage modules
        
        Returns:
            True if mastery achieved, False otherwise
        """
        if not stage.modules:
            return True
        
        # Compute performance ratios for all modules in stage
        ratios = []
        for module_id in stage.modules:
            P_S = student_scores.get(module_id, 0.0)
            P_T = teacher_scores.get(module_id, 1.0)
            
            if P_T > 0:
                ratio = P_S / P_T
            else:
                ratio = 1.0  # If teacher score is 0, consider mastered
            
            ratios.append(ratio)
            stage.mastery_scores[module_id] = ratio
        
        # Check minimum ratio (Eq. 9)
        min_ratio = min(ratios) if ratios else 0.0
        mastery_achieved = min_ratio >= self.config.tau_mastery
        
        stage.mastery_achieved = mastery_achieved
        
        logger.info(f"Stage {stage.stage_id}: min_ratio={min_ratio:.3f}, "
                   f"τ_mastery={self.config.tau_mastery}, "
                   f"achieved={mastery_achieved}")
        
        return mastery_achieved
    
    def get_weak_modules(
        self,
        stage: CurriculumStage
    ) -> List[str]:
        """
        Get modules that haven't reached mastery threshold.
        
        Used for remedial data generation.
        
        Args:
            stage: Curriculum stage
        
        Returns:
            List of weak module IDs
        """
        weak_modules = []
        
        for module_id, score in stage.mastery_scores.items():
            if score < self.config.tau_mastery:
                weak_modules.append(module_id)
        
        return weak_modules
    
    def should_generate_remedial(
        self,
        stage: CurriculumStage
    ) -> bool:
        """
        Check if remedial data should be generated.
        
        As mentioned in Section 3.3: "Otherwise, the remedial data will be
        synthesized to continue learning knowledge in this stage until mastery."
        
        Args:
            stage: Current curriculum stage
        
        Returns:
            True if remedial data should be generated
        """
        if stage.mastery_achieved:
            return False
        
        if stage.remedial_count >= self.config.max_remedial_iterations:
            logger.warning(f"Stage {stage.stage_id} reached max remedial iterations")
            return False
        
        return True
    
    def increment_remedial_count(self, stage: CurriculumStage) -> None:
        """Increment the remedial iteration counter"""
        stage.remedial_count += 1
        logger.debug(f"Stage {stage.stage_id}: remedial iteration {stage.remedial_count}")
    
    def can_advance(self, stage: CurriculumStage) -> bool:
        """
        Check if we can advance to the next stage.
        
        Args:
            stage: Current curriculum stage
        
        Returns:
            True if advancement is allowed
        """
        return stage.mastery_achieved or \
               stage.remedial_count >= self.config.max_remedial_iterations
    
    def get_stage_info(self, stage: CurriculumStage) -> Dict[str, Any]:
        """
        Get information about a stage for prompting.
        
        Args:
            stage: Curriculum stage
        
        Returns:
            Dictionary with stage information
        """
        return {
            "stage_id": stage.stage_id,
            "index": stage.index,
            "modules": stage.modules,
            "prerequisites": list(stage.prerequisites),
            "avg_difficulty": stage.avg_difficulty,
            "difficulty_increment": stage.difficulty_increment,
            "remedial_count": stage.remedial_count
        }
    
    def organize(
        self,
        target_modules: List[str],
        domain: str = ""
    ) -> Curriculum:
        """
        Main organization pipeline.
        
        This is the entry point that runs the complete organization process.
        
        Args:
            target_modules: List of target module IDs from Identifier
            domain: Domain name
        
        Returns:
            Organized curriculum
        """
        logger.info(f"Starting knowledge organization for {len(target_modules)} modules")
        
        curriculum = self.create_curriculum(target_modules, domain)
        
        logger.info(f"Organization complete. Curriculum has {len(curriculum.stages)} stages")
        
        return curriculum
    
    def get_curriculum_summary(self) -> str:
        """Get a summary of the curriculum for logging"""
        if not self.curriculum:
            return "No curriculum created"
        
        summary = [f"Curriculum for {self.curriculum.domain}:"]
        summary.append(f"  Total stages: {len(self.curriculum.stages)}")
        summary.append(f"  τ_ZPD: {self.curriculum.tau_zpd}")
        summary.append(f"  τ_mastery: {self.curriculum.tau_mastery}")
        
        for stage in self.curriculum.stages:
            status = "✓" if stage.mastery_achieved else "○"
            summary.append(
                f"  {status} Stage {stage.index + 1}: "
                f"{len(stage.modules)} modules, "
                f"difficulty={stage.avg_difficulty:.3f}"
            )
        
        return "\n".join(summary)


def create_learning_schedule(
    curriculum: Curriculum,
    samples_per_stage: int = 100
) -> List[Dict[str, Any]]:
    """
    Create a learning schedule from a curriculum.
    
    Args:
        curriculum: Organized curriculum
        samples_per_stage: Number of training samples per stage
    
    Returns:
        List of schedule entries with stage info and sample counts
    """
    schedule = []
    
    for stage in curriculum.stages:
        entry = {
            "stage_id": stage.stage_id,
            "modules": stage.modules,
            "samples": samples_per_stage,
            "prerequisites": list(stage.prerequisites),
            "difficulty": stage.avg_difficulty
        }
        schedule.append(entry)
    
    return schedule


if __name__ == "__main__":
    # Test the organizer
    from ..config.config import OrganizerConfig
    from ..utils.graph_utils import KnowledgeNode, KnowledgeDependencyGraph
    
    # Create test graph
    graph = KnowledgeDependencyGraph()
    
    nodes = [
        KnowledgeNode("math/basics", "Math", "Basics", student_score=0.9, teacher_score=1.0),
        KnowledgeNode("math/algebra", "Math", "Algebra", student_score=0.6, teacher_score=0.95),
        KnowledgeNode("math/calculus", "Math", "Calculus", student_score=0.3, teacher_score=0.9),
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    graph.add_edge("math/basics", "math/algebra", 0.7)
    graph.add_edge("math/algebra", "math/calculus", 0.8)
    
    # Create organizer
    config = OrganizerConfig()
    organizer = KnowledgeOrganizer(config, graph)
    
    # Create curriculum
    target = ["math/basics", "math/algebra", "math/calculus"]
    curriculum = organizer.organize(target, domain="math")
    
    print(organizer.get_curriculum_summary())