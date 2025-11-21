"""
Knowledge Identifier Module

This module implements the Identifier component of the IOA framework
as described in Section 3.2 of the paper.

The Identifier answers "What to teach" by:
1. Knowledge Deficiency Diagnosis - evaluating performance gaps (Eq. 2)
2. Dependency Graph Construction - building prerequisite relationships (Eq. 3)
3. Targeted Knowledge Selection - prioritizing critical gaps (Eq. 4, 5, 6)

Key equations implemented:
- Eq. 2: Performance gap Δ(k) = (P_T(k) - P_S(k)) / P_T(k)
- Eq. 3: Dependency strength
- Eq. 4: Severity(k) = α·Δ(k) + (1-α)·avg(Dependency)
- Eq. 5, 6: K_target = Top-m(K_deficient, Severity)
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from ..config.config import IdentifierConfig
from ..utils.llm_client import LLMClient
from ..utils.graph_utils import (
    KnowledgeNode,
    KnowledgeDependencyGraph,
)
from ..data.seed_data import SeedDataset, get_probe_tasks

logger = logging.getLogger(__name__)


class KnowledgeIdentifier:
    """
    Identifies knowledge deficiencies in student models and prioritizes
    critical knowledge gaps for targeted distillation.
    """
    
    def __init__(
        self,
        config: IdentifierConfig,
        teacher_client: LLMClient,
        seed_dataset: SeedDataset
    ):
        """
        Initialize the Knowledge Identifier.
        
        Args:
            config: Identifier configuration with thresholds
            teacher_client: LLM client for teacher model
            seed_dataset: Seed dataset for probe tasks
        """
        self.config = config
        self.teacher_client = teacher_client
        self.seed_dataset = seed_dataset
        
        # Storage for results
        self.knowledge_modules: List[KnowledgeNode] = []
        self.dependency_graph: Optional[KnowledgeDependencyGraph] = None
        self.performance_gaps: Dict[str, float] = {}
        self.deficient_modules: List[str] = []
        self.target_modules: List[str] = []
    
    def decompose_knowledge_domain(self, domain: str) -> List[KnowledgeNode]:
        """
        Decompose a capability domain into constituent knowledge modules.
        
        As described in Section 3.2: "For a given target capability domain D,
        we employ a hierarchical decomposition strategy that organizes knowledge
        across multiple granular levels."
        
        Args:
            domain: Target capability domain (e.g., "math_problem_solving")
        
        Returns:
            List of KnowledgeNode objects representing the domain structure
        """
        logger.info(f"Decomposing knowledge domain: {domain}")
        
        # Query teacher LLM to structure knowledge units
        prompt = self._get_decomposition_prompt(domain)
        
        try:
            response = self.teacher_client.generate_json(
                prompt=prompt,
                system_prompt=self._get_decomposition_system_prompt(),
                temperature=0.3
            )
            
            modules = self._parse_decomposition_response(response, domain)
            logger.info(f"Decomposed {domain} into {len(modules)} knowledge modules")
            
        except Exception as e:
            logger.warning(f"Failed to decompose domain via LLM: {e}")
            logger.info("Using fallback knowledge structure")
            modules = self._get_fallback_decomposition(domain)
        
        return modules
    
    def _get_decomposition_system_prompt(self) -> str:
        """Get system prompt for knowledge decomposition"""
        return """You are an expert educational curriculum designer. Your task is to decompose 
a learning domain into a hierarchical structure of knowledge modules.

For each module, provide:
- A unique identifier (category/module)
- The category it belongs to
- A descriptive name
- Difficulty level (introductory, intermediate, advanced)

Output as a JSON array of objects with keys: id, category, name, difficulty"""
    
    def _get_decomposition_prompt(self, domain: str) -> str:
        """Get prompt for decomposing a specific domain"""
        domain_descriptions = {
            "math_problem_solving": """mathematical problem solving including (but not limited to):
- Arithmetic (e.g., basic operations, fractions, decimals)
- Algebra (e.g., linear equations, quadratic equations, polynomials, functions)
- Geometry (e.g., basic shapes, coordinate geometry, trigonometry)
- Calculus (e.g., limits, derivatives, integrals)
- Discrete Mathematics (e.g., combinatorics, graph theory, logic)
- Number Theory (e.g., prime numbers, divisibility, modular arithmetic)
- Probability and Statistics (e.g., probability fundamentals, statistical analysis, modeling & inference)""",
            
            "code_generation": """programming and code generation including (but not limited to):
- Python basics (e.g., syntax, data types, control flow)
- Data structures (e.g., lists, dictionaries, sets, trees)
- Algorithms (e.g., orting, searching, dynamic programming)
- Object-oriented programming (e.g., classes, inheritance, polymorphism)
- String processing (e.g., string manipulation, pattern matching, text parsing & clearning)
- File I/O and APIs (e.g., file operations, structured data handling, API interaction)""",
            
            "instruction_following": """instruction following capabilities including (but not limited to):
- Question answering (e.g., factual, analytical)
- Summarization (e.g., text summarization, extractive & abstractive)
- Creative writing (e.g., story generation, poetry, dialogues)
- Brainstorming (e.g., idea generation, problem solving)
- Multi-turn dialogue (e.g., conversational context, follow-up questions)""",
            
            "academic_knowledge_reasoning": """academic knowledge reasoning including (but not limited to):
- Physics (e.g., mechanics, electricity, thermodynamics, optics)
- Geology (e.g., earth structure, minerals, plate tectonics, geological time)
- Chemistry (e.g., organic chemistry, inorganic chemistry, physical chemistry, analytical, biochemistry)
- Biology (e.g., molecular biology, cell biology, genetics, ecology, evolutionary biology)
- Engineering fundamentals (e.g., electrical engineering, civil engineering, materials engineering)"""
        }
        
        description = domain_descriptions.get(
            domain,
            f"the domain of {domain}"
        )
        
        return f"""Please decompose the following learning domain into a hierarchical structure 
of fine-grained knowledge modules:

Domain: {domain}
Description: {description}

Please refer to above description and create 25-35 knowledge modules organized by category (not limited to above mentioned in the description), each with:
- Unique ID in format "category/module"
- Category name
- Descriptive module name
- Difficulty level

Return as a JSON array. Example format:
[
    {{"id": "algebra/linear_equations", "category": "Algebra", "name": "Linear Equations", "difficulty": "introductory"}},
    {{"id": "algebra/quadratic_equations", "category": "Algebra", "name": "Quadratic Equations", "difficulty": "intermediate"}}
]"""
    
    def _parse_decomposition_response(
        self,
        response: Any,
        domain: str
    ) -> List[KnowledgeNode]:
        """Parse LLM response into KnowledgeNode objects"""
        modules = []
        
        if isinstance(response, list):
            for item in response:
                node = KnowledgeNode(
                    module_id=item.get("id", ""),
                    category=item.get("category", ""),
                    name=item.get("name", ""),
                    difficulty_level=item.get("difficulty", "introductory")
                )
                if node.module_id:
                    modules.append(node)
        
        return modules
    
    def _get_fallback_decomposition(self, domain: str) -> List[KnowledgeNode]:
        """Get fallback knowledge structure when LLM decomposition fails"""
        structures = {
            "math_problem_solving": [
                ("arithmetic/basic", "Arithmetic", "Basic Operations"),
                ("arithmetic/fractions", "Arithmetic", "Fractions"),
                ("algebra/linear", "Algebra", "Linear Equations"),
                ("algebra/quadratic", "Algebra", "Quadratic Equations"),
                ("algebra/functions", "Algebra", "Functions"),
                ("geometry/basic", "Geometry", "Basic Shapes"),
                ("geometry/coordinate", "Geometry", "Coordinate Geometry"),
                ("geometry/trigonometry", "Geometry", "Trigonometry"),
                ("calculus/limits", "Calculus", "Limits"),
                ("calculus/derivatives", "Calculus", "Derivatives"),
                ("calculus/integrals", "Calculus", "Integrals"),
            ],
            "code_generation": [
                ("python/basics", "Python", "Python Basics"),
                ("python/control-flow", "Python", "Control Flow"),
                ("python/functions", "Python", "Functions"),
                ("data-structures/lists", "Data Structures", "Lists"),
                ("data-structures/dicts", "Data Structures", "Dictionaries"),
                ("algorithms/sorting", "Algorithms", "Sorting"),
                ("algorithms/searching", "Algorithms", "Searching"),
                ("string-processing/basic", "String Processing", "Basic Operations"),
            ],
            "instruction_following": [
                ("qa/factual", "Question Answering", "Factual QA"),
                ("qa/analytical", "Question Answering", "Analytical QA"),
                ("summarization/basic", "Summarization", "Basic Summarization"),
                ("creative/stories", "Creative Writing", "Story Writing"),
                ("dialogue/multi-turn", "Dialogue", "Multi-turn Conversation"),
            ],
            "academic_knowledge_reasoning": [
                ("physics/mechanics", "Physics", "Mechanics"),
                ("physics/electricity", "Physics", "Electricity"),
                ("chemistry/basics", "Chemistry", "Basic Chemistry"),
                ("biology/cells", "Biology", "Cell Biology"),
            ]
        }
        
        modules = []
        for module_id, category, name in structures.get(domain, []):
            modules.append(KnowledgeNode(
                module_id=module_id,
                category=category,
                name=name
            ))
        
        return modules
    
    def evaluate_performance_gap(
        self,
        module: KnowledgeNode,
        teacher_model: Any,
        student_model: Any
    ) -> float:
        """
        Evaluate performance gap between teacher and student on a knowledge module.
        
        Implements Equation 2:
        Δ(k) = (P_T(k) - P_S(k)) / P_T(k)
        
        Args:
            module: Knowledge module to evaluate
            teacher_model: Teacher model for inference
            student_model: Student model for inference
        
        Returns:
            Performance gap Δ(k) in [0, 1]
        """
        # Get probe tasks for this module
        probe_tasks = get_probe_tasks(self.seed_dataset, module.module_id)
        
        if not probe_tasks:
            # Try parent category
            for category_module in self.seed_dataset.module_index.keys():
                if category_module.startswith(module.category.lower()):
                    probe_tasks = get_probe_tasks(self.seed_dataset, category_module)
                    break
        
        if not probe_tasks:
            logger.warning(f"No probe tasks for module {module.module_id}")
            return 0.0
        
        # Evaluate teacher and student
        teacher_correct = 0
        student_correct = 0
        
        for task in probe_tasks:
            # Teacher evaluation
            teacher_response = self._evaluate_task(teacher_model, task)
            if self._check_correctness(teacher_response, task["output"]):
                teacher_correct += 1
            
            # Student evaluation
            student_response = self._evaluate_task(student_model, task)
            if self._check_correctness(student_response, task["output"]):
                student_correct += 1
        
        # Compute scores
        P_T = teacher_correct / len(probe_tasks) if probe_tasks else 0
        P_S = student_correct / len(probe_tasks) if probe_tasks else 0
        
        # Store scores in node
        module.teacher_score = P_T
        module.student_score = P_S
        
        # Compute gap (Eq. 2)
        if P_T > 0:
            gap = (P_T - P_S) / P_T
        else:
            gap = 0.0
        
        module.performance_gap = gap
        
        logger.debug(f"Module {module.module_id}: P_T={P_T:.3f}, P_S={P_S:.3f}, Δ={gap:.3f}")
        
        return gap
    
    def _evaluate_task(self, model: Any, task: Dict[str, str]) -> str:
        """Evaluate a model on a probe task"""
        # This should be implemented based on the model type
        # For now, return empty string if model is None
        if model is None:
            return ""
        
        # If model has generate method (HuggingFace)
        if hasattr(model, 'generate'):
            # Use model for inference
            pass
        
        return ""
    
    def _check_correctness(self, response: str, reference: str) -> bool:
        """Check if response is correct against reference"""
        from ..data.data_utils import compute_rouge_l, compute_exact_match
        
        # For math/code, use exact match
        if compute_exact_match(response, reference):
            return True
        
        # For instruction following, use ROUGE-L threshold
        rouge = compute_rouge_l(response, reference)
        return rouge >= 0.5
    
    def compute_dependency_strength(
        self,
        module_i: KnowledgeNode,
        module_j: KnowledgeNode,
        student_performances: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Compute dependency strength between two knowledge modules.
        
        Implements Equation 3:
        Dependency(k_i → k_j) = 
            (P_S(k_j | P_S(k_i)/P_T(k_i) >= τ_high) - P_S(k_j | P_S(k_i)/P_T(k_i) < τ_low))
            / (P_S(k_j | P_S(k_i)/P_T(k_i) >= τ_high) + ε)
        
        Args:
            module_i: Prerequisite module
            module_j: Dependent module
            student_performances: Dict mapping module_id to (student_score, teacher_score)
        
        Returns:
            Dependency strength in [0, 1]
        """
        # Get performance ratios
        P_S_i, P_T_i = student_performances.get(
            module_i.module_id,
            (module_i.student_score, module_i.teacher_score)
        )
        
        if P_T_i == 0:
            return 0.0
        
        ratio_i = P_S_i / P_T_i
        
        # This is a simplified version - in practice, you'd need
        # conditional performance data which requires multiple evaluations
        # For now, use heuristic based on category relationships
        
        # Check if modules are in related categories
        same_category = module_i.category == module_j.category
        
        # Check difficulty progression
        difficulty_order = {"introductory": 0, "intermediate": 1, "advanced": 2}
        diff_i = difficulty_order.get(module_i.difficulty_level, 0)
        diff_j = difficulty_order.get(module_j.difficulty_level, 0)
        
        # Higher dependency if same category and j is harder than i
        base_strength = 0.0
        if same_category and diff_j > diff_i:
            base_strength = 0.5 + (diff_j - diff_i) * 0.2
        elif same_category:
            base_strength = 0.3
        
        # Adjust based on mastery level
        if ratio_i >= self.config.tau_high:
            # High mastery of prerequisite
            strength = base_strength
        elif ratio_i < self.config.tau_low:
            # Low mastery of prerequisite
            strength = base_strength * 0.5
        else:
            # Medium mastery
            strength = base_strength * 0.75
        
        return min(strength, 1.0)
    
    def build_dependency_graph(
        self,
        modules: List[KnowledgeNode]
    ) -> KnowledgeDependencyGraph:
        """
        Construct the knowledge dependency graph.
        
        As described in Section 3.2: "We construct a directed acyclic graph
        G = (V, E) where vertices V represent knowledge modules and edges E
        encode prerequisite dependencies."
        
        Args:
            modules: List of knowledge modules
        
        Returns:
            KnowledgeDependencyGraph with dependency relationships
        """
        logger.info(f"Building dependency graph for {len(modules)} modules")
        
        graph = KnowledgeDependencyGraph()
        
        # Add all nodes
        for module in modules:
            graph.add_node(module)
        
        # Compute student performances for dependency calculation
        student_performances = {
            m.module_id: (m.student_score, m.teacher_score)
            for m in modules
        }
        
        # Compute pairwise dependencies
        for i, module_i in enumerate(modules):
            for j, module_j in enumerate(modules):
                if i == j:
                    continue
                
                strength = self.compute_dependency_strength(
                    module_i, module_j, student_performances
                )
                
                # Add edge if strength exceeds threshold (τ_dep)
                if strength > self.config.tau_dep:
                    graph.add_edge(module_i.module_id, module_j.module_id, strength)
        
        logger.info(f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        return graph
    
    def compute_severity_score(
        self,
        module: KnowledgeNode,
        graph: KnowledgeDependencyGraph
    ) -> float:
        """
        Compute deficiency severity score for prioritization.
        
        Implements Equation 4:
        Severity(k) = α·Δ(k) + (1-α)·(1/|N(k)|)·Σ_{k'∈N(k)} Dependency(k→k')
        
        Args:
            module: Knowledge module
            graph: Dependency graph
        
        Returns:
            Severity score
        """
        # First term: performance gap
        gap_term = self.config.alpha * module.performance_gap
        
        # Second term: structural importance (average dependency strength)
        avg_dependency = graph.compute_average_dependency_strength(module.module_id)
        structure_term = (1 - self.config.alpha) * avg_dependency
        
        severity = gap_term + structure_term
        module.severity_score = severity
        
        return severity
    
    def identify_deficient_modules(
        self,
        modules: List[KnowledgeNode]
    ) -> List[KnowledgeNode]:
        """
        Identify modules with significant performance deficiency.
        
        Implements Equation 6:
        K_deficient = {k : Δ(k) > τ_gap}
        
        Args:
            modules: List of knowledge modules with computed performance gaps
        
        Returns:
            List of deficient modules
        """
        deficient = [
            m for m in modules
            if m.performance_gap > self.config.tau_gap
        ]
        
        logger.info(f"Identified {len(deficient)}/{len(modules)} deficient modules "
                   f"(τ_gap={self.config.tau_gap})")
        
        return deficient
    
    def select_target_modules(
        self,
        deficient_modules: List[KnowledgeNode],
        graph: KnowledgeDependencyGraph
    ) -> List[str]:
        """
        Select target modules for distillation based on severity ranking.
        
        Implements Equation 5:
        K_target = Top-m(K_deficient, Severity(·))
        
        Args:
            deficient_modules: Modules with Δ(k) > τ_gap
            graph: Dependency graph
        
        Returns:
            List of target module IDs ranked by severity
        """
        # Compute severity scores
        for module in deficient_modules:
            self.compute_severity_score(module, graph)
        
        # Sort by severity (descending)
        sorted_modules = sorted(
            deficient_modules,
            key=lambda m: m.severity_score,
            reverse=True
        )
        
        # Select top m modules (typically 20-30% as mentioned in Section 3.2)
        m = max(1, int(len(sorted_modules) * self.config.target_module_percentage))
        target_modules = sorted_modules[:m]
        
        target_ids = [m.module_id for m in target_modules]
        
        logger.info(f"Selected {len(target_ids)} target modules from "
                   f"{len(deficient_modules)} deficient modules")
        
        for i, module in enumerate(target_modules[:5]):
            logger.debug(f"  {i+1}. {module.module_id}: severity={module.severity_score:.3f}")
        
        return target_ids
    
    def identify(
        self,
        domain: str,
        teacher_model: Any,
        student_model: Any
    ) -> Tuple[List[str], KnowledgeDependencyGraph]:
        """
        Main identification pipeline.
        
        This is the entry point that runs the complete identification process:
        1. Decompose domain into knowledge modules
        2. Evaluate performance gaps
        3. Build dependency graph
        4. Select target modules
        
        Args:
            domain: Target capability domain
            teacher_model: Teacher model for evaluation
            student_model: Student model for evaluation
        
        Returns:
            Tuple of (target_module_ids, dependency_graph)
        """
        logger.info(f"Starting knowledge identification for domain: {domain}")
        
        # Step 1: Decompose domain
        self.knowledge_modules = self.decompose_knowledge_domain(domain)
        
        # Step 2: Evaluate performance gaps
        logger.info("Evaluating performance gaps...")
        for module in self.knowledge_modules:
            gap = self.evaluate_performance_gap(module, teacher_model, student_model)
            self.performance_gaps[module.module_id] = gap
        
        # Step 3: Build dependency graph
        self.dependency_graph = self.build_dependency_graph(self.knowledge_modules)
        
        # Step 4: Identify deficient modules
        deficient = self.identify_deficient_modules(self.knowledge_modules)
        self.deficient_modules = [m.module_id for m in deficient]
        
        # Step 5: Select target modules
        self.target_modules = self.select_target_modules(
            deficient, self.dependency_graph
        )
        
        logger.info(f"Identification complete. Target modules: {len(self.target_modules)}")
        
        return self.target_modules, self.dependency_graph
    
    def get_module_by_id(self, module_id: str) -> Optional[KnowledgeNode]:
        """Get a knowledge module by its ID"""
        for module in self.knowledge_modules:
            if module.module_id == module_id:
                return module
        return None


if __name__ == "__main__":
    # Test the identifier
    from ..config.config import IdentifierConfig
    
    config = IdentifierConfig()
    print(f"Identifier config loaded: τ_gap={config.tau_gap}, α={config.alpha}")