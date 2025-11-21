"""
Graph Utilities for Knowledge Dependency Management

This module provides utilities for constructing and manipulating
knowledge dependency graphs as described in Section 3.2 and 3.3.

Key operations:
- Dependency graph construction (Section 3.2, Eq. 3)
- Topological ordering for curriculum construction (Section 3.3, Eq. 7)
- Connected component analysis for knowledge clustering
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Represents a knowledge module in the dependency graph"""
    
    # Unique identifier for the knowledge module
    module_id: str
    
    # Category (e.g., "Algebra", "Calculus")
    category: str
    
    # Name of the specific knowledge unit
    name: str
    
    # Performance scores
    teacher_score: float = 0.0
    student_score: float = 0.0
    
    # Computed metrics
    performance_gap: float = 0.0  # Δ(k) from Eq. 2
    severity_score: float = 0.0   # From Eq. 4
    
    # Metadata
    difficulty_level: str = "introductory"  # introductory, intermediate, advanced
    
    def __hash__(self):
        return hash(self.module_id)
    
    def __eq__(self, other):
        if isinstance(other, KnowledgeNode):
            return self.module_id == other.module_id
        return False


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between knowledge modules"""
    
    # Source node (prerequisite)
    source: str
    
    # Target node (dependent)
    target: str
    
    # Dependency strength from Eq. 3
    strength: float = 0.0


class KnowledgeDependencyGraph:
    """
    Directed Acyclic Graph (DAG) for knowledge module dependencies.
    
    This implements G = (V, E) as described in Section 3.2, where:
    - V (vertices) represents knowledge modules
    - E (edges) encodes prerequisite dependencies
    """
    
    def __init__(self):
        # Nodes indexed by module_id
        self.nodes: Dict[str, KnowledgeNode] = {}
        
        # Adjacency list: node_id -> list of (target_id, strength)
        self.adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        # Reverse adjacency for finding prerequisites
        self.reverse_adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        # Edges for easy access
        self.edges: List[DependencyEdge] = []
    
    def add_node(self, node: KnowledgeNode) -> None:
        """Add a knowledge module to the graph"""
        if node.module_id not in self.nodes:
            self.nodes[node.module_id] = node
            logger.debug(f"Added node: {node.module_id}")
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        strength: float
    ) -> None:
        """
        Add a dependency edge from source to target.
        
        This means source is a prerequisite for target.
        
        Args:
            source_id: ID of the prerequisite knowledge module
            target_id: ID of the dependent knowledge module
            strength: Dependency strength from Eq. 3
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source_id}, {target_id}")
        
        # Add to adjacency lists
        self.adjacency[source_id].append((target_id, strength))
        self.reverse_adjacency[target_id].append((source_id, strength))
        
        # Store edge
        edge = DependencyEdge(source=source_id, target=target_id, strength=strength)
        self.edges.append(edge)
        
        logger.debug(f"Added edge: {source_id} -> {target_id} (strength={strength:.3f})")
    
    def get_prerequisites(self, node_id: str) -> List[str]:
        """
        Get all prerequisite modules for a given node.
        
        Args:
            node_id: ID of the knowledge module
        
        Returns:
            List of prerequisite module IDs
        """
        return [source for source, _ in self.reverse_adjacency.get(node_id, [])]
    
    def get_dependents(self, node_id: str) -> List[str]:
        """
        Get all modules that depend on a given node.
        
        Args:
            node_id: ID of the knowledge module
        
        Returns:
            List of dependent module IDs
        """
        return [target for target, _ in self.adjacency.get(node_id, [])]
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """
        Get all neighboring modules (both prerequisites and dependents).
        
        This is N(k) in Eq. 4.
        
        Args:
            node_id: ID of the knowledge module
        
        Returns:
            Set of neighboring module IDs
        """
        neighbors = set()
        neighbors.update(self.get_prerequisites(node_id))
        neighbors.update(self.get_dependents(node_id))
        return neighbors
    
    def get_dependency_strength(self, source_id: str, target_id: str) -> float:
        """
        Get the dependency strength between two modules.
        
        Args:
            source_id: Prerequisite module ID
            target_id: Dependent module ID
        
        Returns:
            Dependency strength, or 0 if no edge exists
        """
        for target, strength in self.adjacency.get(source_id, []):
            if target == target_id:
                return strength
        return 0.0
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sort on the dependency graph.
        
        This is used for curriculum sequence construction (Section 3.3, Eq. 7)
        to ensure prerequisites are learned before dependent knowledge.
        
        Returns:
            List of module IDs in topological order
        
        Raises:
            ValueError: If the graph contains cycles
        """
        # Calculate in-degrees
        in_degree = {node_id: 0 for node_id in self.nodes}
        for source_id in self.adjacency:
            for target_id, _ in self.adjacency[source_id]:
                in_degree[target_id] += 1
        
        # Initialize queue with nodes having no prerequisites
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            # Reduce in-degree of dependents
            for target_id, _ in self.adjacency.get(node_id, []):
                in_degree[target_id] -= 1
                if in_degree[target_id] == 0:
                    queue.append(target_id)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles - not a valid DAG")
        
        return result
    
    def get_all_prerequisites_recursive(self, node_id: str) -> Set[str]:
        """
        Get all prerequisites recursively (transitive closure).
        
        Args:
            node_id: ID of the knowledge module
        
        Returns:
            Set of all prerequisite module IDs
        """
        visited = set()
        stack = [node_id]
        
        while stack:
            current = stack.pop()
            for prereq in self.get_prerequisites(current):
                if prereq not in visited:
                    visited.add(prereq)
                    stack.append(prereq)
        
        return visited
    
    def compute_average_dependency_strength(self, node_id: str) -> float:
        """
        Compute average dependency strength for neighbors.
        
        This is used in the severity score calculation (Eq. 4):
        (1/|N(k)|) * Σ_{k' ∈ N(k)} Dependency(k → k')
        
        Args:
            node_id: ID of the knowledge module
        
        Returns:
            Average dependency strength
        """
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        
        total_strength = 0.0
        for neighbor_id in neighbors:
            # Check both directions
            strength = self.get_dependency_strength(node_id, neighbor_id)
            if strength == 0:
                strength = self.get_dependency_strength(neighbor_id, node_id)
            total_strength += strength
        
        return total_strength / len(neighbors)
    
    def find_connected_components(self) -> List[Set[str]]:
        """
        Find connected components in the graph (treating as undirected).
        
        Useful for grouping semantically similar knowledge modules.
        
        Returns:
            List of sets, each containing module IDs in a component
        """
        visited = set()
        components = []
        
        for node_id in self.nodes:
            if node_id not in visited:
                component = set()
                stack = [node_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        # Add all neighbors
                        for neighbor in self.get_neighbors(current):
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                components.append(component)
        
        return components
    
    def subgraph(self, node_ids: Set[str]) -> "KnowledgeDependencyGraph":
        """
        Extract a subgraph containing only the specified nodes.
        
        Args:
            node_ids: Set of node IDs to include
        
        Returns:
            New graph containing only the specified nodes and their edges
        """
        subgraph = KnowledgeDependencyGraph()
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # Add edges
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                subgraph.add_edge(edge.source, edge.target, edge.strength)
        
        return subgraph
    
    def get_root_nodes(self) -> List[str]:
        """
        Get nodes with no prerequisites (entry points).
        
        Returns:
            List of root node IDs
        """
        return [
            node_id for node_id in self.nodes
            if not self.get_prerequisites(node_id)
        ]
    
    def get_leaf_nodes(self) -> List[str]:
        """
        Get nodes with no dependents (terminal nodes).
        
        Returns:
            List of leaf node IDs
        """
        return [
            node_id for node_id in self.nodes
            if not self.get_dependents(node_id)
        ]
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes


def build_curriculum_stages(
    graph: KnowledgeDependencyGraph,
    target_modules: List[str]
) -> List[List[str]]:
    """
    Build curriculum stages from the dependency graph.
    
    This implements Equation 7 from Section 3.3:
    s_i = {k ∈ K_target : ∀k' ∈ Prerequisites(k), k' ∈ ∪_{j<i} s_j}
    
    Args:
        graph: Knowledge dependency graph
        target_modules: List of modules to include in curriculum
    
    Returns:
        List of stages, where each stage is a list of module IDs
    """
    # Create subgraph with only target modules
    target_set = set(target_modules)
    subgraph = graph.subgraph(target_set)
    
    stages = []
    completed = set()
    remaining = set(target_modules)
    
    while remaining:
        # Find modules whose prerequisites are all completed
        current_stage = []
        
        for module_id in remaining:
            prerequisites = set(subgraph.get_prerequisites(module_id))
            # Only consider prerequisites that are in target_modules
            relevant_prereqs = prerequisites.intersection(target_set)
            
            if relevant_prereqs.issubset(completed):
                current_stage.append(module_id)
        
        if not current_stage:
            # No progress possible - might have cycles or missing prerequisites
            logger.warning(f"Cannot schedule remaining modules: {remaining}")
            # Add remaining modules as final stage
            current_stage = list(remaining)
        
        stages.append(current_stage)
        completed.update(current_stage)
        remaining.difference_update(current_stage)
    
    return stages


def group_similar_modules(
    modules: List[str],
    graph: KnowledgeDependencyGraph,
    max_group_size: int = 3
) -> List[List[str]]:
    """
    Group semantically similar modules within the same stage.
    
    As mentioned in Section 3.3, each stage contains a subset of
    semantically similar knowledge modules.
    
    Args:
        modules: List of module IDs to group
        graph: Knowledge dependency graph
        max_group_size: Maximum modules per group
    
    Returns:
        List of groups, each containing similar module IDs
    """
    if not modules:
        return []
    
    # Group by category
    category_groups: Dict[str, List[str]] = defaultdict(list)
    for module_id in modules:
        if module_id in graph.nodes:
            category = graph.nodes[module_id].category
            category_groups[category].append(module_id)
    
    # Split large groups
    result = []
    for category, group in category_groups.items():
        for i in range(0, len(group), max_group_size):
            result.append(group[i:i + max_group_size])
    
    return result


def compute_stage_difficulty(
    stage: List[str],
    graph: KnowledgeDependencyGraph
) -> float:
    """
    Compute average difficulty of a curriculum stage.
    
    This is used in Equation 8 for ZPD threshold checking:
    (1/|s_i|) * Σ_{k ∈ s_i} P_S(k)
    
    Args:
        stage: List of module IDs in the stage
        graph: Knowledge dependency graph
    
    Returns:
        Average student performance (as difficulty proxy)
    """
    if not stage:
        return 0.0
    
    total_score = 0.0
    for module_id in stage:
        if module_id in graph.nodes:
            total_score += graph.nodes[module_id].student_score
    
    return total_score / len(stage)


def check_zpd_constraint(
    current_stage: List[str],
    next_stage: List[str],
    graph: KnowledgeDependencyGraph,
    tau_zpd: float = 0.15
) -> bool:
    """
    Check if difficulty increment satisfies ZPD constraint.
    
    This implements Equation 8 from Section 3.3:
    (1/|s_{i+1}|) Σ P_S(k) - (1/|s_i|) Σ P_S(k) ≤ τ_ZPD * (1/|s_i|) Σ P_S(k)
    
    Args:
        current_stage: Current stage modules
        next_stage: Next stage modules
        graph: Knowledge dependency graph
        tau_zpd: ZPD threshold (default 0.15)
    
    Returns:
        True if constraint is satisfied
    """
    current_difficulty = compute_stage_difficulty(current_stage, graph)
    next_difficulty = compute_stage_difficulty(next_stage, graph)
    
    if current_difficulty == 0:
        return True  # No constraint for first stage
    
    difficulty_increment = next_difficulty - current_difficulty
    max_increment = tau_zpd * current_difficulty
    
    return difficulty_increment <= max_increment


def visualize_graph(
    graph: KnowledgeDependencyGraph,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize the knowledge dependency graph.
    
    Args:
        graph: Knowledge dependency graph
        output_path: Path to save the visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Convert to networkx graph
        G = nx.DiGraph()
        
        for node_id, node in graph.nodes.items():
            G.add_node(node_id, label=node.name)
        
        for edge in graph.edges:
            G.add_edge(edge.source, edge.target, weight=edge.strength)
        
        # Draw
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        nx.draw(
            G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=2000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20
        )
        
        # Edge labels
        edge_labels = {(e.source, e.target): f"{e.strength:.2f}" for e in graph.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Knowledge Dependency Graph")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Graph saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib or networkx not installed for visualization")


if __name__ == "__main__":
    # Test the graph utilities
    graph = KnowledgeDependencyGraph()
    
    # Add some test nodes
    nodes = [
        KnowledgeNode("algebra-basics", "Algebra", "Basic Operations"),
        KnowledgeNode("algebra-linear", "Algebra", "Linear Equations"),
        KnowledgeNode("algebra-quadratic", "Algebra", "Quadratic Equations"),
        KnowledgeNode("calculus-limits", "Calculus", "Limits"),
        KnowledgeNode("calculus-derivatives", "Calculus", "Derivatives"),
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Add dependencies
    graph.add_edge("algebra-basics", "algebra-linear", 0.8)
    graph.add_edge("algebra-linear", "algebra-quadratic", 0.7)
    graph.add_edge("algebra-basics", "calculus-limits", 0.6)
    graph.add_edge("calculus-limits", "calculus-derivatives", 0.9)
    graph.add_edge("algebra-quadratic", "calculus-derivatives", 0.5)
    
    # Test topological sort
    order = graph.topological_sort()
    print(f"Topological order: {order}")
    
    # Test curriculum stages
    stages = build_curriculum_stages(
        graph,
        ["algebra-linear", "algebra-quadratic", "calculus-derivatives"]
    )
    print(f"Curriculum stages: {stages}")
    
    # Test connected components
    components = graph.find_connected_components()
    print(f"Connected components: {components}")