"""
Utility modules for IOA Framework

This package contains utility functions and classes:
- LLMClient: API client for teacher model access
- KnowledgeDependencyGraph: DAG for knowledge module dependencies (Section 3.2)
- Curriculum construction utilities (Section 3.3, Eq. 7, 8)
"""

from .llm_client import LLMClient, create_teacher_client
from .graph_utils import (
    KnowledgeNode,
    DependencyEdge,
    KnowledgeDependencyGraph,
    build_curriculum_stages,
    group_similar_modules,
    compute_stage_difficulty,
    check_zpd_constraint
)

__all__ = [
    # LLM client
    "LLMClient",
    "create_teacher_client",
    # Graph data structures
    "KnowledgeNode",
    "DependencyEdge",
    "KnowledgeDependencyGraph",
    # Curriculum utilities
    "build_curriculum_stages",
    "group_similar_modules",
    "compute_stage_difficulty",
    "check_zpd_constraint"
]