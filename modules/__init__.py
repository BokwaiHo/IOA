"""
Core modules for IOA Framework

This package contains the three main components:
- Identifier: Knowledge deficiency diagnosis (Section 3.2)
- Organizer: Progressive curriculum design (Section 3.3)
- Adapter: Knowledge representation adaptation (Section 3.4)
"""

from .identifier import KnowledgeIdentifier
from .organizer import (
    KnowledgeOrganizer,
    CurriculumStage,
    Curriculum,
    create_learning_schedule
)
from .adapter import (
    KnowledgeAdapter,
    adapt_knowledge_for_curriculum
)

__all__ = [
    # Identifier
    "KnowledgeIdentifier",
    # Organizer
    "KnowledgeOrganizer",
    "CurriculumStage",
    "Curriculum",
    "create_learning_schedule",
    # Adapter
    "KnowledgeAdapter",
    "adapt_knowledge_for_curriculum"
]