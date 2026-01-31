"""itext2KG module for Knowledge Graph construction using LLM-powered extraction."""

from .kg_constructor import (
    KGConstructor,
    KGConfig,
    KGResult,
    Entity,
    Relationship,
)

__all__ = [
    "KGConstructor",
    "KGConfig",
    "KGResult",
    "Entity",
    "Relationship",
]
