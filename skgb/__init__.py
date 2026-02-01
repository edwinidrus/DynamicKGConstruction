"""SKGB - Semantic Knowledge Graph Builder.

This package orchestrates:
- Docling ingestion (PDF -> Markdown)
- Semantic chunking (Markdown -> chunks)
- itext2kg incremental KG construction (chunks -> Knowledge Graph)
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
