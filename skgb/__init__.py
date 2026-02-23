"""SKGB - Semantic Knowledge Graph Builder.

This package orchestrates:
- Docling ingestion (PDF -> Markdown)
- Semantic chunking (Markdown -> chunks)
- itext2kg incremental KG construction (chunks -> Knowledge Graph)

Usage (Python / Jupyter / Colab):
    from DynamicKGConstruction.skgb import SKGBConfig, run_pipeline
    from pathlib import Path

    cfg = SKGBConfig.from_out_dir("./my_run")
    result = run_pipeline(Path("document.pdf"), cfg)
    print(result.kg_output_dir)
"""

from __future__ import annotations

from .config import SKGBConfig
from .pipeline import run_pipeline, PipelineResult
from .adapters.docling_adapter import docling_convert_to_text
from .models import ModelRegistry, LLMProvider, detect_provider

__all__ = [
    "__version__",
    "SKGBConfig",
    "run_pipeline",
    "PipelineResult",
    "docling_convert_to_text",
    "ModelRegistry",
    "LLMProvider",
    "detect_provider",
]

__version__ = "0.1.0"
