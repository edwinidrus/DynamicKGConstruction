from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SKGBConfig:
    # I/O
    out_dir: Path
    build_docling_dir: Path
    chunks_output_dir: Path
    kg_output_dir: Path

    # Ollama
    ollama_base_url: str
    llm_model: str
    embeddings_model: str
    temperature: float

    # itext2kg thresholds
    ent_threshold: float
    rel_threshold: float
    max_workers: int

    # chunking
    min_chunk_words: int
    max_chunk_words: int
    overlap_words: int
    preserve_metadata: bool

    @staticmethod
    def from_out_dir(
        out_dir: str | Path,
        *,
        ollama_base_url: str | None = None,
        llm_model: str | None = None,
        embeddings_model: str | None = None,
        temperature: float = 0.0,
        ent_threshold: float = 0.8,
        rel_threshold: float = 0.7,
        max_workers: int = 4,
        min_chunk_words: int = 200,
        max_chunk_words: int = 800,
        overlap_words: int = 50,
        preserve_metadata: bool = True,
    ) -> "SKGBConfig":
        out = Path(out_dir)
        return SKGBConfig(
            out_dir=out,
            build_docling_dir=out / "build_docling",
            chunks_output_dir=out / "chunks_output",
            kg_output_dir=out / "kg_output",
            ollama_base_url=(
                ollama_base_url
                or os.environ.get("OLLAMA_BASE_URL")
                or os.environ.get("OLLAMA_HOST")
                or "http://localhost:11434"
            ),
            llm_model=llm_model or os.environ.get("LLM_MODEL") or "qwen2.5:32b",
            embeddings_model=(
                embeddings_model
                or os.environ.get("EMBEDDINGS_MODEL")
                or "nomic-embed-text"
            ),
            temperature=temperature,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            max_workers=max_workers,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            overlap_words=overlap_words,
            preserve_metadata=preserve_metadata,
        )
