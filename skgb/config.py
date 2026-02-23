from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SKGBConfig:
    # I/O
    out_dir: Path
    build_docling_dir: Path
    chunks_output_dir: Path
    kg_output_dir: Path

    # LLM — provider is auto-detected from llm_model name
    ollama_base_url: str
    llm_model: str
    provider: str            # "ollama" | "anthropic" | "openai"
    api_key: Optional[str]   # API key for Anthropic / OpenAI LLMs

    # Embeddings — provider is auto-detected from embeddings_model name
    embeddings_model: str
    embeddings_provider: str           # "ollama" | "openai"
    embeddings_api_key: Optional[str]  # API key for OpenAI embeddings

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
        # Cloud provider keys
        api_key: str | None = None,
        embeddings_api_key: str | None = None,
    ) -> "SKGBConfig":
        """Create a config, auto-detecting providers from model names.

        Parameters
        ----------
        llm_model:
            Any Ollama model name (e.g. ``"qwen2.5:32b"``), a Claude model
            (e.g. ``"claude-sonnet-4-6"``), or an OpenAI model (e.g. ``"gpt-4o"``).
            Provider is detected automatically from the name.
        embeddings_model:
            Any Ollama embeddings model (e.g. ``"nomic-embed-text"``) or an
            OpenAI embeddings model (e.g. ``"text-embedding-3-small"``).
            Note: Anthropic has no public embeddings API — use Ollama or OpenAI.
        api_key:
            API key for the LLM provider (Anthropic or OpenAI).  Falls back to
            ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` environment variables.
        embeddings_api_key:
            API key for the embeddings provider (OpenAI only).  Falls back to
            ``OPENAI_API_KEY``.
        """
        from .models import detect_provider  # deferred to avoid circular issues

        out = Path(out_dir)

        _llm_model = llm_model or os.environ.get("LLM_MODEL") or "qwen2.5:32b"
        _emb_model = (
            embeddings_model
            or os.environ.get("EMBEDDINGS_MODEL")
            or "nomic-embed-text"
        )

        # Auto-detect providers
        _provider = detect_provider(_llm_model).value
        _emb_provider = detect_provider(_emb_model).value

        # Resolve API keys (parameter > env var)
        _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        _emb_api_key = embeddings_api_key or os.environ.get("OPENAI_API_KEY")

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
            llm_model=_llm_model,
            provider=_provider,
            api_key=_api_key,
            embeddings_model=_emb_model,
            embeddings_provider=_emb_provider,
            embeddings_api_key=_emb_api_key,
            temperature=temperature,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            max_workers=max_workers,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            overlap_words=overlap_words,
            preserve_metadata=preserve_metadata,
        )
