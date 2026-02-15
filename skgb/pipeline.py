from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .config import SKGBConfig


@dataclass(frozen=True)
class PipelineResult:
    build_docling_dir: Path
    chunks_json_path: Path
    kg_output_dir: Path
    neo4j_cypher_path: Path


def _ensure_dirs(cfg: SKGBConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.build_docling_dir.mkdir(parents=True, exist_ok=True)
    cfg.chunks_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.kg_output_dir.mkdir(parents=True, exist_ok=True)


def run_pipeline(
    input_path: Path, cfg: SKGBConfig, *, recursive: bool = True
) -> PipelineResult:
    _ensure_dirs(cfg)

    from .adapters.docling_adapter import docling_convert_to_markdown
    from .adapters.chunking_adapter import chunk_markdown_files
    from .adapters.itext2kg_adapter import build_kg_from_atomic_facts
    from .export.file_export import export_kg_outputs
    from .export.neo4j_export import write_neo4j_load_cypher

    md_paths = docling_convert_to_markdown(
        input_path=input_path,
        output_dir=cfg.build_docling_dir,
        recursive=recursive,
    )
    if not md_paths:
        raise SystemExit(f"No documents were converted from input: {input_path}")

    all_chunks: List[Dict[str, Any]] = chunk_markdown_files(
        md_paths=md_paths,
        min_chunk_words=cfg.min_chunk_words,
        max_chunk_words=cfg.max_chunk_words,
        overlap_words=cfg.overlap_words,
        preserve_metadata=cfg.preserve_metadata,
    )

    chunks_json_path = cfg.chunks_output_dir / "all_chunks.json"
    chunks_json_path.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    t_obs = datetime.datetime.now().strftime("%Y-%m-%d")
    atomic_facts_dict: Dict[str, List[str]] = {t_obs: []}
    for ch in all_chunks:
        section_title = ch.get("section_title") or ""
        content = ch.get("content") or ""
        atomic_facts_dict[t_obs].append(f"[{section_title}] {content}".strip())

    kg = build_kg_from_atomic_facts(
        atomic_facts_dict=atomic_facts_dict,
        ollama_base_url=cfg.ollama_base_url,
        llm_model=cfg.llm_model,
        embeddings_model=cfg.embeddings_model,
        temperature=cfg.temperature,
        ent_threshold=cfg.ent_threshold,
        rel_threshold=cfg.rel_threshold,
        max_workers=cfg.max_workers,
    )

    export_kg_outputs(
        kg=kg,
        kg_output_dir=cfg.kg_output_dir,
        total_chunks=len(all_chunks),
        ent_threshold=cfg.ent_threshold,
        rel_threshold=cfg.rel_threshold,
        llm_model=cfg.llm_model,
        embeddings_model=cfg.embeddings_model,
    )

    neo4j_cypher_path = write_neo4j_load_cypher(cfg.kg_output_dir)

    return PipelineResult(
        build_docling_dir=cfg.build_docling_dir,
        chunks_json_path=chunks_json_path,
        kg_output_dir=cfg.kg_output_dir,
        neo4j_cypher_path=neo4j_cypher_path,
    )
