from __future__ import annotations

import asyncio
from typing import Dict, List


def _run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        # In rare cases (already-running loop), fall back to creating a task.
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


async def _build_async(
    *,
    atomic_facts_dict: Dict[str, List[str]],
    ollama_base_url: str,
    llm_model: str,
    embeddings_model: str,
    temperature: float,
    ent_threshold: float,
    rel_threshold: float,
    max_workers: int,
):
    try:
        import itext2kg  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "itext2kg is not installed in this Python environment. "
            "Install dependencies first (see DynamicKGConstruction/requirements.txt)."
        ) from e

    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except Exception as e:
        raise RuntimeError(
            "langchain-ollama is not installed in this Python environment. "
            "Install dependencies first (see DynamicKGConstruction/requirements.txt)."
        ) from e

    from itext2kg.atom import Atom

    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
        base_url=ollama_base_url,
    )
    embeddings = OllamaEmbeddings(
        model=embeddings_model,
        base_url=ollama_base_url,
    )

    atom = Atom(llm_model=llm, embeddings_model=embeddings)
    kg = await atom.build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps=atomic_facts_dict,
        ent_threshold=ent_threshold,
        rel_threshold=rel_threshold,
        max_workers=max_workers,
    )
    return kg


def build_kg_from_atomic_facts(
    *,
    atomic_facts_dict: Dict[str, List[str]],
    ollama_base_url: str,
    llm_model: str,
    embeddings_model: str,
    temperature: float = 0.0,
    ent_threshold: float = 0.8,
    rel_threshold: float = 0.7,
    max_workers: int = 4,
):
    """Build a KnowledgeGraph using itext2kg ATOM (async under the hood)."""
    return _run(
        _build_async(
            atomic_facts_dict=atomic_facts_dict,
            ollama_base_url=ollama_base_url,
            llm_model=llm_model,
            embeddings_model=embeddings_model,
            temperature=temperature,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            max_workers=max_workers,
        )
    )
