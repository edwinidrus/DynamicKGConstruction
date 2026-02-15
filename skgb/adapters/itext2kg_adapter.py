from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def _run(coro):
    """Run async coroutine, handling Jupyter/Colab event loop conflicts."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        # In Jupyter/Colab environments, there's already an event loop running
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        import nest_asyncio
        nest_asyncio.apply()
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
    from itext2kg.graphs.knowledge_graph import KnowledgeGraph

    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
        base_url=ollama_base_url,
    )
    embeddings = OllamaEmbeddings(
        model=embeddings_model,
        base_url=ollama_base_url,
    )

    # Log what we're processing
    total_facts = sum(len(facts) for facts in atomic_facts_dict.values())
    logger.info(f"Building KG from {total_facts} atomic facts across {len(atomic_facts_dict)} timestamps")

    atom = Atom(llm_model=llm, embeddings_model=embeddings)
    
    try:
        kg = await atom.build_graph_from_different_obs_times(
            atomic_facts_with_obs_timestamps=atomic_facts_dict,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            max_workers=max_workers,
        )
    except IndexError as e:
        # Handle the itext2kg bug where parallel_atomic_merge returns current[0]
        # on an empty list - this happens when all atomic KGs are empty
        logger.warning(
            "itext2kg IndexError during atomic KG building - likely no valid entities/relations extracted. "
            "This can happen when: (1) LLM returns malformed quintuples, (2) embeddings fail to match entities, "
            "(3) the model provider is not recognized (check for 'Unknown provider' warnings above). "
            f"Error: {e}"
        )
        # Return empty KG instead of crashing
        kg = KnowledgeGraph()
    except Exception as e:
        # Catch any other exceptions from itext2kg and provide helpful context
        error_msg = str(e)
        if "list index out of range" in error_msg.lower():
            logger.warning(
                f"itext2kg failed with IndexError: {e}. "
                "This typically means no valid knowledge graph entities could be extracted."
            )
            kg = KnowledgeGraph()
        else:
            logger.error(f"itext2kg failed with unexpected error: {e}")
            raise
    
    # Log results
    if kg and not kg.is_empty():
        logger.info(f"Successfully built KG with {len(kg.entities)} entities and {len(kg.relations)} relations")
    else:
        logger.warning("Knowledge graph is empty - no entities or relations extracted")
    
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
    try:
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
    except IndexError as e:
        # Catch IndexError that propagates through the event loop
        # This is the itext2kg parallel_atomic_merge bug where current[0] fails on empty list
        logger.warning(
            f"itext2kg IndexError (caught at sync level): {e}. "
            "No valid knowledge graph could be built from the atomic facts. "
            "Returning empty KnowledgeGraph."
        )
        from itext2kg.graphs.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph()
