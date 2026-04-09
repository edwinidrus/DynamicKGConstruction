from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RetrievalConfig, load_retrieval_config
from .entity_graph import build_answer_prompt, build_fallback_answer
from .factory import RetrievalRuntime, build_runtime


@dataclass(frozen=True)
class RetrievedItem:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    items: list[RetrievedItem]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphAnswer:
    answer: str
    context: list[RetrievedItem] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_items(items: list[Any]) -> list[RetrievedItem]:
    converted: list[RetrievedItem] = []
    for item in items:
        converted.append(
            RetrievedItem(
                content=str(getattr(item, "content", "")),
                metadata=dict(getattr(item, "metadata", {}) or {}),
            )
        )
    return converted


def _resolve_runtime(
    *,
    runtime: RetrievalRuntime | None,
    config: RetrievalConfig | None,
    env_file: str | Path | None,
    retrieval_query: str | None,
) -> tuple[RetrievalRuntime, bool]:
    if runtime is not None:
        return runtime, False
    return build_runtime(config, env_file=env_file, retrieval_query=retrieval_query), True


def _answer_with_entity_graph(
    question: str,
    *,
    top_k: int,
    response_fallback: str,
    runtime: RetrievalRuntime,
    return_context: bool,
) -> GraphAnswer:
    retriever_result = runtime.retriever.search(
        query_text=question,
        top_k=top_k,
    )
    raw_items = list(getattr(retriever_result, "items", []) or [])
    context_items = _to_items(raw_items)
    metadata = {"top_k": top_k, "retrieval_strategy": runtime.config.retrieval_strategy}
    metadata.update(dict(getattr(retriever_result, "metadata", {}) or {}))

    if not context_items:
        return GraphAnswer(
            answer=response_fallback,
            context=context_items if return_context else None,
            metadata=metadata,
        )

    prompt = build_answer_prompt(question, raw_items)
    try:
        llm_response = runtime.llm.invoke(prompt)
        answer = str(getattr(llm_response, "content", "") or "").strip() or response_fallback
    except Exception as exc:
        metadata["llm_error"] = str(exc)
        answer = build_fallback_answer(question, raw_items, response_fallback)

    return GraphAnswer(
        answer=answer,
        context=context_items if return_context else None,
        metadata=metadata,
    )


def build_rag(
    config: RetrievalConfig | None = None,
    *,
    env_file: str | Path | None = None,
    retrieval_query: str | None = None,
    validate: bool = True,
) -> RetrievalRuntime:
    """Build a reusable retrieval runtime for notebook sessions."""

    return build_runtime(
        config,
        env_file=env_file,
        retrieval_query=retrieval_query,
        validate=validate,
    )


def search_context(
    question: str,
    *,
    top_k: int | None = None,
    runtime: RetrievalRuntime | None = None,
    config: RetrievalConfig | None = None,
    env_file: str | Path | None = None,
    retrieval_query: str | None = None,
) -> SearchResult:
    """Run only the retriever and return notebook-friendly context items."""

    resolved_runtime, owns_runtime = _resolve_runtime(
        runtime=runtime,
        config=config,
        env_file=env_file,
        retrieval_query=retrieval_query,
    )
    try:
        resolved_top_k = top_k or resolved_runtime.config.retriever_top_k
        retriever_result = resolved_runtime.retriever.search(
            query_text=question,
            top_k=resolved_top_k,
        )
        metadata = dict(getattr(retriever_result, "metadata", {}) or {})
        metadata.setdefault("top_k", resolved_top_k)
        return SearchResult(
            items=_to_items(list(getattr(retriever_result, "items", []) or [])),
            metadata=metadata,
        )
    finally:
        if owns_runtime:
            resolved_runtime.close()


def ask_graph(
    question: str,
    *,
    top_k: int | None = None,
    return_context: bool = False,
    response_fallback: str | None = None,
    runtime: RetrievalRuntime | None = None,
    config: RetrievalConfig | None = None,
    env_file: str | Path | None = None,
    retrieval_query: str | None = None,
) -> GraphAnswer:
    """Run full GraphRAG question answering against the existing Neo4j graph."""

    resolved_runtime, owns_runtime = _resolve_runtime(
        runtime=runtime,
        config=config,
        env_file=env_file,
        retrieval_query=retrieval_query,
    )
    try:
        resolved_top_k = top_k or resolved_runtime.config.retriever_top_k
        resolved_fallback = response_fallback or resolved_runtime.config.response_fallback

        if resolved_runtime.config.retrieval_strategy == "entity_graph" or resolved_runtime.rag is None:
            return _answer_with_entity_graph(
                question,
                top_k=resolved_top_k,
                response_fallback=resolved_fallback,
                runtime=resolved_runtime,
                return_context=return_context,
            )

        rag_result = resolved_runtime.rag.search(
            query_text=question,
            retriever_config={"top_k": resolved_top_k},
            return_context=return_context,
            response_fallback=resolved_fallback,
        )

        retriever_result = getattr(rag_result, "retriever_result", None)
        context = None
        metadata = {"top_k": resolved_top_k}
        if retriever_result is not None:
            metadata.update(dict(getattr(retriever_result, "metadata", {}) or {}))
            context = _to_items(list(getattr(retriever_result, "items", []) or []))

        return GraphAnswer(
            answer=str(getattr(rag_result, "answer", "")),
            context=context,
            metadata=metadata,
        )
    finally:
        if owns_runtime:
            resolved_runtime.close()


__all__ = [
    "GraphAnswer",
    "RetrievedItem",
    "SearchResult",
    "ask_graph",
    "build_rag",
    "load_retrieval_config",
    "search_context",
]
