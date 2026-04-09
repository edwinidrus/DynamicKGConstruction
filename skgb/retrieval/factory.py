from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RetrievalConfig, load_retrieval_config
from .entity_graph import EntityGraphRetriever


def _cypher_ident(value: str) -> str:
    return f"`{value.replace('`', '``')}`"


def build_default_retrieval_query(cfg: RetrievalConfig) -> str:
    from_doc_rel = _cypher_ident(cfg.chunk_to_document_relationship)
    from_chunk_rel = _cypher_ident(cfg.entity_to_chunk_relationship)
    return f"""
WITH node AS chunk, score
OPTIONAL MATCH (chunk)-[:{from_doc_rel}]->(doc)
OPTIONAL MATCH (entity)-[:{from_chunk_rel}]->(chunk)
WITH chunk, score, doc, entity,
     CASE
         WHEN entity IS NULL THEN NULL
         ELSE [label IN labels(entity) WHERE NOT label IN ['Chunk', 'Document']][0]
     END AS entity_label
WITH chunk, score, doc,
     collect(DISTINCT CASE
         WHEN entity IS NULL THEN NULL
         WHEN entity_label IS NULL THEN coalesce(entity.name, entity.id, elementId(entity))
         ELSE entity_label + ': ' + coalesce(entity.name, entity.id, elementId(entity))
     END) AS entities
RETURN
    coalesce(chunk.text, chunk.content, chunk.chunk_text, chunk.body, '') AS text,
    score,
    coalesce(doc.source, doc.title, doc.path, doc.doc_name, doc.name, doc.id, '') AS source,
    [entity_name IN entities WHERE entity_name IS NOT NULL] AS entities
""".strip()


def _build_result_formatter():
    try:
        from neo4j_graphrag.types import RetrieverResultItem
    except ImportError as exc:
        raise RuntimeError(
            "neo4j-graphrag is required for retrieval. Install it with "
            "`pip install -r skgb/retrieval/requirements.txt`."
        ) from exc

    def formatter(record: Any) -> Any:
        text = str(record.get("text") or "").strip()
        source = str(record.get("source") or "").strip()
        entities = [str(item).strip() for item in (record.get("entities") or []) if item]

        parts = []
        if text:
            parts.append(text)
        if entities:
            parts.append(f"Related entities: {', '.join(entities)}")
        if source:
            parts.append(f"Source: {source}")
        if not parts:
            parts.append(str(record))

        return RetrieverResultItem(
            content="\n".join(parts),
            metadata={
                "score": record.get("score"),
                "source": source or None,
                "entities": entities,
            },
        )

    return formatter


def create_driver(cfg: RetrievalConfig):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "The Neo4j Python driver is required for retrieval. Install it with "
            "`pip install -r skgb/retrieval/requirements.txt`."
        ) from exc

    if cfg.neo4j_auth_disabled:
        return GraphDatabase.driver(cfg.neo4j_uri)

    return GraphDatabase.driver(
        cfg.neo4j_uri,
        auth=(cfg.neo4j_username, cfg.neo4j_password),
    )


def create_embedder(cfg: RetrievalConfig):
    if not cfg.ollama_embeddings_model:
        raise ValueError(
            "An embeddings model is required when using vector retrieval. "
            "Set OLLAMA_EMBEDDINGS_MODEL or switch to RETRIEVAL_STRATEGY=entity_graph."
        )

    try:
        from neo4j_graphrag.embeddings import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "neo4j-graphrag[ollama] is required for retrieval. Install it with "
            "`pip install -r skgb/retrieval/requirements.txt`."
        ) from exc

    embedder_kwargs: dict[str, Any] = {"host": cfg.ollama_host}
    headers = cfg.ollama_headers()
    if headers:
        embedder_kwargs["headers"] = headers

    return OllamaEmbeddings(model=cfg.ollama_embeddings_model, **embedder_kwargs)


def create_llm(cfg: RetrievalConfig):
    try:
        from neo4j_graphrag.llm import OllamaLLM
    except ImportError as exc:
        raise RuntimeError(
            "neo4j-graphrag[ollama] is required for retrieval. Install it with "
            "`pip install -r skgb/retrieval/requirements.txt`."
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "host": cfg.ollama_host,
        "model_params": {"options": {"temperature": cfg.llm_temperature}},
    }
    headers = cfg.ollama_headers()
    if headers:
        llm_kwargs["headers"] = headers

    return OllamaLLM(model_name=cfg.ollama_llm_model, **llm_kwargs)


def create_retriever(cfg: RetrievalConfig, *, driver, embedder, retrieval_query: str | None = None):
    if cfg.retrieval_strategy == "entity_graph":
        return EntityGraphRetriever(driver=driver, neo4j_database=cfg.neo4j_database)

    try:
        from neo4j_graphrag.retrievers import VectorCypherRetriever
    except ImportError as exc:
        raise RuntimeError(
            "neo4j-graphrag is required for retrieval. Install it with "
            "`pip install -r skgb/retrieval/requirements.txt`."
        ) from exc

    return VectorCypherRetriever(
        driver=driver,
        index_name=cfg.neo4j_vector_index,
        retrieval_query=retrieval_query or build_default_retrieval_query(cfg),
        embedder=embedder,
        result_formatter=_build_result_formatter(),
        neo4j_database=cfg.neo4j_database,
    )


def create_graphrag(*, retriever, llm):
    try:
        from neo4j_graphrag.generation import GraphRAG
    except ImportError as exc:
        raise RuntimeError(
            "neo4j-graphrag is required for retrieval. Install it with "
            "`pip install -r skgb/retrieval/requirements.txt`."
        ) from exc

    return GraphRAG(retriever=retriever, llm=llm)


def verify_runtime(driver, cfg: RetrievalConfig, *, retriever: Any | None = None) -> None:
    driver.verify_connectivity()

    if cfg.retrieval_strategy == "entity_graph":
        if retriever is None:
            retriever = EntityGraphRetriever(driver=driver, neo4j_database=cfg.neo4j_database)
        retriever.validate()
        return

    query = """
    SHOW INDEXES YIELD name, type
WHERE name = $name AND type = 'VECTOR'
RETURN count(*) AS index_count
""".strip()
    with driver.session(database=cfg.neo4j_database) as session:
        record = session.run(query, name=cfg.neo4j_vector_index).single()
    index_count = 0 if record is None else int(record.get("index_count") or 0)
    if index_count <= 0:
        raise ValueError(
            f"Neo4j vector index '{cfg.neo4j_vector_index}' was not found in database "
            f"'{cfg.neo4j_database}'."
        )


@dataclass
class RetrievalRuntime:
    """Live retrieval objects that can be reused across notebook cells."""

    config: RetrievalConfig
    driver: Any
    embedder: Any
    retriever: Any
    llm: Any
    rag: Any

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "RetrievalRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def build_runtime(
    config: RetrievalConfig | None = None,
    *,
    env_file: str | Path | None = None,
    retrieval_query: str | None = None,
    validate: bool = True,
) -> RetrievalRuntime:
    """Build a reusable Neo4j GraphRAG runtime."""

    cfg = config or load_retrieval_config(env_file)
    driver = create_driver(cfg)
    try:
        embedder = create_embedder(cfg) if cfg.retrieval_strategy == "vector" else None
        llm = create_llm(cfg)
        retriever = create_retriever(
            cfg,
            driver=driver,
            embedder=embedder,
            retrieval_query=retrieval_query,
        )
        if validate:
            verify_runtime(driver, cfg, retriever=retriever)

        rag = create_graphrag(retriever=retriever, llm=llm) if cfg.retrieval_strategy == "vector" else None
        return RetrievalRuntime(
            config=cfg,
            driver=driver,
            embedder=embedder,
            retriever=retriever,
            llm=llm,
            rag=rag,
        )
    except Exception:
        driver.close()
        raise


__all__ = [
    "RetrievalRuntime",
    "build_default_retrieval_query",
    "build_runtime",
    "create_driver",
    "create_embedder",
    "create_graphrag",
    "create_llm",
    "create_retriever",
    "verify_runtime",
]
