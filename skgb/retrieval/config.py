from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_RESPONSE_FALLBACK = (
    "I cannot answer this question because no relevant graph context was retrieved."
)


def _parse_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_file.exists():
        return values

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value

    return values


def _pick_value(values: Mapping[str, str], *keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = values.get(key)
        if value is not None and value != "":
            return value
    return default


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _normalize_ollama_host(host: str) -> str:
    normalized = host.strip().rstrip("/")
    for suffix in ("/api", "/v1"):
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized.rstrip("/")


def _normalize_retrieval_strategy(value: str | None) -> str:
    normalized = (value or "entity_graph").strip().lower()
    if normalized not in {"entity_graph", "vector"}:
        raise ValueError(
            "Unsupported retrieval strategy. Use 'entity_graph' or 'vector'."
        )
    return normalized


@dataclass(frozen=True)
class RetrievalConfig:
    env_file: Path
    neo4j_uri: str
    neo4j_username: str | None
    neo4j_password: str | None
    neo4j_auth_disabled: bool
    neo4j_database: str
    neo4j_vector_index: str | None
    neo4j_fulltext_index: str | None
    ollama_host: str
    ollama_api_key: str | None
    ollama_llm_model: str
    ollama_embeddings_model: str | None
    retriever_top_k: int
    response_fallback: str
    llm_temperature: float
    chunk_to_document_relationship: str
    entity_to_chunk_relationship: str
    retrieval_strategy: str

    @classmethod
    def from_env_file(cls, env_file: str | Path | None = None) -> "RetrievalConfig":
        resolved_env_file = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
        file_values = _parse_env_file(resolved_env_file)
        merged_values: dict[str, str] = {**file_values, **os.environ}
        retrieval_strategy = _normalize_retrieval_strategy(
            _pick_value(
                merged_values,
                "RETRIEVAL_STRATEGY",
                "retrieval_strategy",
                default="entity_graph",
            )
        )
        neo4j_auth_value = _pick_value(merged_values, "NEO4J_AUTH", "neo4j_auth")
        neo4j_auth_disabled = (neo4j_auth_value or "").strip().lower() == "none"

        ollama_llm_model = _pick_value(
            merged_values,
            "OLLAMA_LLM_MODEL",
            "ollama_llm_model",
            "OLLAMA_API_MODEL",
            "ollama_api_model",
            "LLM_MODEL",
        )

        ollama_embeddings_model = _pick_value(
            merged_values,
            "OLLAMA_EMBEDDINGS_MODEL",
            "ollama_embeddings_model",
            "EMBEDDINGS_MODEL",
            default=ollama_llm_model,
        )

        values = {
            "neo4j_uri": _pick_value(merged_values, "NEO4J_URI", "neo4j_uri"),
            "neo4j_vector_index": _pick_value(
                merged_values,
                "NEO4J_VECTOR_INDEX",
                "neo4j_vector_index",
                "NEO4J_INDEX_NAME",
                "neo4j_index_name",
            ),
            "ollama_host": _pick_value(
                merged_values,
                "OLLAMA_HOST",
                "ollama_host",
                "OLLAMA_API_URL",
                "ollama_api_url",
                "OLLAMA_BASE_URL",
            ),
            "ollama_llm_model": ollama_llm_model,
            "ollama_embeddings_model": ollama_embeddings_model,
        }

        neo4j_username = _pick_value(
            merged_values,
            "NEO4J_USERNAME",
            "neo4j_username",
            "NEO4J_USER",
            "neo4j_user",
        )
        neo4j_password = _pick_value(merged_values, "NEO4J_PASSWORD", "neo4j_password")

        if not neo4j_auth_disabled:
            values["neo4j_username"] = neo4j_username
            values["neo4j_password"] = neo4j_password

        required_fields = {"neo4j_uri", "ollama_host", "ollama_llm_model"}
        if retrieval_strategy == "vector":
            required_fields.update({"neo4j_vector_index", "ollama_embeddings_model"})

        missing = [
            name for name, value in values.items() if name in required_fields and value in (None, "")
        ]
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(
                "Missing retrieval configuration values: "
                f"{missing_keys}. Update .env at the repository root or environment variables."
            )

        return cls(
            env_file=resolved_env_file,
            neo4j_uri=values["neo4j_uri"] or "",
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_auth_disabled=neo4j_auth_disabled,
            neo4j_database=_pick_value(
                merged_values, "NEO4J_DATABASE", "neo4j_database", default="neo4j"
            )
            or "neo4j",
            neo4j_vector_index=values["neo4j_vector_index"] or None,
            neo4j_fulltext_index=_pick_value(
                merged_values,
                "NEO4J_FULLTEXT_INDEX",
                "neo4j_fulltext_index",
            ),
            ollama_host=_normalize_ollama_host(values["ollama_host"] or ""),
            ollama_api_key=_pick_value(
                merged_values,
                "OLLAMA_API_KEY",
                "ollama_api_key",
            ),
            ollama_llm_model=values["ollama_llm_model"] or "",
            ollama_embeddings_model=values["ollama_embeddings_model"] or None,
            retriever_top_k=_parse_int(
                _pick_value(merged_values, "RETRIEVER_TOP_K", "retriever_top_k"),
                default=5,
            ),
            response_fallback=_pick_value(
                merged_values,
                "RESPONSE_FALLBACK",
                "response_fallback",
                default=DEFAULT_RESPONSE_FALLBACK,
            )
            or DEFAULT_RESPONSE_FALLBACK,
            llm_temperature=_parse_float(
                _pick_value(merged_values, "LLM_TEMPERATURE", "llm_temperature"),
                default=0.0,
            ),
            chunk_to_document_relationship=_pick_value(
                merged_values,
                "CHUNK_TO_DOCUMENT_RELATIONSHIP",
                "chunk_to_document_relationship",
                default="FROM_DOCUMENT",
            )
            or "FROM_DOCUMENT",
            entity_to_chunk_relationship=_pick_value(
                merged_values,
                "ENTITY_TO_CHUNK_RELATIONSHIP",
                "entity_to_chunk_relationship",
                default="FROM_CHUNK",
            )
            or "FROM_CHUNK",
            retrieval_strategy=retrieval_strategy,
        )

    def ollama_headers(self) -> dict[str, str] | None:
        if not self.ollama_api_key:
            return None
        return {"Authorization": f"Bearer {self.ollama_api_key}"}


def load_retrieval_config(env_file: str | Path | None = None) -> RetrievalConfig:
    return RetrievalConfig.from_env_file(env_file)


__all__ = [
    "DEFAULT_ENV_FILE",
    "DEFAULT_RESPONSE_FALLBACK",
    "REPO_ROOT",
    "RetrievalConfig",
    "load_retrieval_config",
]
