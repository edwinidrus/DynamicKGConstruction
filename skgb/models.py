"""Centralized model registry for SKGB.

Supports three LLM / embeddings providers:

* **Ollama** (local, default) — any model name not matching Claude or GPT patterns.
* **Anthropic / Claude**      — model names starting with ``claude-``.
* **OpenAI**                  — model names starting with ``gpt-``, ``o1-``, ``o3-``,
                                 ``text-embedding-``, etc.

Typical usage
-------------
::

    from DynamicKGConstruction.skgb.models import ModelRegistry

    # Claude LLM + Ollama embeddings (recommended: quality + privacy)
    llm = ModelRegistry.create_llm(
        "claude-sonnet-4-6",
        api_key="sk-ant-...",          # or set ANTHROPIC_API_KEY env var
    )
    emb = ModelRegistry.create_embeddings(
        "nomic-embed-text",
        ollama_base_url="http://localhost:11434",
    )

    # Fully local (Ollama for both)
    llm = ModelRegistry.create_llm("qwen2.5:32b",
                                   ollama_base_url="http://localhost:11434")
    emb = ModelRegistry.create_embeddings("nomic-embed-text",
                                          ollama_base_url="http://localhost:11434")

    # OpenAI for both
    llm = ModelRegistry.create_llm("gpt-4o", api_key="sk-...")
    emb = ModelRegistry.create_embeddings("text-embedding-3-small", api_key="sk-...")
"""

from __future__ import annotations

import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    """Supported LLM / embeddings providers."""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_ANTHROPIC_RE = re.compile(r"^claude[-_]", re.IGNORECASE)
_OPENAI_RE = re.compile(
    r"^(gpt-|o1-|o3-|chatgpt-|text-embedding-|text-davinci-)",
    re.IGNORECASE,
)


def detect_provider(model_name: str) -> LLMProvider:
    """Auto-detect the provider from a model name.

    Returns :attr:`LLMProvider.ANTHROPIC` for ``claude-*`` models,
    :attr:`LLMProvider.OPENAI` for ``gpt-*`` / ``text-embedding-*`` models,
    and :attr:`LLMProvider.OLLAMA` for everything else.
    """
    if _ANTHROPIC_RE.match(model_name):
        return LLMProvider.ANTHROPIC
    if _OPENAI_RE.match(model_name):
        return LLMProvider.OPENAI
    return LLMProvider.OLLAMA


# ---------------------------------------------------------------------------
# Model tier  (controls itext2kg batch sizes and retry behaviour)
# ---------------------------------------------------------------------------

#: Per-tier configuration for the ATOM KG-building loop.
MODEL_TIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "large":  {"batch_size": 5, "max_retries": 3, "retry_delay": 5,  "batch_delay": 10},
    "medium": {"batch_size": 3, "max_retries": 4, "retry_delay": 3,  "batch_delay": 5},
    "small":  {"batch_size": 2, "max_retries": 5, "retry_delay": 2,  "batch_delay": 3},
}

# Ordered list of (regex-pattern, tier) pairs — first match wins.
_TIER_PATTERNS: List[Tuple[str, str]] = [
    # Anthropic Claude
    (r"claude-opus",                              "large"),
    (r"claude-sonnet",                            "large"),
    (r"claude-haiku",                             "medium"),
    # OpenAI
    (r"gpt-4o|gpt-4",                            "large"),
    (r"gpt-3\.5",                                "medium"),
    # Ollama — Qwen 2.5
    (r"qwen2\.5:(32b|72b)",                      "large"),
    (r"qwen2\.5:(14b|7b)",                       "medium"),
    (r"qwen2\.5:(3b|1\.5b)",                     "small"),
    # Ollama — LLaMA / Mixtral
    (r"llama3\.\d+:70b|llama3\.3:70b|mixtral:8x22b", "large"),
    (r"llama3\.\d+:8b|mistral|gemma2:27b",        "medium"),
    # Small / embedded
    (r"phi|tinyllama|gemma2:(9b|2b)",             "small"),
]


def get_model_tier(model_name: str) -> str:
    """Return ``'large'``, ``'medium'``, or ``'small'`` for *model_name*."""
    lower = model_name.lower()
    for pattern, tier in _TIER_PATTERNS:
        if re.search(pattern, lower):
            return tier
    return "medium"  # safe default for unknown models


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Return the tier-specific batch/retry config dict for *model_name*."""
    return MODEL_TIER_CONFIGS[get_model_tier(model_name)].copy()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Factory for LangChain LLM and Embeddings objects.

    All heavy imports (``langchain_ollama``, ``langchain_anthropic``, …) are
    deferred to call time so that importing this module never fails even when
    optional packages are not installed.
    """

    # ------------------------------------------------------------------
    # LLM factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_llm(
        model_name: str,
        *,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        """Return a LangChain chat model for *model_name*.

        Provider is detected automatically:

        * ``claude-*``  → :class:`~langchain_anthropic.ChatAnthropic`
        * ``gpt-*`` etc.→ :class:`~langchain_openai.ChatOpenAI`
        * everything else → :class:`~langchain_ollama.ChatOllama`

        Parameters
        ----------
        model_name:
            Model identifier, e.g. ``"claude-sonnet-4-6"``, ``"qwen2.5:32b"``.
        temperature:
            Sampling temperature; ``0.0`` for deterministic output.
        api_key:
            API key for cloud providers.  Falls back to ``ANTHROPIC_API_KEY`` /
            ``OPENAI_API_KEY`` environment variables.
        ollama_base_url:
            Base URL of a running Ollama server (Ollama models only).
        """
        provider = detect_provider(model_name)

        # ---- Anthropic / Claude ----
        if provider == LLMProvider.ANTHROPIC:
            try:
                from langchain_anthropic import ChatAnthropic  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-anthropic is required for Claude models. "
                    "Install it: pip install 'langchain-anthropic>=0.1.0'"
                ) from exc
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required for Anthropic/Claude models. "
                    "Pass api_key= to SKGBConfig.from_out_dir() or set the "
                    "ANTHROPIC_API_KEY environment variable."
                )
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=key,
                **kwargs,
            )

        # ---- OpenAI ----
        if provider == LLMProvider.OPENAI:
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-openai is required for OpenAI models. "
                    "Install it: pip install 'langchain-openai>=0.2.14,<0.3.0'"
                ) from exc
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI models. "
                    "Pass api_key= to SKGBConfig.from_out_dir() or set the "
                    "OPENAI_API_KEY environment variable."
                )
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=key,
                **kwargs,
            )

        # ---- Ollama (default) ----
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "langchain-ollama is required for Ollama models. "
                "Install it: pip install 'langchain-ollama>=0.1.0,<1.0.0'"
            ) from exc
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=ollama_base_url,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Embeddings factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_embeddings(
        model_name: str,
        *,
        api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        """Return a LangChain embeddings object for *model_name*.

        Provider is detected automatically:

        * ``text-embedding-*`` / ``gpt-*`` → :class:`~langchain_openai.OpenAIEmbeddings`
        * ``claude-*`` → raises :exc:`ValueError` (no public Anthropic embeddings API;
          use ``nomic-embed-text`` via Ollama or an OpenAI embeddings model instead).
        * everything else → :class:`~langchain_ollama.OllamaEmbeddings`

        Parameters
        ----------
        model_name:
            Embeddings model identifier, e.g. ``"nomic-embed-text"``,
            ``"text-embedding-3-small"``.
        api_key:
            API key for OpenAI embeddings.  Falls back to ``OPENAI_API_KEY``.
        ollama_base_url:
            Base URL of a running Ollama server (Ollama embeddings only).
        """
        provider = detect_provider(model_name)

        # Anthropic has no public standalone embeddings API
        if provider == LLMProvider.ANTHROPIC:
            raise ValueError(
                "Anthropic does not provide a public standalone embeddings API. "
                "Use an Ollama embeddings model (recommended: 'nomic-embed-text') "
                "or an OpenAI embeddings model (e.g. 'text-embedding-3-small') "
                "for the embeddings_model field in SKGBConfig."
            )

        # ---- OpenAI embeddings ----
        if provider == LLMProvider.OPENAI:
            try:
                from langchain_openai import OpenAIEmbeddings  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-openai is required for OpenAI embeddings. "
                    "Install it: pip install 'langchain-openai>=0.2.14,<0.3.0'"
                ) from exc
            key = api_key or os.environ.get("OPENAI_API_KEY")
            return OpenAIEmbeddings(model=model_name, api_key=key, **kwargs)

        # ---- Ollama embeddings (default) ----
        try:
            from langchain_ollama import OllamaEmbeddings  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "langchain-ollama is required for Ollama embeddings. "
                "Install it: pip install 'langchain-ollama>=0.1.0,<1.0.0'"
            ) from exc
        return OllamaEmbeddings(model=model_name, base_url=ollama_base_url, **kwargs)
