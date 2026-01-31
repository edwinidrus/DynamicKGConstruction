"""
Knowledge Graph Constructor Module

This module provides a callable class for constructing knowledge graphs
from semantic chunks using itext2kg and LLM-powered extraction.

Designed for PhD thesis: Dynamic Knowledge Graph Construction Pipeline
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from types import SimpleNamespace

import networkx as nx
import pandas as pd

# Apply nest_asyncio for Jupyter/async compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


@dataclass
class Entity:
    """Represents an entity (node) in the knowledge graph."""
    name: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "properties": self.properties,
        }


@dataclass
class Relationship:
    """Represents a relationship (edge) in the knowledge graph."""
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "properties": self.properties,
        }


@dataclass
class KGConfig:
    """
    Configuration for Knowledge Graph construction.
    
    LLM Provider Options:
        - "ollama" (default): Local Ollama server
        - "openai": OpenAI API (requires OPENAI_API_KEY env var)
        - "anthropic": Anthropic API (requires ANTHROPIC_API_KEY env var)
        - "custom": Pass custom LLM/embeddings objects via constructor
    
    Example:
        # Default Ollama configuration
        config = KGConfig()
        
        # Custom Ollama model
        config = KGConfig(llm_model="llama3.1:70b", embeddings_model="mxbai-embed-large")
        
        # OpenAI configuration
        config = KGConfig(
            llm_provider="openai",
            llm_model="gpt-4o",
            embeddings_model="text-embedding-3-small"
        )
    """
    # LLM Provider settings
    llm_provider: str = "ollama"  # Options: "ollama", "openai", "anthropic", "custom"
    
    # LLM Model settings (defaults are for Ollama)
    llm_model: str = "qwen2.5:32b"  # For Ollama: qwen2.5:32b, llama3.1:8b, etc.
    embeddings_model: str = "nomic-embed-text"  # For Ollama: nomic-embed-text, mxbai-embed-large, etc.
    temperature: float = 0.0
    
    # Ollama-specific settings (used when llm_provider="ollama")
    ollama_base_url: str = "http://localhost:11434"
    
    # OpenAI-specific settings (used when llm_provider="openai")
    openai_api_key: Optional[str] = None  # If None, reads from OPENAI_API_KEY env var
    openai_base_url: Optional[str] = None  # For OpenAI-compatible APIs (e.g., Azure, local)
    
    # Anthropic-specific settings (used when llm_provider="anthropic")
    anthropic_api_key: Optional[str] = None  # If None, reads from ANTHROPIC_API_KEY env var
    
    # ATOM thresholds
    entity_threshold: float = 0.8 # Threshold for entity extraction
    relation_threshold: float = 0.7 # Threshold for relationship extraction
    max_workers: int = 4 # Max parallel workers for ATOM processing
    
    # Output settings
    output_dir: str = "kg_output"
    
    def to_dict(self) -> Dict[str, Any]:
        # Exclude API keys from serialization for security
        result = asdict(self)
        result.pop("openai_api_key", None)
        result.pop("anthropic_api_key", None)
        return result


@dataclass
class KGResult:
    """Result container for Knowledge Graph construction."""
    entities: List[Entity]
    relationships: List[Relationship]
    stats: Dict[str, Any]
    config: KGConfig
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    @property
    def num_entities(self) -> int:
        return len(self.entities)
    
    @property
    def num_relationships(self) -> int:
        return len(self.relationships)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "stats": self.stats,
            "config": self.config.to_dict(),
            "timestamp": self.timestamp,
        }
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph."""
        G = nx.DiGraph()
        for entity in self.entities:
            G.add_node(entity.name, label=entity.label, **entity.properties)
        for rel in self.relationships:
            G.add_edge(rel.source, rel.target, relation=rel.relation, **rel.properties)
        return G


class KGConstructor:
    """
    Knowledge Graph Constructor using itext2kg ATOM.
    
    This class provides methods to construct knowledge graphs from semantic chunks
    using LLM-powered entity and relationship extraction.
    
    Example:
        >>> constructor = KGConstructor()
        >>> result = await constructor.build_from_chunks(chunks)
        >>> result.to_dict()
    
    For FastAPI integration:
        >>> @app.post("/build-kg")
        >>> async def build_kg(chunks: List[dict]):
        >>>     constructor = KGConstructor()
        >>>     result = await constructor.build_from_chunks(chunks)
        >>>     return result.to_dict()
    """
    
    def __init__(
        self,
        config: Optional[KGConfig] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ):
        """
        Initialize the Knowledge Graph Constructor.
        
        Args:
            config: Configuration object. If None, uses default settings (Ollama).
            llm: Optional custom LLM object (for provider="custom").
            embeddings: Optional custom embeddings object (for provider="custom").
        
        Examples:
            # Default: Use Ollama with default models
            constructor = KGConstructor()
            
            # Custom Ollama models
            config = KGConfig(llm_model="llama3.1:70b")
            constructor = KGConstructor(config)
            
            # Use OpenAI
            config = KGConfig(
                llm_provider="openai",
                llm_model="gpt-4o",
                embeddings_model="text-embedding-3-small"
            )
            constructor = KGConstructor(config)
            
            # Custom LLM objects
            config = KGConfig(llm_provider="custom")
            constructor = KGConstructor(config, llm=my_llm, embeddings=my_embeddings)
        """
        self.config = config or KGConfig()
        self._llm = llm
        self._embeddings = embeddings
        self._atom = None
        self._initialized = False
    
    def _initialize_models(self) -> None:
        """
        Initialize LLM and embedding models lazily based on configured provider.
        
        Supported providers:
            - ollama: Local Ollama server (default)
            - openai: OpenAI API
            - anthropic: Anthropic API
            - custom: Use custom LLM/embeddings passed to constructor
        """
        if self._initialized:
            return
        
        # Import itext2kg (required for all providers)
        try:
            from itext2kg.atom import Atom
        except ImportError:
            raise ImportError(
                "itext2kg is required. Install with: pip install itext2kg"
            )
        
        provider = self.config.llm_provider.lower()
        
        if provider == "ollama":
            self._initialize_ollama()
        elif provider == "openai":
            self._initialize_openai()
        elif provider == "anthropic":
            self._initialize_anthropic()
        elif provider == "custom":
            if self._llm is None or self._embeddings is None:
                raise ValueError(
                    "For 'custom' provider, pass llm and embeddings to constructor: "
                    "KGConstructor(config, llm=your_llm, embeddings=your_embeddings)"
                )
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Supported: 'ollama', 'openai', 'anthropic', 'custom'"
            )
        
        self._atom = Atom(
            llm_model=self._llm,
            embeddings_model=self._embeddings,
        )
        
        self._initialized = True
    
    def _initialize_ollama(self) -> None:
        """Initialize Ollama LLM and embeddings."""
        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for Ollama provider. "
                "Install with: pip install langchain-ollama"
            )
        
        self._llm = ChatOllama(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            base_url=self.config.ollama_base_url,
        )
        
        self._embeddings = OllamaEmbeddings(
            model=self.config.embeddings_model,
            base_url=self.config.ollama_base_url,
        )
    
    def _initialize_openai(self) -> None:
        """Initialize OpenAI LLM and embeddings."""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. "
                "Install with: pip install langchain-openai"
            )
        
        import os
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set via config.openai_api_key or OPENAI_API_KEY env var."
            )
        
        llm_kwargs = {
            "model": self.config.llm_model,
            "temperature": self.config.temperature,
            "api_key": api_key,
        }
        embed_kwargs = {
            "model": self.config.embeddings_model,
            "api_key": api_key,
        }
        
        if self.config.openai_base_url:
            llm_kwargs["base_url"] = self.config.openai_base_url
            embed_kwargs["base_url"] = self.config.openai_base_url
        
        self._llm = ChatOpenAI(**llm_kwargs)
        self._embeddings = OpenAIEmbeddings(**embed_kwargs)
    
    def _initialize_anthropic(self) -> None:
        """Initialize Anthropic LLM (with OpenAI embeddings as fallback)."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic provider. "
                "Install with: pip install langchain-anthropic"
            )
        
        import os
        api_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set via config.anthropic_api_key or ANTHROPIC_API_KEY env var."
            )
        
        self._llm = ChatAnthropic(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            api_key=api_key,
        )
        
        # Anthropic doesn't have embeddings, use OpenAI or Ollama as fallback
        try:
            from langchain_openai import OpenAIEmbeddings
            openai_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if openai_key:
                self._embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=openai_key,
                )
            else:
                raise ValueError("No OpenAI key for embeddings")
        except (ImportError, ValueError):
            # Fallback to Ollama embeddings
            try:
                from langchain_ollama import OllamaEmbeddings
                self._embeddings = OllamaEmbeddings(
                    model=self.config.embeddings_model,
                    base_url=self.config.ollama_base_url,
                )
            except ImportError:
                raise ImportError(
                    "Anthropic provider requires embeddings. Install langchain-openai or langchain-ollama."
                )
    
    def _prepare_atomic_facts(
        self,
        chunks: List[Union[Dict[str, Any], SimpleNamespace]],
        timestamp: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Convert chunks to atomic facts format required by ATOM.
        
        Args:
            chunks: List of chunk objects or dictionaries.
            timestamp: Observation timestamp. Defaults to current date.
            
        Returns:
            Dictionary mapping timestamps to lists of atomic facts.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        
        atomic_facts_dict: Dict[str, List[str]] = {}
        
        for chunk in chunks:
            # Handle both dict and SimpleNamespace
            if isinstance(chunk, dict):
                section_title = chunk.get("section_title", chunk.get("header", ""))
                content = chunk.get("content", "")
            else:
                section_title = getattr(chunk, "section_title", getattr(chunk, "header", ""))
                content = getattr(chunk, "content", "")
            
            fact_text = f"[{section_title}] {content}" if section_title else content
            
            if timestamp not in atomic_facts_dict:
                atomic_facts_dict[timestamp] = []
            
            atomic_facts_dict[timestamp].append(fact_text)
        
        return atomic_facts_dict
    
    def _convert_kg_to_result(self, kg_obj: Any, chunks_count: int) -> KGResult:
        """
        Convert itext2kg KnowledgeGraph object to KGResult.
        
        Args:
            kg_obj: The KnowledgeGraph object from itext2kg.
            chunks_count: Number of input chunks.
            
        Returns:
            KGResult object with entities and relationships.
        """
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        
        # Extract entities
        for entity in kg_obj.entities:
            entity_name = getattr(entity, "name", "")
            entity_label = getattr(entity, "label", "Entity")
            
            props = {}
            if hasattr(entity, "properties") and entity.properties:
                if hasattr(entity.properties, "embeddings") and entity.properties.embeddings is not None:
                    props["has_embeddings"] = True
            
            entities.append(Entity(
                name=entity_name,
                label=entity_label,
                properties=props,
            ))
        
        # Extract relationships
        for rel in kg_obj.relationships:
            start_entity = getattr(rel, "startEntity", None)
            end_entity = getattr(rel, "endEntity", None)
            
            source = getattr(start_entity, "name", "") if start_entity else ""
            target = getattr(end_entity, "name", "") if end_entity else ""
            rel_name = getattr(rel, "name", "")
            
            props = {}
            if hasattr(rel, "properties") and rel.properties:
                rel_props = rel.properties
                if hasattr(rel_props, "t_start") and rel_props.t_start:
                    props["t_start"] = rel_props.t_start
                if hasattr(rel_props, "t_end") and rel_props.t_end:
                    props["t_end"] = rel_props.t_end
                if hasattr(rel_props, "t_obs") and rel_props.t_obs:
                    props["t_obs"] = rel_props.t_obs
                if hasattr(rel_props, "atomic_facts") and rel_props.atomic_facts:
                    props["atomic_facts"] = rel_props.atomic_facts
            
            relationships.append(Relationship(
                source=source,
                target=target,
                relation=rel_name,
                properties=props,
            ))
        
        # Compute statistics
        entity_labels = {}
        for e in entities:
            entity_labels[e.label] = entity_labels.get(e.label, 0) + 1
        
        relation_types = {}
        for r in relationships:
            relation_types[r.relation] = relation_types.get(r.relation, 0) + 1
        
        stats = {
            "num_input_chunks": chunks_count,
            "num_entities": len(entities),
            "num_relationships": len(relationships),
            "entity_labels": entity_labels,
            "relation_types": relation_types,
        }
        
        return KGResult(
            entities=entities,
            relationships=relationships,
            stats=stats,
            config=self.config,
        )
    
    async def build_from_chunks(
        self,
        chunks: List[Union[Dict[str, Any], SimpleNamespace]],
        timestamp: Optional[str] = None,
    ) -> KGResult:
        """
        Build a knowledge graph from semantic chunks.
        
        Args:
            chunks: List of chunk objects with 'section_title'/'header' and 'content'.
            timestamp: Observation timestamp for temporal tracking.
            
        Returns:
            KGResult containing entities, relationships, and statistics.
        """
        self._initialize_models()
        
        atomic_facts = self._prepare_atomic_facts(chunks, timestamp)
        
        kg = await self._atom.build_graph_from_different_obs_times(
            atomic_facts_with_obs_timestamps=atomic_facts,
            ent_threshold=self.config.entity_threshold,
            rel_threshold=self.config.relation_threshold,
            max_workers=self.config.max_workers,
        )
        
        return self._convert_kg_to_result(kg, len(chunks))
    
    def build_from_chunks_sync(
        self,
        chunks: List[Union[Dict[str, Any], SimpleNamespace]],
        timestamp: Optional[str] = None,
    ) -> KGResult:
        """
        Synchronous wrapper for build_from_chunks.
        
        Use this method when calling from synchronous code.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.build_from_chunks(chunks, timestamp))
    
    async def build_from_json_file(
        self,
        json_path: str,
        timestamp: Optional[str] = None,
    ) -> KGResult:
        """
        Build a knowledge graph from a JSON file containing chunks.
        
        Args:
            json_path: Path to JSON file with chunk data.
            timestamp: Observation timestamp for temporal tracking.
            
        Returns:
            KGResult containing entities, relationships, and statistics.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        
        # Convert to SimpleNamespace for consistency
        chunks = [SimpleNamespace(**chunk) for chunk in chunks_data]
        
        return await self.build_from_chunks(chunks, timestamp)
    
    def export_to_json(self, result: KGResult, output_path: Optional[str] = None) -> str:
        """
        Export KGResult to JSON file.
        
        Args:
            result: The KGResult to export.
            output_path: Output file path. Defaults to config output_dir.
            
        Returns:
            Path to the saved JSON file.
        """
        if output_path is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            output_path = os.path.join(self.config.output_dir, "knowledge_graph.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def export_to_csv(self, result: KGResult, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export KGResult to CSV files (nodes and edges).
        
        Args:
            result: The KGResult to export.
            output_dir: Output directory. Defaults to config output_dir.
            
        Returns:
            Dictionary with paths to nodes and edges CSV files.
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export nodes
        nodes_path = os.path.join(output_dir, "kg_nodes.csv")
        nodes_df = pd.DataFrame([e.to_dict() for e in result.entities])
        nodes_df.to_csv(nodes_path, index=False)
        
        # Export edges
        edges_path = os.path.join(output_dir, "kg_edges.csv")
        edges_df = pd.DataFrame([r.to_dict() for r in result.relationships])
        edges_df.to_csv(edges_path, index=False)
        
        return {"nodes": nodes_path, "edges": edges_path}
    
    def export_to_graphml(self, result: KGResult, output_path: Optional[str] = None) -> str:
        """
        Export KGResult to GraphML format for Gephi/Cytoscape.
        
        Args:
            result: The KGResult to export.
            output_path: Output file path. Defaults to config output_dir.
            
        Returns:
            Path to the saved GraphML file.
        """
        if output_path is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            output_path = os.path.join(self.config.output_dir, "knowledge_graph.graphml")
        
        G = result.to_networkx()
        nx.write_graphml(G, output_path)
        
        return output_path
    
    def export_all(self, result: KGResult, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Export KGResult to all supported formats.
        
        Args:
            result: The KGResult to export.
            output_dir: Output directory. Defaults to config output_dir.
            
        Returns:
            Dictionary with all exported file paths and report.
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = self.export_to_json(result, os.path.join(output_dir, "knowledge_graph.json"))
        csv_paths = self.export_to_csv(result, output_dir)
        graphml_path = self.export_to_graphml(result, os.path.join(output_dir, "knowledge_graph.graphml"))
        
        # Generate report
        report = self._generate_report(result)
        report_path = os.path.join(output_dir, "construction_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        return {
            "json": json_path,
            "nodes_csv": csv_paths["nodes"],
            "edges_csv": csv_paths["edges"],
            "graphml": graphml_path,
            "report": report_path,
        }
    
    def _generate_report(self, result: KGResult) -> str:
        """Generate a text report of the KG construction."""
        provider = self.config.llm_provider.upper()
        report = f"""
KNOWLEDGE GRAPH CONSTRUCTION REPORT
{'='*60}

Total Entities: {result.num_entities}
Total Relations: {result.num_relationships}

Processing Date: {result.timestamp}

LLM Configuration:
  - Provider: {provider}
  - LLM Model: {self.config.llm_model}
  - Embeddings Model: {self.config.embeddings_model}
  - Temperature: {self.config.temperature}

ATOM Parameters:
  - Entity threshold: {self.config.entity_threshold}
  - Relation threshold: {self.config.relation_threshold}

Entity Labels:
{self._format_dict(result.stats.get('entity_labels', {}))}

Top Relation Types:
{self._format_dict(dict(sorted(result.stats.get('relation_types', {}).items(), key=lambda x: x[1], reverse=True)[:10]))}

{'='*60}
"""
        return report
    
    @staticmethod
    def _format_dict(d: Dict[str, int], indent: int = 2) -> str:
        """Format a dictionary for display."""
        lines = []
        for key, value in d.items():
            lines.append(f"{' ' * indent}- {key}: {value}")
        return "\n".join(lines) if lines else f"{' ' * indent}(none)"
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Ollama models.
        
        Returns:
            Dictionary with test results.
        """
        self._initialize_models()
        
        results = {"llm": False, "embeddings": False, "errors": []}
        
        try:
            response = self._llm.invoke("Say 'ready' if you can respond.")
            results["llm"] = True
            results["llm_response"] = response.content
        except Exception as e:
            results["errors"].append(f"LLM error: {str(e)}")
        
        try:
            embedding = self._embeddings.embed_query("test")
            results["embeddings"] = True
            results["embedding_dim"] = len(embedding)
        except Exception as e:
            results["errors"].append(f"Embeddings error: {str(e)}")
        
        return results


# Convenience functions for direct use

async def build_kg_from_chunks(
    chunks: List[Union[Dict[str, Any], SimpleNamespace]],
    config: Optional[KGConfig] = None,
) -> KGResult:
    """
    Convenience function to build KG from chunks.
    
    Args:
        chunks: List of chunk objects.
        config: Optional configuration.
        
    Returns:
        KGResult with entities and relationships.
    """
    constructor = KGConstructor(config)
    return await constructor.build_from_chunks(chunks)


async def build_kg_from_json(
    json_path: str,
    config: Optional[KGConfig] = None,
) -> KGResult:
    """
    Convenience function to build KG from JSON file.
    
    Args:
        json_path: Path to JSON file with chunks.
        config: Optional configuration.
        
    Returns:
        KGResult with entities and relationships.
    """
    constructor = KGConstructor(config)
    return await constructor.build_from_json_file(json_path)
