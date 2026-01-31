# itext2KG Module - Knowledge Graph Construction Pipeline

## Overview

This module implements the **Knowledge Graph (KG) Construction Pipeline** for a PhD thesis project. The pipeline transforms unstructured documents into a structured knowledge graph using LLM-powered extraction.

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────┐     ┌───────────────────┐     ┌──────────────┐
│   Documents     │────▶│   Docling   │────▶│ Semantic Chunking │────▶│   itext2kg   │
│ (PDF/DOCX/TXT)  │     │  (Ingest)   │     │   (by Header)     │     │    (ATOM)    │
└─────────────────┘     └─────────────┘     └───────────────────┘     └──────────────┘
                                                                              │
                                                                              ▼
                                                                      ┌──────────────┐
                                                                      │ Knowledge    │
                                                                      │ Graph (Neo4J)│
                                                                      └──────────────┘
```

## Components

### 1. Document Ingestion (Docling)
- **Module**: `docling_ingest/docling_multiple_document.py`
- **Purpose**: Converts multiple document formats (PDF, DOCX, TXT) into processable text
- **Output**: Structured document content with metadata

### 2. Semantic Chunking
- **Module**: `chunking_semantic/chunking_semantic_by_header.py`
- **Purpose**: Splits documents into meaningful semantic chunks based on headers/sections
- **Output**: `all_chunks.json` - Array of chunk objects

### 3. Knowledge Graph Construction (itext2kg + ATOM)
- **Module**: `itext2KG/KG_Construction_pipeline_pure_llm(Working).ipynb`
- **Purpose**: Extracts entities and relationships from chunks to build KG
- **Framework**: Uses `itext2kg` library with ATOM (Atomic Text to Knowledge Graph)

---

## Data Structures

### Input: Chunk Object
```json
{
  "id": "chunk_001",
  "section_title": "Abstract",
  "content": "Text content of the chunk...",
  "metadata": {
    "doc_name": "document.pdf",
    "page": 1
  }
}
```

### Output: Knowledge Graph Object
```python
class KnowledgeGraph:
    entities: List[Entity]      # Nodes in the graph
    relationships: List[Relationship]  # Edges in the graph

class Entity:
    name: str       # Entity name/identifier
    label: str      # Entity type/category
    properties: EntityProperties  # Additional metadata

class Relationship:
    name: str           # Relationship type
    startEntity: Entity # Source node
    endEntity: Entity   # Target node
    properties: RelationshipProperties  # Temporal data, atomic facts
```

---

## Model Configuration

### LLM (Large Language Model)
```python
llm = ChatOllama(
    model="qwen2.5:32b",      # Model for entity/relation extraction
    temperature=0,            # Deterministic output
    base_url="http://localhost:11434"
)
```

### Embeddings Model
```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # For entity/relation similarity matching
    base_url="http://localhost:11434"
)
```

### ATOM Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `ent_threshold` | 0.8 | Entity similarity threshold for deduplication |
| `rel_threshold` | 0.7 | Relation similarity threshold for deduplication |
| `max_workers` | 4 | Parallel processing workers |

---

## Processing Steps

### Step 1: Load Chunks
```python
with open('all_chunks.json', 'r') as f:
    chunks_data = json.load(f)
all_chunks = [SimpleNamespace(**chunk) for chunk in chunks_data]
```

### Step 2: Prepare Atomic Facts
```python
atomic_facts_dict = {}
for chunk in all_chunks:
    fact_text = f"[{chunk.section_title}] {chunk.content}"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    atomic_facts_dict.setdefault(timestamp, []).append(fact_text)
```

### Step 3: Initialize ATOM
```python
atom = Atom(
    llm_model=llm,
    embeddings_model=embeddings
)
```

### Step 4: Build Knowledge Graph
```python
kg = await atom.build_graph_from_different_obs_times(
    atomic_facts_with_obs_timestamps=atomic_facts_dict,
    ent_threshold=0.8,
    rel_threshold=0.7,
    max_workers=4
)
```

---

## Output Formats

### 1. JSON Export
```python
# kg_output/knowledge_graph.json
{
  "nodes": [{"name": "...", "label": "..."}],
  "edges": [{"source": "...", "target": "...", "relation": "..."}]
}
```

### 2. CSV Export
- `kg_output/kg_nodes.csv` - Entity data
- `kg_output/kg_edges.csv` - Relationship data

### 3. GraphML Export
- `kg_output/knowledge_graph.graphml` - For Gephi/Cytoscape visualization

### 4. Visualization
- `kg_visualization.html` - Interactive PyVis visualization
- `kg_static.png` - Static NetworkX visualization

---

## Dependencies

### Python Packages
```
langchain-ollama>=0.1.0    # LLM integration (NOT langchain-community)
langchain-core>=0.3.0      # Core langchain components
itext2kg                   # Knowledge graph construction
networkx                   # Graph operations
matplotlib                 # Static visualization
pyvis                      # Interactive visualization
pandas                     # Data manipulation
nest_asyncio               # Async support in Jupyter
```

### External Services
- **Ollama Server**: `http://localhost:11434`
  - Model: `qwen2.5:32b` (LLM)
  - Model: `nomic-embed-text` (Embeddings)

---

## Key Implementation Notes

### 1. LangChain Package Selection
```python
# CORRECT: Use langchain_ollama for structured output support
from langchain_ollama import ChatOllama, OllamaEmbeddings

# WRONG: langchain_community does NOT support structured output
# from langchain_community.llms import Ollama  # DON'T USE
```

### 2. Async Event Loop Handling
```python
import nest_asyncio
nest_asyncio.apply()  # Required for Jupyter/Colab environments
```

### 3. KnowledgeGraph Access Pattern
```python
# KnowledgeGraph is a Pydantic model, NOT a dictionary
entities = kg.entities          # Direct property access
relationships = kg.relationships

# Entity/Relationship attribute access
entity_name = getattr(entity, 'name', 'N/A')
rel_name = getattr(rel, 'name', 'N/A')
```

---

## Agentic AI Integration Points

### Potential Agent Tasks
1. **Document Ingestion Agent**: Automatically detect and process new documents
2. **Chunk Validation Agent**: Verify chunk quality and semantic coherence
3. **Entity Resolution Agent**: Handle entity disambiguation and merging
4. **KG Quality Agent**: Validate extracted relationships for accuracy
5. **Query Agent**: Answer questions using the constructed KG

### API Endpoints for Agents
```python
# Suggested function signatures for agent integration
async def ingest_document(file_path: str) -> List[Chunk]
async def chunk_document(doc_content: str) -> List[Chunk]
async def extract_kg(chunks: List[Chunk]) -> KnowledgeGraph
async def query_kg(question: str, kg: KnowledgeGraph) -> str
async def update_kg(new_facts: List[str], kg: KnowledgeGraph) -> KnowledgeGraph
```

---

## File Structure

```
itext2KG/
├── __init__.py                                        # Module exports
├── kg_constructor.py                                  # Main callable class
├── MODULE_DESCRIPTION.md                              # This file
├── KG_Construction_pipeline_pure_llm(Working).ipynb   # Development notebook
└── kg_output/                                         # Generated outputs
    ├── knowledge_graph.json
    ├── kg_nodes.csv
    ├── kg_edges.csv
    ├── knowledge_graph.graphml
    └── construction_report.txt
```

---

## Callable Class API

### KGConstructor

The main class for building knowledge graphs from chunks.

```python
from itext2KG import KGConstructor, KGConfig

# Initialize with default config
constructor = KGConstructor()

# Or with custom config
config = KGConfig(
    llm_model="qwen2.5:32b",
    embeddings_model="nomic-embed-text",
    entity_threshold=0.8,
    relation_threshold=0.7,
)
constructor = KGConstructor(config)
```

### Async Usage

```python
# Build from list of chunks
result = await constructor.build_from_chunks(chunks)

# Build from JSON file
result = await constructor.build_from_json_file("all_chunks.json")
```

### Sync Usage

```python
# Synchronous wrapper for non-async contexts
result = constructor.build_from_chunks_sync(chunks)
```

### Export Methods

```python
# Export to various formats
constructor.export_to_json(result)
constructor.export_to_csv(result)
constructor.export_to_graphml(result)
constructor.export_all(result)  # All formats at once
```

### FastAPI Integration

```python
from fastapi import FastAPI
from itext2KG import KGConstructor, KGConfig
from typing import List, Dict, Any

app = FastAPI()

@app.post("/build-kg")
async def build_knowledge_graph(chunks: List[Dict[str, Any]]):
    constructor = KGConstructor()
    result = await constructor.build_from_chunks(chunks)
    return result.to_dict()

@app.post("/build-kg/export")
async def build_and_export_kg(chunks: List[Dict[str, Any]]):
    constructor = KGConstructor()
    result = await constructor.build_from_chunks(chunks)
    paths = constructor.export_all(result)
    return {"result": result.to_dict(), "exported_files": paths}
```

### Data Classes

```python
from itext2KG import Entity, Relationship, KGConfig, KGResult

# Entity represents a node
entity = Entity(name="Company A", label="Organization")

# Relationship represents an edge
relationship = Relationship(
    source="Company A",
    target="Product X",
    relation="PRODUCES"
)

# KGResult contains the full graph
result.entities      # List[Entity]
result.relationships # List[Relationship]
result.stats         # Dict with statistics
result.to_dict()     # Convert to dictionary
result.to_networkx() # Convert to NetworkX graph
```

---

## Future Enhancements

1. **Incremental KG Updates**: Add new documents without full reconstruction
2. **Temporal Reasoning**: Leverage ATOM's temporal properties (t_start, t_end, t_obs)
3. **Neo4j Integration**: Direct export to Neo4j graph database
4. **Multi-modal Support**: Handle images and tables from documents
5. **Agent Orchestration**: MCP-based agent coordination for pipeline automation

---

## References

- [itext2kg Documentation](https://github.com/AuvaLab/itext2kg)
- [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama)
- [Docling](https://github.com/DS4SD/docling)
- [ATOM: Temporal Knowledge Graph Construction](https://arxiv.org/abs/2312.15639)
