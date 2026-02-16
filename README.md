# DynamicKGConstruction

**Automated Knowledge Graph Construction from Documents using Local LLMs**

DynamicKGConstruction turns unstructured documents (PDF, DOCX, PPTX, HTML, images, and more) into structured, queryable knowledge graphs — fully offline, using [Ollama](https://ollama.com) for local LLM inference.

The core framework, **SKGB (Semantic Knowledge Graph Builder)**, orchestrates a three-stage pipeline:

```
Documents  ──►  Docling  ──►  Semantic Chunks  ──►  itext2kg (ATOM)  ──►  Knowledge Graph
               (parsing)      (header-aware)        (entity/relation       (JSON, CSV,
                               splitting)            extraction)           GraphML, Neo4j)
```

---

## Features

- **17+ document formats** — PDF, DOCX, PPTX, XLSX, HTML, Markdown, images, audio, and more via Docling
- **Fully local & private** — runs entirely on your machine with Ollama (no API keys required)
- **Semantic chunking** — header-aware splitting with configurable size and overlap, stable SHA-256 chunk IDs
- **Incremental KG construction** — uses the ATOM method from itext2kg for entity/relation extraction and deduplication
- **Multiple export formats** — JSON, CSV, GraphML, interactive HTML visualization (PyVis), and Neo4j Cypher scripts
- **CLI + Python API + Colab notebook** — use it however fits your workflow

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running ([install guide](https://ollama.com/download))

```bash
# Pull the required models
ollama pull qwen2.5:32b        # LLM for extraction (~20 GB)
ollama pull nomic-embed-text   # Embeddings (~274 MB)
```

> **Tip:** For smaller setups, use `qwen2.5` (7B, ~4.7 GB) or `qwen2.5:14b` instead.

### Installation

```bash
git clone https://github.com/edwinidrus/DynamicKGConstruction.git
cd DynamicKGConstruction
pip install -r requirements.txt
```

### Run the Pipeline

**CLI:**

```bash
python -m DynamicKGConstruction.skgb run \
  --input "path/to/document.pdf" \
  --out "output/"
```

**Python:**

```python
from pathlib import Path
from DynamicKGConstruction.skgb import SKGBConfig, run_pipeline

cfg = SKGBConfig.from_out_dir("output/")
result = run_pipeline(Path("document.pdf"), cfg)

print(f"Nodes: {result.kg_output_dir / 'kg_nodes.csv'}")
print(f"Edges: {result.kg_output_dir / 'kg_edges.csv'}")
print(f"Visualization: {result.kg_output_dir / 'kg_visualization.html'}")
```

---

## Using with Ollama (Step by Step)

This is the recommended workflow. A complete, runnable example is provided in the [Colab notebook](DynamicKGConstruction/notebooks/skgb_colab_demo.ipynb).

### 1. Start Ollama and Pull Models

```bash
# Start the Ollama server (if not already running)
ollama serve

# Pull models
ollama pull qwen2.5:32b
ollama pull nomic-embed-text
```

### 2. Configure the Pipeline

```python
from pathlib import Path
from DynamicKGConstruction.skgb import SKGBConfig, run_pipeline

cfg = SKGBConfig.from_out_dir(
    "skgb_output",
    llm_model="qwen2.5:32b",              # Ollama LLM model
    embeddings_model="nomic-embed-text",   # Ollama embeddings model
    ollama_base_url="http://localhost:11434",
    temperature=0.0,                       # Deterministic output
    ent_threshold=0.8,                     # Entity deduplication threshold
    rel_threshold=0.7,                     # Relation deduplication threshold
    max_workers=2,                         # Parallel workers
    min_chunk_words=200,                   # Minimum words per chunk
    max_chunk_words=800,                   # Maximum words per chunk
    overlap_words=0,                       # Word overlap between chunks
)
```

### 3. Run the Pipeline

```python
pdf_path = Path("input_docs/my_paper.pdf")
result = run_pipeline(pdf_path, cfg)

print(f"Markdown:      {result.build_docling_dir}")
print(f"Chunks:        {result.chunks_json_path}")
print(f"KG outputs:    {result.kg_output_dir}")
print(f"Neo4j Cypher:  {result.neo4j_cypher_path}")
```

### 4. Explore the Results

```python
import json
import pandas as pd

# Load the knowledge graph
kg = json.loads((result.kg_output_dir / "knowledge_graph.json").read_text())
print(f"Entities: {len(kg['nodes'])}, Relations: {len(kg['edges'])}")

# Work with DataFrames
df_nodes = pd.read_csv(result.kg_output_dir / "kg_nodes.csv")
df_edges = pd.read_csv(result.kg_output_dir / "kg_edges.csv")

# Open the interactive visualization in your browser
# result.kg_output_dir / "kg_visualization.html"
```

### 5. Load into Neo4j (Optional)

```bash
python -m DynamicKGConstruction.skgb export-neo4j \
  --kg-output "skgb_output/kg_output"
```

This generates a `neo4j_load.cypher` script using `LOAD CSV`. Copy the CSV files to your Neo4j import directory and run the Cypher script.

---

## Google Colab

A ready-to-run notebook is provided at [`notebooks/skgb_colab_demo.ipynb`](DynamicKGConstruction/notebooks/skgb_colab_demo.ipynb). It:

1. Installs Ollama inside Colab (CPU or T4 GPU)
2. Pulls `qwen2.5:32b` and `nomic-embed-text`
3. Runs the full pipeline on an uploaded PDF
4. Displays an interactive knowledge graph visualization
5. Exports results as a downloadable ZIP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edwinidrus/DynamicKGConstruction/blob/main/DynamicKGConstruction/notebooks/skgb_colab_demo.ipynb)

---

## Output Files

The pipeline writes all results to `<out_dir>/kg_output/`:

| File | Description |
|------|-------------|
| `knowledge_graph.json` | Full KG as JSON (nodes + edges with metadata) |
| `kg_nodes.csv` | Nodes table for downstream analysis or Neo4j |
| `kg_edges.csv` | Edges table with temporal fields and atomic facts |
| `knowledge_graph.graphml` | GraphML for Gephi, Cytoscape, or NetworkX |
| `kg_visualization.html` | Interactive graph visualization (PyVis) |
| `construction_report.txt` | Summary: entity/relation counts, parameters used |
| `neo4j_load.cypher` | Ready-to-run Cypher LOAD CSV script |

---

## Configuration Reference

All parameters can be set via `SKGBConfig.from_out_dir()` or environment variables:

| Parameter | Default | Env Variable | Description |
|-----------|---------|--------------|-------------|
| `llm_model` | `qwen2.5:32b` | `LLM_MODEL` | Ollama LLM for extraction |
| `embeddings_model` | `nomic-embed-text` | `EMBEDDINGS_MODEL` | Ollama embeddings model |
| `ollama_base_url` | `http://localhost:11434` | `OLLAMA_BASE_URL` | Ollama server URL |
| `temperature` | `0.0` | — | LLM temperature (0 = deterministic) |
| `ent_threshold` | `0.8` | — | Entity similarity threshold for deduplication |
| `rel_threshold` | `0.7` | — | Relation similarity threshold for deduplication |
| `max_workers` | `4` | — | Parallel processing workers |
| `min_chunk_words` | `200` | — | Minimum words per semantic chunk |
| `max_chunk_words` | `800` | — | Maximum words per semantic chunk |
| `overlap_words` | `50` | — | Word overlap between adjacent chunks |

---

## Project Structure

```
DynamicKGConstruction/
├── requirements.txt
├── notebooks/
│   └── skgb_colab_demo.ipynb      # Interactive Colab demo
└── skgb/                           # Core framework
    ├── cli.py                      # CLI entry point (run, export-neo4j)
    ├── config.py                   # SKGBConfig dataclass
    ├── pipeline.py                 # Pipeline orchestration
    ├── adapters/
    │   ├── docling_adapter.py      # Document → Markdown
    │   ├── chunking_adapter.py     # Markdown → semantic chunks
    │   └── itext2kg_adapter.py     # Chunks → knowledge graph
    └── export/
        ├── file_export.py          # JSON / CSV / GraphML / HTML
        └── neo4j_export.py         # Neo4j Cypher script generation
```

---

## Known Constraints

- **langchain < 0.4.0** — pinned for itext2kg 1.0.0 compatibility
- **numpy < 2.0** — required by itext2kg's scipy dependency in some environments
- Neo4j export uses a generic `:REL` relationship type with the actual predicate in `r.relation` (avoids invalid Cypher type names from dynamic predicates)

---

## Acknowledgments

This project builds on the excellent work of two open-source teams:

- **[Docling](https://github.com/docling-project/docling)** by the Docling Project — for robust, multi-format document parsing that makes ingestion of PDFs, Office documents, and more seamless and reliable. Thank you for making document understanding accessible.

- **[itext2kg](https://github.com/AuvaLab/itext2kg)** by AuvaLab — for the ATOM (Augmented Text-to-KG Ontology Mapping) method that powers incremental knowledge graph construction with entity and relation deduplication. Thank you for advancing the state of automated KG building.

---

## License

This project is part of ongoing PhD research. See [LICENSE](LICENSE) for details.
