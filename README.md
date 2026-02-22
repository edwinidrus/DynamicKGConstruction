# DynamicKGConstruction

**Automated Knowledge Graph Construction from Documents using Local LLMs**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-PhD%20Research-orange.svg)](LICENSE)

DynamicKGConstruction turns unstructured documents (PDF, DOCX, PPTX, XLSX, HTML, Markdown, images, audio, and more) into structured, queryable knowledge graphs — fully offline, using [Ollama](https://ollama.com) for local LLM inference.

The core framework, **SKGB (Semantic Knowledge Graph Builder)**, orchestrates a three-stage pipeline:

```
Documents  ──►  Docling  ──►  Semantic Chunks  ──►  itext2kg (ATOM)  ──►  Knowledge Graph
               (parsing)      (header-aware)        (entity/relation       (JSON, CSV,
                               splitting)            extraction)           GraphML, Neo4j)
```

---

## Why DynamicKG?

Traditional knowledge graph construction requires:
- Expensive API calls (OpenAI, Anthropic, etc.)
- Cloud dependencies and data privacy concerns
- Complex setup and configuration

DynamicKG solves this by running **entirely locally** on your machine:

- **Zero API costs** — use any Ollama model
- **Complete privacy** — your documents never leave your machine
- **17+ formats supported** — PDF, DOCX, PPTX, XLSX, HTML, Markdown, images, audio
- **Incremental construction** — ATOM method builds and deduplicates entities/relations progressively
- **Multiple export formats** — JSON, CSV, GraphML, interactive HTML, Neo4j Cypher

---

## Features

- **17+ document formats** — PDF, DOCX, PPTX, XLSX, HTML, Markdown, images, audio, and more via Docling
- **Fully local & private** — runs entirely on your machine with Ollama (no API keys required)
- **Semantic chunking** — header-aware splitting with configurable size and overlap, stable SHA-256 chunk IDs
- **Incremental KG construction** — uses the ATOM method from itext2kg for entity/relation extraction and deduplication
- **Multiple export formats** — JSON, CSV, GraphML, interactive HTML visualization (PyVis), and Neo4j Cypher scripts
- **CLI + Python API + Colab** — use it however fits your workflow

---

## How It Works

The pipeline consists of four stages:

### 1. Document Parsing (Docling)
Documents are converted to Markdown using [Docling](https://github.com/docling-project/docling), preserving structure and layout from 17+ formats.

### 2. Semantic Chunking
Markdown is split into semantic chunks based on headers. Each chunk:
- Has a hierarchical section path (e.g., `Introduction/Background/Related Work`)
- Is sized between `min_chunk_words` and `max_chunk_words` (default: 200-800)
- Uses SHA-256 hashing for stable, reproducible IDs

### 3. KG Construction (itext2kg ATOM)
Each chunk is processed to extract entities and relations using the ATOM method:
1. **Atomic Fact Extraction** — LLM generates (subject, predicate, object) quintuples
2. **Entity Deduplication** — Similar entities are merged using embedding similarity
3. **Relation Deduplication** — Similar relations are merged
4. **Incremental Merge** — New facts are added to the existing graph

### 4. Export
The final knowledge graph is exported in multiple formats for downstream use.

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

1. Installs Ollama inside Colab (CPU or T4/V100 GPU)
2. Pulls `qwen2.5:7b` (default) or your chosen model
3. Runs the full pipeline on an uploaded PDF
4. Displays an interactive knowledge graph visualization
5. Exports results as a downloadable ZIP

### Colab GPU Setup

For faster processing, enable GPU acceleration in Colab:

1. Runtime → Change runtime type → Hardware accelerator → GPU
2. Select T4 or V100 for best performance
3. With GPU, you can use larger models like `qwen2.5:14b` or `qwen2.5:32b`

### Colab Memory Considerations

- **Free tier (CPU only)**: Use `qwen2.5:7b` for reasonable performance
- **Colab Pro (GPU)**: `qwen2.5:14b` works well on T4
- **Large documents**: Process in batches or reduce chunk size

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

## Troubleshooting

### Ollama Connection Issues

**Error: `ConnectionRefusedError` or `Cannot connect to Ollama`**
- Ensure Ollama is running: `ollama serve`
- Check the URL in config matches your setup (default: `http://localhost:11434`)
- Verify firewall settings allow local connections

**Error: `model not found`**
- Pull the required models: `ollama pull qwen2.5:32b` and `ollama pull nomic-embed-text`
- List installed models: `ollama list`

### Memory Issues

**Out of memory errors**
- Use a smaller LLM model (e.g., `qwen2.5:7b` instead of `qwen2.5:32b`)
- Reduce `max_workers` in config to limit parallel processing
- Process documents one at a time instead of batching

### langchain Compatibility

**Import errors or version conflicts**
- This project pins `langchain < 0.4.0` for itext2kg compatibility
- Do NOT upgrade langchain to 1.x
- If you have conflicts, create a fresh virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

### itext2kg Errors

**`IndexError: list index out of range` in itext2kg**
- The adapter includes patches for known itext2kg v1.0.0 bugs
- This typically occurs with empty input or failed extraction
- Check your input documents are valid and have extractable text

---

## Frequently Asked Questions

**Q: Can I use models other than qwen2.5?**
A: Yes! Any Ollama model that supports JSON mode output should work. You may need to adjust `temperature` and prompt formatting.

**Q: How do I process multiple documents?**
A: Pass a directory path to `--input` or use `recursive=True` in the Python API. The pipeline will process all supported files.

**Q: Can I use this without Ollama?**
A: Not directly — SKGB requires Ollama for LLM inference. However, you could modify the adapters to use a different LLM provider.

**Q: How are entities deduplicated?**
A: The ATOM method uses embedding-based similarity. Entities with similarity above `ent_threshold` (default 0.8) are merged. Relations use `rel_threshold` (default 0.7).

**Q: What's the difference between the JSON and GraphML exports?**
A: JSON is best for programmatic access and Neo4j import. GraphML works with tools like Gephi, Cytoscape, and NetworkX for advanced visualization.

---

## Troubleshooting

### Ollama Connection Issues

**Error: `ConnectionError: [Errno 111] Connection refused`**

Make sure Ollama is running:
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434
```

**Error: `Model not found`**

Pull the required models:
```bash
ollama pull qwen2.5:32b
ollama pull nomic-embed-text
```

### Memory Issues

**Out of memory during extraction?**

- Use a smaller model: `qwen2.5:7b` or `qwen2.5:14b`
- Reduce `max_workers` in config to limit parallel processing
- Reduce `max_chunk_words` to process smaller chunks

### langchain Version Conflict

**Error: `ImportError` or version conflicts?**

Ensure langchain is pinned:
```bash
pip install 'langchain<0.4.0'
```

### itext2kg Crashes

The adapter includes patches for known itext2kg 1.0.0 bugs:
- `parallel_atomic_merge` IndexError fix
- `build_atomic_kg_from_quintuples` error handling
- `build_graph` exception handling with `return_exceptions=True`

If you encounter new issues, check the [issue tracker](https://github.com/edwinidrus/DynamicKGConstruction/issues).

---

## FAQ

**Q: Can I use models other than qwen2.5?**
A: Yes, any Ollama model that supports chat/completions and embeddings should work. Adjust `llm_model` and `embeddings_model` in config.

**Q: How do I process multiple documents?**
A: Pass a directory path to `--input` or set `recursive=True` in the Python API.

**Q: Can I use this with OpenAI instead of Ollama?**
A: Currently no — the project is designed for local Ollama. PRs welcome for additional provider support.

**Q: How long does processing take?**
A: Depends on document size, model, and hardware. A 10-page PDF with qwen2.5:32b on CPU typically takes 5-15 minutes.

---

## Troubleshooting

### Common Issues

**Ollama not running**
```
Error: Connection refused to http://localhost:11434
```
Make sure Ollama is running: `ollama serve`

**Model not found**
```
Error: model 'qwen2.5:32b' not found
```
Pull the required models: `ollama pull qwen2.5:32b && ollama pull nomic-embed-text`

**langchain version conflict**
```
ImportError: cannot import name 'xxx' from 'langchain'
```
This project requires `langchain < 0.4.0`. Reinstall: `pip install 'langchain<0.4.0'`

**Out of memory**
- Use a smaller LLM model (e.g., `qwen2.5:7b` instead of `qwen2.5:32b`)
- Reduce `max_workers` to limit parallel processing
- Process documents in batches

### Performance Tips

- Use `qwen2.5:14b` or `qwen2.5:7b` for faster processing on consumer hardware
- Increase `max_workers` (up to 8) on multi-core systems for faster chunk processing
- Use GPU acceleration with CUDA-capable GPUs for best performance

---

## Acknowledgments

This project builds on the excellent work of two open-source teams:

- **[Docling](https://github.com/docling-project/docling)** by the Docling Project — for robust, multi-format document parsing that makes ingestion of PDFs, Office documents, and more seamless and reliable. Thank you for making document understanding accessible.

- **[itext2kg](https://github.com/AuvaLab/itext2kg)** by AuvaLab — for the ATOM (Augmented Text-to-KG Ontology Mapping) method that powers incremental knowledge graph construction with entity and relation deduplication. Thank you for advancing the state of automated KG building.

---

## License

This project is part of ongoing PhD research. See [LICENSE](LICENSE) for details.
