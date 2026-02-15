# SKGB (Semantic Knowledge Graph Builder)

SKGB runs an end-to-end pipeline:

**PDF(s) → Docling ingestion → header-based semantic chunking → itext2kg (ATOM) incremental KG**

---

## 1. CLI Usage (Bash / Terminal)

From repo root:

```bash
python3 skgb.py run \
  --input "DynamicKGConstruction/examples/robotic for resilient supply chain.pdf" \
  --out "DynamicKGConstruction/examples/skgb_run"
```

Or as a Python module:

```bash
python3 -m DynamicKGConstruction.skgb run --input "..." --out "..."
```

Generate Neo4j import script for an existing run:

```bash
python3 skgb.py export-neo4j --kg-output "DynamicKGConstruction/examples/skgb_run/kg_output"
```

---

## 2. Python Usage (Scripts / Jupyter Notebooks)

```python
from pathlib import Path
from DynamicKGConstruction.skgb import SKGBConfig, run_pipeline

# Configure the run
cfg = SKGBConfig.from_out_dir(
    "./my_skgb_run",
    llm_model="qwen2.5:32b",           # or any Ollama model
    embeddings_model="nomic-embed-text",
    ollama_base_url="http://localhost:11434",
)

# Run the pipeline
result = run_pipeline(
    input_path=Path("path/to/document.pdf"),
    cfg=cfg,
    recursive=True,  # search subfolders if input is a directory
)

print(f"KG outputs: {result.kg_output_dir}")
print(f"Chunks:     {result.chunks_json_path}")
```

---

## 3. Google Colab Usage

```python
# Cell 1: Clone repo and install dependencies
!git clone https://github.com/edwinidrus/DynamicKGConstruction.git
%cd DynamicKGConstruction
!pip install -r requirements.txt

# Cell 2: Add repo to Python path
import sys
sys.path.insert(0, "/content/DynamicKGConstruction")

# Cell 3: Run SKGB
from pathlib import Path
from DynamicKGConstruction.skgb import SKGBConfig, run_pipeline

cfg = SKGBConfig.from_out_dir(
    "/content/skgb_run",
    ollama_base_url="http://localhost:11434",  # or your Ollama server
)

result = run_pipeline(
    input_path=Path("/content/DynamicKGConstruction/examples/robotic for resilient supply chain.pdf"),
    cfg=cfg,
)

print(result.kg_output_dir)
```

> **Note:** In Colab, you need Ollama running somewhere accessible (local tunnel, cloud VM, etc.)

---

## Outputs

`<out>/kg_output/` contains:

| File | Description |
|------|-------------|
| `knowledge_graph.json` | Full KG as JSON (nodes + edges) |
| `kg_nodes.csv` | Nodes CSV for Neo4j import |
| `kg_edges.csv` | Edges CSV for Neo4j import |
| `knowledge_graph.graphml` | GraphML for Gephi/Cytoscape |
| `kg_visualization.html` | Interactive graph visualization (PyVis) |
| `construction_report.txt` | Summary report with stats |
| `neo4j_load.cypher` | Ready-to-run Neo4j LOAD CSV script |

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_model` | `qwen2.5:32b` | Ollama LLM model |
| `embeddings_model` | `nomic-embed-text` | Ollama embeddings model |
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `temperature` | `0.0` | LLM temperature |
| `ent_threshold` | `0.8` | Entity matching threshold |
| `rel_threshold` | `0.7` | Relation matching threshold |
| `min_chunk_words` | `200` | Min words per chunk |
| `max_chunk_words` | `800` | Max words per chunk |
| `overlap_words` | `50` | Overlap between chunks |

---

## Dependencies

Install from `DynamicKGConstruction/requirements.txt`:

```bash
pip install -r DynamicKGConstruction/requirements.txt
```

**Important:** Do not upgrade `langchain` to 1.x — `itext2kg` requires `langchain<0.4`.
