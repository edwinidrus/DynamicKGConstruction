# SKGB (Semantic Knowledge Graph Builder)

This is the new framework module that runs your end-to-end pipeline:

PDF -> Docling ingestion -> header-based semantic chunking -> itext2kg (ATOM) incremental KG

## Run

From repo root:

```bash
python3 skgb.py run --input "DynamicKGConstruction/examples/robotic for resilient supply chain.pdf" --out "DynamicKGConstruction/examples/skgb_run"
```

Or as a module:

```bash
python3 -m DynamicKGConstruction.skgb run --input "..." --out "..."
```

## Outputs

`<out>/kg_output/` contains (matching your notebook):
- `knowledge_graph.json`
- `kg_nodes.csv`
- `kg_edges.csv`
- `knowledge_graph.graphml`
- `construction_report.txt`

Neo4j import helper:
- `neo4j_load.cypher` (uses `LOAD CSV` and a single relationship type `:REL`)

## Dependencies

This environment currently does not have the required Python packages installed.
Install them using your preferred method (venv/conda/docker). The dependency list
is in `DynamicKGConstruction/requirements.txt`.

Important: do not upgrade `langchain` to 1.x when using `itext2kg`.
