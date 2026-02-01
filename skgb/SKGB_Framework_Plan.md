# SKGB - Semantic Knowledge Graph Builder (Framework Plan)

## Goal
Build a reusable framework that converts PDF documents into an accurate, incremental Knowledge Graph using:
- Default LLM orchestration: Ollama + `qwen2.5:32b` (configurable)
- Pipeline: Docling ingestion -> header-based semantic chunking -> itext2kg KG Build
- Outputs: same as the current notebook (`knowledge_graph.json`, `kg_nodes.csv`, `kg_edges.csv`, `knowledge_graph.graphml`, `construction_report.txt`, `kg_visualization.html`)
- Neo4j: CSV + Cypher `LOAD CSV` export only (no direct Bolt writes, no RDF)

## Existing Building Blocks (Source Paths)
- Docling ingestion:
  - `DynamicKGConstruction/docling_ingest/docling_multiple_document.py`
- Semantic chunking:
  - `DynamicKGConstruction/chunking_semantic/chunking_semantic_by_header.py`
- itext2kg pipeline reference (notebook to refactor into modules):
  - `DynamicKGConstruction/pdf2KG/KG_Construction_pipeline_pure_llm(Working).ipynb`

## End-to-End Pipeline (SKGB Run)

### 1) Ingest (Docling)
Input:
- One PDF or a folder of PDFs

Process:
- Use Docling to convert each document to Markdown

Output (per run):
- `build_docling/*.md`
- A small manifest file mapping `doc_id -> source_path -> md_path`

Notes:
- `doc_id` should be stable (e.g., SHA256 of file bytes or (path+mtime+size) hash)

### 2) Semantic Chunking (Header-Based)
Input:
- Docling-produced Markdown

Process:
- Chunk by header hierarchy, preserving parent/child section context
- Ensure chunk IDs are stable across re-runs:
  - `chunk_id = {doc_id}__{section_path}__{chunk_index}`
- Chunk schema should match the notebook expectation (minimum):
  - `id`, `section_title`, `content`, `metadata` (include `doc_name`, optional page/provenance)

Important improvement for Docling Markdown:
- Support headers as `^#{1,6}\s+...` (Docling may not use only `##`)
- Use `#` count as structural level; optionally keep the current semantic "major section" heuristics as an extra field

Output:
- `chunks_output/all_chunks.json` (list of chunk records)

### 3) Atomic Facts Preparation (Notebook-Compatible)
Input:
- `chunks_output/all_chunks.json`

Process:
- Create `atomic_facts_dict`:
  - Key: `t_obs` (observation timestamp, e.g. run date or doc date)
  - Value: list of strings: `"[{section_title}] {content}"`

Output:
- In-memory `atomic_facts_dict` (optionally persisted as `atomic_facts.json` for debugging)

### 4) KG Construction (itext2kg ATOM, Incremental)
Input:
- `atomic_facts_dict`
- LLM + embeddings configuration

Defaults:
- LLM provider: Ollama
- LLM model: `qwen2.5:32b`
- Embeddings model: `nomic-embed-text`
- temperature: `0`

Process:
- Initialize `Atom` from itext2kg with thresholds (from notebook defaults):
  - entity threshold: `0.8`
  - relation threshold: `0.7`
- Build KG incrementally:
  - Process chunks in a deterministic order
  - Merge into a global KG for the run

Critical dependency note:
- The notebook shows `langchain 1.x` conflicts with `itext2kg` (expects `langchain<0.4`).
- SKGB must pin compatible versions and must NOT "upgrade langchain to latest".

Output:
- In-memory `kg` object (itext2kg KnowledgeGraph)

### 5) Export Outputs (Match Notebook)
Output folder:
- `kg_output/`

Files (must match notebook naming):
- `kg_output/knowledge_graph.json`
- `kg_output/kg_nodes.csv`
- `kg_output/kg_edges.csv`
- `kg_output/knowledge_graph.graphml`
- `kg_output/kg_visualization.html`
- `kg_output/construction_report.txt`

Optional (nice-to-have, also in notebook):
- `kg_output/kg_static.png` (matplotlib)

Export logic:
- Reuse the notebook's `kg_to_dict()` logic:
  - nodes: `name`, `label`, plus minimal flags
  - edges: `source`, `target`, `relation`, optional temporal fields and `atomic_facts`
- Write GraphML from a NetworkX DiGraph with `label` on nodes and `relation` on edges
- Write a construction report with totals + model names + thresholds + processing timestamp

## Neo4j Support (CSV + Cypher LOAD CSV ONLY)

### What SKGB Produces
- `kg_output/kg_nodes.csv`
- `kg_output/kg_edges.csv`
- `kg_output/neo4j_load.cypher` (generated script that imports the CSVs)

### Neo4j Import Strategy
Assumptions:
- CSVs are accessible to Neo4j in its import directory (or via Neo4j Desktop "import" folder)

Cypher pattern:
- Create uniqueness constraint on node name:
  - `CREATE CONSTRAINT ... FOR (n:Entity) REQUIRE n.name IS UNIQUE;`
- Load nodes:
  - `LOAD CSV WITH HEADERS FROM 'file:///kg_nodes.csv' AS row ...`
- Load edges:
  - `LOAD CSV WITH HEADERS FROM 'file:///kg_edges.csv' AS row ...`
  - `MATCH (s:Entity {name: row.source}), (t:Entity {name: row.target})`
  - `MERGE (s)-[r:RELTYPE]->(t)` (either generic `:RELATED` with `r.type=row.relation`, or sanitize relation into a valid type)

Recommendation:
- Use a generic relationship type (e.g., `:REL`) and store the real predicate in a property:
  - `MERGE (s)-[r:REL {relation: row.relation}]->(t)`
This avoids Cypher failing due to invalid dynamic relationship type names.

## SKGB Code Layout (New Framework Module)
Create:
- `DynamicKGConstruction/skgb/`
  - `cli.py` (commands: `run`, `export-neo4j`)
  - `config.py` (pydantic settings; env overrides)
  - `pipeline.py` (orchestrator)
  - `adapters/docling_adapter.py`
  - `adapters/chunking_adapter.py`
  - `adapters/itext2kg_adapter.py`
  - `export/file_export.py` (JSON/CSV/GraphML/report)
  - `export/neo4j_export.py` (writes `neo4j_load.cypher`)
  - `schemas.py` (chunk schema + kg export schema)

## CLI (Expected Usage)
- End-to-end run:
  - `skgb run --input <pdf_or_dir> --out <run_dir> --llm-model qwen2.5:32b --embeddings-model nomic-embed-text`
- Generate Neo4j Cypher (and ensure CSV paths are correct for your Neo4j setup):
  - `skgb export-neo4j --run <run_dir> --neo4j-import-name kg_output`

## Milestones
1) MVP:
- Wrap existing Docling + chunker + itext2kg logic into SKGB modules
- Produce notebook-equivalent outputs in `kg_output/`
- Generate `neo4j_load.cypher`

2) Hardening:
- Stable IDs + caching (re-run only changed docs/chunks)
- Better header parsing for Docling Markdown
- Deterministic processing and better provenance

3) Quality:
- Evaluation harness (spot-check triples; regression tests)
- Configurable prompts/constraints for higher extraction accuracy
