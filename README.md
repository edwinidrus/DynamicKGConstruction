# DynamicKGConstruction

A comprehensive toolkit for document processing, semantic chunking, and knowledge graph construction — enabling context-aware retrieval and reasoning for resilient and sustainable supply chains.

---

## 🙏 Attribution & Ethical Acknowledgment

This project stands on the shoulders of giants. We are deeply grateful to the open-source communities whose exceptional work makes this project possible.

### Docling - Document Conversion Foundation

> **This repository integrates [Docling](https://github.com/docling-project/docling)**, the powerful document conversion library developed by the **Docling Project team**.
>
> We extend our sincere gratitude and ethical acknowledgment to the creators and maintainers of Docling for their excellent work in building a robust, open-source document processing framework that supports 17+ document formats. Their commitment to open science and accessible document processing tools has been invaluable to the research community.
>
> 📦 **Docling Repository:** [https://github.com/docling-project/docling](https://github.com/docling-project/docling)
>
> 📄 **Docling License:** Please refer to Docling's license for the underlying document conversion library.

### iText2KG - Knowledge Graph Construction Engine

> **This repository integrates [iText2KG](https://github.com/AuvaLab/itext2kg)**, an innovative knowledge graph construction framework developed by the **AuvaLab team**.
>
> We express our heartfelt appreciation to the iText2KG creators for developing a state-of-the-art approach to extracting structured knowledge from unstructured text using LLM-powered entity and relationship extraction. Their ATOM (Atomic Text Operations for Meaning) methodology represents a significant advancement in automated knowledge graph construction.
>
> 📦 **iText2KG Repository:** [https://github.com/AuvaLab/itext2kg](https://github.com/AuvaLab/itext2kg)
>
> 📄 **iText2KG License:** Please refer to iText2KG's license for the underlying knowledge graph construction library.

### Our Commitment to Ethical Open Source

We believe in:
- **Transparent Attribution**: Clearly acknowledging all libraries and tools we build upon
- **Respecting Licenses**: Adhering to the terms of all integrated open-source projects
- **Giving Back**: Contributing improvements upstream when possible
- **Academic Integrity**: Properly citing these works in any academic publications

---

## 📋 Overview

DynamicKGConstruction provides a complete pipeline from raw documents to structured knowledge graphs:

1. **Multi-Format Document Conversion** (`docling_ingest/`) - Convert documents (PDF, DOCX, PPTX, XLSX, HTML, images, CSV, and more) to structured Markdown using Docling
2. **Semantic Chunking** (`chunking_semantic/`) - Intelligently split documents into meaningful chunks based on headers, structure, and semantic coherence
3. **Knowledge Graph Construction** (`itext2KG/`) - Extract entities and relationships from chunks using LLM-powered analysis to build structured knowledge graphs

Perfect for **RAG (Retrieval-Augmented Generation)** systems, document analysis, research paper processing, and knowledge base construction.

## 🚀 Features

### Multi-Format Document Processing (`docling_ingest/`)
*Powered by [Docling](https://github.com/docling-project/docling)*
- 📄 Batch process multiple documents from a folder
- 🔄 Convert **17+ formats** to structured Markdown (PDF, DOCX, PPTX, XLSX, HTML, images, CSV, etc.)
- 📊 Preserve document structure (headings, lists, tables)
- 💾 Automatically save processed files with organized naming
- 🔧 Configurable format options (OCR, table extraction, etc.)

### Semantic Chunking (`chunking_semantic/`)
- 🎯 **Header-based chunking** - Splits documents by section headers (## markers)
- 🧩 **Smart size optimization** - Merges small chunks, splits large ones
- 🔗 **Context preservation** - Maintains parent-child relationships between sections
- 📏 **Configurable parameters** - Customize chunk sizes and overlap
- 🔄 **Overlapping chunks** - Adds context overlap for better semantic continuity
- 📊 **Multiple export formats** - JSON, Markdown, and plain text output

### Knowledge Graph Construction (`itext2KG/`)
*Powered by [iText2KG](https://github.com/AuvaLab/itext2kg)*
- 🧠 **LLM-powered extraction** - Uses advanced language models for entity/relationship extraction
- 🔌 **Multi-provider support** - Ollama (local), OpenAI, Anthropic, or custom LLMs
- 📊 **ATOM methodology** - Atomic Text Operations for Meaning extraction
- 🕸️ **Graph exports** - JSON, CSV, GraphML formats for visualization tools
- ⚡ **Async/sync interfaces** - Flexible integration with any application architecture
- 📈 **Statistics & reports** - Detailed construction reports and metrics

## 📦 Installation

### Prerequisites
```bash
# Python 3.10 or higher required
python --version
```

### Install Dependencies
```bash
# Core: Docling for multi-format document processing
pip install docling docling-core

# Core: iText2KG for knowledge graph construction
pip install itext2kg

# LLM Providers (choose based on your needs)
pip install langchain-ollama      # For local Ollama models (recommended)
pip install langchain-openai      # For OpenAI API
pip install langchain-anthropic   # For Anthropic/Claude API

# Additional dependencies
pip install networkx pandas nest_asyncio

# Optional: For OCR support
pip install docling[ocr]

# Optional: For audio transcription
pip install docling[audio]
```

### Create a Conda Environment (Recommended)
```bash
conda create -n dynamickg python=3.10 pip ipykernel
conda activate dynamickg
pip install docling docling-core itext2kg
pip install langchain-ollama networkx pandas nest_asyncio
```

## 🛠️ Usage

### 🐳 Quick Start with Docker (Recommended)

The easiest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/edwinidrus/DynamicKGConstruction.git
cd DynamicKGConstruction

# Start all services (API + Ollama LLM)
docker-compose up -d

# Pull LLM models (first time only)
docker-compose --profile setup up ollama-pull

# Access the API
open http://localhost:8000/docs
```

### 🚀 Quick Start with uvicorn

```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access the API documentation
open http://localhost:8000/docs
```

### 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for Docker/K8s |
| `/chunk` | POST | Chunk markdown content |
| `/chunk/file` | POST | Upload and chunk a file |
| `/convert` | POST | Convert document to Markdown |
| `/convert/batch` | POST | Convert multiple documents |
| `/kg/build` | POST | Build knowledge graph from chunks |
| `/kg/test-connection` | GET | Test LLM connection |
| `/pipeline` | POST | Run full pipeline (convert → chunk → KG) |
| `/pipeline/status/{job_id}` | GET | Check pipeline job status |
| `/files/markdown` | GET | List converted files |
| `/files/chunks` | GET | List chunk files |
| `/files/kg` | GET | List KG files |

### Example API Calls

**Chunk Content:**
```bash
curl -X POST "http://localhost:8000/chunk" \
  -H "Content-Type: application/json" \
  -d '{"content": "## Introduction\nThis is a test document.\n\n## Methods\nWe used Python."}'
```

**Convert a Document:**
```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@document.pdf"
```

**Build Knowledge Graph:**
```bash
curl -X POST "http://localhost:8000/kg/build" \
  -H "Content-Type: application/json" \
  -d '{"chunks": [{"header": "Introduction", "content": "AI is transforming..."}]}'
```

### 1. Convert Documents to Markdown

```python
from docling_ingest import convert_documents_to_markdown
from docling.datamodel.base_models import InputFormat

# Process all documents in a folder (supports 17+ formats)
output_files = convert_documents_to_markdown(
    input_path="path/to/your/documents",
    output_dir="build_docling",
    allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML],
    enable_ocr=False,
    enable_table_structure=True,
)
print(f"Processed {len(output_files)} files")
```

### 2. Chunk Documents Semantically

```python
from chunking_semantic.chunking_semantic_by_header import SemanticChunker

chunker = SemanticChunker(
    min_chunk_size=200,
    max_chunk_size=800,
    overlap_size=50,
    preserve_metadata=True
)

chunks = chunker.parse_document('path/to/document.md')

for chunk in chunks:
    print(f"Header: {chunk.header}")
    print(f"Word count: {chunk.word_count}")
```

### 3. Build Knowledge Graph from Chunks

```python
from itext2KG import KGConstructor, KGConfig

# Configure the KG constructor (default: Ollama local models)
config = KGConfig(
    llm_provider="ollama",           # or "openai", "anthropic"
    llm_model="qwen2.5:32b",
    embeddings_model="nomic-embed-text",
    entity_threshold=0.8,
    relation_threshold=0.7,
)

# Build the knowledge graph
constructor = KGConstructor(config)
result = await constructor.build_from_chunks(chunks)

# Access results
print(f"Entities: {result.num_entities}")
print(f"Relationships: {result.num_relationships}")

# Export to various formats
constructor.export_to_json(result, "kg_output/knowledge_graph.json")
constructor.export_to_csv(result, "kg_output/")
constructor.export_to_graphml(result, "kg_output/knowledge_graph.graphml")
```

### 4. Using OpenAI or Anthropic

```python
from itext2KG import KGConstructor, KGConfig

# OpenAI configuration
config = KGConfig(
    llm_provider="openai",
    llm_model="gpt-4o",
    embeddings_model="text-embedding-3-small",
    # API key from environment: OPENAI_API_KEY
)

# Anthropic configuration
config = KGConfig(
    llm_provider="anthropic",
    llm_model="claude-3-sonnet-20240229",
    # API key from environment: ANTHROPIC_API_KEY
)

constructor = KGConstructor(config)
```

## 🏗️ Project Structure

```
DynamicKGConstruction/
├── main.py                            # FastAPI application entry point
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── docling_ingest/                    # Document conversion module
│   ├── __init__.py
│   └── docling_multiple_document.py   # Multi-format conversion (wraps Docling)
├── chunking_semantic/                 # Semantic chunking module
│   ├── __init__.py
│   └── chunking_semantic_by_header.py # Header-based chunking engine
├── itext2KG/                          # Knowledge graph construction module
│   ├── __init__.py
│   └── kg_constructor.py              # KG builder (wraps iText2KG)
├── examples/                          # Example data for testing
│   └── robotics - paper.txt           # Sample document
├── AGENTS.md                          # Guidelines for AI coding agents
└── README.md

# Output directories (created at runtime, git-ignored):
# ├── build_docling/                   # Converted Markdown files
# ├── chunks_output/                   # Semantic chunks (JSON, MD, TXT)
# └── kg_output/                       # Knowledge graphs (JSON, CSV, GraphML)
```

## 📊 Supported Input Formats

| Format | Extensions |
|--------|------------|
| PDF | `.pdf` |
| Word | `.docx`, `.dotx`, `.docm`, `.dotm` |
| PowerPoint | `.pptx`, `.potx`, `.ppsx`, `.pptm` |
| Excel | `.xlsx`, `.xlsm` |
| HTML | `.html`, `.htm`, `.xhtml` |
| Markdown | `.md` |
| CSV | `.csv` |
| Images | `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.webp` |
| Audio | `.wav`, `.mp3`, `.m4a` (requires ASR models) |
| AsciiDoc | `.adoc`, `.asciidoc`, `.asc` |

## ⚙️ Configuration Options

### SemanticChunker Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_chunk_size` | 200 | Minimum words per chunk |
| `max_chunk_size` | 1000 | Maximum words per chunk |
| `overlap_size` | 50 | Overlapping words between chunks |
| `preserve_metadata` | True | Keep document metadata |

### KGConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_provider` | "ollama" | LLM provider: "ollama", "openai", "anthropic" |
| `llm_model` | "qwen2.5:32b" | Model name for the provider |
| `embeddings_model` | "nomic-embed-text" | Embeddings model name |
| `entity_threshold` | 0.8 | Entity similarity threshold |
| `relation_threshold` | 0.7 | Relationship similarity threshold |
| `output_dir` | "kg_output" | Output directory for exports |

## 🎓 Use Cases

### Research & Academia
- Process research papers and extract structured knowledge
- Build literature review knowledge bases
- Create embeddings for academic document retrieval

### Supply Chain & Industry
- Extract entities and relationships from technical documents
- Build knowledge graphs for inventory management
- Enable context-aware reasoning for decision support

### RAG Systems
- Prepare documents for vector databases
- Generate high-quality semantic chunks
- Improve retrieval accuracy with structured knowledge

## 🤝 Contributing

This is a PhD research project. Contributions, suggestions, and feedback are welcome!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: This project integrates the following open-source libraries. Please also refer to their respective licenses:
- [Docling](https://github.com/docling-project/docling) - Document conversion
- [iText2KG](https://github.com/AuvaLab/itext2kg) - Knowledge graph construction

## 👨‍🔬 Author

**Edwin** - PhD Researcher

## 🔍 Keywords

`knowledge graph` `document processing` `semantic chunking` `PDF extraction` `NLP` `RAG` `embeddings` `itext2kg` `docling` `LLM` `entity extraction` `relationship extraction`

---

## 📚 Citations

If you use this project in academic work, please consider citing the underlying libraries:

### Docling
Please refer to the [Docling repository](https://github.com/docling-project/docling) for citation information.

### iText2KG
Please refer to the [iText2KG repository](https://github.com/AuvaLab/itext2kg) for citation information.

---

**Note**: This toolkit is designed for structured documents with clear header hierarchies. For best results with semantic chunking, ensure your documents use `##` markers for section headers.
