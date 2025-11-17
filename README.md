# DynamicKGConstruction
A toolkit for tools inventory management and knowledge graph construction, enabling context-aware retrieval and reasoning for resilient and sustainable supply chains.

# Docling Chunking - Semantic Document Processing

A powerful Python toolkit for converting PDF documents to structured text and intelligently chunking them into semantic sections for better document understanding, embedding generation, and information retrieval.

## 📋 Overview

This repository provides two main functionalities:

1. **PDF to Text Conversion** - Extract and convert PDF documents to structured text format using Docling
2. **Semantic Chunking** - Intelligently split documents into meaningful chunks based on headers, structure, and semantic coherence

Perfect for **RAG (Retrieval-Augmented Generation)** systems, document analysis, and research paper processing.

## 🚀 Features

### PDF Processing (`docling/`)
- 📄 Batch process multiple PDF files from a folder
- 🔄 Convert PDFs to structured Markdown format
- 📊 Preserve document structure (headings, lists, tables)
- 💾 Automatically save processed files with organized naming

### Semantic Chunking (`chunking_semantic/`)
- 🎯 **Header-based chunking** - Splits documents by section headers (## markers)
- 🧩 **Smart size optimization** - Merges small chunks, splits large ones
- 🔗 **Context preservation** - Maintains parent-child relationships between sections
- 📏 **Configurable parameters** - Customize chunk sizes and overlap
- 🔄 **Overlapping chunks** - Adds context overlap for better semantic continuity
- 📊 **Multiple export formats** - JSON, Markdown, and plain text output
- 🏷️ **Hierarchy detection** - Automatically identifies major sections and subsections

## 📦 Installation

### Prerequisites
```bash
# Python 3.8 or higher required
python --version
```

### Install Dependencies
```bash
# Install Docling for PDF processing
pip install docling docling-core

# Standard library dependencies (usually pre-installed)
# - pathlib
# - re
# - typing
# - dataclasses
```

## 🛠️ Usage

### Quick Start with main.py

The easiest way to get started is using the provided `main.py` script:

```bash
# 1. Edit main.py and update the PDF_FOLDER path
# 2. Run the complete pipeline
python main.py
```

This will:
1. ✅ Convert all PDFs in your folder to text
2. ✅ Create semantic chunks from each document
3. ✅ Export chunks in JSON, Markdown, and Text formats
4. ✅ Display detailed statistics

### 1. Convert PDFs to Text

```python
from docling.docling_multiple_document import process_pdfs_to_text

# Process all PDFs in a folder
folder_path = "C:/path/to/your/pdfs"
output_files = process_pdfs_to_text(folder_path, output_dir="build")

# Output: List of generated text file paths
print(f"Processed {len(output_files)} files")
```

### 2. Chunk Documents Semantically

```python
from chunking_semantic.chunking_semantic_by_header import SemanticChunker

# Initialize chunker with your preferences
chunker = SemanticChunker(
    min_chunk_size=200,      # Minimum words per chunk
    max_chunk_size=800,      # Maximum words per chunk
    overlap_size=50,         # Overlap between chunks
    preserve_metadata=True   # Keep document metadata
)

# Parse document into semantic chunks
chunks = chunker.parse_document('path/to/document.txt')

# Access chunk information
for chunk in chunks:
    print(f"Header: {chunk.header}")
    print(f"Level: {chunk.level}")
    print(f"Word count: {chunk.word_count}")
    print(f"Content preview: {chunk.content[:100]}...")
```

### 3. Export Chunks in Different Formats

```python
# Export as JSON (for APIs, databases)
json_output = chunker.export_chunks(chunks, 'json')
with open('chunks.json', 'w', encoding='utf-8') as f:
    f.write(json_output)

# Export as Markdown (for documentation)
md_output = chunker.export_chunks(chunks, 'markdown')
with open('chunks.md', 'w', encoding='utf-8') as f:
    f.write(md_output)

# Export as plain text (for analysis)
text_output = chunker.export_chunks(chunks, 'text')
with open('chunks.txt', 'w', encoding='utf-8') as f:
    f.write(text_output)
```

## 🏗️ Project Structure

```
docling_chungking/
├── chunking_semantic/
│   ├── __init__.py
│   └── chunking_semantic_by_header.py    # Semantic chunking engine
├── docling/
│   ├── __init__.py
│   └── docling_multiple_document.py      # PDF processing utilities
├── main.py                               # Complete pipeline example
└── README.md
```

## 📊 Chunking Algorithm

The semantic chunker uses an intelligent multi-step process:

1. **Header Extraction** - Identifies all section headers (## markers)
2. **Hierarchy Detection** - Determines header levels (major sections vs. subsections)
3. **Initial Chunking** - Creates chunks based on document structure
4. **Size Optimization**:
   - Merges chunks smaller than `min_chunk_size`
   - Splits chunks larger than `max_chunk_size`
5. **Context Overlap** - Adds overlapping text between chunks for continuity
6. **Metadata Preservation** - Maintains document information and relationships

### Chunk Properties

Each chunk contains:
- `header` - Section title
- `content` - Section text
- `level` - Hierarchy level (0=meta, 1=major, 2=subsection)
- `word_count` - Number of words
- `start_line` & `end_line` - Line range in original document
- `parent_header` - Parent section for context

## 🎓 Use Cases

### Research & Academia
- Process research papers and extract sections
- Create embeddings for academic literature
- Build knowledge bases from scientific documents

### RAG Systems
- Prepare documents for vector databases
- Generate context-aware embeddings
- Improve retrieval accuracy with semantic chunks

### Document Analysis
- Analyze document structure and organization
- Extract specific sections automatically
- Compare documents based on structure

## ⚙️ Configuration Options

### SemanticChunker Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_chunk_size` | 200 | Minimum words per chunk (smaller chunks get merged) |
| `max_chunk_size` | 1000 | Maximum words per chunk (larger chunks get split) |
| `overlap_size` | 50 | Number of overlapping words between chunks |
| `preserve_metadata` | True | Keep document metadata in output |

### Recommended Settings

**For Embeddings (RAG/Search)**
```python
chunker = SemanticChunker(
    min_chunk_size=150,
    max_chunk_size=500,
    overlap_size=50
)
```

**For Document Analysis**
```python
chunker = SemanticChunker(
    min_chunk_size=300,
    max_chunk_size=1500,
    overlap_size=100
)
```

**For Fast Processing**
```python
chunker = SemanticChunker(
    min_chunk_size=100,
    max_chunk_size=600,
    overlap_size=0
)
```

## 🎬 Running the Complete Pipeline

### Using main.py

1. **Edit the configuration** in `main.py`:
```python
PDF_FOLDER = "C:/path/to/your/pdfs"  # Update this path
```

2. **Run the script**:
```bash
python main.py
```

3. **Check the output**:
   - Text files: `build/` directory
   - Chunk files: `chunks_output/` directory (JSON, MD, TXT formats)

### Example Functions in main.py

The script includes three example functions:

1. **`main()`** - Complete pipeline (PDF → Text → Chunks → Export)
2. **`example_custom_chunking()`** - Shows different chunking configurations
3. **`example_single_file()`** - Process one file at a time

Uncomment the function calls at the bottom of `main.py` to try different examples.

## 📈 Example Output

### Statistics
```
Document parsed into 15 semantic chunks

Statistics:
  Total chunks: 15
  Total words: 8,542
  Average words per chunk: 569.5
  Smallest chunk: 203 words
  Largest chunk: 798 words
```

### Chunk Structure
```
1. Introduction...
   Level: 1 | Words: 450 | Lines: 10-55
   Parent: 

2. Literature Review...
   Level: 1 | Words: 623 | Lines: 56-110
   Parent: 

3. 2.1 | Theoretical Framework...
   Level: 2 | Words: 301 | Lines: 111-145
   Parent: Literature Review
```

## 🤝 Contributing

This is a PhD research project. Contributions, suggestions, and feedback are welcome!

## 📝 License

This project is part of PhD research. Please contact the author for usage permissions.

## 👨‍🔬 Author

**Edwin** - PhD Researcher

## 🔍 Keywords

`document processing` `semantic chunking` `PDF extraction` `NLP` `RAG` `embeddings` `document analysis` `text segmentation` `Docling` `research tools`

---

**Note**: This tool is specifically designed for academic papers and structured documents with clear header hierarchies. For best results, ensure your documents use `##` markers for section headers.

