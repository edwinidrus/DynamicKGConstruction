"""
DynamicKGConstruction - Main Pipeline Script

Complete pipeline: Documents → Markdown → Semantic Chunks

This project is a wrapper around Docling (https://github.com/docling-project/docling).
All credit for the underlying document conversion goes to the Docling Project team.
"""

from pathlib import Path
from docling_ingest import convert_documents_to_markdown
from chunking_semantic.chunking_semantic_by_header import SemanticChunker


# ============================================================
# CONFIGURATION - Update this path to your documents folder
# ============================================================
DOCUMENT_FOLDER = "path/to/your/documents"  # Update this!
OUTPUT_DIR = "build_docling"
CHUNKS_OUTPUT_DIR = "chunks_output"


def main():
    """
    Complete pipeline: Documents → Markdown → Semantic Chunks → Export
    """
    print("=" * 60)
    print("DynamicKGConstruction - Document Processing Pipeline")
    print("=" * 60)
    
    # Validate input path
    input_path = Path(DOCUMENT_FOLDER)
    if not input_path.exists():
        print(f"\n❌ Error: Document folder not found: {DOCUMENT_FOLDER}")
        print("   Please update DOCUMENT_FOLDER in main.py")
        return
    
    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    chunks_dir = Path(CHUNKS_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    chunks_dir.mkdir(exist_ok=True)
    
    # Step 1: Convert documents to Markdown
    print(f"\n📄 Step 1: Converting documents from {DOCUMENT_FOLDER}")
    print("-" * 40)
    
    md_files = convert_documents_to_markdown(
        input_path=str(input_path),
        output_dir=str(output_dir),
        enable_ocr=False,
        enable_table_structure=True,
    )
    
    if not md_files:
        print("⚠️ No documents were converted. Check input folder.")
        return
    
    print(f"✅ Converted {len(md_files)} documents to Markdown")
    
    # Step 2: Chunk each Markdown file
    print(f"\n🧩 Step 2: Creating semantic chunks")
    print("-" * 40)
    
    chunker = SemanticChunker(
        min_chunk_size=200,
        max_chunk_size=800,
        overlap_size=50,
        preserve_metadata=True,
    )
    
    total_chunks = 0
    for md_file in md_files:
        md_path = Path(md_file)
        print(f"  Processing: {md_path.name}")
        
        # Read markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create chunks
        chunks = chunker.parse_document(content)
        total_chunks += len(chunks)
        
        # Export chunks in multiple formats
        base_name = md_path.stem
        
        # JSON export
        json_output = chunker.export_chunks(chunks, "json")
        json_path = chunks_dir / f"{base_name}_chunks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_output)
        
        # Markdown export
        md_output = chunker.export_chunks(chunks, "markdown")
        md_chunks_path = chunks_dir / f"{base_name}_chunks.md"
        with open(md_chunks_path, "w", encoding="utf-8") as f:
            f.write(md_output)
        
        # Text export
        text_output = chunker.export_chunks(chunks, "text")
        text_path = chunks_dir / f"{base_name}_chunks.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_output)
        
        print(f"    → {len(chunks)} chunks created")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print(f"  Documents processed: {len(md_files)}")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Markdown output: {output_dir}/")
    print(f"  Chunks output: {chunks_dir}/")


def example_single_file(file_path: str):
    """
    Process a single document file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Convert single file
    output_dir = Path("build_docling")
    output_dir.mkdir(exist_ok=True)
    
    md_files = convert_documents_to_markdown(
        input_path=str(file_path.parent),
        output_dir=str(output_dir),
    )
    
    print(f"✅ Converted: {md_files}")


if __name__ == "__main__":
    main()
    
    # Uncomment to process a single file:
    # example_single_file("path/to/your/document.pdf")
