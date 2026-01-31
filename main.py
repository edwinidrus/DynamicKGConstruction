"""
DynamicKGConstruction - FastAPI Application

Complete pipeline: Documents → Markdown → Semantic Chunks → Knowledge Graph

This project integrates:
- Docling (https://github.com/docling-project/docling) for document conversion
- iText2KG (https://github.com/AuvaLab/itext2kg) for knowledge graph construction

All credit for the underlying libraries goes to their respective teams.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or with Docker Compose:
    docker-compose up
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field


# =============================================================================
# Configuration from Environment Variables
# =============================================================================
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "build_docling")
CHUNKS_OUTPUT_DIR = os.getenv("CHUNKS_OUTPUT_DIR", "chunks_output")
KG_OUTPUT_DIR = os.getenv("KG_OUTPUT_DIR", "kg_output")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:32b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# =============================================================================
# Pydantic Models for API
# =============================================================================
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


class ChunkConfig(BaseModel):
    min_chunk_size: int = Field(default=200, ge=50, le=1000)
    max_chunk_size: int = Field(default=800, ge=100, le=5000)
    overlap_size: int = Field(default=50, ge=0, le=500)
    preserve_metadata: bool = True


class KGConfigRequest(BaseModel):
    llm_provider: str = Field(
        default="ollama", description="LLM provider: ollama, openai, anthropic"
    )
    llm_model: str = Field(default="qwen2.5:32b", description="Model name")
    embeddings_model: str = Field(
        default="nomic-embed-text", description="Embeddings model"
    )
    entity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    relation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ChunkRequest(BaseModel):
    content: str = Field(..., description="Markdown content to chunk")
    config: Optional[ChunkConfig] = None


class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    total_chunks: int
    total_words: int


class KGBuildRequest(BaseModel):
    chunks: List[Dict[str, Any]] = Field(
        ..., description="List of chunks to build KG from"
    )
    config: Optional[KGConfigRequest] = None


class KGResponse(BaseModel):
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    stats: Dict[str, Any]
    num_entities: int
    num_relationships: int


class DocumentProcessRequest(BaseModel):
    enable_ocr: bool = False
    enable_table_structure: bool = True
    chunk_config: Optional[ChunkConfig] = None
    build_kg: bool = False
    kg_config: Optional[KGConfigRequest] = None


class PipelineResponse(BaseModel):
    job_id: str
    status: str
    message: str
    documents_processed: int = 0
    chunks_created: int = 0
    kg_entities: int = 0
    kg_relationships: int = 0
    output_files: Dict[str, List[str]] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[PipelineResponse] = None


# =============================================================================
# In-memory job storage (use Redis in production)
# =============================================================================
jobs: Dict[str, JobStatus] = {}


# =============================================================================
# Lifespan Context Manager
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    # Startup: Create output directories
    for dir_path in [OUTPUT_DIR, CHUNKS_OUTPUT_DIR, KG_OUTPUT_DIR, UPLOAD_DIR]:
        Path(dir_path).mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("DynamicKGConstruction API Started")
    print("=" * 60)
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Chunks Dir: {CHUNKS_OUTPUT_DIR}")
    print(f"  KG Dir: {KG_OUTPUT_DIR}")
    print(f"  LLM Provider: {LLM_PROVIDER}")
    print(f"  LLM Model: {LLM_MODEL}")
    print("=" * 60)

    yield

    # Shutdown: Cleanup if needed
    print("DynamicKGConstruction API Shutting Down")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="DynamicKGConstruction API",
    description="""
    A comprehensive API for document processing, semantic chunking, and knowledge graph construction.
    
    ## Features
    - **Document Conversion**: Convert PDF, DOCX, PPTX, and more to Markdown (powered by Docling)
    - **Semantic Chunking**: Intelligently split documents into meaningful chunks
    - **Knowledge Graph Construction**: Extract entities and relationships (powered by iText2KG)
    
    ## Attribution
    This project integrates:
    - [Docling](https://github.com/docling-project/docling) for document conversion
    - [iText2KG](https://github.com/AuvaLab/itext2kg) for knowledge graph construction
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DynamicKGConstruction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint for Docker/Kubernetes."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/config", tags=["Info"])
async def get_config():
    """Get current configuration (excluding sensitive data)."""
    return {
        "output_dir": OUTPUT_DIR,
        "chunks_output_dir": CHUNKS_OUTPUT_DIR,
        "kg_output_dir": KG_OUTPUT_DIR,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "embeddings_model": EMBEDDINGS_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
    }


# =============================================================================
# Chunking Endpoints
# =============================================================================
@app.post("/chunk", response_model=ChunkResponse, tags=["Chunking"])
async def chunk_content(request: ChunkRequest):
    """
    Chunk markdown content into semantic segments.

    This endpoint takes markdown text and splits it into meaningful chunks
    based on headers and semantic coherence.
    """
    from chunking_semantic.chunking_semantic_by_header import SemanticChunker

    config = request.config or ChunkConfig()

    chunker = SemanticChunker(
        min_chunk_size=config.min_chunk_size,
        max_chunk_size=config.max_chunk_size,
        overlap_size=config.overlap_size,
        preserve_metadata=config.preserve_metadata,
    )

    chunks = chunker.parse_document(request.content)
    chunks_dicts = [chunk.to_dict() for chunk in chunks]
    total_words = sum(chunk.word_count for chunk in chunks)

    return ChunkResponse(
        chunks=chunks_dicts,
        total_chunks=len(chunks),
        total_words=total_words,
    )


@app.post("/chunk/file", response_model=ChunkResponse, tags=["Chunking"])
async def chunk_file(
    file: UploadFile = File(...),
    min_chunk_size: int = Query(200, ge=50, le=1000),
    max_chunk_size: int = Query(800, ge=100, le=5000),
    overlap_size: int = Query(50, ge=0, le=500),
):
    """
    Upload a markdown file and chunk it into semantic segments.
    """
    from chunking_semantic.chunking_semantic_by_header import SemanticChunker

    content = await file.read()
    text_content = content.decode("utf-8")

    chunker = SemanticChunker(
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        overlap_size=overlap_size,
        preserve_metadata=True,
    )

    chunks = chunker.parse_document(text_content)
    chunks_dicts = [chunk.to_dict() for chunk in chunks]
    total_words = sum(chunk.word_count for chunk in chunks)

    return ChunkResponse(
        chunks=chunks_dicts,
        total_chunks=len(chunks),
        total_words=total_words,
    )


# =============================================================================
# Document Conversion Endpoints
# =============================================================================
@app.post("/convert", tags=["Document Conversion"])
async def convert_document(
    file: UploadFile = File(...),
    enable_ocr: bool = Query(False),
    enable_table_structure: bool = Query(True),
):
    """
    Convert a document (PDF, DOCX, etc.) to Markdown.

    Supported formats: PDF, DOCX, PPTX, XLSX, HTML, CSV, Images
    """
    from docling_ingest import convert_documents_to_markdown

    # Save uploaded file
    upload_path = Path(UPLOAD_DIR) / f"{uuid.uuid4()}_{file.filename}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Convert document
        md_files = convert_documents_to_markdown(
            input_path=str(upload_path.parent),
            output_dir=OUTPUT_DIR,
            enable_ocr=enable_ocr,
            enable_table_structure=enable_table_structure,
        )

        if not md_files:
            raise HTTPException(status_code=400, detail="Failed to convert document")

        # Read converted markdown
        md_path = Path(md_files[0])
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        return {
            "filename": file.filename,
            "markdown_file": str(md_path),
            "markdown_content": markdown_content,
            "content_length": len(markdown_content),
        }

    finally:
        # Cleanup uploaded file
        if upload_path.exists():
            upload_path.unlink()


@app.post("/convert/batch", tags=["Document Conversion"])
async def convert_documents_batch(
    files: List[UploadFile] = File(...),
    enable_ocr: bool = Query(False),
    enable_table_structure: bool = Query(True),
):
    """
    Convert multiple documents to Markdown.
    """
    from docling_ingest import convert_documents_to_markdown

    # Create temp directory for uploads
    temp_dir = Path(UPLOAD_DIR) / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save all uploaded files
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

        # Convert all documents
        md_files = convert_documents_to_markdown(
            input_path=str(temp_dir),
            output_dir=OUTPUT_DIR,
            enable_ocr=enable_ocr,
            enable_table_structure=enable_table_structure,
        )

        results = []
        for md_file in md_files:
            md_path = Path(md_file)
            results.append(
                {
                    "markdown_file": str(md_path),
                    "filename": md_path.name,
                }
            )

        return {
            "documents_processed": len(results),
            "markdown_files": results,
        }

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# =============================================================================
# Knowledge Graph Endpoints
# =============================================================================
@app.post("/kg/build", response_model=KGResponse, tags=["Knowledge Graph"])
async def build_knowledge_graph(request: KGBuildRequest):
    """
    Build a knowledge graph from semantic chunks.

    Requires an LLM provider (Ollama, OpenAI, or Anthropic).
    """
    try:
        from itext2KG import KGConstructor, KGConfig
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="iText2KG not installed. Run: pip install itext2kg langchain-ollama",
        )

    config_req = request.config or KGConfigRequest()

    config = KGConfig(
        llm_provider=config_req.llm_provider,
        llm_model=config_req.llm_model,
        embeddings_model=config_req.embeddings_model,
        entity_threshold=config_req.entity_threshold,
        relation_threshold=config_req.relation_threshold,
        output_dir=KG_OUTPUT_DIR,
        ollama_base_url=OLLAMA_BASE_URL,
    )

    try:
        constructor = KGConstructor(config)
        result = constructor.build_from_chunks_sync(request.chunks)

        return KGResponse(
            entities=[e.to_dict() for e in result.entities],
            relationships=[r.to_dict() for r in result.relationships],
            stats=result.stats,
            num_entities=result.num_entities,
            num_relationships=result.num_relationships,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG construction failed: {str(e)}")


@app.get("/kg/test-connection", tags=["Knowledge Graph"])
async def test_kg_connection():
    """
    Test connection to the LLM provider.
    """
    try:
        from itext2KG import KGConstructor, KGConfig
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="iText2KG not installed. Run: pip install itext2kg langchain-ollama",
        )

    config = KGConfig(
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
        embeddings_model=EMBEDDINGS_MODEL,
        ollama_base_url=OLLAMA_BASE_URL,
    )

    try:
        constructor = KGConstructor(config)
        result = constructor.test_connection()
        return {
            "llm_connected": result.get("llm", False),
            "embeddings_connected": result.get("embeddings", False),
            "errors": result.get("errors", []),
            "config": {
                "provider": LLM_PROVIDER,
                "model": LLM_MODEL,
                "embeddings_model": EMBEDDINGS_MODEL,
            },
        }
    except Exception as e:
        return {
            "llm_connected": False,
            "embeddings_connected": False,
            "errors": [str(e)],
        }


# =============================================================================
# Full Pipeline Endpoint
# =============================================================================
@app.post("/pipeline", response_model=PipelineResponse, tags=["Pipeline"])
async def run_pipeline(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    enable_ocr: bool = Query(False),
    enable_table_structure: bool = Query(True),
    min_chunk_size: int = Query(200),
    max_chunk_size: int = Query(800),
    overlap_size: int = Query(50),
    build_kg: bool = Query(False),
    llm_provider: str = Query("ollama"),
    llm_model: str = Query("qwen2.5:32b"),
):
    """
    Run the complete pipeline: Documents → Markdown → Chunks → Knowledge Graph

    This endpoint processes documents through the entire pipeline:
    1. Convert documents to Markdown
    2. Chunk into semantic segments
    3. Optionally build a knowledge graph
    """
    job_id = str(uuid.uuid4())

    # Initialize job status
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Job queued",
    )

    # Save files for background processing
    temp_dir = Path(UPLOAD_DIR) / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    for file in files:
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        file_paths.append(str(file_path))

    # Run pipeline in background
    background_tasks.add_task(
        _run_pipeline_background,
        job_id=job_id,
        temp_dir=str(temp_dir),
        enable_ocr=enable_ocr,
        enable_table_structure=enable_table_structure,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        overlap_size=overlap_size,
        build_kg=build_kg,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    return PipelineResponse(
        job_id=job_id,
        status="processing",
        message="Pipeline started. Use /pipeline/status/{job_id} to check progress.",
    )


@app.get("/pipeline/status/{job_id}", response_model=JobStatus, tags=["Pipeline"])
async def get_pipeline_status(job_id: str):
    """Get the status of a pipeline job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


async def _run_pipeline_background(
    job_id: str,
    temp_dir: str,
    enable_ocr: bool,
    enable_table_structure: bool,
    min_chunk_size: int,
    max_chunk_size: int,
    overlap_size: int,
    build_kg: bool,
    llm_provider: str,
    llm_model: str,
):
    """Background task to run the full pipeline."""
    from docling_ingest import convert_documents_to_markdown
    from chunking_semantic.chunking_semantic_by_header import SemanticChunker

    try:
        jobs[job_id].status = "processing"
        jobs[job_id].message = "Converting documents..."
        jobs[job_id].progress = 0.1

        # Step 1: Convert documents
        md_files = convert_documents_to_markdown(
            input_path=temp_dir,
            output_dir=OUTPUT_DIR,
            enable_ocr=enable_ocr,
            enable_table_structure=enable_table_structure,
        )

        jobs[job_id].progress = 0.4
        jobs[job_id].message = "Chunking documents..."

        # Step 2: Chunk documents
        chunker = SemanticChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size,
            preserve_metadata=True,
        )

        all_chunks = []
        chunks_files = []

        for md_file in md_files:
            md_path = Path(md_file)
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = chunker.parse_document(content)
            chunks_dicts = [chunk.to_dict() for chunk in chunks]
            all_chunks.extend(chunks_dicts)

            # Save chunks
            base_name = md_path.stem
            json_path = Path(CHUNKS_OUTPUT_DIR) / f"{base_name}_chunks.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunks_dicts, f, indent=2, ensure_ascii=False)
            chunks_files.append(str(json_path))

        jobs[job_id].progress = 0.7

        # Step 3: Build KG (optional)
        kg_entities = 0
        kg_relationships = 0
        kg_files = []

        if build_kg and all_chunks:
            jobs[job_id].message = "Building knowledge graph..."

            try:
                from itext2KG import KGConstructor, KGConfig

                config = KGConfig(
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    embeddings_model=EMBEDDINGS_MODEL,
                    output_dir=KG_OUTPUT_DIR,
                    ollama_base_url=OLLAMA_BASE_URL,
                )

                constructor = KGConstructor(config)
                result = constructor.build_from_chunks_sync(all_chunks)

                kg_entities = result.num_entities
                kg_relationships = result.num_relationships

                # Export KG
                export_paths = constructor.export_all(result, KG_OUTPUT_DIR)
                kg_files = list(export_paths.values())

            except Exception as e:
                jobs[job_id].message = f"KG construction failed: {str(e)}"

        jobs[job_id].progress = 1.0
        jobs[job_id].status = "completed"
        jobs[job_id].message = "Pipeline completed successfully"
        jobs[job_id].result = PipelineResponse(
            job_id=job_id,
            status="completed",
            message="Pipeline completed successfully",
            documents_processed=len(md_files),
            chunks_created=len(all_chunks),
            kg_entities=kg_entities,
            kg_relationships=kg_relationships,
            output_files={
                "markdown": md_files,
                "chunks": chunks_files,
                "knowledge_graph": kg_files,
            },
        )

    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].message = f"Pipeline failed: {str(e)}"
        jobs[job_id].progress = 0.0

    finally:
        # Cleanup temp directory
        temp_path = Path(temp_dir)
        if temp_path.exists():
            shutil.rmtree(temp_path)


# =============================================================================
# File Download Endpoints
# =============================================================================
@app.get("/files/markdown", tags=["Files"])
async def list_markdown_files():
    """List all converted markdown files."""
    md_dir = Path(OUTPUT_DIR)
    if not md_dir.exists():
        return {"files": []}

    files = [str(f) for f in md_dir.glob("*.md")]
    return {"files": files, "count": len(files)}


@app.get("/files/chunks", tags=["Files"])
async def list_chunk_files():
    """List all chunk files."""
    chunks_dir = Path(CHUNKS_OUTPUT_DIR)
    if not chunks_dir.exists():
        return {"files": []}

    files = [str(f) for f in chunks_dir.glob("*.json")]
    return {"files": files, "count": len(files)}


@app.get("/files/kg", tags=["Files"])
async def list_kg_files():
    """List all knowledge graph files."""
    kg_dir = Path(KG_OUTPUT_DIR)
    if not kg_dir.exists():
        return {"files": []}

    files = [str(f) for f in kg_dir.iterdir() if f.is_file()]
    return {"files": files, "count": len(files)}


@app.get("/files/download/{file_type}/{filename}", tags=["Files"])
async def download_file(file_type: str, filename: str):
    """Download a specific file."""
    dir_map = {
        "markdown": OUTPUT_DIR,
        "chunks": CHUNKS_OUTPUT_DIR,
        "kg": KG_OUTPUT_DIR,
    }

    if file_type not in dir_map:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = Path(dir_map[file_type]) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


# =============================================================================
# CLI Entry Point (for backwards compatibility)
# =============================================================================
def run_cli():
    """Run the original CLI pipeline (for backwards compatibility)."""
    from docling_ingest import convert_documents_to_markdown
    from chunking_semantic.chunking_semantic_by_header import SemanticChunker

    print("=" * 60)
    print("DynamicKGConstruction - CLI Mode")
    print("=" * 60)
    print("Use 'uvicorn main:app --reload' for the API server")
    print("Or run with Docker: docker-compose up")
    print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        import uvicorn

        uvicorn.run(
            "main:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            reload=os.getenv("RELOAD", "false").lower() == "true", # Reload option if needed, with default value "false"
        )
