"""
Docling Ingest Module - Multi-format document conversion to Markdown.

This module is a wrapper around the Docling library (https://github.com/docling-project/docling).
All credit for the underlying document conversion goes to the Docling Project team.

Supports: PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images, Audio, and more.
"""

from .docling_multiple_document import (
    ALL_SUPPORTED_EXTENSIONS,
    DEFAULT_ALLOWED_FORMATS,
    SUPPORTED_FORMATS,
    convert_documents_to_markdown,
    create_document_converter,
    explore_documents_in_folder,
    get_all_supported_formats,
    get_extensions_for_formats,
)

__all__ = [
    "ALL_SUPPORTED_EXTENSIONS",
    "DEFAULT_ALLOWED_FORMATS",
    "SUPPORTED_FORMATS",
    "convert_documents_to_markdown",
    "create_document_converter",
    "explore_documents_in_folder",
    "get_all_supported_formats",
    "get_extensions_for_formats",
]
