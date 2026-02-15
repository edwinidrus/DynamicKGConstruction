"""Docling adapter – converts documents to Markdown.

This module inlines the logic that previously lived in
``docling_ingest.docling_multiple_document`` so that ``skgb/`` is fully
self-contained with no dependency on legacy directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

# ---------------------------------------------------------------------------
# Supported formats & helpers
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS: Dict[InputFormat, List[str]] = {
    InputFormat.PDF: [".pdf"],
    InputFormat.DOCX: [".docx", ".dotx", ".docm", ".dotm"],
    InputFormat.PPTX: [".pptx", ".potx", ".ppsx", ".pptm"],
    InputFormat.XLSX: [".xlsx", ".xlsm"],
    InputFormat.HTML: [".html", ".htm", ".xhtml"],
    InputFormat.MD: [".md"],
    InputFormat.ASCIIDOC: [".adoc", ".asciidoc", ".asc"],
    InputFormat.CSV: [".csv"],
    InputFormat.IMAGE: [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"],
    InputFormat.AUDIO: [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"],
    InputFormat.VTT: [".vtt"],
    InputFormat.XML_USPTO: [".xml"],
    InputFormat.XML_JATS: [".xml", ".nxml"],
    InputFormat.JSON_DOCLING: [".json"],
    InputFormat.METS_GBS: [".tar.gz"],
}

DEFAULT_ALLOWED_FORMATS: List[InputFormat] = [
    InputFormat.PDF,
    InputFormat.DOCX,
    InputFormat.PPTX,
    InputFormat.XLSX,
    InputFormat.HTML,
    InputFormat.MD,
    InputFormat.ASCIIDOC,
    InputFormat.CSV,
    InputFormat.IMAGE,
]

ALL_SUPPORTED_EXTENSIONS: List[str] = [
    ext for extensions in SUPPORTED_FORMATS.values() for ext in extensions
]


def _get_extensions_for_formats(formats: Iterable[InputFormat]) -> List[str]:
    extensions: List[str] = []
    for fmt in formats:
        extensions.extend(SUPPORTED_FORMATS.get(fmt, []))
    return extensions


def _explore_documents_in_folder(
    folder_path: str,
    extensions: Optional[Iterable[str]] = None,
    recursive: bool = True,
) -> List[str]:
    if extensions is None:
        extensions = ALL_SUPPORTED_EXTENSIONS

    extensions = [ext.lower() for ext in extensions]
    paths: List[str] = []

    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in extensions):
                    paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in extensions):
                paths.append(os.path.join(folder_path, file))

    return paths


def _create_document_converter(
    allowed_formats: Optional[Iterable[InputFormat]] = None,
    enable_ocr: bool = False,
    enable_table_structure: bool = True,
) -> DocumentConverter:
    if allowed_formats is None:
        allowed_formats = DEFAULT_ALLOWED_FORMATS

    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = enable_ocr
    pdf_pipeline_options.do_table_structure = enable_table_structure

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
    }

    optional_format_options = {
        "WordFormatOption": InputFormat.DOCX,
        "PowerpointFormatOption": InputFormat.PPTX,
        "ExcelFormatOption": InputFormat.XLSX,
        "HTMLFormatOption": InputFormat.HTML,
        "MarkdownFormatOption": InputFormat.MD,
        "AsciiDocFormatOption": InputFormat.ASCIIDOC,
        "CsvFormatOption": InputFormat.CSV,
        "ImageFormatOption": InputFormat.IMAGE,
        "AudioFormatOption": InputFormat.AUDIO,
    }

    try:
        import docling.document_converter as dc  # type: ignore

        for class_name, input_format in optional_format_options.items():
            option_cls = getattr(dc, class_name, None)
            if option_cls is not None:
                format_options[input_format] = option_cls()
    except Exception:
        pass

    return DocumentConverter(
        allowed_formats=list(allowed_formats),
        format_options=format_options,
    )


def _convert_documents_to_markdown(
    input_path: str,
    output_dir: str = "build",
    allowed_formats: Optional[Iterable[InputFormat]] = None,
    enable_ocr: bool = False,
    enable_table_structure: bool = True,
    recursive: bool = True,
) -> List[str]:
    """Convert documents in a folder to Markdown using Docling."""
    formats = list(allowed_formats) if allowed_formats else DEFAULT_ALLOWED_FORMATS
    extensions = _get_extensions_for_formats(formats)

    doc_paths = _explore_documents_in_folder(
        folder_path=input_path,
        extensions=extensions,
        recursive=recursive,
    )

    if not doc_paths:
        print(f"No supported documents found in {input_path}")
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    converter = _create_document_converter(
        allowed_formats=formats,
        enable_ocr=enable_ocr,
        enable_table_structure=enable_table_structure,
    )

    output_files: List[str] = []

    for doc_path in doc_paths:
        try:
            print(f"Processing: {doc_path}")
            doc = converter.convert(doc_path).document
            md_text = MarkdownDocSerializer(doc=doc).serialize().text

            source_path = Path(doc_path)
            suffixes = [s.lower() for s in source_path.suffixes]
            if suffixes[-2:] == [".tar", ".gz"]:
                ext_label = "tar.gz"
                stem = source_path.name[: -len(".tar.gz")]
            else:
                ext_label = source_path.suffix.lower().lstrip(".") or "doc"
                stem = source_path.stem

            output_path = out_dir / f"{stem}_{ext_label}.md"
            output_path.write_text(md_text, encoding="utf-8")

            output_files.append(str(output_path))
            print(f"✓ Saved parsed text to: {output_path}")
        except Exception as exc:
            print(f"✗ Error processing {doc_path}: {exc}")

    print(f"\nCompleted processing {len(doc_paths)} files.")
    return output_files


# ---------------------------------------------------------------------------
# Public adapter API used by the SKGB pipeline
# ---------------------------------------------------------------------------

def docling_convert_to_markdown(
    *,
    input_path: Path,
    output_dir: Path,
    recursive: bool = True,
    enable_ocr: bool = False,
    enable_table_structure: bool = True,
) -> List[Path]:
    """Convert PDF(s) to Markdown using the Docling library."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        input_folder = str(input_path.parent)
    else:
        input_folder = str(input_path)

    out_files = _convert_documents_to_markdown(
        input_path=input_folder,
        output_dir=str(output_dir),
        enable_ocr=enable_ocr,
        enable_table_structure=enable_table_structure,
        recursive=recursive,
    )

    md_paths = [Path(p) for p in out_files]
    if input_path.is_file():
        stem = input_path.stem
        md_paths = [p for p in md_paths if p.name.startswith(stem + "_")]

    return md_paths
