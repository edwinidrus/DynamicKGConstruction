"""Multi-format document ingestion with Docling.

Note: This module is intentionally *not* inside a package named `docling`.
The upstream dependency is also named `docling`, and using that name locally
would shadow the installed library.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer


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


def get_all_supported_formats() -> List[InputFormat]:
    """Return all supported input formats."""
    return list(SUPPORTED_FORMATS.keys())


def get_extensions_for_formats(formats: Iterable[InputFormat]) -> List[str]:
    """Return all extensions for the given formats."""
    extensions: List[str] = []
    for fmt in formats:
        extensions.extend(SUPPORTED_FORMATS.get(fmt, []))
    return extensions


def explore_documents_in_folder(
    folder_path: str,
    extensions: Optional[Iterable[str]] = None,
    recursive: bool = True,
) -> List[str]:
    """Return paths of all supported document files in a folder."""
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


def create_document_converter(
    allowed_formats: Optional[Iterable[InputFormat]] = None,
    enable_ocr: bool = False,
    enable_table_structure: bool = True,
) -> DocumentConverter:
    """Create a DocumentConverter with multi-format support."""
    if allowed_formats is None:
        allowed_formats = DEFAULT_ALLOWED_FORMATS

    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = enable_ocr
    pdf_pipeline_options.do_table_structure = enable_table_structure

    # Format options are optional for many formats, but providing them (when
    # available in the installed docling version) improves consistency.
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
        # Keep converter usable even if the installed docling version differs.
        pass

    return DocumentConverter(
        allowed_formats=list(allowed_formats),
        format_options=format_options,
    )


def convert_documents_to_markdown(
    input_path: str,
    output_dir: str = "build",
    allowed_formats: Optional[Iterable[InputFormat]] = None,
    enable_ocr: bool = False,
    enable_table_structure: bool = True,
    recursive: bool = True,
) -> List[str]:
    """Convert documents in a folder to Markdown using Docling."""
    formats = list(allowed_formats) if allowed_formats else DEFAULT_ALLOWED_FORMATS
    extensions = get_extensions_for_formats(formats)

    doc_paths = explore_documents_in_folder(
        folder_path=input_path,
        extensions=extensions,
        recursive=recursive,
    )

    if not doc_paths:
        print(f"No supported documents found in {input_path}")
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    converter = create_document_converter(
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
                ext_label = (source_path.suffix.lower().lstrip(".") or "doc")
                stem = source_path.stem

            output_path = out_dir / f"{stem}_{ext_label}.md"
            output_path.write_text(md_text, encoding="utf-8")

            output_files.append(str(output_path))
            print(f"✓ Saved parsed text to: {output_path}")
        except Exception as exc:
            print(f"✗ Error processing {doc_path}: {exc}")

    print(f"\nCompleted processing {len(doc_paths)} files.")
    return output_files
