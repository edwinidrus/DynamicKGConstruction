from __future__ import annotations

from pathlib import Path
from typing import List


def docling_convert_to_markdown(
    *,
    input_path: Path,
    output_dir: Path,
    recursive: bool = True,
    enable_ocr: bool = False,
    enable_table_structure: bool = True,
) -> List[Path]:
    """Convert PDF(s) to Markdown using the existing Docling ingestion module."""
    try:
        import docling  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Docling is not installed in this Python environment. "
            "Install dependencies first (see DynamicKGConstruction/requirements.txt)."
        ) from e

    # The existing function accepts a folder path; accept a single file too.
    from ...docling_ingest.docling_multiple_document import convert_documents_to_markdown

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # Convert just the containing folder; Docling ingestion already filters supported formats.
        input_folder = str(input_path.parent)
    else:
        input_folder = str(input_path)

    out_files = convert_documents_to_markdown(
        input_path=input_folder,
        output_dir=str(output_dir),
        enable_ocr=enable_ocr,
        enable_table_structure=enable_table_structure,
        recursive=recursive,
    )

    # If user passed a single file, filter to that doc stem best-effort.
    md_paths = [Path(p) for p in out_files]
    if input_path.is_file():
        stem = input_path.stem
        md_paths = [p for p in md_paths if p.name.startswith(stem + "_")]

    return md_paths
