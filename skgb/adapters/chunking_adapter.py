from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class _Header:
    line_idx: int
    level: int
    title: str
    path: Tuple[str, ...]


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _doc_id_for_path(p: Path) -> str:
    # Stable-ish ID for caching/provenance. For MVP, hash the markdown bytes.
    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _extract_headers(lines: List[str]) -> List[_Header]:
    headers: List[_Header] = []
    stack: List[Tuple[int, str]] = []  # (level, title)

    for idx, raw in enumerate(lines):
        m = _HEADER_RE.match(raw.rstrip("\n"))
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()

        # Maintain a level-based stack.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = tuple(t for _, t in stack)
        headers.append(_Header(line_idx=idx, level=level, title=title, path=path))

    return headers


def _split_large_text_by_paragraphs(text: str, max_words: int) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    parts: List[str] = []
    buf: List[str] = []
    count = 0

    for para in paras:
        w = len(para.split())
        if buf and count + w > max_words:
            parts.append("\n\n".join(buf).strip())
            buf = [para]
            count = w
        else:
            buf.append(para)
            count += w

    if buf:
        parts.append("\n\n".join(buf).strip())
    return parts or [text.strip()]


def _add_overlap(chunks: List[Dict[str, Any]], overlap_words: int) -> None:
    if overlap_words <= 0 or len(chunks) <= 1:
        return
    for i in range(len(chunks)):
        content = (chunks[i].get("content") or "").strip()
        if not content:
            continue

        if i > 0:
            prev = (chunks[i - 1].get("content") or "").split()
            if len(prev) > overlap_words:
                prefix = " ".join(prev[-overlap_words:])
                content = f"[...{prefix}]\n\n{content}"

        if i < len(chunks) - 1:
            nxt = (chunks[i + 1].get("content") or "").split()
            if len(nxt) > overlap_words:
                suffix = " ".join(nxt[:overlap_words])
                content = f"{content}\n\n[{suffix}...]"

        chunks[i]["content"] = content


def chunk_markdown_files(
    *,
    md_paths: List[Path],
    min_chunk_words: int = 200,
    max_chunk_words: int = 800,
    overlap_words: int = 0,
    preserve_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """Chunk text files with ``#`` headers into the ``all_chunks.json`` schema.

    Accepts both ``.md`` (Markdown) and ``.txt`` (plain text with ``#``
    hierarchical headers) produced by the Docling adapter.  Sections are
    detected via ``# … ######`` header lines.
    """
    all_chunks: List[Dict[str, Any]] = []

    for md_path in md_paths:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        headers = _extract_headers(lines)

        doc_id = _doc_id_for_path(md_path)
        doc_name = md_path.stem

        # Optional metadata chunk (before first header)
        if preserve_metadata:
            first_header_line = headers[0].line_idx if headers else None
            if first_header_line is None:
                meta = text.strip()
            else:
                meta = "".join(lines[:first_header_line]).strip()
            if meta:
                all_chunks.append(
                    {
                        "id": f"{doc_name}__{doc_id}__metadata__0",
                        "section_title": "Document Metadata",
                        "content": meta,
                        "metadata": {
                            "doc_name": doc_name,
                            "doc_id": doc_id,
                            "md_path": str(md_path),
                            "header_level": 0,
                            "header_path": "Document Metadata",
                        },
                    }
                )

        # If no headers, treat entire doc as one chunk
        if not headers:
            body = text.strip()
            if body:
                all_chunks.append(
                    {
                        "id": f"{doc_name}__{doc_id}__body__0",
                        "section_title": "Body",
                        "content": body,
                        "metadata": {
                            "doc_name": doc_name,
                            "doc_id": doc_id,
                            "md_path": str(md_path),
                            "header_level": 0,
                            "header_path": "Body",
                        },
                    }
                )
            continue

        for hi, h in enumerate(headers):
            start = h.line_idx + 1
            end = (
                (headers[hi + 1].line_idx - 1)
                if (hi + 1) < len(headers)
                else (len(lines) - 1)
            )
            content = "".join(lines[start : end + 1]).strip()
            if not content:
                continue

            parts = _split_large_text_by_paragraphs(content, max_chunk_words)
            for pi, part in enumerate(parts):
                if len(part.split()) < min_chunk_words and len(parts) > 1:
                    # Allow small subparts when splitting; we won't merge across headers in MVP.
                    pass

                section_title = h.title
                if len(parts) > 1:
                    section_title = f"{h.title} (Part {pi + 1})"

                header_path = " > ".join(h.path)
                chunk_id = f"{doc_name}__{doc_id}__{hashlib.sha1(header_path.encode('utf-8')).hexdigest()[:8]}__{pi}"
                all_chunks.append(
                    {
                        "id": chunk_id,
                        "section_title": section_title,
                        "content": part,
                        "metadata": {
                            "doc_name": doc_name,
                            "doc_id": doc_id,
                            "md_path": str(md_path),
                            "header_level": h.level,
                            "header_path": header_path,
                        },
                    }
                )

    _add_overlap(all_chunks, overlap_words)
    return all_chunks
