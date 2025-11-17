"""
Improved Semantic Chunking Method for Academic Papers
Based on header hierarchy (## markers) with enhanced semantic awareness
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a semantic chunk of text"""
    header: str
    content: str
    level: int
    start_line: int
    end_line: int
    word_count: int
    parent_header: str = ""

    def to_dict(self):
        return {
            "header": self.header,
            "content": self.content,
            "level": self.level,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "word_count": self.word_count,
            "parent_header": self.parent_header
        }


class SemanticChunker:
    """
    Enhanced semantic chunker that creates meaningful chunks based on:
    1. Document structure (headers with ##)
    2. Semantic coherence (keeping related content together)
    3. Context preservation (maintaining parent-child relationships)
    4. Optimal chunk size for embeddings
    """

    def __init__(self,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 1000,
                 overlap_size: int = 50,
                 preserve_metadata: bool = True):
        """
        Initialize the semantic chunker

        Args:
            min_chunk_size: Minimum words per chunk (smaller chunks may be merged)
            max_chunk_size: Maximum words per chunk (larger chunks will be split)
            overlap_size: Number of words to overlap between chunks for context
            preserve_metadata: Whether to keep metadata lines (Source, Processing Date, etc.)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.preserve_metadata = preserve_metadata

    def parse_document(self, file_path: str) -> List[Chunk]:
        """
        Parse document into semantic chunks based on headers

        Args:
            file_path: Path to the text file

        Returns:
            List of Chunk objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find all headers and their positions
        headers = self._extract_headers(lines)

        # Create initial chunks based on headers
        initial_chunks = self._create_header_chunks(lines, headers)

        # Process chunks for optimal size
        processed_chunks = self._optimize_chunk_sizes(initial_chunks)

        # Add overlapping context for better semantic continuity
        final_chunks = self._add_overlap(processed_chunks)

        return final_chunks

    def _extract_headers(self, lines: List[str]) -> List[Tuple[int, str, int]]:
        """
        Extract headers with their line numbers and hierarchy level

        Returns:
            List of tuples: (line_number, header_text, level)
        """
        headers = []
        header_pattern = re.compile(r'^##\s+(.+)$')

        for i, line in enumerate(lines):
            match = header_pattern.match(line.strip())
            if match:
                header_text = match.group(1).strip()
                # Determine level based on context (all ## are level 1 in this format)
                # but we can infer hierarchy from content type
                level = self._determine_header_level(header_text)
                headers.append((i, header_text, level))

        return headers

    def _determine_header_level(self, header_text: str) -> int:
        """
        Determine semantic level of header based on content

        Major sections (level 1): Introduction, Methods, Results, Discussion, Conclusion
        Subsections (level 2): numbered sections like "2.1", "3.2"
        Special sections (level 0): ABSTRACT, ORIGINAL ARTICLE
        """
        major_sections = [
            'introduction', 'literature review', 'theoretical framework',
            'hypothesis development', 'research design', 'data and variables',
            'empirical results', 'robustness checks', 'mechanism analysis',
            'heterogeneity analysis', 'discussion', 'conclusion',
            'managerial implications', 'references', 'supporting information'
        ]

        header_lower = header_text.lower()

        # Meta sections
        if header_lower in ['original article', 'abstract']:
            return 0

        # Check for numbered subsections (e.g., "2.1", "3.2")
        if re.match(r'^\d+\.?\d*\s+\|', header_text) or re.match(r'^\d+\.?\d+\s+[A-Z]', header_text):
            return 2

        # Major sections
        for section in major_sections:
            if section in header_lower:
                return 1

        # Default level
        return 1

    def _create_header_chunks(self, lines: List[str], headers: List[Tuple[int, str, int]]) -> List[Chunk]:
        """
        Create initial chunks based on header positions
        """
        chunks = []

        # Handle document metadata (before first header)
        if headers:
            first_header_line = headers[0][0]
            if first_header_line > 0:
                metadata_content = ''.join(lines[:first_header_line]).strip()
                if metadata_content and self.preserve_metadata:
                    chunks.append(Chunk(
                        header="Document Metadata",
                        content=metadata_content,
                        level=-1,
                        start_line=0,
                        end_line=first_header_line - 1,
                        word_count=len(metadata_content.split())
                    ))

        # Create chunks for each section
        for i, (line_num, header_text, level) in enumerate(headers):
            # Determine end of this section
            if i < len(headers) - 1:
                end_line = headers[i + 1][0] - 1
            else:
                end_line = len(lines) - 1

            # Extract content
            content_lines = lines[line_num + 1:end_line + 1]
            content = ''.join(content_lines).strip()

            # Skip empty sections
            if not content:
                continue

            # Determine parent header for context
            parent_header = ""
            if level > 0:
                for j in range(i - 1, -1, -1):
                    if headers[j][2] < level:
                        parent_header = headers[j][1]
                        break

            chunks.append(Chunk(
                header=header_text,
                content=content,
                level=level,
                start_line=line_num,
                end_line=end_line,
                word_count=len(content.split()),
                parent_header=parent_header
            ))

        return chunks

    def _optimize_chunk_sizes(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Optimize chunk sizes by merging small chunks and splitting large ones
        """
        optimized = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # If chunk is too large, split it
            if current_chunk.word_count > self.max_chunk_size:
                split_chunks = self._split_large_chunk(current_chunk)
                optimized.extend(split_chunks)

            # If chunk is too small, try to merge with next chunk
            elif current_chunk.word_count < self.min_chunk_size and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]

                # Only merge if next chunk is also small or is a subsection
                if (next_chunk.word_count < self.min_chunk_size or
                    next_chunk.level > current_chunk.level):
                    merged = self._merge_chunks(current_chunk, next_chunk)
                    optimized.append(merged)
                    i += 2  # Skip next chunk as it's been merged
                    continue
                else:
                    optimized.append(current_chunk)
            else:
                optimized.append(current_chunk)

            i += 1

        return optimized

    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """
        Split a large chunk into smaller semantic units
        """
        # Split by paragraphs first
        paragraphs = chunk.content.split('\n\n')

        sub_chunks = []
        current_content = []
        current_word_count = 0
        part_num = 1

        for para in paragraphs:
            para_words = len(para.split())

            if current_word_count + para_words > self.max_chunk_size and current_content:
                # Create a sub-chunk
                sub_chunks.append(Chunk(
                    header=f"{chunk.header} (Part {part_num})",
                    content='\n\n'.join(current_content).strip(),
                    level=chunk.level,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    word_count=current_word_count,
                    parent_header=chunk.parent_header
                ))
                current_content = [para]
                current_word_count = para_words
                part_num += 1
            else:
                current_content.append(para)
                current_word_count += para_words

        # Add remaining content
        if current_content:
            sub_chunks.append(Chunk(
                header=f"{chunk.header} (Part {part_num})" if part_num > 1 else chunk.header,
                content='\n\n'.join(current_content).strip(),
                level=chunk.level,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                word_count=current_word_count,
                parent_header=chunk.parent_header
            ))

        return sub_chunks if sub_chunks else [chunk]

    def _merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """
        Merge two consecutive chunks
        """
        merged_content = f"{chunk1.content}\n\n{chunk2.content}".strip()
        merged_header = chunk1.header

        # If merging a parent with child, use more descriptive header
        if chunk2.parent_header == chunk1.header:
            merged_header = f"{chunk1.header} - {chunk2.header}"

        return Chunk(
            header=merged_header,
            content=merged_content,
            level=min(chunk1.level, chunk2.level),
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            word_count=chunk1.word_count + chunk2.word_count,
            parent_header=chunk1.parent_header
        )

    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add overlapping text between chunks for better context preservation
        """
        if len(chunks) <= 1 or self.overlap_size == 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            # Add prefix from previous chunk
            if i > 0:
                prev_words = chunks[i - 1].content.split()
                if len(prev_words) > self.overlap_size:
                    overlap_prefix = ' '.join(prev_words[-self.overlap_size:])
                    chunk.content = f"[...{overlap_prefix}]\n\n{chunk.content}"

            # Add suffix from next chunk
            if i < len(chunks) - 1:
                next_words = chunks[i + 1].content.split()
                if len(next_words) > self.overlap_size:
                    overlap_suffix = ' '.join(next_words[:self.overlap_size])
                    chunk.content = f"{chunk.content}\n\n[{overlap_suffix}...]"

            overlapped_chunks.append(chunk)

        return overlapped_chunks

    def export_chunks(self, chunks: List[Chunk], output_format: str = 'json') -> str:
        """
        Export chunks to various formats

        Args:
            chunks: List of chunks to export
            output_format: 'json', 'markdown', or 'text'
        """
        if output_format == 'json':
            import json
            return json.dumps([chunk.to_dict() for chunk in chunks], indent=2, ensure_ascii=False)

        elif output_format == 'markdown':
            output = []
            for i, chunk in enumerate(chunks, 1):
                output.append(f"# Chunk {i}: {chunk.header}\n")
                output.append(f"**Level:** {chunk.level} | **Words:** {chunk.word_count}")
                if chunk.parent_header:
                    output.append(f" | **Parent:** {chunk.parent_header}")
                output.append(f"\n\n{chunk.content}\n")
                output.append("\n" + "="*80 + "\n\n")
            return ''.join(output)

        elif output_format == 'text':
            output = []
            for i, chunk in enumerate(chunks, 1):
                output.append(f"CHUNK {i}: {chunk.header}")
                output.append(f"Lines: {chunk.start_line}-{chunk.end_line} | Words: {chunk.word_count}")
                output.append(f"\n{chunk.content}\n")
                output.append("\n" + "-"*80 + "\n\n")
            return ''.join(output)

        else:
            raise ValueError(f"Unsupported format: {output_format}")


def main():
    """Example usage of the semantic chunker"""

    # Initialize chunker with custom parameters
    chunker = SemanticChunker(
        min_chunk_size=200,      # Minimum 200 words per chunk
        max_chunk_size=800,      # Maximum 800 words per chunk
        overlap_size=50,         # 50 words overlap between chunks
        preserve_metadata=True   # Keep document metadata
    )

    # Parse the document
    input_file = '/content/robotics - paper.txt'
    chunks = chunker.parse_document(input_file)

    # Print summary
    print(f"Document parsed into {len(chunks)} semantic chunks")
    print(f"\nChunk Summary:")
    print("-" * 80)

    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk.header[:60]}...")
        print(f"   Level: {chunk.level} | Words: {chunk.word_count} | Lines: {chunk.start_line}-{chunk.end_line}")
        if chunk.parent_header:
            print(f"   Parent: {chunk.parent_header}")
        print()

    # Export to different formats
    print("\n" + "="*80)
    print("Exporting chunks...")

    # JSON format
    json_output = chunker.export_chunks(chunks, 'json')
    with open('/content/chunks_output.json', 'w', encoding='utf-8') as f:
        f.write(json_output)
    print("✓ JSON export: chunks_output.json")

    # Markdown format
    md_output = chunker.export_chunks(chunks, 'markdown')
    with open('/content/chunks_output.md', 'w', encoding='utf-8') as f:
        f.write(md_output)
    print("✓ Markdown export: chunks_output.md")

    # Text format
    text_output = chunker.export_chunks(chunks, 'text')
    with open('/content/chunks_output.txt', 'w', encoding='utf-8') as f:
        f.write(text_output)
    print("✓ Text export: chunks_output.txt")

    print("\nAll exports completed successfully!")

    # Statistics
    total_words = sum(chunk.word_count for chunk in chunks)
    avg_words = total_words / len(chunks) if chunks else 0

    print(f"\nStatistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total words: {total_words}")
    print(f"  Average words per chunk: {avg_words:.1f}")
    print(f"  Smallest chunk: {min(chunk.word_count for chunk in chunks)} words")
    print(f"  Largest chunk: {max(chunk.word_count for chunk in chunks)} words")


if __name__ == "__main__":
    main()