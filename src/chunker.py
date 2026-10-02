from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.parser import DocumentPage
import re

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TechnicalDocumentChunker:
    """
    Recursive section & character chunker tailored for technical documentation.
    Preserves section context, requirement IDs, and handles overlap smoothly.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Delimiters from strongest semantic boundary to weakest
        self.separators = [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            ""
        ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators until pieces fit in chunk_size."""
        if not separators:
            # Fallback: slice directly
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        chunks = []
        current_chunk = ""

        for part in splits:
            if not part.strip():
                continue

            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(part) > self.chunk_size:
                    # Part itself is too large, recurse with remaining separators
                    sub_chunks = self._split_text(part, remaining_seps)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _apply_overlap(self, pieces: List[str]) -> List[str]:
        """Combine pieces to reach target chunk_size while maintaining overlap."""
        if not pieces:
            return []

        merged_chunks = []
        i = 0
        while i < len(pieces):
            current = pieces[i]
            # Try combining with subsequent small pieces up to chunk_size
            j = i + 1
            while j < len(pieces) and len(current) + len(pieces[j]) + 1 <= self.chunk_size:
                current += "\n" + pieces[j]
                j += 1

            merged_chunks.append(current.strip())
            
            # Move index forward, respecting overlap
            if j == i + 1:
                i += 1
            else:
                # Step forward with overlap
                i = max(i + 1, j - 1)

        return merged_chunks

    def chunk_document_pages(self, pages: List[DocumentPage]) -> List[Chunk]:
        """Chunks pages and enriches them with contextual headers for high-fidelity retrieval."""
        all_chunks = []
        global_chunk_idx = 0

        for page in pages:
            raw_text = page.text.strip()
            if not raw_text:
                continue

            pieces = self._split_text(raw_text, self.separators)
            page_chunks = self._apply_overlap(pieces)

            for local_idx, chunk_text in enumerate(page_chunks):
                if len(chunk_text.strip()) < 30:
                    continue  # Ignore trivial tiny chunks

                chunk_id = f"{page.source_file}_p{page.page_number}_c{local_idx}"
                
                # Context injection header for embedding accuracy
                context_header = f"[Source: {page.source_file} | Page: {page.page_number}]\n"
                enriched_text = context_header + chunk_text

                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=enriched_text,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    metadata={
                        "source_file": page.source_file,
                        "page_number": page.page_number,
                        "chunk_index": global_chunk_idx,
                        "char_length": len(chunk_text),
                        **page.metadata
                    }
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        return all_chunks
