from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os

@dataclass
class DocumentPage:
    text: str
    source_file: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_pdf(file_path: Path) -> List[DocumentPage]:
    """Parse a PDF file using PyMuPDF (fitz) or fallback to pypdf."""
    pages = []
    file_name = file_path.name

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(file_path))
        for page_idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    DocumentPage(
                        text=text,
                        source_file=file_name,
                        page_number=page_idx + 1,
                        metadata={"total_pages": len(doc), "file_path": str(file_path)}
                    )
                )
        doc.close()
        return pages
    except ImportError:
        pass

    # Fallback to pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(
                    DocumentPage(
                        text=text,
                        source_file=file_name,
                        page_number=page_idx + 1,
                        metadata={"total_pages": len(reader.pages), "file_path": str(file_path)}
                    )
                )
        return pages
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF {file_path}: {e}")


def parse_text_file(file_path: Path) -> List[DocumentPage]:
    """Parse plain text or markdown files."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        
        if not content:
            return []

        return [
            DocumentPage(
                text=content,
                source_file=file_path.name,
                page_number=1,
                metadata={"total_pages": 1, "file_path": str(file_path)}
            )
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path}: {e}")


def parse_document(file_path: Path) -> List[DocumentPage]:
    """Universal parser based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in [".txt", ".md", ".rst", ".c", ".h", ".arxml", ".xml"]:
        return parse_text_file(file_path)
    else:
        # Attempt plain text read as fallback
        return parse_text_file(file_path)
