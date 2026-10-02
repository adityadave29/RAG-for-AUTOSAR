import sys
import time
from pathlib import Path
from typing import Optional

from src.config import DOCS_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.parser import parse_document
from src.chunker import TechnicalDocumentChunker
from src.vector_store import VectorStoreManager

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None

def log_info(msg: str):
    if console:
        console.print(f"[bold blue][INFO][/bold blue] {msg}")
    else:
        print(f"[INFO] {msg}")

def log_success(msg: str):
    if console:
        console.print(f"[bold green][SUCCESS][/bold green] {msg}")
    else:
        print(f"[SUCCESS] {msg}")

def log_warning(msg: str):
    if console:
        console.print(f"[bold yellow][WARNING][/bold yellow] {msg}")
    else:
        print(f"[WARNING] {msg}")

def run_ingestion(docs_path: Optional[Path] = None, clear_existing: bool = False):
    """
    Main ingestion pipeline:
    1. Scan docs directory
    2. Parse documents into pages
    3. Chunk pages with context preservation
    4. Store chunks and generate embeddings in ChromaDB
    """
    target_dir = docs_path or DOCS_DIR
    target_dir = Path(target_dir)

    if not target_dir.exists():
        log_warning(f"Docs folder '{target_dir}' does not exist. Creating it.")
        target_dir.mkdir(parents=True, exist_ok=True)
        return

    # Scan for files
    supported_exts = {".pdf", ".txt", ".md", ".rst", ".c", ".h", ".arxml", ".xml"}
    files = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    if not files:
        log_warning(f"No documents found in '{target_dir}'. Place AUTOSAR PDFs or text files here.")
        return

    log_info(f"Found {len(files)} document(s) in '{target_dir}'. Starting ingestion...")
    start_time = time.time()

    # Step 1: Initialize Vector Store
    vector_store = VectorStoreManager()
    if clear_existing:
        log_info("Clearing existing ChromaDB collection...")
        vector_store.clear()

    # Step 2: Initialize Chunker
    chunker = TechnicalDocumentChunker(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    all_chunks = []
    file_summary = []

    # Step 3: Parse and Chunk each document
    for file_path in files:
        try:
            log_info(f"Parsing: {file_path.name}")
            pages = parse_document(file_path)
            chunks = chunker.chunk_document_pages(pages)
            all_chunks.extend(chunks)

            file_summary.append({
                "file": file_path.name,
                "pages": len(pages),
                "chunks": len(chunks)
            })
        except Exception as e:
            log_warning(f"Failed to process {file_path.name}: {e}")

    if not all_chunks:
        log_warning("No chunks generated from documents.")
        return

    # Step 4: Index into ChromaDB
    log_info(f"Indexing {len(all_chunks)} total chunks into ChromaDB...")
    indexed_count = vector_store.add_chunks(all_chunks)

    elapsed = round(time.time() - start_time, 2)
    stats = vector_store.get_stats()

    # Print summary table
    if console:
        table = Table(title="AUTOSAR Ingestion Summary")
        table.add_column("Document Name", style="cyan")
        table.add_column("Pages / Sections", justify="right", style="magenta")
        table.add_column("Chunks Generated", justify="right", style="green")

        for row in file_summary:
            table.add_row(row["file"], str(row["pages"]), str(row["chunks"]))

        console.print(table)
        console.print(f"[bold green]✓ Ingestion Complete![/bold green] Total chunks indexed: {indexed_count} in {elapsed}s.")
        console.print(f"Collection status: [bold cyan]{stats['total_chunks']}[/bold cyan] total vectors stored in `{stats['persist_dir']}`.\n")
    else:
        print("\n--- Ingestion Summary ---")
        for row in file_summary:
            print(f"- {row['file']}: {row['pages']} pages -> {row['chunks']} chunks")
        print(f"[SUCCESS] Ingestion Complete! {indexed_count} chunks indexed in {elapsed}s.")
        print(f"Total in collection: {stats['total_chunks']} chunks.\n")

if __name__ == "__main__":
    run_ingestion()
