import argparse
import sys
from pathlib import Path
from src.ingest import run_ingestion
from src.vector_store import VectorStoreManager
from src.retriever import AUTOSARRetriever
from src.generator import GeminiGenerator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    console = Console()
except ImportError:
    console = None

def main():
    parser = argparse.ArgumentParser(
        description="AUTOSAR RAG Pipeline - Ingestion, Query Refinement, and Retrieval-Augmented Generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from docs/ into ChromaDB")
    ingest_parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Path to folder containing documents (defaults to ./docs)"
    )
    ingest_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing ChromaDB index before ingestion"
    )

    # 2. Status command
    subparsers.add_parser("status", help="Check status and total vector count of ChromaDB collection")

    # 3. Query command
    query_parser = subparsers.add_parser("query", help="Ask a question against indexed AUTOSAR specs")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of final context chunks to use")
    query_parser.add_argument("--candidates", type=int, default=12, help="Initial candidate pool for re-ranking")
    query_parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder re-ranking")
    query_parser.add_argument("--no-refine", action="store_true", help="Disable LLM query refinement")
    query_parser.add_argument("--only-retrieval", action="store_true", help="Only show retrieved chunks without calling LLM")

    args = parser.parse_args()

    if args.command == "ingest":
        docs_path = Path(args.docs_dir) if args.docs_dir else None
        run_ingestion(docs_path=docs_path, clear_existing=args.clear)

    elif args.command == "status":
        vs = VectorStoreManager()
        stats = vs.get_stats()
        if console:
            table = Table(title="ChromaDB Collection Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            for k, v in stats.items():
                table.add_row(str(k), str(v))
            console.print(table)
        else:
            print("\n--- ChromaDB Collection Status ---")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()

    elif args.command == "query":
        generator = GeminiGenerator()
        retriever = AUTOSARRetriever()
        use_reranker = not args.no_rerank
        use_refiner = not args.no_refine

        search_query = args.question

        # Step 1: Query Refinement via LLM
        if use_refiner:
            if console:
                console.print(f"\n[bold magenta]🔮 Step 1: Refining query with LLM for optimal AUTOSAR retrieval...[/bold magenta]")
            refined = generator.refine_query(args.question)
            if refined and refined != args.question:
                search_query = refined
                if console:
                    console.print(f"[bold cyan]✨ Optimized Search Query:[/bold cyan] [yellow]{search_query}[/yellow]\n")

        # Step 2: Retrieval
        if console:
            console.print(f"[bold blue]🔍 Step 2: Searching vector database...[/bold blue]")
        chunks = retriever.retrieve(
            query=search_query,
            top_k_candidates=args.candidates,
            top_k_final=args.top_k,
            use_reranker=use_reranker
        )

        if not chunks:
            if console:
                console.print("[bold red]No matching documents found in vector database.[/bold red]")
            else:
                print("No matching documents found.")
            return

        # Display retrieved chunks
        if console:
            table = Table(title="Retrieved Relevant Chunks")
            table.add_column("#", justify="right", style="cyan", width=3)
            table.add_column("Source Document", style="magenta")
            table.add_column("Page", justify="center", style="yellow", width=6)
            table.add_column("Score", justify="right", style="green", width=8)
            table.add_column("Snippet Preview", style="white")

            for i, c in enumerate(chunks, 1):
                src = c.get("metadata", {}).get("source_file", "Unknown")
                page = str(c.get("metadata", {}).get("page_number", "N/A"))
                score = str(c.get("rerank_score", c.get("score", "N/A")))
                snippet = c["text"].replace("\n", " ")[:90] + "..."
                table.add_row(str(i), src, page, score, snippet)

            console.print(table)
            console.print()

        if args.only_retrieval:
            return

        # Step 3: LLM Technical Generation
        context = retriever.format_context(chunks)
        if console:
            console.print("[bold green]🤖 Step 3: Generating technical response with Gemini...[/bold green]\n")

        answer = generator.generate_answer(args.question, context)

        if console:
            console.print(Panel(Markdown(answer), title="AUTOSAR Technical Answer", expand=False))
        else:
            print("\n=== AUTOSAR Technical Answer ===")
            print(answer)
            print("================================\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
