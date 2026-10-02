from typing import List, Dict, Any, Optional
from src.vector_store import VectorStoreManager
from src.reranker import DocumentReranker

class AUTOSARRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Dense Semantic Search (ChromaDB) -> Top K1 candidates (e.g. 10)
    Stage 2: Cross-Encoder Re-ranking -> Top K2 most relevant chunks (e.g. 4)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        reranker: Optional[DocumentReranker] = None
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.reranker = reranker or DocumentReranker()

    def retrieve(
        self,
        query: str,
        top_k_candidates: int = 10,
        top_k_final: int = 4,
        use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve and rank the most relevant AUTOSAR document chunks."""
        # Stage 1: Dense Vector Search
        initial_candidates = self.vector_store.query(
            query_text=query,
            top_k=top_k_candidates
        )

        if not initial_candidates:
            return []

        # Stage 2: Re-ranking
        if use_reranker:
            final_chunks = self.reranker.rerank(
                query=query,
                candidate_chunks=initial_candidates,
                top_k=top_k_final
            )
        else:
            final_chunks = initial_candidates[:top_k_final]

        return final_chunks

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a clean context block for LLM prompt."""
        formatted_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source_file", "Unknown Document")
            page = chunk.get("metadata", {}).get("page_number", "N/A")
            score_info = f"Relevance Score: {chunk.get('rerank_score', chunk.get('score', 'N/A'))}"

            formatted_blocks.append(
                f"--- [Document Context #{i}] Source: {source} (Page {page}) | {score_info} ---\n"
                f"{chunk['text']}\n"
            )

        return "\n\n".join(formatted_blocks)
