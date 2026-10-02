from typing import List, Dict, Any, Optional
import os

class DocumentReranker:
    """
    Cross-Encoder Re-ranker.
    Scores (query, passage) pairs together to accurately select the top-k most relevant chunks.
    Uses FlashRank by default (ultra-fast, CPU-friendly), with fallback to SentenceTransformers CrossEncoder.
    """
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self.ranker = None
        self._backend = None
        self._init_ranker()

    def _init_ranker(self):
        # Try FlashRank first (lightweight & fast)
        try:
            from flashrank import Ranker
            self.ranker = Ranker(model_name=self.model_name, cache_dir="./data/models")
            self._backend = "flashrank"
            return
        except Exception:
            pass

        # Try SentenceTransformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            self.ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._backend = "sentence_transformers"
            return
        except Exception:
            pass

        # Fallback to no re-ranking if packages are not available
        self._backend = "passthrough"

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """Re-rank candidate chunks based on joint relevance to the query."""
        if not candidate_chunks:
            return []

        if len(candidate_chunks) <= top_k and self._backend == "passthrough":
            return candidate_chunks

        if self._backend == "flashrank":
            try:
                from flashrank import RerankRequest
                passages = [
                    {"id": idx, "text": chunk["text"], "meta": chunk.get("metadata", {})}
                    for idx, chunk in enumerate(candidate_chunks)
                ]
                request = RerankRequest(query=query, passages=passages)
                results = self.ranker.rerank(request)
                
                reranked = []
                for res in results[:top_k]:
                    orig_idx = res["id"]
                    item = dict(candidate_chunks[orig_idx])
                    item["rerank_score"] = round(float(res["score"]), 4)
                    reranked.append(item)
                return reranked
            except Exception:
                pass

        elif self._backend == "sentence_transformers":
            try:
                pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
                scores = self.ranker.predict(pairs)
                scored_chunks = list(zip(candidate_chunks, scores))
                scored_chunks.sort(key=lambda x: x[1], reverse=True)

                reranked = []
                for chunk, score in scored_chunks[:top_k]:
                    item = dict(chunk)
                    item["rerank_score"] = round(float(score), 4)
                    reranked.append(item)
                return reranked
            except Exception:
                pass

        # Passthrough fallback: return top_k candidates by initial score
        return candidate_chunks[:top_k]
