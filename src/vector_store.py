from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from src.chunker import Chunk
from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME

class VectorStoreManager:
    """Manages ChromaDB vector store initialization, indexing, and dense retrieval."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME
    ):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.model_name = model_name

        # Initialize local persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Configure local sentence-transformers embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Add chunks to ChromaDB in batches."""
        if not chunks:
            return 0

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "source_file": str(c.metadata.get("source_file", "")),
                    "page_number": int(c.metadata.get("page_number", 1)),
                    "chunk_index": int(c.metadata.get("chunk_index", 0)),
                }
                for c in batch
            ]

            # Upsert into ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            total_added += len(batch)

        return total_added

    def query(self, query_text: str, top_k: int = 10, where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Dense similarity search returning ranked results."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )

        formatted_results = []
        if not results or not results["documents"] or not results["documents"][0]:
            return formatted_results

        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            formatted_results.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "score": round(1.0 - dist, 4)  # Convert cosine distance to similarity
            })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Return collection count and status."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": self.persist_dir,
            "embedding_model": self.model_name
        }

    def clear(self) -> None:
        """Reset and delete collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
