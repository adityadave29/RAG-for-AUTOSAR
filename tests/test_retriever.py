import unittest
import tempfile
from pathlib import Path
from src.chunker import Chunk
from src.vector_store import VectorStoreManager
from src.reranker import DocumentReranker
from src.retriever import AUTOSARRetriever

class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.vs = VectorStoreManager(
            persist_dir=self.temp_path,
            collection_name="test_retriever_col"
        )
        self.reranker = DocumentReranker()
        self.retriever = AUTOSARRetriever(vector_store=self.vs, reranker=self.reranker)

        # Seed sample chunks
        chunks = [
            Chunk(
                chunk_id="c1",
                text="[Source: SWS_CAN.pdf | Page: 1] [SWS_Can_00017] Can_SetControllerMode switches CAN modes.",
                source_file="SWS_CAN.pdf",
                page_number=1,
                chunk_index=0,
                metadata={"source_file": "SWS_CAN.pdf", "page_number": 1}
            ),
            Chunk(
                chunk_id="c2",
                text="[Source: SWS_RTE.pdf | Page: 10] RTE_Send transmits data via Sender-Receiver port.",
                source_file="SWS_RTE.pdf",
                page_number=10,
                chunk_index=1,
                metadata={"source_file": "SWS_RTE.pdf", "page_number": 10}
            )
        ]
        self.vs.add_chunks(chunks)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_two_stage_retrieval(self):
        results = self.retriever.retrieve("How to switch CAN controller modes?", top_k_candidates=2, top_k_final=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "c1")

    def test_format_context(self):
        chunks = [
            {
                "id": "c1",
                "text": "Can_Init initializes the CAN module.",
                "metadata": {"source_file": "SWS_CAN.pdf", "page_number": 3},
                "score": 0.95
            }
        ]
        context = self.retriever.format_context(chunks)
        self.assertIn("Source: SWS_CAN.pdf (Page 3)", context)
        self.assertIn("Can_Init", context)

if __name__ == "__main__":
    unittest.main()
