import unittest
import tempfile
from pathlib import Path
from src.chunker import Chunk
from src.vector_store import VectorStoreManager

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.vs = VectorStoreManager(
            persist_dir=self.temp_path,
            collection_name="test_collection"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_query_chunks(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                text="[Source: SWS_CAN.pdf | Page: 10] Can_SetControllerMode sets the controller mode.",
                source_file="SWS_CAN.pdf",
                page_number=10,
                chunk_index=0,
                metadata={"source_file": "SWS_CAN.pdf", "page_number": 10, "chunk_index": 0}
            ),
            Chunk(
                chunk_id="c2",
                text="[Source: SWS_RTE.pdf | Page: 20] RTE provides communication between Software Components.",
                source_file="SWS_RTE.pdf",
                page_number=20,
                chunk_index=1,
                metadata={"source_file": "SWS_RTE.pdf", "page_number": 20, "chunk_index": 1}
            )
        ]

        added = self.vs.add_chunks(chunks)
        self.assertEqual(added, 2)

        stats = self.vs.get_stats()
        self.assertEqual(stats["total_chunks"], 2)

        # Query CAN related chunk
        results = self.vs.query("CAN controller mode transition", top_k=2)
        self.assertGreater(len(results), 0)
        top_res = results[0]
        self.assertEqual(top_res["id"], "c1")
        self.assertIn("Can_SetControllerMode", top_res["text"])
        self.assertGreater(top_res["score"], 0.0)

    def test_clear_collection(self):
        chunk = Chunk(
            chunk_id="c1",
            text="[Source: test.pdf | Page: 1] Testing vector clearing.",
            source_file="test.pdf",
            page_number=1,
            chunk_index=0,
            metadata={"source_file": "test.pdf", "page_number": 1, "chunk_index": 0}
        )
        self.vs.add_chunks([chunk])
        self.assertEqual(self.vs.get_stats()["total_chunks"], 1)

        self.vs.clear()
        self.assertEqual(self.vs.get_stats()["total_chunks"], 0)

if __name__ == "__main__":
    unittest.main()
