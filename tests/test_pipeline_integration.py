import unittest
import tempfile
from pathlib import Path
from src.parser import parse_document
from src.chunker import TechnicalDocumentChunker
from src.vector_store import VectorStoreManager
from src.reranker import DocumentReranker
from src.retriever import AUTOSARRetriever

class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_ingest_and_retrieve(self):
        # 1. Create a dummy AUTOSAR SWS specification
        spec_file = self.temp_path / "AUTOSAR_SWS_DiagnosticEventManager.md"
        spec_content = """# Specification of Diagnostic Event Manager (Dem)

## 1. Introduction
The Dem module is responsible for processing and storing diagnostic events and Diagnostic Trouble Codes (DTCs).

## 2. API Specification
[SWS_Dem_00010] The function `Dem_SetEventStatus(EventId, EventStatus)` shall report the status of a diagnostic event.
- `EventId`: Identification of the diagnostic event.
- `EventStatus`: Status value (e.g., `DEM_EVENT_STATUS_PASSED`, `DEM_EVENT_STATUS_FAILED`).

[SWS_Dem_00015] If the EventId is invalid, `Dem_SetEventStatus` shall report the development error `DEM_E_PARAM_DATA`.
"""
        spec_file.write_text(spec_content, encoding="utf-8")

        # 2. Parse
        pages = parse_document(spec_file)
        self.assertEqual(len(pages), 1)

        # 3. Chunk
        chunker = TechnicalDocumentChunker(chunk_size=400, chunk_overlap=80)
        chunks = chunker.chunk_document_pages(pages)
        self.assertGreater(len(chunks), 0)

        # 4. Vector Store & Index
        vs = VectorStoreManager(
            persist_dir=self.temp_path / "chroma",
            collection_name="test_integration_col"
        )
        added_count = vs.add_chunks(chunks)
        self.assertEqual(added_count, len(chunks))

        # 5. Retrieve & Re-rank
        reranker = DocumentReranker()
        retriever = AUTOSARRetriever(vector_store=vs, reranker=reranker)

        results = retriever.retrieve(
            query="How does Dem_SetEventStatus report event failures?",
            top_k_candidates=5,
            top_k_final=2
        )

        self.assertGreater(len(results), 0)
        top_chunk = results[0]
        self.assertIn("Dem_SetEventStatus", top_chunk["text"])
        self.assertIn("SWS_Dem_00010", top_chunk["text"])
        self.assertIn("AUTOSAR_SWS_DiagnosticEventManager.md", top_chunk["metadata"]["source_file"])

if __name__ == "__main__":
    unittest.main()
