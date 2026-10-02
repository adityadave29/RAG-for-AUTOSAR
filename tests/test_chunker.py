import unittest
from src.parser import DocumentPage
from src.chunker import TechnicalDocumentChunker, Chunk

class TestChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = TechnicalDocumentChunker(chunk_size=300, chunk_overlap=50)

    def test_chunking_with_context_header(self):
        page = DocumentPage(
            text="## 1. Overview\nThe CAN Interface module provides an abstract interface to the CAN Driver.\n\n## 2. Functions\nCanIf_Init initializes the CanIf module.",
            source_file="AUTOSAR_SWS_CANInterface.pdf",
            page_number=5,
            metadata={"total_pages": 50}
        )

        chunks = self.chunker.chunk_document_pages([page])
        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertEqual(chunk.source_file, "AUTOSAR_SWS_CANInterface.pdf")
            self.assertEqual(chunk.page_number, 5)
            # Context header validation
            self.assertTrue(chunk.text.startswith("[Source: AUTOSAR_SWS_CANInterface.pdf | Page: 5]"))

    def test_short_text_filtering(self):
        # Very short text (< 30 chars) should be filtered out to prevent noise
        page = DocumentPage(
            text="Short text",
            source_file="test.txt",
            page_number=1
        )
        chunks = self.chunker.chunk_document_pages([page])
        self.assertEqual(len(chunks), 0)

    def test_large_document_multiple_chunks(self):
        long_text = "\n\n".join([f"Paragraph {i}: This is requirement [SWS_Com_{i:05d}] describing CAN communication handling in AUTOSAR stack." for i in range(20)])
        page = DocumentPage(
            text=long_text,
            source_file="AUTOSAR_SWS_COM.pdf",
            page_number=12
        )
        chunks = self.chunker.chunk_document_pages([page])
        self.assertGreater(len(chunks), 1)
        # Verify chunk indices are sequential
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, idx)

if __name__ == "__main__":
    unittest.main()
