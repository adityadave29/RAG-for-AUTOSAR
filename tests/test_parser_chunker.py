import unittest
from pathlib import Path
from src.parser import parse_document
from src.chunker import TechnicalDocumentChunker

class TestIngestionComponents(unittest.TestCase):
    def test_text_parser_and_chunker(self):
        sample_path = Path("docs/AUTOSAR_SWS_CAN_Sample.md")
        self.assertTrue(sample_path.exists(), "Sample doc should exist")

        pages = parse_document(sample_path)
        self.assertEqual(len(pages), 1)
        self.assertIn("Can_SetControllerMode", pages[0].text)

        chunker = TechnicalDocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document_pages(pages)
        self.assertGreater(len(chunks), 0)

        # Check context injection header
        first_chunk = chunks[0]
        self.assertTrue(first_chunk.text.startswith("[Source: AUTOSAR_SWS_CAN_Sample.md"))
        self.assertEqual(first_chunk.source_file, "AUTOSAR_SWS_CAN_Sample.md")

if __name__ == "__main__":
    unittest.main()
