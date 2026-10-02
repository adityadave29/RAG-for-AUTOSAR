import unittest
import tempfile
from pathlib import Path
from src.parser import parse_document, parse_text_file, DocumentPage

class TestParser(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_markdown_file(self):
        md_file = self.temp_path / "sample_spec.md"
        content = "# SWS CAN\n\n[SWS_Can_00001] CAN Driver shall initialize hardware."
        md_file.write_text(content, encoding="utf-8")

        pages = parse_document(md_file)
        self.assertEqual(len(pages), 1)
        self.assertIsInstance(pages[0], DocumentPage)
        self.assertEqual(pages[0].source_file, "sample_spec.md")
        self.assertEqual(pages[0].page_number, 1)
        self.assertIn("[SWS_Can_00001]", pages[0].text)

    def test_parse_empty_file(self):
        empty_file = self.temp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        pages = parse_document(empty_file)
        self.assertEqual(len(pages), 0)

    def test_parse_c_header_file(self):
        header_file = self.temp_path / "Can.h"
        header_content = "typedef enum { CAN_UNINIT, CAN_READY } Can_StateType;"
        header_file.write_text(header_content, encoding="utf-8")

        pages = parse_document(header_file)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].source_file, "Can.h")
        self.assertIn("Can_StateType", pages[0].text)

if __name__ == "__main__":
    unittest.main()
