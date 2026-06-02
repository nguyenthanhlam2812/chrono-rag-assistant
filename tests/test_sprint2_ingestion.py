"""
Sprint 2A - Parser and Ingestion Tests.

Covers:
- .txt parse
- .md parse
- .html parse with script/style removal
- Unsupported extension (skip without crash)
- Missing file (skip without crash)
- PDF parser smoke test (tiny in-memory PDF via PyMuPDF)
- Regression: current 5 metadata documents still load
"""

import sys
import unittest
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.document_parser import parse_file, SUPPORTED_EXTENSIONS
from src.ingest.document_loader import load_documents


class TestParserTxt(unittest.TestCase):
    """Tests for plain-text parsing."""

    def test_parse_txt(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write("Hello, ChronoRAG!")
            f.flush()
            result = parse_file(Path(f.name))
        self.assertEqual(result, "Hello, ChronoRAG!")

    def test_parse_txt_unicode(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write("Ti\u1ebfng Vi\u1ec7t -- \u00f1 -- \u65e5\u672c\u8a9e")
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("Ti\u1ebfng Vi\u1ec7t", result)

    def test_parse_txt_empty(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write("")
            f.flush()
            result = parse_file(Path(f.name))
        self.assertEqual(result, "")


class TestParserMd(unittest.TestCase):
    """Tests for Markdown parsing (treated as plain text)."""

    def test_parse_md_basic(self):
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write("# Heading\n\nSome **bold** text.\n")
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("# Heading", result)
        self.assertIn("**bold**", result)

    def test_parse_md_preserves_code_blocks(self):
        content = "# Title\n\n```python\nprint('hello')\n```\n"
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(content)
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("print('hello')", result)


class TestParserHtml(unittest.TestCase):
    """Tests for HTML parsing with script/style removal."""

    def test_parse_html_basic(self):
        html = "<html><body><p>Hello world</p></body></html>"
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="wb", delete=False
        ) as f:
            f.write(html.encode("utf-8"))
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("Hello world", result)

    def test_parse_html_strips_script(self):
        html = (
            "<html><head><script>alert('x')</script></head>"
            "<body><p>Content</p></body></html>"
        )
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="wb", delete=False
        ) as f:
            f.write(html.encode("utf-8"))
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("Content", result)
        self.assertNotIn("alert", result)
        self.assertNotIn("<script>", result)

    def test_parse_html_strips_style(self):
        html = (
            "<html><head><style>body{color:red}</style></head>"
            "<body><p>Visible</p></body></html>"
        )
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="wb", delete=False
        ) as f:
            f.write(html.encode("utf-8"))
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("Visible", result)
        self.assertNotIn("color:red", result)

    def test_parse_html_strips_both_script_and_style(self):
        html = (
            "<html><head>"
            "<script>var x = 1;</script>"
            "<style>.cls { display: none; }</style>"
            "</head><body>"
            "<nav>Navbar</nav>"
            "<p>Main content here</p>"
            "<footer>Footer</footer>"
            "</body></html>"
        )
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="wb", delete=False
        ) as f:
            f.write(html.encode("utf-8"))
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("Main content here", result)
        self.assertNotIn("var x = 1", result)
        self.assertNotIn("display: none", result)

    def test_parse_htm_extension(self):
        html = "<p>Works with .htm too</p>"
        with tempfile.NamedTemporaryFile(
            suffix=".htm", mode="wb", delete=False
        ) as f:
            f.write(html.encode("utf-8"))
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("Works with .htm too", result)

    def test_parse_html_encoding_detection(self):
        """HTML parser reads bytes and lets BeautifulSoup detect encoding."""
        html_bytes = (
            '<html><head><meta charset="utf-8"></head>'
            "<body><p>\u00dcn\u00efc\u00f6d\u00e9</p></body></html>"
        ).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="wb", delete=False
        ) as f:
            f.write(html_bytes)
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("\u00dcn\u00efc\u00f6d\u00e9", result)


class TestParserUnsupportedExtension(unittest.TestCase):
    """Unsupported file extensions should raise ValueError."""

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(
            suffix=".xyz", mode="w", delete=False
        ) as f:
            f.write("data")
            f.flush()
            with self.assertRaises(ValueError):
                parse_file(Path(f.name))

    def test_unsupported_docx(self):
        with tempfile.NamedTemporaryFile(
            suffix=".docx", mode="w", delete=False
        ) as f:
            f.write("data")
            f.flush()
            with self.assertRaises(ValueError):
                parse_file(Path(f.name))


class TestParserMissingFile(unittest.TestCase):
    """Missing files should raise FileNotFoundError."""

    def test_missing_file(self):
        fake_path = Path(tempfile.gettempdir()) / "nonexistent_file_12345.txt"
        with self.assertRaises(FileNotFoundError):
            parse_file(fake_path)

    def test_missing_pdf(self):
        fake_path = Path(tempfile.gettempdir()) / "nonexistent_paper.pdf"
        with self.assertRaises(FileNotFoundError):
            parse_file(fake_path)


class TestParserPdf(unittest.TestCase):
    """PDF parser smoke test - creates a tiny PDF in memory via PyMuPDF."""

    @classmethod
    def setUpClass(cls):
        """Try to import fitz; skip all PDF tests if unavailable."""
        try:
            import fitz  # noqa: F401
            cls.fitz_available = True
        except ImportError:
            cls.fitz_available = False

    def _make_tiny_pdf(self, text: str = "Hello from PDF") -> Path:
        """Create a minimal single-page PDF and return its path."""
        import fitz

        doc = fitz.open()  # new empty document
        page = doc.new_page()
        page.insert_text((72, 72), text)
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        import os
        os.close(fd)  # close fd so PyMuPDF can write
        doc.save(tmp_path)
        doc.close()
        return Path(tmp_path)

    def test_parse_pdf_smoke(self):
        if not self.fitz_available:
            self.skipTest("PyMuPDF (fitz) not installed")
        pdf_path = self._make_tiny_pdf("Hello from PDF")
        result = parse_file(pdf_path)
        self.assertIsInstance(result, str)
        self.assertIn("Hello from PDF", result)

    def test_parse_pdf_multipage(self):
        if not self.fitz_available:
            self.skipTest("PyMuPDF (fitz) not installed")
        import fitz
        import os

        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1}")
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(tmp_path)
        doc.close()
        result = parse_file(Path(tmp_path))
        self.assertIn("Page 1", result)
        self.assertIn("Page 2", result)
        self.assertIn("Page 3", result)

    def test_parse_pdf_empty(self):
        if not self.fitz_available:
            self.skipTest("PyMuPDF (fitz) not installed")
        import fitz
        import os

        doc = fitz.open()
        doc.new_page()  # blank page, no text
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(tmp_path)
        doc.close()
        result = parse_file(Path(tmp_path))
        self.assertIsInstance(result, str)


class TestParserReturnType(unittest.TestCase):
    """All parser functions must return str."""

    def test_return_type_txt(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write("text")
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIsInstance(result, str)

    def test_return_type_md(self):
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write("# md")
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIsInstance(result, str)

    def test_return_type_html(self):
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="wb", delete=False
        ) as f:
            f.write(b"<p>html</p>")
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIsInstance(result, str)


class TestSupportedExtensions(unittest.TestCase):
    """Verify the extension set is correct."""

    def test_supported_set(self):
        self.assertIn(".txt", SUPPORTED_EXTENSIONS)
        self.assertIn(".md", SUPPORTED_EXTENSIONS)
        self.assertIn(".html", SUPPORTED_EXTENSIONS)
        self.assertIn(".htm", SUPPORTED_EXTENSIONS)
        self.assertIn(".pdf", SUPPORTED_EXTENSIONS)
        self.assertNotIn(".docx", SUPPORTED_EXTENSIONS)
        self.assertNotIn(".csv", SUPPORTED_EXTENSIONS)


class TestDocumentLoaderRegression(unittest.TestCase):
    """Regression: current 5 metadata documents still load correctly."""

    @classmethod
    def setUpClass(cls):
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.metadata_csv = cls.raw_dir / "metadata.csv"
        if not cls.metadata_csv.exists():
            raise unittest.SkipTest("metadata.csv not found - cannot run regression test")

    def test_five_documents_load(self):
        documents = load_documents(self.metadata_csv, self.raw_dir)
        expected_ids = {"agent_001", "agent_005", "agent_006", "agent_012", "agent_013"}
        loaded_ids = {doc["doc_id"] for doc in documents}
        self.assertTrue(
            expected_ids.issubset(loaded_ids),
            f"Expected {expected_ids} to be a subset of {loaded_ids}",
        )

    def test_all_documents_have_text(self):
        documents = load_documents(self.metadata_csv, self.raw_dir)
        for doc in documents:
            self.assertIsInstance(doc["text"], str, f"{doc['doc_id']} text is not str")
            self.assertGreater(
                len(doc["text"]), 0, f"{doc['doc_id']} has empty text"
            )

    def test_documents_have_required_fields(self):
        required = {
            "doc_id", "title", "topic", "source_type", "source_url",
            "year", "local_path", "text", "retrieved_at",
        }
        documents = load_documents(self.metadata_csv, self.raw_dir)
        for doc in documents:
            for field in required:
                self.assertIn(field, doc, f"{doc['doc_id']} missing field '{field}'")

    def test_no_read_text_on_pdf(self):
        """Ensure document_loader.py does not import or call read_text."""
        import inspect
        from src.ingest import document_loader as mod

        source = inspect.getsource(mod)
        self.assertNotIn(
            "read_text(",
            source,
            "document_loader.py must not call read_text() - PDFs would break",
        )


class TestDocumentLoaderSkips(unittest.TestCase):
    """Loader should skip missing files and unsupported extensions gracefully."""

    def test_loader_skips_missing_file(self):
        """A metadata row pointing to a non-existent file should not crash."""
        import csv

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Write a minimal metadata CSV pointing to a missing file
            meta_path = tmpdir / "metadata.csv"
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "doc_id", "title", "topic", "source_type",
                        "source_url", "published_date", "year", "authors",
                        "local_path", "retrieved_at",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "doc_id": "missing_001",
                    "title": "Gone",
                    "topic": "test",
                    "source_type": "paper",
                    "source_url": "",
                    "published_date": "",
                    "year": "2024",
                    "authors": "",
                    "local_path": "does_not_exist.txt",
                    "retrieved_at": "2024-01-01",
                })
            docs = load_documents(meta_path, tmpdir)
            self.assertEqual(len(docs), 0)

    def test_loader_skips_unsupported_extension(self):
        """A metadata row with .xyz should skip gracefully."""
        import csv

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create the file so it exists but has an unsupported extension
            bad_file = tmpdir / "weird.xyz"
            bad_file.write_text("stuff", encoding="utf-8")

            meta_path = tmpdir / "metadata.csv"
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "doc_id", "title", "topic", "source_type",
                        "source_url", "published_date", "year", "authors",
                        "local_path", "retrieved_at",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "doc_id": "unsupported_001",
                    "title": "Bad ext",
                    "topic": "test",
                    "source_type": "other",
                    "source_url": "",
                    "published_date": "",
                    "year": "2024",
                    "authors": "",
                    "local_path": "weird.xyz",
                    "retrieved_at": "2024-01-01",
                })
            docs = load_documents(meta_path, tmpdir)
            self.assertEqual(len(docs), 0)


if __name__ == "__main__":
    unittest.main()
