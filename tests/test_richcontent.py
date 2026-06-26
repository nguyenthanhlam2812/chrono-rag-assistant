"""
Rich-content extraction tests: tables (PDF/HTML), formula preservation,
images (HTML alt-text + PDF OCR fallback), and table-aware chunking.

These cover the ingest/preprocess additions that let the corpus carry
tabular data and math without losing structure, while keeping the default
(flags off) pipeline output byte-for-byte unchanged.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.ingest.document_parser as dp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.document_parser import (
    parse_file,
    _html_table_to_markdown,
    _extract_html_main,
)
from src.preprocessing.cleaner import clean_text
from src.preprocessing.chunker import chunk_document, _table_spans


def _trafilatura_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("trafilatura") is not None


# ---------------------------------------------------------------------------
# Formula preservation (cleaner)
# ---------------------------------------------------------------------------

class TestFormulaPreservation(unittest.TestCase):
    """Math delimiters must survive paper/survey cleaning intact."""

    def test_display_math_dollar(self):
        r = clean_text(r"Loss $$\mathcal{L} = \sum_{i} x_i$$ done.", source_type="paper")
        self.assertIn(r"\sum", r)
        self.assertIn(r"\mathcal{L}", r)

    def test_bracket_display_math(self):
        r = clean_text(r"Then \[\frac{a}{b} = \alpha\] follows.", source_type="paper")
        self.assertIn(r"\frac{a}{b}", r)
        self.assertIn(r"\alpha", r)

    def test_paren_inline_math(self):
        r = clean_text(r"We use \(\beta_0 + \beta_1 x\) as the model.", source_type="survey")
        self.assertIn(r"\beta_0", r)

    def test_equation_environment(self):
        r = clean_text(
            r"\begin{equation}\nabla f = 0\end{equation} is the condition.",
            source_type="paper",
        )
        self.assertIn(r"\nabla", r)
        self.assertIn(r"\begin{equation}", r)

    def test_stray_layout_command_still_removed(self):
        """Non-math stray commands must still be stripped (regression)."""
        r = clean_text(r"$$\sum x$$ text \noindent more.", source_type="paper")
        self.assertIn(r"\sum", r)
        self.assertNotIn(r"\noindent", r)

    def test_currency_not_corrupted(self):
        r = clean_text("It costs $5 and then $10 total.", source_type="paper")
        self.assertIn("$5", r)
        self.assertIn("$10", r)

    def test_no_math_is_unchanged(self):
        """Text without math must clean exactly as before (no placeholders leak)."""
        r = clean_text("Plain paper text with no formulas at all.", source_type="paper")
        self.assertEqual(r, "Plain paper text with no formulas at all.")
        self.assertNotIn("", r)


# ---------------------------------------------------------------------------
# HTML tables and images
# ---------------------------------------------------------------------------

class TestHtmlTableToMarkdown(unittest.TestCase):
    def _parse(self, html: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".html", mode="wb", delete=False) as f:
            f.write(html.encode("utf-8"))
            f.flush()
            return parse_file(Path(f.name))

    def test_table_becomes_markdown(self):
        html = (
            "<table><tr><th>Model</th><th>Year</th></tr>"
            "<tr><td>BERT</td><td>2018</td></tr></table>"
        )
        result = self._parse(html)
        self.assertIn("| Model | Year |", result)
        self.assertIn("| --- | --- |", result)
        self.assertIn("| BERT | 2018 |", result)

    def test_image_alt_preserved(self):
        html = "<p>See</p><img src='x.png' alt='Architecture diagram'>"
        result = self._parse(html)
        self.assertIn("[Image: Architecture diagram]", result)

    def test_image_without_alt_dropped(self):
        html = "<p>Body</p><img src='x.png'>"
        result = self._parse(html)
        self.assertIn("Body", result)
        self.assertNotIn("x.png", result)

    def test_plain_html_unchanged(self):
        """HTML with no table/img behaves exactly like before."""
        result = self._parse("<html><body><p>Hello world</p></body></html>")
        self.assertEqual(result, "Hello world")

    def test_ragged_rows_padded(self):
        table = (
            "<table><tr><th>A</th><th>B</th><th>C</th></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )
        md = _html_table_to_markdown(__import__("bs4").BeautifulSoup(table, "html.parser").table)
        # Every row has 3 columns -> 4 pipes per line.
        for line in md.splitlines():
            self.assertEqual(line.count("|"), 4)


# ---------------------------------------------------------------------------
# PDF tables and OCR (PyMuPDF)
# ---------------------------------------------------------------------------

class TestTrafilaturaMainContent(unittest.TestCase):
    """trafilatura should keep article body and drop nav/sidebar boilerplate."""

    ARTICLE = (
        "<html><body>"
        "<nav><a href='/'>Home</a><a href='/docs'>Docs</a>"
        "<a href='/api'>API Reference</a><a href='/faq'>FAQs</a></nav>"
        "<article><h1>Getting Started</h1>"
        "<p>AutoGen is an open-source programming framework for building AI "
        "agents and facilitating cooperation among multiple agents to solve "
        "tasks of considerable complexity in real applications.</p>"
        "<p>It offers conversable agents, tool use, and human-in-the-loop "
        "workflows for accelerating agentic AI research and development.</p>"
        "</article>"
        "<footer>Copyright 2024 Example Corp</footer></body></html>"
    )

    def setUp(self):
        if not _trafilatura_available():
            self.skipTest("trafilatura not installed")

    def test_main_content_via_extract(self):
        result = _extract_html_main(self.ARTICLE.encode("utf-8"))
        self.assertIsNotNone(result)
        self.assertIn("Getting Started", result)
        self.assertIn("conversable agents", result)
        # Nav/footer boilerplate should be gone.
        self.assertNotIn("API Reference", result)
        self.assertNotIn("Copyright 2024", result)

    def test_parse_file_routes_through_trafilatura(self):
        with tempfile.NamedTemporaryFile(suffix=".html", mode="wb", delete=False) as f:
            f.write(self.ARTICLE.encode("utf-8"))
            f.flush()
            result = parse_file(Path(f.name))
        self.assertIn("Getting Started", result)
        self.assertNotIn("API Reference", result)


class TestPdfTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import fitz  # noqa: F401
            cls.fitz = True
        except ImportError:
            cls.fitz = False

    def _make_table_pdf(self) -> Path:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 50), "Intro paragraph.")
        cells = [("Model", "Year"), ("BERT", "2018"), ("GPT", "2020")]
        x0, y0, cw, ch = 72, 80, 120, 24
        for r, row in enumerate(cells):
            for c, val in enumerate(row):
                x, y = x0 + c * cw, y0 + r * ch
                page.draw_rect(fitz.Rect(x, y, x + cw, y + ch))
                page.insert_text((x + 4, y + 16), val)
        page.insert_text((72, 200), "Conclusion paragraph.")
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(tmp)
        doc.close()
        return Path(tmp)

    def test_extract_tables_emits_markdown(self):
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_table_pdf()
        txt = parse_file(pdf, extract_tables=True)
        compact = txt.replace(" ", "")
        self.assertIn("|BERT|2018|", compact)
        self.assertIn("Intro paragraph", txt)
        self.assertIn("Conclusion paragraph", txt)
        os.unlink(pdf)

    def test_extract_tables_no_duplication(self):
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_table_pdf()
        txt = parse_file(pdf, extract_tables=True)
        # Cell text appears only in the Markdown table, not also as prose.
        self.assertEqual(txt.count("BERT"), 1)
        os.unlink(pdf)

    def test_default_off_is_plain_text(self):
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_table_pdf()
        txt = parse_file(pdf)  # flags default off
        self.assertNotIn("|---|", txt.replace(" ", ""))
        self.assertIn("Intro paragraph", txt)
        os.unlink(pdf)

    def test_ocr_flag_degrades_gracefully(self):
        """ocr=True on a blank page must not crash even without Tesseract."""
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        import fitz
        doc = fitz.open()
        doc.new_page()  # blank, no embedded text
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(tmp)
        doc.close()
        result = parse_file(Path(tmp), ocr=True)
        self.assertIsInstance(result, str)
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Docling PDF backend (opt-in)
# ---------------------------------------------------------------------------

class TestDoclingBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import fitz  # noqa: F401
            cls.fitz = True
        except ImportError:
            cls.fitz = False

    def _make_text_pdf(self) -> Path:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello from a real PDF document.")
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(tmp)
        doc.close()
        return Path(tmp)

    def test_default_backend_does_not_call_docling(self):
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_text_pdf()
        with mock.patch.object(dp, "_parse_pdf_docling") as docling:
            out = parse_file(pdf)  # default backend = pymupdf
        docling.assert_not_called()
        self.assertIn("Hello from a real PDF", out)
        os.unlink(pdf)

    def test_docling_backend_dispatches(self):
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_text_pdf()
        with mock.patch.object(dp, "_parse_pdf_docling", return_value="DOCLING_OUT") as d:
            out = parse_file(pdf, pdf_backend="docling")
        d.assert_called_once()
        self.assertEqual(out, "DOCLING_OUT")
        os.unlink(pdf)

    def test_docling_falls_back_to_pymupdf_when_absent(self):
        """If docling cannot be imported, fall back to PyMuPDF (no crash)."""
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_text_pdf()
        with mock.patch.object(dp, "_get_docling_converter", side_effect=ImportError):
            out = dp._parse_pdf_docling(pdf)
        self.assertIn("Hello from a real PDF", out)
        os.unlink(pdf)

    def test_docling_failure_status_falls_back(self):
        """A FAILURE conversion falls back to PyMuPDF for complete content."""
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_text_pdf()
        status = mock.Mock(name="FAILURE")
        status.name = "FAILURE"
        conv = mock.Mock()
        conv.convert.return_value = mock.Mock(status=status)
        with mock.patch.object(dp, "_get_docling_converter", return_value=conv):
            out = dp._parse_pdf_docling(pdf)
        self.assertIn("Hello from a real PDF", out)  # from PyMuPDF fallback
        os.unlink(pdf)

    def test_docling_partial_success_returns_markdown(self):
        """PARTIAL_SUCCESS still returns docling output (with a warning logged)."""
        if not self.fitz:
            self.skipTest("PyMuPDF not installed")
        pdf = self._make_text_pdf()
        status = mock.Mock()
        status.name = "PARTIAL_SUCCESS"
        document = mock.Mock()
        document.export_to_markdown.return_value = "# Partial Markdown"
        conv = mock.Mock()
        conv.convert.return_value = mock.Mock(status=status, document=document)
        with mock.patch.object(dp, "_get_docling_converter", return_value=conv):
            out = dp._parse_pdf_docling(pdf)
        self.assertEqual(out, "# Partial Markdown")
        os.unlink(pdf)


# ---------------------------------------------------------------------------
# Table-aware chunking
# ---------------------------------------------------------------------------

class TestTableAwareChunking(unittest.TestCase):
    def test_no_table_spans_for_plain_text(self):
        self.assertEqual(_table_spans("just some plain text\nwith two lines"), [])

    def test_plain_text_chunking_unchanged(self):
        """Table-free docs keep the exact sliding-window stride."""
        text = "a" * 2500
        doc = {"doc_id": "d", "text": text}
        chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=200)
        starts = [c["start_char"] for c in chunks]
        self.assertEqual(starts, [0, 800, 1600, 2400])

    def test_table_not_split_across_chunks(self):
        import re
        table = "\n".join(f"| r{i} | c{i} |" for i in range(40))
        text = "x" * 900 + "\n| H1 | H2 |\n| --- | --- |\n" + table + "\n" + "y" * 900
        doc = {"doc_id": "t", "text": text}
        chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=200)

        def rows(s):
            return {l for l in s.splitlines() if re.match(r"^\s*\|.*\|\s*$", l)}

        all_rows = rows(text)
        self.assertTrue(
            any(all_rows.issubset(rows(c["text"])) for c in chunks),
            "No single chunk contains the whole table block",
        )
        # Bounds stay valid.
        for c in chunks:
            self.assertGreater(c["end_char"], c["start_char"])


if __name__ == "__main__":
    unittest.main()
