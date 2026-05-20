"""
Sprint 2B - Source-Type-Aware Text Cleaning Tests.

Covers:
- Markdown code block with version/date is preserved.
- HTML script/style content is removed.
- Normal text keeps year/date/version strings.
- Markdown badge noise is removed.
- HTML comments are removed.
- LaTeX artefacts normalised for paper/survey.
- Fenced code blocks NOT removed for docs/github/blog.
- Control characters stripped.
- Whitespace normalised.
- Backward compat: clean_text() still works with no source_type.
- agent_005 processed chunks contain no img.shields.io or <!-- noise.
- agent_006 processed chunks contain no <script>, navbar, svg.
- Existing offline pipeline still runs.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cleaner import clean_text


# -----------------------------------------------------------------------
# Preservation tests - timeline signals must survive cleaning
# -----------------------------------------------------------------------

class TestPreservesTimelineSignals(unittest.TestCase):
    """Years, dates, versions, model names must not be destroyed."""

    def test_preserves_years(self):
        text = "Published in 2017 and updated in 2020, then 2024."
        result = clean_text(text, source_type="blog")
        self.assertIn("2017", result)
        self.assertIn("2020", result)
        self.assertIn("2024", result)

    def test_preserves_iso_dates(self):
        text = "Released on 2023-09-01 and again on 2024-01-15."
        result = clean_text(text, source_type="docs")
        self.assertIn("2023-09-01", result)
        self.assertIn("2024-01-15", result)

    def test_preserves_month_dates(self):
        text = "January 2024 saw a major update. Updated March 2023."
        result = clean_text(text, source_type="github")
        self.assertIn("January 2024", result)
        self.assertIn("March 2023", result)

    def test_preserves_version_strings(self):
        text = "Upgrade from v0.2.0 to v0.4.0. Use GPT-4 and BERT-base."
        result = clean_text(text, source_type="docs")
        self.assertIn("v0.2.0", result)
        self.assertIn("v0.4.0", result)
        self.assertIn("GPT-4", result)
        self.assertIn("BERT-base", result)

    def test_preserves_version_in_paper(self):
        text = "We evaluate GPT-4 (v2024-01-25) on the benchmark."
        result = clean_text(text, source_type="paper")
        self.assertIn("GPT-4", result)
        self.assertIn("2024-01-25", result)


# -----------------------------------------------------------------------
# Code block preservation
# -----------------------------------------------------------------------

class TestCodeBlockPreservation(unittest.TestCase):
    """Fenced code blocks must NOT be removed globally."""

    def test_code_block_preserved_docs(self):
        text = (
            "## Install\n\n"
            "```bash\n"
            "pip install autogen-agentchat~=0.2\n"
            "```\n\n"
            "Released in 2023.\n"
        )
        result = clean_text(text, source_type="docs")
        self.assertIn("```bash", result)
        self.assertIn("pip install autogen-agentchat~=0.2", result)
        self.assertIn("2023", result)

    def test_code_block_preserved_github(self):
        text = (
            "```python\n"
            "print('hello v2.0.0')\n"
            "```\n"
        )
        result = clean_text(text, source_type="github")
        self.assertIn("print('hello v2.0.0')", result)

    def test_code_block_preserved_blog(self):
        text = "```\ncurl -fsSL https://setup.agpt.co/install.sh\n```\n"
        result = clean_text(text, source_type="blog")
        self.assertIn("curl -fsSL", result)

    def test_code_block_preserved_paper(self):
        """Even for paper, fenced code blocks should survive (no global removal)."""
        text = "```python\nimport torch  # v2.0\n```\n"
        result = clean_text(text, source_type="paper")
        self.assertIn("import torch", result)


# -----------------------------------------------------------------------
# HTML noise removal
# -----------------------------------------------------------------------

class TestHtmlNoiseRemoval(unittest.TestCase):
    """Script, style, svg, nav, footer, header blocks must be stripped."""

    def test_removes_script(self):
        text = "Hello <script>alert('x')</script> world."
        result = clean_text(text)
        self.assertNotIn("alert", result)
        self.assertNotIn("<script>", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_removes_style(self):
        text = "Visible <style>body{color:red}</style> text."
        result = clean_text(text)
        self.assertNotIn("color:red", result)
        self.assertNotIn("<style>", result)
        self.assertIn("Visible", result)

    def test_removes_svg(self):
        text = "Before <svg width='100'><path d='M0 0'/></svg> after."
        result = clean_text(text)
        self.assertNotIn("<svg", result)
        self.assertNotIn("<path", result)
        self.assertIn("Before", result)
        self.assertIn("after", result)

    def test_removes_nav(self):
        text = "<nav>Menu Item 1 | Menu Item 2</nav>\nActual content here."
        result = clean_text(text)
        self.assertNotIn("Menu Item", result)
        self.assertIn("Actual content here", result)

    def test_removes_footer(self):
        text = "Body text.\n<footer>Copyright 2024</footer>"
        result = clean_text(text)
        self.assertNotIn("<footer>", result)
        self.assertIn("Body text", result)

    def test_removes_stray_html_tags(self):
        text = "<div class='wrapper'><p>Hello</p></div>"
        result = clean_text(text)
        self.assertNotIn("<div", result)
        self.assertNotIn("<p>", result)
        self.assertNotIn("</p>", result)
        self.assertIn("Hello", result)

    def test_replaces_html_entities(self):
        text = "A &amp; B &ndash; C &nbsp; D"
        result = clean_text(text)
        self.assertIn("A & B", result)
        self.assertIn("-", result)


# -----------------------------------------------------------------------
# HTML comment removal
# -----------------------------------------------------------------------

class TestHtmlCommentRemoval(unittest.TestCase):
    """HTML comments <!-- ... --> must be stripped."""

    def test_removes_single_line_comment(self):
        text = "Before <!-- comment --> After"
        result = clean_text(text)
        self.assertNotIn("<!--", result)
        self.assertNotIn("comment", result)
        self.assertIn("Before", result)
        self.assertIn("After", result)

    def test_removes_multiline_comment(self):
        text = "A\n<!-- keep\nthese\nlinks -->\nB"
        result = clean_text(text)
        self.assertNotIn("<!--", result)
        self.assertNotIn("keep", result)
        self.assertIn("A", result)
        self.assertIn("B", result)


# -----------------------------------------------------------------------
# Markdown badge removal
# -----------------------------------------------------------------------

class TestMarkdownBadgeRemoval(unittest.TestCase):
    """Shield.io badge lines should be cleaned."""

    def test_removes_badge_line(self):
        text = (
            "# Title\n\n"
            "[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=foo)](https://discord.gg/autogpt)\n"
            "Real content here.\n"
        )
        result = clean_text(text, source_type="github")
        self.assertNotIn("img.shields.io", result)
        self.assertIn("# Title", result)
        self.assertIn("Real content here", result)

    def test_removes_inline_badge(self):
        text = "Status: [![Build](https://img.shields.io/badge/build-passing-green)](url) done."
        result = clean_text(text)
        self.assertNotIn("img.shields.io", result)
        self.assertIn("Status:", result)

    def test_preserves_normal_image(self):
        text = "![Architecture](https://example.com/arch.png)"
        result = clean_text(text)
        self.assertIn("example.com/arch.png", result)


# -----------------------------------------------------------------------
# LaTeX / PDF artefact cleaning (paper/survey only)
# -----------------------------------------------------------------------

class TestLatexCleaning(unittest.TestCase):
    """LaTeX artefacts should be normalised for paper/survey."""

    def test_unwraps_textbf(self):
        result = clean_text(r"We use \textbf{BERT} for encoding.", source_type="paper")
        self.assertIn("BERT", result)
        self.assertNotIn("\\textbf", result)

    def test_unwraps_cite(self):
        result = clean_text(r"As shown in \cite{vaswani2017attention}.", source_type="paper")
        self.assertIn("vaswani2017attention", result)
        self.assertNotIn("\\cite", result)

    def test_removes_stray_commands(self):
        result = clean_text(r"Text \noindent more text.", source_type="paper")
        self.assertNotIn("\\noindent", result)
        self.assertIn("Text", result)
        self.assertIn("more text", result)

    def test_fixes_ligatures(self):
        result = clean_text("e\ufb03cient and e\ufb00ective", source_type="paper")
        self.assertIn("efficient", result)
        self.assertIn("effective", result)

    def test_latex_not_applied_to_docs(self):
        """LaTeX normalisation should NOT run for docs/github/blog."""
        text = r"Use \textbf{bold} in your docs."
        result = clean_text(text, source_type="docs")
        # The raw \textbf should remain (it's not HTML noise)
        self.assertIn("\\textbf", result)


# -----------------------------------------------------------------------
# Whitespace normalisation
# -----------------------------------------------------------------------

class TestWhitespaceNormalisation(unittest.TestCase):
    """Excessive whitespace and control chars should be collapsed."""

    def test_collapses_spaces(self):
        text = "Hello     world    here."
        result = clean_text(text)
        self.assertEqual(result, "Hello world here.")

    def test_collapses_blank_lines(self):
        text = "A\n\n\n\n\nB"
        result = clean_text(text)
        self.assertEqual(result, "A\n\nB")

    def test_strips_control_chars(self):
        text = "Hello\x00\x01\x02world"
        result = clean_text(text)
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertNotIn("\x00", result)

    def test_preserves_tabs_in_text(self):
        """Tabs are collapsed to single space, not removed."""
        text = "Col1\t\tCol2"
        result = clean_text(text)
        self.assertIn("Col1", result)
        self.assertIn("Col2", result)


# -----------------------------------------------------------------------
# Backward compatibility
# -----------------------------------------------------------------------

class TestBackwardCompat(unittest.TestCase):
    """clean_text() must still work when called with no source_type."""

    def test_no_source_type(self):
        result = clean_text("Hello   world")
        self.assertEqual(result, "Hello world")

    def test_none_source_type(self):
        result = clean_text("Hello   world", source_type=None)
        self.assertEqual(result, "Hello world")

    def test_empty_string(self):
        result = clean_text("")
        self.assertEqual(result, "")

    def test_empty_string_with_type(self):
        result = clean_text("", source_type="paper")
        self.assertEqual(result, "")


# -----------------------------------------------------------------------
# Integration: real corpus documents
# -----------------------------------------------------------------------

class TestRealCorpusCleaning(unittest.TestCase):
    """Verify cleaning on the actual agent_005 / agent_006 content."""

    @classmethod
    def setUpClass(cls):
        raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.agent_005_path = raw_dir / "ai_agent" / "agent_005.md"
        cls.agent_006_path = raw_dir / "ai_agent" / "agent_006.md"
        if not cls.agent_005_path.exists() or not cls.agent_006_path.exists():
            raise unittest.SkipTest("Raw corpus files not found")
        cls.agent_005_raw = cls.agent_005_path.read_text(encoding="utf-8")
        cls.agent_006_raw = cls.agent_006_path.read_text(encoding="utf-8")

    def test_agent_005_no_badge_noise(self):
        """agent_005 cleaned text should not contain img.shields.io."""
        cleaned = clean_text(self.agent_005_raw, source_type="github")
        self.assertNotIn("img.shields.io", cleaned)

    def test_agent_005_no_html_comments(self):
        cleaned = clean_text(self.agent_005_raw, source_type="github")
        self.assertNotIn("<!--", cleaned)

    def test_agent_005_preserves_content(self):
        """Core content and version/date signals survive cleaning."""
        cleaned = clean_text(self.agent_005_raw, source_type="github")
        self.assertIn("AutoGPT", cleaned)
        self.assertIn("powerful platform", cleaned)
        # Version-like strings in software requirements
        self.assertIn("20.10.0", cleaned)

    def test_agent_005_preserves_code_blocks(self):
        cleaned = clean_text(self.agent_005_raw, source_type="github")
        self.assertIn("curl -fsSL", cleaned)

    def test_agent_006_no_html_noise(self):
        cleaned = clean_text(self.agent_006_raw, source_type="docs")
        self.assertNotIn("<script", cleaned)
        self.assertNotIn("<svg", cleaned)

    def test_agent_006_preserves_content(self):
        cleaned = clean_text(self.agent_006_raw, source_type="docs")
        self.assertIn("AutoGen is an open-source programming framework", cleaned)

    def test_agent_006_preserves_version_string(self):
        """The 0.2 version reference in code block / URL must survive."""
        cleaned = clean_text(self.agent_006_raw, source_type="docs")
        self.assertIn("0.2", cleaned)

    def test_agent_006_preserves_code_blocks(self):
        cleaned = clean_text(self.agent_006_raw, source_type="docs")
        self.assertIn("pip install autogen-agentchat", cleaned)


# -----------------------------------------------------------------------
# Integration: pipeline end-to-end (regression)
# -----------------------------------------------------------------------

class TestPipelineRegression(unittest.TestCase):
    """Offline pipeline must still produce valid output after cleaner rewrite."""

    @classmethod
    def setUpClass(cls):
        from workflows.offline_pipeline import run_offline_pipeline
        run_offline_pipeline()
        cls.processed_dir = PROJECT_ROOT / "data" / "processed"

    def test_documents_exist(self):
        self.assertTrue((self.processed_dir / "documents.jsonl").exists())

    def test_chunks_exist(self):
        self.assertTrue((self.processed_dir / "chunks.jsonl").exists())

    def test_sentences_exist(self):
        self.assertTrue((self.processed_dir / "sentences.jsonl").exists())

    def test_chunks_no_badge_noise(self):
        """Processed chunks should not contain img.shields.io badge URLs."""
        import json
        chunks_path = self.processed_dir / "chunks.jsonl"
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self.assertNotIn(
                    "img.shields.io",
                    chunk.get("text", ""),
                    f"Badge noise in chunk {chunk.get('chunk_id')}",
                )

    def test_chunks_no_html_comments(self):
        import json
        chunks_path = self.processed_dir / "chunks.jsonl"
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self.assertNotIn(
                    "<!--",
                    chunk.get("text", ""),
                    f"HTML comment in chunk {chunk.get('chunk_id')}",
                )

    def test_chunks_preserve_autogpt_content(self):
        import json
        chunks_path = self.processed_dir / "chunks.jsonl"
        agent_005_texts = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                if chunk.get("doc_id") == "agent_005":
                    agent_005_texts.append(chunk["text"])
        full_text = " ".join(agent_005_texts)
        self.assertIn("AutoGPT", full_text)

    def test_chunks_preserve_autogen_content(self):
        import json
        chunks_path = self.processed_dir / "chunks.jsonl"
        agent_006_texts = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                if chunk.get("doc_id") == "agent_006":
                    agent_006_texts.append(chunk["text"])
        full_text = " ".join(agent_006_texts)
        self.assertIn("AutoGen is an open-source programming framework", full_text)


if __name__ == "__main__":
    unittest.main()
