"""
Document parser module for ChronoRAG.

Provides file-type-specific text extraction. Each parser function
accepts a pathlib.Path and returns the extracted text as a plain str.
Unsupported extensions and missing files are signalled via
FileNotFoundError and ValueError respectively so the caller
(document_loader) can log and skip gracefully.
"""

from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger("document_parser")

# ---------------------------------------------------------------------------
# Extension -> parser mapping
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf"}


def parse_file(
    file_path: Path,
    doc_id: Optional[str] = None,
    *,
    extract_tables: bool = False,
    ocr: bool = False,
    pdf_backend: str = "pymupdf",
) -> str:
    """Dispatch to the correct parser based on file extension.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the raw document file.
    doc_id : str, optional
        Document identifier used for logging context.
    extract_tables : bool, default False
        For PDFs (PyMuPDF backend), detect tables and serialise them as
        Markdown (table cells are removed from the prose flow to avoid
        duplication). HTML tables are always serialised to Markdown.
    ocr : bool, default False
        For PDFs (PyMuPDF backend), run OCR on pages that carry no embedded
        text (scanned pages). Requires the Tesseract binary; degrades
        gracefully if absent.
    pdf_backend : str, default "pymupdf"
        ``"pymupdf"`` (fast, default) or ``"docling"`` (stronger academic-PDF
        structure/table/heading recovery; falls back to PyMuPDF if docling is
        not installed). The ``extract_tables``/``ocr`` flags apply only to the
        PyMuPDF backend -- docling handles tables and OCR natively.

    Returns
    -------
    str
        Extracted plain-text content.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist on disk.
    ValueError
        If the file extension is not in ``SUPPORTED_EXTENSIONS``.
    RuntimeError
        If the underlying parser encounters an unrecoverable error.
    """
    label = doc_id or str(file_path)

    if not file_path.exists():
        logger.warning(f"[{label}] File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(
            f"[{label}] Unsupported file extension '{ext}' for: {file_path}"
        )
        raise ValueError(f"Unsupported file extension: {ext}")

    try:
        if ext in (".txt", ".md"):
            return _parse_text(file_path)
        elif ext in (".html", ".htm"):
            return _parse_html(file_path)
        elif ext == ".pdf":
            if pdf_backend == "docling":
                return _parse_pdf_docling(file_path)
            return _parse_pdf(file_path, extract_tables=extract_tables, ocr=ocr)
        else:
            # Defensive: should never reach here.
            raise ValueError(f"Unsupported file extension: {ext}")
    except (FileNotFoundError, ValueError):
        raise  # re-raise domain errors as-is
    except Exception as exc:
        logger.error(
            f"[{label}] Parser error for {file_path} (ext={ext}): {exc}"
        )
        raise RuntimeError(
            f"Parser failed for {file_path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------

def _parse_text(file_path: Path) -> str:
    """Parse plain-text or Markdown files (UTF-8)."""
    return file_path.read_text(encoding="utf-8")


def _parse_html(file_path: Path) -> str:
    """Parse HTML files.

    Prefers trafilatura for main-content extraction (drops nav/sidebar/
    boilerplate, keeps tables and image captions as Markdown). Falls back to
    the BeautifulSoup pipeline when trafilatura is unavailable or finds no
    article content (e.g. fragments / non-article pages).
    """
    raw_bytes = file_path.read_bytes()
    main = _extract_html_main(raw_bytes)
    if main:
        return main
    return _parse_html_bs4(raw_bytes)


def _extract_html_main(raw_bytes: bytes) -> Optional[str]:
    """Extract the main article content as Markdown via trafilatura.

    Returns None if trafilatura is not installed, the page has no detectable
    article body, or extraction errors out -- callers then fall back to
    BeautifulSoup.
    """
    try:
        import trafilatura
    except ImportError:
        return None

    try:
        html = raw_bytes.decode("utf-8", errors="replace")
        result = trafilatura.extract(
            html,
            output_format="markdown",
            include_tables=True,
            include_images=True,
            include_comments=False,
            favor_recall=True,
        )
    except Exception as exc:  # noqa: BLE001 - never let extraction crash ingest
        logger.warning(f"trafilatura extraction failed; using BeautifulSoup: {exc}")
        return None

    return result or None


def _parse_html_bs4(raw_bytes: bytes) -> str:
    """BeautifulSoup HTML fallback.

    Strips ``<script>`` and ``<style>`` elements, turns ``<table>`` into a
    Markdown table, and replaces ``<img>`` with its alt/title text before
    extracting the remaining prose.
    """
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(raw_bytes, "html.parser")

    # Remove script and style blocks
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Replace <img> with its alt/title so the caption survives as text.
    for img in soup.find_all("img"):
        alt = (img.get("alt") or img.get("title") or "").strip()
        img.replace_with(NavigableString(f"[Image: {alt}]" if alt else ""))

    # Serialise <table> to Markdown so row/column structure is preserved.
    for table in soup.find_all("table"):
        md = _html_table_to_markdown(table)
        table.replace_with(NavigableString("\n" + md + "\n" if md else ""))

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _html_table_to_markdown(table) -> str:
    """Convert a BeautifulSoup ``<table>`` node into a Markdown pipe-table."""
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        values = [
            " ".join(cell.get_text(separator=" ").split()).replace("|", "\\|")
            for cell in cells
        ]
        rows.append(values)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    header = rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _parse_pdf(
    file_path: Path,
    extract_tables: bool = False,
    ocr: bool = False,
) -> str:
    """Parse PDF files using PyMuPDF (imported as ``fitz``).

    Extracts plain text page-by-page.  Does **not** extract internal
    metadata -- metadata still comes from ``metadata.csv``.

    When *extract_tables* is set, detected tables are serialised to Markdown
    and their cells are dropped from the prose flow (no duplication). When
    *ocr* is set, pages with no embedded text are OCR'd as a fallback.
    """
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(file_path)) as doc:
        for page in doc:
            page_text = _extract_pdf_page(
                page, fitz, extract_tables=extract_tables, ocr=ocr
            )
            if page_text:
                pages.append(page_text)

    return "\n".join(pages)


def _extract_pdf_page(page, fitz, extract_tables: bool, ocr: bool) -> str:
    """Extract one PDF page's text, optionally with tables and OCR."""
    native = page.get_text("text")

    # Scanned page (no embedded text) -> optional OCR fallback.
    if ocr and not native.strip():
        return _ocr_pdf_page(page) or ""

    if not extract_tables:
        return native

    return _pdf_page_with_tables(page, fitz)


def _pdf_page_with_tables(page, fitz) -> str:
    """Return page text with tables as Markdown, in reading order.

    Text blocks whose centre falls inside a detected table are skipped so the
    same cells are not emitted twice (once as prose, once as Markdown).
    Any failure falls back to plain ``get_text``.
    """
    try:
        tables = list(page.find_tables().tables)
    except Exception as exc:  # noqa: BLE001 - detection is best-effort
        logger.warning(f"find_tables failed; using plain text: {exc}")
        return page.get_text("text")

    if not tables:
        return page.get_text("text")

    table_rects = [fitz.Rect(t.bbox) for t in tables]
    items = []  # (top_y, text) tuples sorted into reading order

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, btext = block[0], block[1], block[2], block[3], block[4]
        if not btext.strip():
            continue
        center = fitz.Point((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        if any(rect.contains(center) for rect in table_rects):
            continue  # belongs to a table; emitted as Markdown below
        items.append((y0, btext.strip()))

    for table, rect in zip(tables, table_rects):
        try:
            md = table.to_markdown()
        except Exception:  # noqa: BLE001
            md = ""
        if md and md.strip():
            items.append((rect.y0, md.strip()))

    items.sort(key=lambda it: it[0])
    return "\n".join(text for _, text in items)


def _ocr_pdf_page(page) -> Optional[str]:
    """OCR a single page via PyMuPDF/Tesseract. Returns None if unavailable."""
    try:
        textpage = page.get_textpage_ocr(full=True)
        return page.get_text("text", textpage=textpage)
    except Exception as exc:  # noqa: BLE001 - tesseract may be absent
        logger.warning(f"OCR unavailable or failed for a page: {exc}")
        return None


# Cached across calls so the layout/table models load only once per run.
_DOCLING_CONVERTER = None


def _get_docling_converter():
    """Lazily build and cache a docling DocumentConverter.

    OCR is disabled (academic PDFs carry embedded text; RapidOCR is the slow,
    memory-hungry stage that caused std::bad_alloc on large pages). Table
    structure recovery -- a key reason to use docling -- stays on.
    """
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        _DOCLING_CONVERTER = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _DOCLING_CONVERTER


def _parse_pdf_docling(file_path: Path) -> str:
    """Parse a PDF with docling, exporting structure-aware Markdown.

    docling recovers headings, reading order, and tables far better than plain
    text extraction. If docling is not installed or conversion fails, this
    falls back to the PyMuPDF backend so ingest never breaks.
    """
    try:
        converter = _get_docling_converter()
    except ImportError:
        logger.warning(
            "docling not installed; falling back to PyMuPDF for %s", file_path
        )
        return _parse_pdf(file_path)

    try:
        result = converter.convert(str(file_path))
    except Exception as exc:  # noqa: BLE001 - any docling failure -> fallback
        logger.error(
            f"docling failed for {file_path}; falling back to PyMuPDF: {exc}"
        )
        return _parse_pdf(file_path)

    # Surface incomplete conversions instead of silently returning short text.
    status_name = getattr(getattr(result, "status", None), "name", "")
    if status_name == "FAILURE":
        logger.error(
            f"docling reported FAILURE for {file_path}; falling back to PyMuPDF"
        )
        return _parse_pdf(file_path)
    if status_name == "PARTIAL_SUCCESS":
        logger.warning(
            f"docling only partially converted {file_path} (some pages dropped, "
            f"likely out of memory). Content may be incomplete -- consider "
            f"pdf_backend=pymupdf or a machine with more RAM."
        )

    return result.document.export_to_markdown()
