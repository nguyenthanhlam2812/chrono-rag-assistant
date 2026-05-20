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


def parse_file(file_path: Path, doc_id: Optional[str] = None) -> str:
    """Dispatch to the correct parser based on file extension.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the raw document file.
    doc_id : str, optional
        Document identifier used for logging context.

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
            return _parse_pdf(file_path)
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

    Reads raw bytes so that BeautifulSoup can auto-detect encoding.
    Strips ``<script>`` and ``<style>`` elements before extracting text.
    """
    from bs4 import BeautifulSoup

    raw_bytes = file_path.read_bytes()
    soup = BeautifulSoup(raw_bytes, "html.parser")

    # Remove script and style blocks
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _parse_pdf(file_path: Path) -> str:
    """Parse PDF files using PyMuPDF (imported as ``fitz``).

    Extracts plain text page-by-page.  Does **not** extract internal
    metadata -- metadata still comes from ``metadata.csv``.
    """
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(file_path)) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                pages.append(page_text)

    return "\n".join(pages)
