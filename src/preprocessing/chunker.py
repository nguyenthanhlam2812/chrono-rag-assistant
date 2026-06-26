import re
from typing import List, Dict, Any, Tuple

# A Markdown table row: a line that starts and ends with a pipe.
_RE_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$")


def _table_spans(text: str) -> List[Tuple[int, int]]:
    """Char spans (start, end) of contiguous Markdown table blocks.

    A block is two or more consecutive pipe-rows (header + separator/data).
    Returns an empty list when the text contains no tables, so callers can
    keep their original behaviour byte-for-byte on table-free documents.
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    run_start = None
    run_rows = 0
    for line in text.splitlines(keepends=True):
        if _RE_TABLE_ROW.match(line):
            if run_start is None:
                run_start = pos
            run_rows += 1
        else:
            if run_start is not None and run_rows >= 2:
                spans.append((run_start, pos))
            run_start = None
            run_rows = 0
        pos += len(line)
    if run_start is not None and run_rows >= 2:
        spans.append((run_start, pos))
    return spans


def chunk_document(doc: Dict[str, Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split a document into sliding window character-based chunks.
    Preserves metadata and generates unique chunk IDs.

    If a chunk boundary would fall inside a Markdown table, the boundary is
    pushed to the end of that table so rows are never split across chunks.
    Documents without tables are chunked exactly as before.
    """
    text = doc.get("text", "")
    doc_id = doc.get("doc_id", "")
    topic = doc.get("topic", "")
    title = doc.get("title", "")
    source_url = doc.get("source_url", "")
    year = doc.get("year", None)
    
    chunks = []
    
    if not text:
        return chunks
        
    start = 0
    chunk_index = 1
    spans = _table_spans(text)

    # In case the document is shorter than the chunk size
    if len(text) <= chunk_size:
        chunks.append({
            "chunk_id": f"{doc_id}_c{chunk_index:04d}",
            "doc_id": doc_id,
            "topic": topic,
            "title": title,
            "chunk_index": chunk_index,
            "text": text,
            "start_char": 0,
            "end_char": len(text),
            "source_url": source_url,
            "year": year
        })
        return chunks
        
    while start < len(text):
        end = start + chunk_size
        # If the boundary lands inside a table, extend to the table's end so
        # rows are never split. No-op when the document has no tables.
        for s0, s1 in spans:
            if s0 < end < s1:
                end = s1
                break
        chunk_text = text[start:end]

        chunks.append({
            "chunk_id": f"{doc_id}_c{chunk_index:04d}",
            "doc_id": doc_id,
            "topic": topic,
            "title": title,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "start_char": start,
            "end_char": min(end, len(text)),
            "source_url": source_url,
            "year": year
        })
        
        start += (chunk_size - chunk_overlap)
        chunk_index += 1
        
    return chunks
