from typing import List, Dict, Any

def chunk_document(doc: Dict[str, Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split a document into sliding window character-based chunks.
    Preserves metadata and generates unique chunk IDs.
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
