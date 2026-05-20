import json
import re
from pathlib import Path
from typing import Dict, Any, List

def count_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())

def find_noise_markers(text: str) -> List[str]:
    """Return obvious processed-text noise markers using case-insensitive checks."""
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    markers = []
    noise_patterns = {
        "<script": "contains '<script'",
        "<style": "contains '<style'",
        "<svg": "contains '<svg'",
        "<!--": "contains HTML comment '<!--'",
        "img.shields.io": "contains 'img.shields.io'",
    }
    for pat, msg in noise_patterns.items():
        if pat in text_lower:
            markers.append(msg)
    return markers

def validate_processed_data(
    documents_path: Path,
    chunks_path: Path,
    sentences_path: Path,
    min_sentence_len: int = 15
) -> Dict[str, Any]:
    general_errors = []
    general_warnings = []
    
    # Maps for tracking errors/warnings per document
    doc_errors: Dict[str, List[str]] = {}
    doc_warnings: Dict[str, List[str]] = {}
    
    # Load documents
    documents = []
    doc_ids = set()
    doc_dict = {}
    
    if not documents_path.exists():
        general_errors.append(f"Documents file not found: {documents_path}")
    else:
        with open(documents_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                    doc_id = doc.get("doc_id")
                    if doc_id:
                        if doc_id in doc_ids:
                            general_errors.append(f"documents.jsonl: Duplicate doc_id found: {doc_id}")
                        doc_ids.add(doc_id)
                        doc_dict[doc_id] = doc
                        doc_errors[doc_id] = []
                        doc_warnings[doc_id] = []
                except json.JSONDecodeError as e:
                    general_errors.append(f"documents.jsonl: Line {line_idx} is not valid JSON: {str(e)}")

    # Load chunks
    chunks = []
    chunk_ids = set()
    chunk_dict = {}
    
    if not chunks_path.exists():
        general_errors.append(f"Chunks file not found: {chunks_path}")
    else:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id:
                        if chunk_id in chunk_ids:
                            general_errors.append(f"chunks.jsonl: Duplicate chunk_id found: {chunk_id}")
                        chunk_ids.add(chunk_id)
                        chunk_dict[chunk_id] = chunk
                except json.JSONDecodeError as e:
                    general_errors.append(f"chunks.jsonl: Line {line_idx} is not valid JSON: {str(e)}")

    # Load sentences
    sentences = []
    sentence_ids = set()
    
    if not sentences_path.exists():
        general_errors.append(f"Sentences file not found: {sentences_path}")
    else:
        with open(sentences_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    sent = json.loads(line)
                    sentences.append(sent)
                    sent_id = sent.get("sentence_id")
                    if sent_id:
                        if sent_id in sentence_ids:
                            general_errors.append(f"sentences.jsonl: Duplicate sentence_id found: {sent_id}")
                        sentence_ids.add(sent_id)
                except json.JSONDecodeError as e:
                    general_errors.append(f"sentences.jsonl: Line {line_idx} is not valid JSON: {str(e)}")

    # Helper to register doc error/warning
    def add_doc_error(doc_id: str, msg: str):
        if doc_id in doc_errors:
            doc_errors[doc_id].append(msg)
        else:
            general_errors.append(f"Unknown doc_id '{doc_id}': {msg}")

    def add_doc_warning(doc_id: str, msg: str):
        if doc_id in doc_warnings:
            doc_warnings[doc_id].append(msg)
        else:
            general_warnings.append(f"Unknown doc_id '{doc_id}': {msg}")

    # Validate documents
    required_doc_fields = ["doc_id", "title", "topic", "source_type", "source_url", "year", "authors", "text", "local_path", "retrieved_at"]
    for idx, doc in enumerate(documents, 1):
        doc_id = doc.get("doc_id")
        d_id_label = doc_id if doc_id else f"at index {idx}"
        
        # Required fields check
        for field in required_doc_fields:
            if field not in doc:
                if doc_id:
                    add_doc_error(doc_id, f"Missing required field: {field}")
                else:
                    general_errors.append(f"documents.jsonl: Document at index {idx} is missing required field: {field}")
        
        if not doc_id:
            continue
            
        # Text quality check
        d_text = doc.get("text")
        if d_text is None:
            add_doc_error(doc_id, "text field is None")
        elif not isinstance(d_text, str):
            add_doc_error(doc_id, "text field is not a string")
        elif not d_text.strip():
            add_doc_error(doc_id, "text field is empty")
        else:
            words = count_words(d_text)
            s_type = doc.get("source_type", "")
            if words < 100:
                add_doc_error(doc_id, f"Document word count is too low ({words} words < 100)")
            elif words < 500:
                if s_type and s_type.lower() in ["paper", "survey", "blog", "docs", "github"]:
                    add_doc_warning(doc_id, f"Document of source_type '{s_type}' has low word count ({words} words < 500)")

        # Noise checks for documents
        if isinstance(d_text, str):
            for msg in find_noise_markers(d_text):
                add_doc_error(doc_id, f"text {msg}")

    # Validate chunks
    required_chunk_fields = ["chunk_id", "doc_id", "topic", "title", "chunk_index", "text", "start_char", "end_char", "source_url", "year"]
    for idx, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        c_id_label = chunk_id if chunk_id else f"at index {idx}"
        
        # Required fields check
        for field in required_chunk_fields:
            if field not in chunk:
                msg = f"chunks.jsonl: Chunk {c_id_label} is missing required field: {field}"
                if doc_id:
                    add_doc_error(doc_id, msg)
                else:
                    general_errors.append(msg)

        if not doc_id:
            continue
            
        # Referential integrity check
        if doc_id not in doc_ids:
            general_errors.append(f"chunks.jsonl: Chunk {c_id_label} references unknown doc_id: {doc_id}")
            continue

        # Position checks
        start_char = chunk.get("start_char")
        end_char = chunk.get("end_char")
        chunk_idx = chunk.get("chunk_index")
        
        if start_char is not None:
            if not isinstance(start_char, int) or start_char < 0:
                add_doc_error(doc_id, f"Chunk {c_id_label} start_char must be non-negative integer, got: {start_char}")
        if end_char is not None and start_char is not None:
            if not isinstance(end_char, int) or end_char <= start_char:
                add_doc_error(doc_id, f"Chunk {c_id_label} end_char ({end_char}) must be greater than start_char ({start_char})")
        if chunk_idx is not None:
            if not isinstance(chunk_idx, int) or chunk_idx < 1:
                add_doc_error(doc_id, f"Chunk {c_id_label} chunk_index must be integer >= 1, got: {chunk_idx}")

        # Text quality & noise check for chunk
        c_text = chunk.get("text")
        if c_text is None:
            add_doc_error(doc_id, f"Chunk {c_id_label} text is None")
        elif not isinstance(c_text, str):
            add_doc_error(doc_id, f"Chunk {c_id_label} text is not a string")
        elif not c_text.strip():
            add_doc_error(doc_id, f"Chunk {c_id_label} text is empty")
        else:
            for msg in find_noise_markers(c_text):
                add_doc_error(doc_id, f"Chunk {c_id_label} {msg}")

    # Validate sentences
    required_sentence_fields = ["sentence_id", "doc_id", "chunk_id", "topic", "sentence_index", "text", "source_url", "year"]
    for idx, sent in enumerate(sentences, 1):
        sent_id = sent.get("sentence_id")
        doc_id = sent.get("doc_id")
        chunk_id = sent.get("chunk_id")
        s_id_label = sent_id if sent_id else f"at index {idx}"
        
        # Required fields check
        for field in required_sentence_fields:
            if field not in sent:
                msg = f"sentences.jsonl: Sentence {s_id_label} is missing required field: {field}"
                if doc_id:
                    add_doc_error(doc_id, msg)
                else:
                    general_errors.append(msg)
                    
        if not doc_id:
            continue

        # Referential integrity check
        if doc_id not in doc_ids:
            general_errors.append(f"sentences.jsonl: Sentence {s_id_label} references unknown doc_id: {doc_id}")
            continue

        if chunk_id and chunk_id not in chunk_ids:
            add_doc_error(doc_id, f"sentences.jsonl: Sentence {s_id_label} references unknown chunk_id: {chunk_id}")

        # Position checks
        sent_idx = sent.get("sentence_index")
        if sent_idx is not None:
            if not isinstance(sent_idx, int) or sent_idx < 1:
                add_doc_error(doc_id, f"Sentence {s_id_label} sentence_index must be integer >= 1, got: {sent_idx}")

        # Text quality, length & noise check for sentence
        s_text = sent.get("text")
        if s_text is None:
            add_doc_error(doc_id, f"Sentence {s_id_label} text is None")
        elif not isinstance(s_text, str):
            add_doc_error(doc_id, f"Sentence {s_id_label} text is not a string")
        elif not s_text.strip():
            add_doc_error(doc_id, f"Sentence {s_id_label} text is empty")
        else:
            if len(s_text) < min_sentence_len:
                add_doc_warning(doc_id, f"Sentence {s_id_label} length ({len(s_text)}) is less than min_sentence_len ({min_sentence_len})")
            
            for msg in find_noise_markers(s_text):
                add_doc_error(doc_id, f"Sentence {s_id_label} {msg}")

    # Build per-document statistics
    per_doc_stats = []
    for doc in documents:
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
            
        topic = doc.get("topic", "")
        source_type = doc.get("source_type", "")
        word_count = count_words(doc.get("text", ""))
        
        # Count chunks and sentences for this doc
        chunk_count = sum(1 for c in chunks if c.get("doc_id") == doc_id)
        sentence_count = sum(1 for s in sentences if s.get("doc_id") == doc_id)
        
        # Status calculation
        status = "PASS"
        if doc_errors.get(doc_id):
            status = "FAIL"
        elif doc_warnings.get(doc_id):
            status = "WARNING"
            
        per_doc_stats.append({
            "doc_id": doc_id,
            "topic": topic,
            "source_type": source_type,
            "word_count": word_count,
            "chunk_count": chunk_count,
            "sentence_count": sentence_count,
            "status": status
        })

    # Combine errors/warnings for the output report
    all_errors = list(general_errors)
    for doc_id, errors_list in doc_errors.items():
        for err in errors_list:
            all_errors.append(f"[{doc_id}] {err}")
            
    all_warnings = list(general_warnings)
    for doc_id, warnings_list in doc_warnings.items():
        for warn in warnings_list:
            all_warnings.append(f"[{doc_id}] {warn}")

    return {
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "total_sentences": len(sentences),
        "errors": all_errors,
        "warnings": all_warnings,
        "per_document_stats": per_doc_stats
    }
