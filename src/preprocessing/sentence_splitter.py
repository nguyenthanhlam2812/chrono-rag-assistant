import re
from typing import List, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("sentence_splitter")

# Regular expression pattern to split text into sentences
# Avoids splitting on abbreviations like e.g., i.e., Dr., etc.
SENTENCE_SPLIT_REGEX = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences using NLTK (if available) or regex fallback.
    """
    if not text:
        return []
        
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        logger.warning(f"NLTK sentence tokenization failed: {e}. Falling back to regex splitting.")
        # Fallback to regex splitting
        sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
        
    return [s.strip() for s in sentences if s.strip()]

def split_chunks_into_sentences(chunks: List[Dict[str, Any]], min_sentence_len: int = 15) -> List[Dict[str, Any]]:
    """
    Split each chunk into sentences and map metadata.
    """
    all_sentences = []
    sentence_index = 1
    
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        chunk_id = chunk.get("chunk_id", "")
        topic = chunk.get("topic", "")
        source_url = chunk.get("source_url", "")
        year = chunk.get("year", None)
        
        raw_sentences = split_into_sentences(chunk.get("text", ""))
        
        for sent_text in raw_sentences:
            # Filter short sentences
            if len(sent_text) < min_sentence_len:
                continue
                
            sent_dict = {
                "sentence_id": f"{doc_id}_s{sentence_index:04d}",
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "topic": topic,
                "sentence_index": sentence_index,
                "text": sent_text,
                "source_url": source_url,
                "year": year
            }
            all_sentences.append(sent_dict)
            sentence_index += 1
            
    return all_sentences
