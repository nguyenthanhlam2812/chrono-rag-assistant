import re
from typing import List, Dict, Any

def is_good_sentence(s: str) -> bool:
    s_clean = s.strip()
    if len(s_clean) < 15:
        return False
    # Check if first alphabetical character is lowercase
    first_alpha = re.search(r'[a-zA-Z]', s_clean)
    if first_alpha and first_alpha.group(0).islower():
        return False
    # Exclude code or remnant signatures
    bad_signals = ['shields.io', 'github.com', 'http', 'www.', '.py', '.html', 'img.']
    if any(bad in s_clean.lower() for bad in bad_signals):
        return False
    return True

class TemplateAnswerer:
    def generate_answer(self, chunks: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
        if not chunks:
            return {
                "answer": "I'm sorry, but I couldn't find any relevant information about that in the local corpus. Please try another question or select a different topic.",
                "citations": []
            }

        # Extract query tokens for sentence selection
        query_tokens = []
        if query:
            query_cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', query.lower())
            tokens = [t for t in query_cleaned.split() if t]
            stopwords = {
                "is", "what", "the", "a", "an", "and", "or", "in", "on", "at", 
                "for", "with", "about", "to", "of", "how", "why", "where", "when", 
                "who", "whom", "whose", "which", "are", "do", "does", "did", "can",
                "could", "should", "would", "you", "your", "my", "our", "their"
            }
            query_tokens = [t for t in tokens if t not in stopwords]
            if not query_tokens:
                query_tokens = tokens

        answer_parts = []
        citations = []
        seen_docs = set()
        total_sentences = 0

        for chunk in chunks[:3]:
            doc_id = chunk.get('doc_id')
            title = chunk.get('title')
            url = chunk.get('source_url', '')

            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                citations.append({
                    "doc_id": doc_id,
                    "title": title,
                    "source_url": url
                })

            text = chunk.get('text', '').strip()
            
            # 1. Multi-line cleanups on the entire text first
            # Remove HTML comments
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            # Remove fenced code blocks
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            
            # 2. Split by block boundaries
            raw_blocks = re.split(r'\n\s*\n|\n\s*(?=[#\-\*\+])|\n\s*(?=\d+\.\s)', text)
            sentences = []
            
            for block in raw_blocks:
                # Replace single line wrap newlines with space
                block = block.replace('\n', ' ')
                # Clean block content
                # Remove bold/italic markdown asterisks
                block_cleaned = re.sub(r'\*+', '', block)
                block_cleaned = re.sub(r'\[\!\[.*?\]\(.*?\)\]\(.*?\)', '', block_cleaned)
                block_cleaned = re.sub(r'\!\[.*?\]\(.*?\)', '', block_cleaned)
                block_cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', block_cleaned)
                block_cleaned = re.sub(r'\&[a-z0-9\#]+;', ' ', block_cleaned)
                block_cleaned = re.sub(r'^#+\s+', '', block_cleaned)
                block_cleaned = re.sub(r'\s+', ' ', block_cleaned).strip()
                
                if not block_cleaned:
                    continue
                    
                # Split block by sentence boundary punctuation
                parts = re.split(r'(?<=[.!?])\s+', block_cleaned)
                for part in parts:
                    part_clean = part.strip()
                    if part_clean:
                        sentences.append(part_clean)

            matched_sentences = []
            other_sentences = []
            
            for s in sentences:
                if not is_good_sentence(s):
                    continue
                
                # Count matches with query tokens
                matches = 0
                if query_tokens:
                    for token in query_tokens:
                        if token in s.lower():
                            matches += 1
                
                if matches > 0:
                    matched_sentences.append((matches, s))
                else:
                    other_sentences.append(s)

            # Sort matched sentences by matches count descending
            matched_sentences.sort(key=lambda x: x[0], reverse=True)

            # Determine how many sentences to select from this chunk
            max_sents = 2 if total_sentences == 0 else 1
            
            selected = []
            for _, s in matched_sentences:
                selected.append(s)
                if len(selected) >= max_sents:
                    break
                    
            if len(selected) < max_sents:
                for s in other_sentences:
                    if s not in selected:
                        selected.append(s)
                        if len(selected) >= max_sents:
                            break

            if selected:
                chunk_summary = " ".join(selected)
                if not chunk_summary.endswith(('.', '!', '?')):
                    chunk_summary += '.'
                answer_parts.append(f"{chunk_summary} [{doc_id}]")
                total_sentences += len(selected)

        answer_text = " ".join(answer_parts)
        if not answer_text:
            return {
                "answer": "I'm sorry, but I couldn't find any relevant information about that in the local corpus. Please try another question or select a different topic.",
                "citations": []
            }

        return {
            "answer": answer_text,
            "citations": citations
        }
