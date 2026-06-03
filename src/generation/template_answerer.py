import re
from typing import List, Dict, Any

from src.indexing.bm25_index import expand_query_tokens
from src.utils.text import STOPWORDS, is_good_sentence


class TemplateAnswerer:
    def generate_answer(self, chunks: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
        if not chunks:
            return {
                "answer": "I'm sorry, but I couldn't find any relevant information about that in the local corpus. Please try another question or select a different topic.",
                "citations": []
            }

        # Extract query tokens for sentence selection
        query_tokens = []
        primary_query_tokens = []
        if query:
            query_cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', query.lower())
            tokens = [t for t in query_cleaned.split() if t]
            query_tokens = [t for t in tokens if t not in STOPWORDS]
            if not query_tokens:
                query_tokens = tokens
            primary_query_tokens = list(query_tokens)
            query_tokens = expand_query_tokens(" ".join(query_tokens))
        is_definition_query = bool(query and query.lower().strip().startswith(
            ("what is", "what are", "explain", "define")
        ))

        answer_parts = []
        citations = []
        seen_docs = set()
        total_sentences = 0

        for chunk in chunks[:3]:
            doc_id = chunk.get('doc_id')
            title = chunk.get('title')
            url = chunk.get('source_url', '')

            text = chunk.get('text', '').strip()
            
            # 1. Multi-line cleanups on the entire text first
            # Remove HTML comments
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            # Remove fenced code blocks
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            
            # 2. Split by block boundaries
            abstract_match = re.search(r'\babstract\b', text, flags=re.IGNORECASE)
            if abstract_match and abstract_match.start() < 1500:
                text = text[abstract_match.end():].strip()

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
                block_cleaned = _repair_pdf_hyphenation(block_cleaned)
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
                if len(s) > 650:
                    continue
                if _looks_incomplete_sentence(s):
                    continue
                
                # Count matches with query tokens
                matches = 0
                if query_tokens:
                    for token in query_tokens:
                        if token in s.lower():
                            matches += 1
                primary_matches = 0
                if primary_query_tokens:
                    for token in primary_query_tokens:
                        if token in s.lower():
                            primary_matches += 1
                    matches += primary_matches * 3

                if is_definition_query and primary_query_tokens and primary_matches == 0:
                    continue
                
                if matches > 0:
                    if is_definition_query and _looks_like_definition_sentence(s):
                        matches += 3
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
                if not is_definition_query:
                    for s in other_sentences:
                        if s not in selected:
                            selected.append(s)
                            if len(selected) >= max_sents:
                                break

            if selected:
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    citations.append({
                        "doc_id": doc_id,
                        "title": title,
                        "source_url": url
                    })
                chunk_summary = " ".join(selected)
                if not chunk_summary.endswith(('.', '!', '?')):
                    chunk_summary += '.'
                answer_parts.append(f"{chunk_summary} [{doc_id}]")
                total_sentences += len(selected)
                if is_definition_query and total_sentences >= 2:
                    break

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


def _looks_like_definition_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(
        signal in lowered
        for signal in (
            " is a ",
            " is an ",
            " was introduced",
            " we propose",
            " we introduce",
            " we present",
            " combines",
            " framework",
            " method",
        )
    )


def _looks_incomplete_sentence(sentence: str) -> bool:
    lowered = sentence.lower().strip(" .,:;")
    return lowered.endswith((
        " and",
        " or",
        " with",
        " to",
        " for",
        " by",
        " as",
        " in",
        " of",
        " the",
        " while",
        " because",
        " e.g",
    ))


def _repair_pdf_hyphenation(text: str) -> str:
    keep_hyphen_prefixes = {
        "fine",
        "general",
        "large",
        "open",
        "pre",
        "question",
        "state",
        "task",
    }

    def replace(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if left.lower() in keep_hyphen_prefixes:
            return f"{left}-{right}"
        return f"{left}{right}"

    return re.sub(r'\b([A-Za-z]{2,})-\s+([A-Za-z]{2,})\b', replace, text)
