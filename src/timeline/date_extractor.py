import re
from typing import Dict, Any, Optional

def extract_date(sentence: str, document_year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract date/year from a sentence using regex patterns.
    
    Returns dict with keys:
    - date_text: str or None (the matched text, e.g. '2020', 'May 2020')
    - normalized_date: str or None (e.g. '2020', '2020-05')
    - extracted_year: int or None
    - date_confidence: float (0.0 to 1.0)
    - date_source: str ('sentence_regex' or 'document_year')
    
    Regex patterns to use (in priority order):
    1. Full date: r'\d{4}-\d{2}-\d{2}' -> confidence 1.0
    2. Month-Year: r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}' -> confidence 0.9
    3. Contextual year: r'(in|around|circa|since|after|before)\s+((?:19|20)\d{2})' -> confidence 0.85
    4. Plain year: r'\b((?:19|20)\d{2})\b' -> confidence 0.7
    5. Fallback: document_year if provided -> confidence 0.4, source='document_year'
    6. No date found -> all None, confidence 0.0
    """
    # 1. Full date
    full_date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', sentence)
    if full_date_match:
        date_text = full_date_match.group(1)
        return {
            "date_text": date_text,
            "normalized_date": date_text,
            "extracted_year": int(date_text[:4]),
            "date_confidence": 1.0,
            "date_source": "sentence_regex"
        }

    # 2. Month-Year
    month_map = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }
    month_year_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', sentence, re.IGNORECASE)
    if month_year_match:
        month_str = month_year_match.group(1).capitalize()
        year_str = month_year_match.group(2)
        month_num = month_map.get(month_str, "01")
        return {
            "date_text": month_year_match.group(0),
            "normalized_date": f"{year_str}-{month_num}",
            "extracted_year": int(year_str),
            "date_confidence": 0.9,
            "date_source": "sentence_regex"
        }

    # 3. Contextual year
    context_year_match = re.search(r'\b(in|around|circa|since|after|before)\s+((?:19|20)\d{2})\b', sentence, re.IGNORECASE)
    if context_year_match:
        year_str = context_year_match.group(2)
        return {
            "date_text": context_year_match.group(0),
            "normalized_date": year_str,
            "extracted_year": int(year_str),
            "date_confidence": 0.85,
            "date_source": "sentence_regex"
        }

    # 4. Plain year
    plain_year_match = re.search(r'\b((?:19|20)\d{2})\b', sentence)
    if plain_year_match:
        year_str = plain_year_match.group(1)
        return {
            "date_text": year_str,
            "normalized_date": year_str,
            "extracted_year": int(year_str),
            "date_confidence": 0.7,
            "date_source": "sentence_regex"
        }

    # 5. Fallback
    if document_year is not None:
        return {
            "date_text": str(document_year),
            "normalized_date": str(document_year),
            "extracted_year": int(document_year),
            "date_confidence": 0.4,
            "date_source": "document_year"
        }

    # 6. No date found
    return {
        "date_text": None,
        "normalized_date": None,
        "extracted_year": None,
        "date_confidence": 0.0,
        "date_source": None
    }
