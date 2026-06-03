from __future__ import annotations

import re
from typing import Any, Dict, Optional

MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

_FULL_DATE_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_CONTEXT_YEAR_RE = re.compile(
    r"\b(in|around|circa|since|after|before|by|during)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_PLAIN_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def extract_date(sentence: str, document_year: Optional[int] = None) -> Dict[str, Any]:
    """Extract a coarse normalized date from an event sentence.

    This intentionally stays rule-based for Sprint 4/6 explainability. It
    returns both the matched surface text and a normalized year/month string.
    """
    text = sentence or ""

    full_date = _FULL_DATE_RE.search(text)
    if full_date:
        date_text = full_date.group(0)
        return _result(
            date_text=date_text,
            normalized_date=date_text,
            extracted_year=int(full_date.group(1)),
            confidence=1.0,
            source="sentence_full_date",
        )

    month_year = _MONTH_YEAR_RE.search(text)
    if month_year:
        month = MONTHS[month_year.group(1).lower()]
        year = month_year.group(2)
        return _result(
            date_text=month_year.group(0),
            normalized_date=f"{year}-{month}",
            extracted_year=int(year),
            confidence=0.9,
            source="sentence_month_year",
        )

    context_year = _CONTEXT_YEAR_RE.search(text)
    if context_year:
        year = context_year.group(2)
        return _result(
            date_text=context_year.group(0),
            normalized_date=year,
            extracted_year=int(year),
            confidence=0.85,
            source="sentence_context_year",
        )

    plain_year = _PLAIN_YEAR_RE.search(text)
    if plain_year:
        year = plain_year.group(1)
        return _result(
            date_text=year,
            normalized_date=year,
            extracted_year=int(year),
            confidence=0.7,
            source="sentence_plain_year",
        )

    if document_year is not None:
        try:
            year_int = int(document_year)
        except (TypeError, ValueError):
            year_int = None
        if year_int is not None:
            return _result(
                date_text=str(year_int),
                normalized_date=str(year_int),
                extracted_year=year_int,
                confidence=0.4,
                source="document_year",
            )

    return _result(
        date_text=None,
        normalized_date=None,
        extracted_year=None,
        confidence=0.0,
        source=None,
    )


def _result(
    date_text: Optional[str],
    normalized_date: Optional[str],
    extracted_year: Optional[int],
    confidence: float,
    source: Optional[str],
) -> Dict[str, Any]:
    return {
        "date_text": date_text,
        "normalized_date": normalized_date,
        "extracted_year": extracted_year,
        "date_confidence": confidence,
        "date_source": source,
    }
