"""Collect Semantic Scholar candidate papers for ChronoRAG expansion.

This is a fallback/companion to the arXiv collector. It searches Semantic
Scholar for open-access PDFs and writes a review manifest. It does not download
PDF files or modify metadata.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "candidate_sources_expansion_s2.csv"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


TOPIC_QUERIES: Dict[str, List[str]] = {
    "rag": [
        "retrieval augmented generation",
        "retrieval-augmented generation large language models",
        "dense passage retrieval open domain question answering",
        "self-rag retrieval augmented generation",
        "corrective retrieval augmented generation",
        "graph rag retrieval augmented generation",
    ],
    "ai_agent": [
        "llm agents tool use",
        "language model agents",
        "autonomous agents large language models",
        "multi-agent language models",
        "react reasoning acting language model",
        "toolformer language model tools",
    ],
    "knowledge_distillation": [
        "knowledge distillation language models",
        "model distillation transformers",
        "distilbert knowledge distillation",
        "tinybert knowledge distillation",
        "mobilebert knowledge distillation",
        "minilm knowledge distillation",
    ],
}


FIELDNAMES = [
    "doc_id",
    "topic",
    "title",
    "source_type",
    "source_url",
    "pdf_url",
    "year",
    "published_date",
    "authors",
    "provider",
    "provider_id",
    "citation_count",
    "fields_of_study",
    "publication_types",
    "priority",
    "status",
    "collection_action",
    "local_path",
    "retrieved_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Semantic Scholar candidate sources.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Candidate CSV output path")
    parser.add_argument("--max-per-topic", type=int, default=170, help="Target rows per topic")
    parser.add_argument("--limit-per-query", type=int, default=100, help="S2 results per query")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--api-key", default="", help="Optional Semantic Scholar API key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()
    seen_titles: Set[str] = set()

    for topic, queries in TOPIC_QUERIES.items():
        topic_rows: List[Dict[str, str]] = []
        for query in queries:
            if len(topic_rows) >= args.max_per_topic:
                break
            papers = search_papers(query, args.limit_per_query, args.api_key)
            topic_rows.extend(
                build_rows(
                    topic=topic,
                    papers=papers,
                    seen_ids=seen_ids,
                    seen_titles=seen_titles,
                    retrieved_at=retrieved_at,
                    start_index=len(topic_rows) + 1,
                )
            )
            topic_rows = topic_rows[: args.max_per_topic]
            if args.sleep > 0:
                time.sleep(args.sleep)
        rows.extend(reindex_topic(topic_rows, topic))
        print(f"{topic}: kept {len(topic_rows)} candidates")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} candidates to {output}")


def search_papers(query: str, limit: int, api_key: str) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "limit": str(min(max(limit, 1), 100)),
        "fields": ",".join(
            [
                "paperId",
                "title",
                "url",
                "year",
                "publicationDate",
                "authors",
                "openAccessPdf",
                "citationCount",
                "fieldsOfStudy",
                "publicationTypes",
            ]
        ),
        "fieldsOfStudy": "Computer Science",
        "openAccessPdf": "",
    }
    headers = {"User-Agent": "ChronoRAG/0.1 student corpus expansion"}
    if api_key.strip():
        headers["x-api-key"] = api_key.strip()
    url = f"{S2_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    try:
        response = requests.get(url, headers=headers, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  Semantic Scholar request failed for '{query}': {exc}")
        return []
    return list(response.json().get("data") or [])


def build_rows(
    *,
    topic: str,
    papers: Iterable[Dict[str, Any]],
    seen_ids: Set[str],
    seen_titles: Set[str],
    retrieved_at: str,
    start_index: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for paper in papers:
        paper_id = str(paper.get("paperId") or "")
        title = clean(paper.get("title") or "")
        title_key = normalize_title(title)
        pdf_url = ((paper.get("openAccessPdf") or {}).get("url") or "").strip()
        if not paper_id or not title or not pdf_url:
            continue
        if paper_id in seen_ids or title_key in seen_titles:
            continue
        if not topic_relevant(topic, title):
            continue
        seen_ids.add(paper_id)
        seen_titles.add(title_key)

        authors = "; ".join(clean(a.get("name") or "") for a in paper.get("authors") or [])
        citation_count = int(paper.get("citationCount") or 0)
        year = str(paper.get("year") or "")
        published_date = str(paper.get("publicationDate") or year or "")
        rows.append(
            {
                "doc_id": f"{topic_prefix(topic)}_{start_index + len(rows):04d}",
                "topic": topic,
                "title": title,
                "source_type": "paper",
                "source_url": str(paper.get("url") or ""),
                "pdf_url": pdf_url,
                "year": year,
                "published_date": published_date,
                "authors": authors,
                "provider": "semantic_scholar",
                "provider_id": paper_id,
                "citation_count": str(citation_count),
                "fields_of_study": "; ".join(paper.get("fieldsOfStudy") or []),
                "publication_types": "; ".join(paper.get("publicationTypes") or []),
                "priority": priority_for(citation_count),
                "status": "candidate",
                "collection_action": "download_pdf_later",
                "local_path": "",
                "retrieved_at": retrieved_at,
                "notes": "Semantic Scholar OA PDF candidate; needs manual review before metadata.csv approval",
            }
        )
    return rows


def topic_relevant(topic: str, title: str) -> bool:
    t = title.lower()
    if topic == "rag":
        return any(term in t for term in ("retrieval", "rag", "open-domain", "question answering", "dense passage"))
    if topic == "ai_agent":
        return any(term in t for term in ("agent", "tool", "autonomous", "reasoning", "acting", "multi-agent"))
    return any(term in t for term in ("distillation", "distil", "tinybert", "mobilebert", "minilm", "compression"))


def reindex_topic(rows: List[Dict[str, str]], topic: str) -> List[Dict[str, str]]:
    prefix = topic_prefix(topic)
    for index, row in enumerate(rows, start=1):
        row["doc_id"] = f"{prefix}_{index:04d}"
    return rows


def topic_prefix(topic: str) -> str:
    return {"rag": "s2rag", "ai_agent": "s2agent", "knowledge_distillation": "s2kd"}[topic]


def priority_for(citation_count: int) -> str:
    if citation_count >= 500:
        return "P0"
    if citation_count >= 100:
        return "P1"
    return "P2"


def clean(value: str) -> str:
    return " ".join(str(value).split())


def normalize_title(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in value).strip()


if __name__ == "__main__":
    main()
