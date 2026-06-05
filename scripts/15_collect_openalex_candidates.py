"""Collect OpenAlex candidate papers for ChronoRAG corpus expansion.

OpenAlex is useful when arXiv/Semantic Scholar rate-limit requests. This script
collects metadata and open-access PDF URLs only; it does not download PDFs or
change metadata.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "candidate_sources_expansion_openalex.csv"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"


TOPIC_QUERIES: Dict[str, List[str]] = {
    "rag": [
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "dense passage retrieval",
        "open-domain question answering retrieval",
        "self-rag",
        "corrective retrieval augmented generation",
        "graph rag",
    ],
    "ai_agent": [
        "llm agents tool use",
        "language model agents",
        "autonomous agents large language models",
        "multi-agent language models",
        "react reasoning acting language model",
        "toolformer language model tools",
        "agentic ai",
    ],
    "knowledge_distillation": [
        "knowledge distillation language models",
        "model distillation transformers",
        "distilbert knowledge distillation",
        "tinybert knowledge distillation",
        "mobilebert knowledge distillation",
        "minilm knowledge distillation",
        "language model compression distillation",
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
    "doi",
    "citation_count",
    "open_access_status",
    "source_name",
    "priority",
    "status",
    "collection_action",
    "local_path",
    "retrieved_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect OpenAlex candidate sources.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Candidate CSV output path")
    parser.add_argument("--max-per-topic", type=int, default=170, help="Target rows per topic")
    parser.add_argument("--per-page", type=int, default=100, help="OpenAlex rows per query")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--mailto", default="", help="Optional email for OpenAlex polite pool")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_rows: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()
    seen_titles: Set[str] = set()

    for topic, queries in TOPIC_QUERIES.items():
        topic_rows: List[Dict[str, str]] = []
        for query in queries:
            if len(topic_rows) >= args.max_per_topic:
                break
            works = search_works(query, args.per_page, args.mailto)
            topic_rows.extend(
                build_rows(
                    topic=topic,
                    works=works,
                    seen_ids=seen_ids,
                    seen_titles=seen_titles,
                    retrieved_at=retrieved_at,
                    start_index=len(topic_rows) + 1,
                )
            )
            topic_rows = topic_rows[: args.max_per_topic]
            if args.sleep > 0:
                time.sleep(args.sleep)
        all_rows.extend(reindex_topic(topic_rows, topic))
        print(f"{topic}: kept {len(topic_rows)} candidates")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} candidates to {output}")


def search_works(query: str, per_page: int, mailto: str) -> List[Dict[str, Any]]:
    params = {
        "search": query,
        "filter": "is_oa:true,has_abstract:true",
        "per-page": str(min(max(per_page, 1), 200)),
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "display_name",
                "publication_year",
                "publication_date",
                "authorships",
                "open_access",
                "primary_location",
                "cited_by_count",
                "type",
            ]
        ),
    }
    if mailto.strip():
        params["mailto"] = mailto.strip()
    try:
        response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  OpenAlex request failed for '{query}': {exc}")
        return []
    return list(response.json().get("results") or [])


def build_rows(
    *,
    topic: str,
    works: Iterable[Dict[str, Any]],
    seen_ids: Set[str],
    seen_titles: Set[str],
    retrieved_at: str,
    start_index: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for work in works:
        work_id = str(work.get("id") or "")
        title = clean(work.get("title") or work.get("display_name") or "")
        title_key = normalize_title(title)
        pdf_url = best_pdf_url(work)
        if not work_id or not title or not pdf_url:
            continue
        if work_id in seen_ids or title_key in seen_titles:
            continue
        if not topic_relevant(topic, title):
            continue
        seen_ids.add(work_id)
        seen_titles.add(title_key)

        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        open_access = work.get("open_access") or {}
        citation_count = int(work.get("cited_by_count") or 0)
        authors = "; ".join(
            clean((authorship.get("author") or {}).get("display_name") or "")
            for authorship in work.get("authorships") or []
        )

        rows.append(
            {
                "doc_id": f"{topic_prefix(topic)}_{start_index + len(rows):04d}",
                "topic": topic,
                "title": title,
                "source_type": "paper",
                "source_url": str(primary_location.get("landing_page_url") or work_id),
                "pdf_url": pdf_url,
                "year": str(work.get("publication_year") or ""),
                "published_date": str(work.get("publication_date") or ""),
                "authors": authors,
                "provider": "openalex",
                "provider_id": work_id.rsplit("/", 1)[-1],
                "doi": str(work.get("doi") or ""),
                "citation_count": str(citation_count),
                "open_access_status": str(open_access.get("oa_status") or ""),
                "source_name": str(source.get("display_name") or ""),
                "priority": priority_for(citation_count),
                "status": "candidate",
                "collection_action": "download_pdf_later",
                "local_path": "",
                "retrieved_at": retrieved_at,
                "notes": "OpenAlex OA PDF candidate; needs manual review before metadata.csv approval",
            }
        )
    return rows


def best_pdf_url(work: Dict[str, Any]) -> str:
    primary_location = work.get("primary_location") or {}
    open_access = work.get("open_access") or {}
    return str(primary_location.get("pdf_url") or open_access.get("oa_url") or "").strip()


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
    return {"rag": "oarag", "ai_agent": "oaagent", "knowledge_distillation": "oakd"}[topic]


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
