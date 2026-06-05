"""Collect arXiv candidate papers for ChronoRAG corpus expansion.

This script only writes a lightweight candidate manifest. It does not download
PDF files. Use it to grow from the current 30-doc MVP toward 100/200/500 paper
corpora without polluting Git or bypassing manual quality review.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "candidate_sources_expansion.csv"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


@dataclass(frozen=True)
class TopicSpec:
    topic: str
    prefix: str
    query_terms: tuple[str, ...]
    strong_terms: tuple[str, ...]


TOPICS: tuple[TopicSpec, ...] = (
    TopicSpec(
        topic="rag",
        prefix="ragx",
        query_terms=(
            "retrieval augmented generation",
            "retrieval-augmented generation",
            "dense passage retrieval",
            "open-domain question answering retrieval",
            "self-rag",
            "corrective retrieval augmented generation",
            "graph rag",
            "retrieval augmented language model",
        ),
        strong_terms=("retrieval augmented", "rag", "dense passage retrieval", "self-rag", "graph rag"),
    ),
    TopicSpec(
        topic="ai_agent",
        prefix="agentx",
        query_terms=(
            "language model agents",
            "llm agents",
            "autonomous agents",
            "multi-agent language models",
            "tool use language models",
            "react reasoning acting",
            "toolformer",
            "agentic ai",
        ),
        strong_terms=("agent", "tool use", "autonomous", "multi-agent", "react", "toolformer"),
    ),
    TopicSpec(
        topic="knowledge_distillation",
        prefix="kdx",
        query_terms=(
            "knowledge distillation",
            "model distillation",
            "distilbert",
            "tinybert",
            "mobilebert",
            "minilm",
            "language model compression",
            "teacher student distillation",
        ),
        strong_terms=("knowledge distillation", "distillation", "distilbert", "tinybert", "mobilebert", "minilm"),
    ),
)


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
    "arxiv_id",
    "categories",
    "priority",
    "status",
    "collection_action",
    "local_path",
    "retrieved_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect arXiv candidate sources for ChronoRAG.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Candidate CSV output path")
    parser.add_argument("--max-per-topic", type=int, default=170, help="Target candidates per topic")
    parser.add_argument("--start", type=int, default=0, help="arXiv start offset")
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds between arXiv API calls")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for transient arXiv/API failures")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Append/update existing output rows instead of replacing the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    existing = load_existing(output) if args.merge_existing else []
    existing_ids = {row.get("arxiv_id", "") for row in existing if row.get("arxiv_id")}
    existing_titles = {normalise_title(row.get("title", "")) for row in existing if row.get("title")}

    rows: List[Dict[str, str]] = list(existing)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for spec_index, spec in enumerate(TOPICS):
        print(f"Collecting {spec.topic} candidates...")
        entries = fetch_topic_entries(
            spec,
            max_results=args.max_per_topic,
            start=args.start,
            retries=args.retries,
        )
        topic_rows = build_rows(spec, entries, existing_ids, existing_titles, retrieved_at)
        rows.extend(topic_rows)
        print(f"  kept {len(topic_rows)} candidates")
        if spec_index < len(TOPICS) - 1 and args.sleep > 0:
            time.sleep(args.sleep)

    rows = dedupe_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["topic"]] = counts.get(row["topic"], 0) + 1
    print(f"Wrote {len(rows)} candidates to {output}")
    print("Topic counts:", counts)


def fetch_topic_entries(
    spec: TopicSpec,
    *,
    max_results: int,
    start: int,
    retries: int,
) -> List[ET.Element]:
    query = build_search_query(spec)
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ChronoRAG/0.1 (student research corpus expansion)"},
    )
    data = b""
    for attempt in range(1, retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            break
        except HTTPError as exc:
            if exc.code == 429 and attempt <= retries:
                wait_seconds = 10 * attempt
                print(f"  arXiv rate-limited request (429). Waiting {wait_seconds}s before retry {attempt}/{retries}...")
                time.sleep(wait_seconds)
                continue
            print(f"  arXiv request failed for {spec.topic}: HTTP {exc.code}. Skipping this topic for now.")
            return []
        except URLError as exc:
            if attempt <= retries:
                wait_seconds = 5 * attempt
                print(f"  Network error: {exc}. Waiting {wait_seconds}s before retry {attempt}/{retries}...")
                time.sleep(wait_seconds)
                continue
            print(f"  Network request failed for {spec.topic}: {exc}. Skipping this topic for now.")
            return []
    root = ET.fromstring(data)
    return list(root.findall(f"{ATOM}entry"))


def build_search_query(spec: TopicSpec) -> str:
    # Keep the query topic-focused and mostly in CS categories. arXiv's Lucene
    # syntax accepts quoted all:"phrase" clauses joined by OR.
    clauses = [f'all:"{term}"' for term in spec.query_terms]
    category_clause = "(cat:cs.CL OR cat:cs.AI OR cat:cs.LG)"
    return f"{category_clause} AND ({' OR '.join(clauses)})"


def build_rows(
    spec: TopicSpec,
    entries: Iterable[ET.Element],
    existing_ids: Set[str],
    existing_titles: Set[str],
    retrieved_at: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen_titles: Set[str] = set()

    for entry in entries:
        title = clean_text(get_text(entry, "title"))
        summary = clean_text(get_text(entry, "summary"))
        arxiv_id = extract_arxiv_id(get_text(entry, "id"))
        title_key = normalise_title(title)
        if not title or not arxiv_id or arxiv_id in existing_ids or title_key in existing_titles or title_key in seen_titles:
            continue
        if not is_relevant(spec, title, summary):
            continue

        published = get_text(entry, "published")[:10]
        year = published[:4] if re.match(r"^\d{4}", published) else ""
        authors = "; ".join(clean_text(author.findtext(f"{ATOM}name", default="")) for author in entry.findall(f"{ATOM}author"))
        categories = "; ".join(
            cat.attrib.get("term", "") for cat in entry.findall(f"{ATOM}category") if cat.attrib.get("term")
        )
        source_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = pdf_link(entry) or f"https://arxiv.org/pdf/{arxiv_id}"

        rows.append(
            {
                "doc_id": f"{spec.prefix}_{len(rows) + 1:04d}",
                "topic": spec.topic,
                "title": title,
                "source_type": "paper",
                "source_url": source_url,
                "pdf_url": pdf_url,
                "year": year,
                "published_date": published,
                "authors": authors,
                "arxiv_id": arxiv_id,
                "categories": categories,
                "priority": priority_for(title, summary),
                "status": "candidate",
                "collection_action": "download_pdf_later",
                "local_path": "",
                "retrieved_at": retrieved_at,
                "notes": "arXiv API candidate; needs manual review before metadata.csv approval",
            }
        )
        seen_titles.add(title_key)

    return rows


def is_relevant(spec: TopicSpec, title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    return any(term.lower() in text for term in spec.strong_terms)


def priority_for(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(term in text for term in ("survey", "benchmark", "state-of-the-art", "comprehensive")):
        return "P1"
    return "P2"


def pdf_link(entry: ET.Element) -> str:
    for link in entry.findall(f"{ATOM}link"):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            return link.attrib.get("href", "")
    return ""


def extract_arxiv_id(url: str) -> str:
    value = (url or "").rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", value)


def get_text(entry: ET.Element, tag: str) -> str:
    return entry.findtext(f"{ATOM}{tag}", default="")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalise_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def load_existing(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_ids: Set[str] = set()
    seen_titles: Set[str] = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        arxiv_id = row.get("arxiv_id", "")
        title_key = normalise_title(row.get("title", ""))
        if arxiv_id in seen_ids or title_key in seen_titles:
            continue
        seen_ids.add(arxiv_id)
        seen_titles.add(title_key)
        out.append(row)
    return out


if __name__ == "__main__":
    main()
