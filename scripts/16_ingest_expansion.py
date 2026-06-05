"""Ingest top expansion candidates into the active corpus.

Reads ``data/raw/candidate_sources_expansion_openalex.csv`` (built earlier by
``scripts/15_collect_openalex_candidates.py``),
applies a quality filter, picks top N per topic, downloads the PDF, and
appends a fresh row to ``data/raw/metadata.csv`` so the regular offline
pipeline (``workflows/offline_pipeline.py``) can pick the new docs up on the
next rebuild.

Usage:
    python scripts/16_ingest_expansion.py --per-topic 5 --apply
    python scripts/16_ingest_expansion.py --per-topic 5    # dry run, just lists

Quality filter (tuned for AI/ML research corpus, not domain-specific apps):
- year in [2020, 2024]
- citation_count >= 30
- pdf_url on arxiv.org or aclanthology.org (open, predictable PDF layout)
- title not in the existing corpus
- title doesn't contain off-topic noise markers
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "data" / "raw" / "candidate_sources_expansion_openalex.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Domains we trust for direct PDF download.
ALLOWED_PDF_HOSTS = ("arxiv.org", "aclanthology.org", "openreview.net")

# Heuristics to drop OpenAlex noise -- single-domain applications, AI-written
# preprints with weird capitalisation, papers whose title is clearly outside
# the RAG / Agent / KD core.
OFF_TOPIC_MARKERS = (
    "legal", "law", "lawyer", "careerx", "medical", "clinical", "patient", "diagnosis",
    "sign language", "wearable", "social media", "personal health",
    "compiler", "dsl", "kernel",
    "trism", "trust, risk",
    "spectral tempering", "geodesic", "manifold-aware",  # the AI-noise 2026 cluster
    "review of trust",
)

# Topics in scope.
TOPICS = ("rag", "ai_agent", "knowledge_distillation")


def _normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _is_arxiv_pdf(url: str) -> bool:
    return any(host in url.lower() for host in ALLOWED_PDF_HOSTS)


def _is_off_topic(title: str) -> bool:
    t = title.lower()
    return any(marker in t for marker in OFF_TOPIC_MARKERS)


def _load_existing_titles() -> set:
    existing = set()
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing.add(_normalise_title(row.get("title", "")))
    return existing


def _next_doc_id(topic: str) -> int:
    """Return the next free numeric suffix for this topic's doc_ids."""
    prefix_map = {"rag": "rag", "ai_agent": "agent", "knowledge_distillation": "kd"}
    prefix = prefix_map[topic]
    max_n = 0
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                m = re.match(rf"{prefix}_(\d+)$", row.get("doc_id", ""))
                if m:
                    max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _pick_candidates(per_topic: int, min_citations: int) -> Dict[str, List[Dict[str, str]]]:
    existing_titles = _load_existing_titles()
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"Candidate file missing: {CANDIDATES_PATH}")

    with CANDIDATES_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    picks: Dict[str, List[Dict[str, str]]] = {t: [] for t in TOPICS}
    pool: Dict[str, List[Tuple[float, Dict[str, str]]]] = {t: [] for t in TOPICS}

    for row in rows:
        topic = row.get("topic")
        if topic not in pool:
            continue
        title = row.get("title", "").strip()
        if not title or _normalise_title(title) in existing_titles:
            continue
        if _is_off_topic(title):
            continue
        pdf_url = (row.get("pdf_url") or "").strip()
        if not pdf_url or not _is_arxiv_pdf(pdf_url):
            continue
        try:
            year = int(row.get("year") or 0)
            citations = int(row.get("citation_count") or 0)
        except ValueError:
            continue
        if year < 2020 or year > 2024:
            continue
        if citations < min_citations:
            continue
        # Score: citations dominate, then recency.
        score = citations * 1000 + year
        pool[topic].append((score, row))

    for topic in TOPICS:
        pool[topic].sort(key=lambda x: -x[0])
        seen_titles: set = set()
        for _, row in pool[topic]:
            key = _normalise_title(row["title"])
            if key in seen_titles:
                continue
            seen_titles.add(key)
            picks[topic].append(row)
            if len(picks[topic]) >= per_topic:
                break
    return picks


def _download_pdf(url: str, out_path: Path, timeout: int = 30) -> Optional[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ChronoRAG/0.1 (+https://github.com/local) ingest"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"  ! download failed: {exc}")
        return None
    if len(data) < 1024:
        print(f"  ! suspiciously small payload ({len(data)} bytes), skipping")
        return None
    out_path.write_bytes(data)
    return out_path


def _format_authors(raw: str) -> str:
    # OpenAlex returns "Last1, F1; Last2, F2" style; metadata.csv uses "F L; F L".
    return (raw or "").replace("\n", " ").strip()


def _append_metadata(rows_to_append: List[Dict[str, str]]) -> None:
    """Append rows preserving the existing metadata.csv column order."""
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    with METADATA_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows_to_append:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest expansion candidates.")
    parser.add_argument("--per-topic", type=int, default=5,
                        help="How many papers to add per topic (default 5)")
    parser.add_argument("--min-citations", type=int, default=30,
                        help="Minimum citation count to keep (default 30)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually download PDFs and edit metadata.csv. "
                             "Without this flag the script only prints picks.")
    args = parser.parse_args()

    picks = _pick_candidates(args.per_topic, args.min_citations)

    print(f"\n=== Candidate picks ({args.per_topic} per topic, citations >= {args.min_citations}) ===")
    for topic, items in picks.items():
        print(f"\n[{topic}] {len(items)} picks")
        for r in items:
            print(f"  [{r['year']}] cc={r.get('citation_count','?'):>4} | {r['title'][:70]}")
            print(f"    {r['pdf_url']}")

    if not args.apply:
        print("\n(dry run) Re-run with --apply to download PDFs and update metadata.csv.")
        return 0

    new_rows: List[Dict[str, str]] = []
    for topic, items in picks.items():
        prefix = {"rag": "rag", "ai_agent": "agent", "knowledge_distillation": "kd"}[topic]
        next_n = _next_doc_id(topic)
        for r in items:
            doc_id = f"{prefix}_{next_n:03d}"
            local_rel = f"{topic}/{doc_id}.pdf"
            local_abs = RAW_DIR / local_rel
            print(f"\n-> {doc_id}  {r['title'][:70]}")
            print(f"  fetching {r['pdf_url']}")
            saved = _download_pdf(r["pdf_url"], local_abs)
            if not saved:
                continue
            new_rows.append({
                "doc_id": doc_id,
                "title": r["title"].strip(),
                "topic": topic,
                "source_type": "paper",
                "source_url": r.get("source_url", "") or r.get("pdf_url", ""),
                "published_date": r.get("published_date", ""),
                "year": r.get("year", ""),
                "authors": _format_authors(r.get("authors", "")),
                "local_path": local_rel.replace("\\", "/"),
                "retrieved_at": date.today().isoformat(),
                "language": "en",
                "num_pages": "",
                "status": "approved",
                "notes": f"Expanded from OpenAlex (cc={r.get('citation_count','?')})",
                "has_clear_timeline_signal": "yes",
            })
            next_n += 1
            time.sleep(1.0)  # be polite to arXiv

    if not new_rows:
        print("\nNo new rows ingested.")
        return 1

    _append_metadata(new_rows)
    print(f"\nOK Appended {len(new_rows)} rows to {METADATA_PATH.relative_to(PROJECT_ROOT)}.")
    print("Next step: re-run the offline pipeline to rebuild artifacts:")
    print("    python workflows/offline_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
