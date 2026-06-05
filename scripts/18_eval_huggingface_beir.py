"""External retrieval benchmark on BEIR/SciFact (HuggingFace).

This is the "literature comparable" companion to ``scripts/15_eval_retrieval.py``:
that one tests our retriever on OUR corpus with hand-curated questions;
this one indexes BEIR/SciFact's corpus with the same BM25 backbone and
reports the standard Recall@k that papers report on BEIR.

Why SciFact: 5183 documents + 1109 queries + 339 graded test qrels, scientific
domain (closest in tone to the AI/ML papers in our corpus), only ~10 MB --
small enough to ship in a CI pass.

Usage:
    python scripts/18_eval_huggingface_beir.py
    python scripts/18_eval_huggingface_beir.py --limit-queries 100   # quick run

Writes:
    data/eval/huggingface_benchmark.json   (consumed by /api/huggingface_eval)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.bm25_index import score_bm25, tokenize_for_bm25  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "huggingface_benchmark.json"
K_VALUES = (1, 3, 5, 10)
DATASET_NAME = "BeIR/scifact"


def _load_beir_scifact():
    """Load BEIR/SciFact from HuggingFace. Returns (corpus, queries, qrels_test)."""
    from datasets import load_dataset  # imported lazily so the rest of the project doesn't pay the import cost

    corpus = load_dataset(DATASET_NAME, "corpus", split="corpus")
    queries = load_dataset(DATASET_NAME, "queries", split="queries")
    qrels = load_dataset(f"{DATASET_NAME}-qrels", split="test")
    return corpus, queries, qrels


def _build_corpus_index(corpus) -> List[Dict[str, Any]]:
    """Materialise the corpus into the dict shape our score_bm25 expects.

    BEIR rows: ``{"_id": str, "title": str, "text": str}``. We map them to
    ``{"chunk_id", "doc_id", "title", "topic", "text"}`` so ``score_bm25`` can
    reuse its existing ``_chunk_text(chunk)`` helper.
    """
    docs: List[Dict[str, Any]] = []
    for row in corpus:
        docs.append({
            "chunk_id": str(row["_id"]),
            "doc_id": str(row["_id"]),
            "topic": "scifact",
            "title": row.get("title") or "",
            "text": row.get("text") or "",
        })
    return docs


def _hit_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> int:
    return 1 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0


def _reciprocal_rank(retrieved_ids: List[str], relevant_ids: set) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(limit_queries: int = 0) -> Dict[str, Any]:
    print(f"Loading {DATASET_NAME} from HuggingFace...")
    corpus, queries, qrels = _load_beir_scifact()
    print(f"  corpus={len(corpus)}  queries={len(queries)}  test_qrels={len(qrels)}")

    # Map query_id -> list of relevant corpus_id (treat score >= 1 as relevant).
    rels: Dict[str, set] = defaultdict(set)
    for row in qrels:
        if int(row.get("score", 0)) >= 1:
            rels[str(row["query-id"])].add(str(row["corpus-id"]))
    print(f"  unique test queries with relevance: {len(rels)}")

    # Build the corpus list once. BM25 index is built lazily inside score_bm25.
    print("Indexing corpus (BM25 lazy build)...")
    docs = _build_corpus_index(corpus)

    # Build a tiny query->text map.
    query_text: Dict[str, str] = {str(row["_id"]): row["text"] for row in queries}

    # Pre-tokenise the corpus once so we can reuse the BM25 index for every query.
    from rank_bm25 import BM25Okapi
    print("Tokenising corpus...")
    tokenised = [tokenize_for_bm25(f'{d["title"]} {d["text"]}') for d in docs]
    bm25 = BM25Okapi(tokenised)

    # Score every relevant query.
    qids = sorted(rels.keys())
    if limit_queries and limit_queries < len(qids):
        qids = qids[:limit_queries]
        print(f"  limiting to first {limit_queries} queries")

    per_q: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for qid in qids:
        q_text = query_text.get(qid, "")
        if not q_text:
            continue
        relevant_ids = rels[qid]
        scores = score_bm25(q_text, docs, bm25=bm25)
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[: max(K_VALUES)]
        retrieved_ids = [docs[i]["doc_id"] for i in ranked]
        per_q.append({
            "qid": qid,
            "query": q_text[:120],
            "relevant": sorted(relevant_ids),
            "retrieved": retrieved_ids,
            "hit": {f"r@{k}": _hit_at_k(retrieved_ids, relevant_ids, k) for k in K_VALUES},
            "mrr": _reciprocal_rank(retrieved_ids, relevant_ids),
        })
    elapsed = time.perf_counter() - started

    if not per_q:
        raise SystemExit("No queries scored -- check dataset structure.")

    summary = {
        "dataset": DATASET_NAME,
        "queries_evaluated": len(per_q),
        "corpus_size": len(docs),
        **{f"recall@{k}": round(mean(r["hit"][f"r@{k}"] for r in per_q), 4) for k in K_VALUES},
        "mrr": round(mean(r["mrr"] for r in per_q), 4),
        "elapsed_seconds": round(elapsed, 3),
    }
    return {
        "summary": summary,
        # Keep first 25 question records as samples; full per-question is large.
        "samples": per_q[:25],
        "config": {
            "k_values": list(K_VALUES),
            "retriever": "BM25Okapi (same backbone as HybridRetriever)",
            "source": f"huggingface://{DATASET_NAME}",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BEIR/SciFact benchmark via HuggingFace.")
    parser.add_argument("--limit-queries", type=int, default=0,
                        help="If > 0, only run the first N queries (handy for smoke tests).")
    args = parser.parse_args()

    report = evaluate(limit_queries=args.limit_queries)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print()
    print(f"Dataset: {s['dataset']}  (corpus={s['corpus_size']}, queries={s['queries_evaluated']})")
    print(f"  Recall@1: {s['recall@1']:.3f}  Recall@3: {s['recall@3']:.3f}  "
          f"Recall@5: {s['recall@5']:.3f}  Recall@10: {s['recall@10']:.3f}  MRR: {s['mrr']:.3f}")
    print(f"  Elapsed: {s['elapsed_seconds']}s")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
