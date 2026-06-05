"""Retrieval benchmark for the ChronoRAG corpus.

Reads ``data/eval/retrieval_eval.jsonl`` (hand-curated question -> expected
doc_ids), runs the live HybridRetriever, and reports Recall@k + MRR overall
and per topic. Writes ``data/eval/retrieval_metrics.json`` so the FastAPI
backend can expose the numbers to the Evaluation panel in the UI.

Usage:
    python scripts/17_eval_retrieval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
METRICS_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_metrics.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "vector_db"

K_VALUES = (1, 3, 5, 10)


def _load_eval_set(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hit_at_k(retrieved_doc_ids: List[str], expected: List[str], k: int) -> int:
    top_k = retrieved_doc_ids[:k]
    return 1 if any(d in top_k for d in expected) else 0


def _reciprocal_rank(retrieved_doc_ids: List[str], expected: List[str]) -> float:
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / idx
    return 0.0


def evaluate(retriever: HybridRetriever, eval_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_question: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for row in eval_set:
        question = row["question"]
        topic = row.get("topic")
        expected = row.get("expected_doc_ids") or []
        chunks = retriever.retrieve(question, topic=topic, top_k=max(K_VALUES))
        # Preserve rank order, dedupe by doc_id (a chunk can repeat the same doc).
        seen, retrieved_doc_ids = set(), []
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id", ""))
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                retrieved_doc_ids.append(doc_id)
        per_question.append({
            "qid": row.get("qid"),
            "topic": topic,
            "intent": row.get("intent"),
            "question": question,
            "expected": expected,
            "retrieved": retrieved_doc_ids[: max(K_VALUES)],
            "hit": {f"r@{k}": _hit_at_k(retrieved_doc_ids, expected, k) for k in K_VALUES},
            "mrr": _reciprocal_rank(retrieved_doc_ids, expected),
        })
    elapsed = time.perf_counter() - started

    def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"count": 0}
        return {
            "count": len(rows),
            **{f"recall@{k}": round(mean(r["hit"][f"r@{k}"] for r in rows), 4) for k in K_VALUES},
            "mrr": round(mean(r["mrr"] for r in rows), 4),
        }

    overall = _summarise(per_question)
    topics_seen = sorted({r["topic"] for r in per_question if r.get("topic")})
    per_topic = {t: _summarise([r for r in per_question if r["topic"] == t]) for t in topics_seen}

    intents_seen = sorted({r["intent"] for r in per_question if r.get("intent")})
    per_intent = {i: _summarise([r for r in per_question if r["intent"] == i]) for i in intents_seen}

    return {
        "summary": overall,
        "per_topic": per_topic,
        "per_intent": per_intent,
        "questions": per_question,
        "config": {
            "k_values": list(K_VALUES),
            "retriever": "HybridRetriever(bm25 + optional FAISS)",
            "eval_set": str(EVAL_PATH.relative_to(PROJECT_ROOT)),
            "total_questions": len(per_question),
            "elapsed_seconds": round(elapsed, 3),
        },
    }


def main() -> int:
    if not EVAL_PATH.exists():
        print(f"Missing eval set: {EVAL_PATH}", file=sys.stderr)
        return 1
    if not CHUNKS_PATH.exists():
        print(f"Missing corpus chunks: {CHUNKS_PATH}", file=sys.stderr)
        return 1

    eval_set = _load_eval_set(EVAL_PATH)
    retriever = HybridRetriever(chunks_path=CHUNKS_PATH, index_dir=INDEX_DIR, use_vector=False)
    report = evaluate(retriever, eval_set)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print(f"Evaluated {s['count']} questions in {report['config']['elapsed_seconds']}s")
    print(f"  Recall@1: {s['recall@1']:.3f}  Recall@3: {s['recall@3']:.3f}  "
          f"Recall@5: {s['recall@5']:.3f}  Recall@10: {s['recall@10']:.3f}  MRR: {s['mrr']:.3f}")
    print()
    print("Per-topic Recall@5:")
    for topic, m in report["per_topic"].items():
        print(f"  {topic:<25} {m.get('recall@5', 0):.3f}  (n={m['count']})")
    print()
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
