"""Sprint 5 retrieval and timeline utility tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.bm25_index import build_bm25_index, score_bm25, tokenize_for_bm25
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expansion import expand_query_for_retrieval
from src.timeline.date_extractor import extract_date


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestDateExtractor(unittest.TestCase):
    def test_extracts_month_year(self):
        result = extract_date("In March 2023, Toolformer was introduced.")
        self.assertEqual(result["normalized_date"], "2023-03")
        self.assertEqual(result["extracted_year"], 2023)
        self.assertGreaterEqual(result["date_confidence"], 0.9)

    def test_falls_back_to_document_year(self):
        result = extract_date("The method improves dense retrieval.", document_year=2020)
        self.assertEqual(result["normalized_date"], "2020")
        self.assertEqual(result["date_source"], "document_year")


class TestBm25Index(unittest.TestCase):
    def test_tokenizer_filters_stopwords_and_keeps_years(self):
        tokens = tokenize_for_bm25("What is RAG in 2020?")
        self.assertNotIn("what", tokens)
        self.assertNotIn("is", tokens)
        self.assertIn("rag", tokens)
        self.assertIn("2020", tokens)

    def test_build_and_score_bm25(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            chunks_path = tmp_path / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "rag_c1",
                    "doc_id": "rag_001",
                    "topic": "rag",
                    "title": "RAG paper",
                    "text": "Retrieval augmented generation was proposed in 2020.",
                },
                {
                    "chunk_id": "agent_c1",
                    "doc_id": "agent_001",
                    "topic": "ai_agent",
                    "title": "Agent paper",
                    "text": "ReAct combines reasoning and acting.",
                },
            ]
            _write_jsonl(chunks_path, rows)
            summary = build_bm25_index(chunks_path, tmp_path / "vector_db")
            self.assertEqual(summary["chunks_indexed"], 2)
            self.assertTrue((tmp_path / "vector_db" / "bm25.pkl").exists())

            scores = score_bm25("When was RAG proposed?", rows)
            self.assertGreater(scores[0], scores[1])


class TestHybridRetriever(unittest.TestCase):
    def test_retrieves_with_bm25_and_topic_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            chunks_path = tmp_path / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "rag_c1",
                    "doc_id": "rag_001",
                    "topic": "rag",
                    "title": "RAG paper",
                    "source_url": "https://example.com/rag",
                    "text": "Retrieval augmented generation was proposed in 2020.",
                },
                {
                    "chunk_id": "agent_c1",
                    "doc_id": "agent_001",
                    "topic": "ai_agent",
                    "title": "ReAct paper",
                    "source_url": "https://example.com/react",
                    "text": "ReAct combines reasoning and acting for language model agents.",
                },
            ]
            _write_jsonl(chunks_path, rows)
            retriever = HybridRetriever(chunks_path=chunks_path, index_dir=tmp_path, use_vector=False)
            result = retriever.retrieve("What is ReAct?", topic="AI Agent", top_k=1)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["chunk_id"], "agent_c1")
            self.assertIn("bm25", result[0]["_retrieval_mode"])

    def test_vietnamese_trend_query_expands_to_topic_context(self):
        query = "xu hướng hiện nay của AI agent là gì"
        expanded = expand_query_for_retrieval(query, topic="AI Agent")
        self.assertIn("LLM agents", expanded)
        self.assertIn("survey", expanded)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            chunks_path = tmp_path / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "agent_c1",
                    "doc_id": "agent_004",
                    "topic": "ai_agent",
                    "title": "A survey of LLM agents",
                    "source_url": "https://example.com/agents",
                    "text": (
                        "Recent LLM agent research studies autonomous agents, "
                        "planning, tool use, and multi-agent frameworks."
                    ),
                },
                {
                    "chunk_id": "rag_c1",
                    "doc_id": "rag_001",
                    "topic": "rag",
                    "title": "RAG paper",
                    "source_url": "https://example.com/rag",
                    "text": "Retrieval augmented generation was proposed in 2020.",
                },
            ]
            _write_jsonl(chunks_path, rows)
            retriever = HybridRetriever(chunks_path=chunks_path, index_dir=tmp_path, use_vector=False)
            result = retriever.retrieve(query, topic="AI Agent", top_k=1)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["chunk_id"], "agent_c1")

    def test_optional_reranker_can_reorder_candidates(self):
        class FakeReranker:
            def predict(self, pairs):
                # Prefer the second candidate even if BM25/RRF put it lower.
                return [0.1, 4.0]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            chunks_path = tmp_path / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "rag_c1",
                    "doc_id": "rag_001",
                    "topic": "rag",
                    "title": "RAG intro",
                    "text": "RAG retrieval augmented generation retrieval retrieval.",
                },
                {
                    "chunk_id": "rag_c2",
                    "doc_id": "rag_002",
                    "topic": "rag",
                    "title": "RAG survey",
                    "text": "Retrieval augmented generation survey applications.",
                },
            ]
            _write_jsonl(chunks_path, rows)
            retriever = HybridRetriever(
                chunks_path=chunks_path,
                index_dir=tmp_path,
                use_vector=False,
                use_reranker=True,
            )
            retriever._reranker = FakeReranker()
            result = retriever.retrieve("RAG", topic="RAG", top_k=1)
            self.assertEqual(result[0]["chunk_id"], "rag_c2")
            self.assertIn("rerank", result[0]["_retrieval_mode"])

    def test_returns_empty_for_unmatched_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            chunks_path = tmp_path / "chunks.jsonl"
            _write_jsonl(
                chunks_path,
                [
                    {
                        "chunk_id": "rag_c1",
                        "doc_id": "rag_001",
                        "topic": "rag",
                        "title": "RAG paper",
                        "text": "Retrieval augmented generation was proposed in 2020.",
                    }
                ],
            )
            retriever = HybridRetriever(chunks_path=chunks_path, index_dir=tmp_path, use_vector=False)
            self.assertEqual(retriever.retrieve("photosynthesis chlorophyll", topic="RAG"), [])


if __name__ == "__main__":
    unittest.main()
