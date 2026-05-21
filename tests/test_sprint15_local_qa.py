import unittest
from pathlib import Path
import tempfile
import json
from src.retrieval.simple_retriever import SimpleRetriever
from src.generation.template_answerer import TemplateAnswerer
from workflows.online_pipeline import get_local_qa_answer, get_mock_answer

class TestSprint15LocalQA(unittest.TestCase):
    def setUp(self):
        # Create a mock chunks file for testing isolated unit cases
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.temp_dir.name) / "chunks.jsonl"
        
        self.mock_chunks = [
            {
                "chunk_id": "agent_005_c0001",
                "doc_id": "agent_005",
                "topic": "ai_agent",
                "title": "AutoGPT GitHub repository README",
                "text": "AutoGPT is a platform that allows you to create, deploy, and manage continuous AI agents.",
                "source_url": "https://github.com/Significant-Gravitas/AutoGPT",
                "year": 2023
            },
            {
                "chunk_id": "agent_006_c0001",
                "doc_id": "agent_006",
                "topic": "ai_agent",
                "title": "Microsoft AutoGen documentation",
                "text": "AutoGen is an open-source programming framework for building AI agents and facilitating cooperation among multiple agents.",
                "source_url": "https://microsoft.github.io/autogen",
                "year": 2023
            },
            {
                "chunk_id": "rag_001_c0001",
                "doc_id": "rag_001",
                "topic": "rag",
                "title": "RAG Paper",
                "text": "Retrieval-Augmented Generation (RAG) combines parametric memory with dense vector search.",
                "source_url": "https://arxiv.org/abs/2005.11401",
                "year": 2020
            }
        ]
        
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            for chunk in self.mock_chunks:
                f.write(json.dumps(chunk) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_simple_retriever_basic(self):
        retriever = SimpleRetriever(self.chunks_path)
        
        # Test basic keyword matching
        results = retriever.retrieve("What is AutoGen?", topic="AI Agent")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["doc_id"], "agent_006")
        
        # Test topic filtering
        results_rag = retriever.retrieve("RAG", topic="RAG")
        self.assertEqual(len(results_rag), 1)
        self.assertEqual(results_rag[0]["doc_id"], "rag_001")

        # Test no matches
        results_none = retriever.retrieve("Unknown keyword")
        self.assertEqual(len(results_none), 0)

    def test_simple_retriever_no_file(self):
        non_existent_path = Path(self.temp_dir.name) / "non_existent.jsonl"
        retriever = SimpleRetriever(non_existent_path)
        results = retriever.retrieve("AutoGen")
        self.assertEqual(len(results), 0)

    def test_template_answerer(self):
        answerer = TemplateAnswerer()
        
        # Test empty input
        fallback = answerer.generate_answer([])
        self.assertIn("couldn't find any relevant information", fallback["answer"])
        self.assertEqual(len(fallback["citations"]), 0)
        
        # Test populated input
        chunks = [self.mock_chunks[1]]
        response = answerer.generate_answer(chunks, query="What is AutoGen?")
        self.assertIn("AutoGen is an open-source programming framework", response["answer"])
        self.assertIn("[agent_006]", response["answer"])
        self.assertEqual(len(response["citations"]), 1)
        self.assertEqual(response["citations"][0]["doc_id"], "agent_006")
        self.assertEqual(response["citations"][0]["title"], "Microsoft AutoGen documentation")

    def test_get_local_qa_answer_fallback(self):
        from unittest.mock import patch
        # When chunks.jsonl is empty or not found, should fall back to mock answer
        with patch('workflows.online_pipeline.PROJECT_ROOT', Path('/nonexistent_directory_for_testing_fallback')):
            res = get_local_qa_answer("RAG", "What is RAG?")
            self.assertIn("Retrieval-Augmented Generation", res["answer"])
            self.assertTrue(len(res["citations"]) > 0)


    def test_real_chunks_autogen(self):
        # Resolve real chunks file path
        repo_root = Path(__file__).resolve().parent.parent
        real_chunks_path = repo_root / 'data' / 'processed' / 'chunks.jsonl'
        
        if not real_chunks_path.exists():
            self.skipTest("Real processed chunks.jsonl not found. Run workflows/offline_pipeline.py first.")
            
        retriever = SimpleRetriever(real_chunks_path)
        answerer = TemplateAnswerer()
        
        # 1. Test "What is AutoGen?" on real processed chunks
        chunks = retriever.retrieve("What is AutoGen?", topic="AI Agent", top_k=3)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0]["doc_id"], "agent_006")
        
        res = answerer.generate_answer(chunks, query="What is AutoGen?")
        self.assertIn("AutoGen is an open-source programming framework", res["answer"])
        self.assertNotIn("b/0.2", res["answer"])
        self.assertNotIn("OPENAI_API_KEY", res["answer"])
        self.assertNotIn("<!--", res["answer"])
        self.assertNotIn("img.shields.io", res["answer"])
        
        # 2. Test "What is AutoGPT?" on real processed chunks
        chunks_gpt = retriever.retrieve("What is AutoGPT?", topic="AI Agent", top_k=3)
        self.assertTrue(len(chunks_gpt) > 0)
        self.assertEqual(chunks_gpt[0]["doc_id"], "agent_005")
        
        res_gpt = answerer.generate_answer(chunks_gpt, query="What is AutoGPT?")
        self.assertIn("AutoGPT is a powerful platform", res_gpt["answer"])
        self.assertNotIn("img.shields.io", res_gpt["answer"])
        self.assertNotIn("<!--", res_gpt["answer"])
        self.assertNotIn("b/0.2", res_gpt["answer"])

        # 3. Test a completely irrelevant query with topic RAG on real processed chunks
        chunks_unrelated = retriever.retrieve("What is photosynthesis?", topic="RAG", top_k=3)
        res_unrelated = answerer.generate_answer(chunks_unrelated, query="What is photosynthesis?")
        self.assertIn("couldn't find any relevant information", res_unrelated["answer"])
        self.assertEqual(len(res_unrelated["citations"]), 0)

if __name__ == '__main__':
    unittest.main()
