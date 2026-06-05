import unittest
import sys
from pathlib import Path
import tempfile
import json
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.simple_retriever import SimpleRetriever
from src.generation.template_answerer import TemplateAnswerer
from src.generation.llm_answerer import maybe_generate_llm_answer
from workflows.online_pipeline import get_local_qa_answer

class TestSprint15LocalQA(unittest.TestCase):
    def setUp(self):
        self._llm_env_patch = patch.dict("os.environ", {"LLM_PROVIDER": "mock"}, clear=False)
        self._llm_env_patch.start()

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
        self._llm_env_patch.stop()

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
        self.assertIn("Không tìm thấy thông tin đủ liên quan", fallback["answer"])
        self.assertEqual(len(fallback["citations"]), 0)
        
        # Test populated input
        chunks = [self.mock_chunks[1]]
        response = answerer.generate_answer(chunks, query="What is AutoGen?")
        self.assertIn("AutoGen is an open-source programming framework", response["answer"])
        self.assertIn("[agent_006]", response["answer"])
        self.assertEqual(len(response["citations"]), 1)
        self.assertEqual(response["citations"][0]["doc_id"], "agent_006")
        self.assertEqual(response["citations"][0]["title"], "Microsoft AutoGen documentation")

    def test_answerer_abstains_when_query_barely_overlaps(self):
        # The chunk is about RAG; the question only shares the incidental word
        # "best". Coverage (1 of 3 content tokens) is below the relevance gate,
        # so the answerer must abstain instead of emitting an off-topic sentence.
        answerer = TemplateAnswerer()
        off_topic_chunk = {
            "chunk_id": "rag_001_c0009",
            "doc_id": "rag_001",
            "topic": "rag",
            "title": "RAG Paper",
            "text": "This is the best paper we have written. It presents the method clearly.",
            "source_url": "https://arxiv.org/abs/2005.11401",
            "year": 2020,
        }
        res = answerer.generate_answer(
            [off_topic_chunk], query="What is the best pizza topping?"
        )
        self.assertIn("Không tìm thấy thông tin đủ liên quan", res["answer"])
        self.assertEqual(len(res["citations"]), 0)

    def test_answerer_abstains_for_vietnamese_greeting(self):
        answerer = TemplateAnswerer()
        res = answerer.generate_answer(
            [self.mock_chunks[2]], query="xin chào, m có thể làm gì?"
        )
        self.assertIn("Không tìm thấy thông tin đủ liên quan", res["answer"])
        self.assertEqual(len(res["citations"]), 0)

    def test_answerer_keeps_relevant_multi_token_answer(self):
        # Guard against over-abstaining: a multi-token query that genuinely
        # overlaps the chunk (framework + agents) must still be answered.
        answerer = TemplateAnswerer()
        res = answerer.generate_answer(
            [self.mock_chunks[1]], query="open source framework for agents"
        )
        self.assertIn("AutoGen is an open-source programming framework", res["answer"])
        self.assertEqual(res["citations"][0]["doc_id"], "agent_006")

    def test_answerer_strips_numbered_section_heading(self):
        # PDF parsing often glues section headings into the next sentence.
        # The answerer must strip the heading so users see clean prose, not
        # "2.3 Active Retrieval Augmented Generation To aid..." artifacts.
        answerer = TemplateAnswerer()
        chunk = {
            "chunk_id": "rag_008_c0017",
            "doc_id": "rag_008",
            "topic": "rag",
            "title": "Active RAG",
            "text": (
                "2.3 Active Retrieval Augmented Generation To aid long-form "
                "generation with retrieval, we propose active retrieval "
                "augmented generation that decides what to retrieve."
            ),
            "source_url": "https://example.org/active-rag",
            "year": 2023,
        }
        res = answerer.generate_answer(
            [chunk], query="What is active retrieval augmented generation?"
        )
        self.assertNotIn("2.3 Active Retrieval", res["answer"])
        self.assertIn("To aid long-form generation", res["answer"])

    def test_topic_intent_overrides_dropdown(self):
        # User asks "AI Agent có từ năm nào" while the UI dropdown is on RAG.
        # The retriever should pull from ai_agent chunks, not the RAG slice.
        from workflows.online_pipeline import _detect_topic_intent

        self.assertEqual(_detect_topic_intent("AI Agent có từ năm nào"), "ai_agent")
        self.assertEqual(_detect_topic_intent("how does autogen work"), "ai_agent")
        self.assertEqual(_detect_topic_intent("What is DistilBERT?"), "knowledge_distillation")
        self.assertEqual(_detect_topic_intent("So sánh distilbert và tinybert"), "knowledge_distillation")
        self.assertEqual(_detect_topic_intent("What is self-RAG?"), "rag")
        # Plain questions with no topic keywords keep using the dropdown.
        self.assertIsNone(_detect_topic_intent("when was that proposed?"))

    def test_short_followup_pulls_in_previous_question(self):
        # "nó làm được gì" alone has no BM25 signal. With history, the
        # retrieval query should be augmented with the previous user message.
        from workflows.online_pipeline import _augment_short_followup

        history = [
            {"role": "user", "content": "AI Agent có từ năm nào"},
            {"role": "assistant", "content": "AI Agent xuất hiện rõ từ 2023..."},
        ]
        out = _augment_short_followup("nó làm được gì", history)
        self.assertIn("AI Agent", out)
        self.assertIn("nó làm được gì", out)
        # Self-contained questions stay untouched.
        self.assertEqual(
            _augment_short_followup("What is RAG?", history),
            "What is RAG?",
        )
        # No history -> no change.
        self.assertEqual(_augment_short_followup("nó là gì", None), "nó là gì")

    def test_llm_answerer_accepts_history_no_crash(self):
        # Smoke test: with mock provider and a history list, the helper must
        # still return None cleanly (no exceptions on the new parameter).
        from unittest.mock import patch

        local_answer = {
            "answer": "AutoGen is an open-source programming framework [agent_006]",
            "citations": [{"doc_id": "agent_006", "title": "AutoGen", "source_url": ""}],
        }
        history = [
            {"role": "user", "content": "What is AutoGen?"},
            {"role": "assistant", "content": "AutoGen is a framework [agent_006]"},
        ]
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}, clear=False):
            res = maybe_generate_llm_answer(
                "How does it work?", [self.mock_chunks[1]], local_answer, history=history
            )
        self.assertIsNone(res)

    def test_meta_question_intercepted_before_retrieval(self):
        # "bạn là ai" / "who are you" must never go through retrieval. Before
        # this guard, "ai" passed the content-token filter and the LLM happily
        # synthesised a paragraph about RAG that had nothing to do with the
        # question.
        from workflows.online_pipeline import get_local_qa_answer

        for q in ("bạn là ai", "Bạn Là Ai?", "who are you", "what can you do?", "/help"):
            res = get_local_qa_answer("rag", q)
            self.assertIn("ChronoRAG", res["answer"], f"meta question failed for: {q!r}")
            self.assertEqual(res["citations"], [])

    def test_meta_question_detector_does_not_swallow_real_questions(self):
        # Defensive: regular topical questions must NOT be detected as meta.
        from workflows.online_pipeline import _looks_like_meta_question

        for q in ("What is RAG?", "How does ReAct work?", "DistilBERT vs TinyBERT", "Tell me about dense passage retrieval"):
            self.assertFalse(_looks_like_meta_question(q), f"false positive on: {q!r}")

    def test_router_answers_capability_and_metrics_before_retrieval(self):
        from unittest.mock import patch
        from workflows.online_pipeline import get_local_qa_answer

        questions = (
            "t có thể hỏi m những cái gì, độ chính xác bao nhiêu",
            "chatbot này hỏi được những gì",
            "model đúng bao nhiêu phần trăm",
            "độ chính xác hiện tại là bao nhiêu",
        )
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}, clear=False):
            results = [get_local_qa_answer("rag", q) for q in questions]

        self.assertIn("RAG", results[0]["answer"])
        self.assertIn("AI Agent", results[0]["answer"])
        self.assertIn("Knowledge Distillation", results[0]["answer"])
        self.assertIn("Binary F1", results[0]["answer"])
        self.assertIn("Binary Accuracy", results[0]["answer"])
        self.assertEqual(results[0]["citations"], [])
        self.assertEqual(results[0]["provider"], "router")

        self.assertIn("3 chủ đề MVP", results[1]["answer"])
        self.assertIn("Binary", results[2]["answer"])
        self.assertIn("test split", results[3]["answer"])

    def test_router_scope_and_llm_status(self):
        from unittest.mock import patch
        from workflows.online_pipeline import get_local_qa_answer

        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "lmstudio",
                "LMSTUDIO_MODEL": "qwen/qwen3-4b",
                "LMSTUDIO_BASE_URL": "http://127.0.0.1:1234/v1",
            },
            clear=False,
        ):
            scope = get_local_qa_answer("rag", "data này đang tập trung về cái gì")
            llm = get_local_qa_answer("rag", "đang dùng model nào")

        self.assertIn("RAG", scope["answer"])
        self.assertIn("AI Agent", scope["answer"])
        self.assertIn("Knowledge Distillation", scope["answer"])
        self.assertIn("qwen/qwen3-4b", llm["answer"])
        self.assertEqual(llm["provider"], "router")

    def test_router_handles_user_identity_question(self):
        from workflows.online_pipeline import get_local_qa_answer

        for question in ("m biết t là ai ko", "ban co biet toi la ai khong", "do you know who I am"):
            res = get_local_qa_answer("rag", question)
            self.assertEqual(res["provider"], "router")
            self.assertIn("không biết danh tính thật", res["answer"])
            self.assertIn("không có hồ sơ cá nhân", res["answer"])
            self.assertEqual(res["citations"], [])

    def test_router_handles_unclear_followup_from_identity_question(self):
        from workflows.online_pipeline import get_local_qa_answer

        history = [
            {"role": "user", "content": "m biết t là ai ko"},
            {"role": "assistant", "content": "Mình không biết danh tính thật của đại ka."},
        ]
        res = get_local_qa_answer("rag", "t hỏi ấy", history=history)
        self.assertEqual(res["provider"], "router")
        self.assertIn("câu trước", res["answer"])
        self.assertIn("không biết danh tính thật", res["answer"])
        self.assertEqual(res["citations"], [])

    def test_repair_pdf_hyphenation(self):
        # Now a shared helper used by both the answerer (chat) and the backend
        # (Analysis sentences). PDF line wraps glue back; real compound hyphens
        # in the keep-list (fine-, pre-, open-...) stay hyphenated.
        from src.utils.text import repair_pdf_hyphenation

        self.assertEqual(
            repair_pdf_hyphenation("FLARE achieves superior or compet- itive performance."),
            "FLARE achieves superior or competitive performance.",
        )
        self.assertEqual(
            repair_pdf_hyphenation("knowl- edge distillation"),
            "knowledge distillation",
        )
        # Compound hyphens must be preserved when the prefix is in the keep set.
        self.assertEqual(
            repair_pdf_hyphenation("a pre- trained model and an open- source release"),
            "a pre-trained model and an open-source release",
        )

    def test_answerer_drops_noise_sentences(self):
        # The chunk mixes a clean sentence with Self-RAG control tokens and a
        # table row. The clean sentence must surface; the junk must not.
        answerer = TemplateAnswerer()
        noisy_chunk = {
            "chunk_id": "rag_007_c0042",
            "doc_id": "rag_007",
            "topic": "rag",
            "title": "Self-RAG paper",
            "text": (
                "Self-RAG introduces a critique framework for retrieval-augmented generation. "
                "[ISREL =Relevant] [ISSUP =Fully Supported] [ISUSE =5] Input given a chat history. "
                "0.4 0.5 0.6 0.7 0.8 0.9 1.0 Eval bpb arxiv 172M 425M 1.5B 7.5B."
            ),
            "source_url": "https://example.org/self-rag",
            "year": 2023,
        }
        res = answerer.generate_answer(
            [noisy_chunk], query="What does Self-RAG introduce?"
        )
        self.assertIn("Self-RAG introduces a critique framework", res["answer"])
        self.assertNotIn("[ISREL", res["answer"])
        self.assertNotIn("Eval bpb", res["answer"])
        self.assertNotIn("172M", res["answer"])

    def test_llm_answerer_without_key_falls_back_to_local(self):
        from unittest.mock import patch

        local_answer = {
            "answer": "AutoGen is an open-source programming framework [agent_006]",
            "citations": [{"doc_id": "agent_006", "title": "AutoGen", "source_url": "https://example.org"}],
        }
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}, clear=False):
            res = maybe_generate_llm_answer("What is AutoGen?", [self.mock_chunks[1]], local_answer)
        self.assertIsNone(res)

    def test_llm_answerer_mock_provider_skips_call(self):
        # Default LLM_PROVIDER=mock means "use the local template answerer only".
        # The helper must return None without ever touching the network.
        from unittest.mock import patch

        local_answer = {
            "answer": "AutoGen is an open-source programming framework [agent_006]",
            "citations": [{"doc_id": "agent_006", "title": "AutoGen", "source_url": "https://example.org"}],
        }
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}, clear=False):
            res = maybe_generate_llm_answer("What is AutoGen?", [self.mock_chunks[1]], local_answer)
        self.assertIsNone(res)

    def test_llm_answerer_does_not_run_when_local_abstained(self):
        # If the local answerer already abstained, calling an LLM would let it
        # invent a plausible-sounding answer — exactly the failure mode we
        # built the abstain gate to prevent. Helper must respect the abstain.
        from unittest.mock import patch
        from src.generation.template_answerer import NO_ANSWER_MESSAGE

        local_answer = {"answer": NO_ANSWER_MESSAGE, "citations": []}
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}, clear=False):
            res = maybe_generate_llm_answer("How do I cook pasta?", [self.mock_chunks[1]], local_answer)
        self.assertIsNone(res)

    def test_llm_answerer_returns_mode_when_configured(self):
        from unittest.mock import patch

        local_answer = {
            "answer": "RAG combines retrieval and generation [rag_001]",
            "citations": [{"doc_id": "rag_001", "title": "RAG Paper", "source_url": "https://example.org"}],
        }
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False):
            with patch(
                "src.generation.llm_answerer._call_chat_completion",
                return_value="RAG combines retrieved documents with generation to ground answers. [rag_001]",
            ):
                res = maybe_generate_llm_answer("Explain RAG", [self.mock_chunks[2]], local_answer)
        self.assertIsNotNone(res)
        self.assertEqual(res["mode"], "llm")
        self.assertEqual(res["provider"], "openai")
        self.assertIn("rag_001", res["answer"])

    def test_llm_output_cleanup_removes_cjk_artifacts(self):
        from src.generation.llm_answerer import _clean_llm_output

        cleaned = _clean_llm_output("RAG uses a vector밀 độ index 检索 knowledge.")
        self.assertIn("chỉ mục vector dày đặc", cleaned)
        self.assertNotIn("밀", cleaned)
        self.assertNotIn("检索", cleaned)

    def test_get_local_qa_answer_fallback(self):
        from unittest.mock import patch
        # When chunks.jsonl is empty or not found, should fall back to mock answer
        with patch('workflows.online_pipeline.PROJECT_ROOT', Path('/nonexistent_directory_for_testing_fallback')):
            res = get_local_qa_answer("RAG", "What is RAG?")
            self.assertIn("Retrieval-Augmented Generation", res["answer"])
            self.assertTrue(len(res["citations"]) > 0)

    def test_vietnamese_definition_question_uses_foundational_rag_source(self):
        res = get_local_qa_answer("RAG", "rag là gì")
        self.assertIn("RAG (Retrieval-Augmented Generation)", res["answer"])
        self.assertIn("dense vector index", res["answer"])
        self.assertEqual(res["citations"][0]["doc_id"], "rag_001")
        self.assertNotIn("active retrieval augmented generation", res["answer"].lower())

    def test_temporal_rag_question_uses_foundational_year(self):
        res = get_local_qa_answer("RAG", "rag ra doi nam bao nhieu")
        self.assertIn("2020", res["answer"])
        self.assertEqual(res["citations"][0]["doc_id"], "rag_001")
        self.assertNotIn("active retrieval", res["answer"].lower())
        self.assertNotIn("corrective", res["answer"].lower())

    def test_temporal_rag_question_works_in_english(self):
        res = get_local_qa_answer("RAG", "when was RAG introduced?")
        self.assertIn("2020", res["answer"])
        self.assertEqual(res["citations"][0]["doc_id"], "rag_001")

    def test_temporal_self_rag_question_uses_self_rag_source(self):
        res = get_local_qa_answer("RAG", "when was Self-RAG introduced?")
        self.assertIn("2023", res["answer"])
        self.assertEqual(res["citations"][0]["doc_id"], "rag_007")


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
        self.assertIn("AutoGPT is an open-source platform", res_gpt["answer"])
        self.assertEqual(res_gpt["citations"][0]["doc_id"], "agent_005")
        self.assertNotIn("img.shields.io", res_gpt["answer"])
        self.assertNotIn("<!--", res_gpt["answer"])
        self.assertNotIn("b/0.2", res_gpt["answer"])

        # 3. Test a completely irrelevant query with topic RAG on real processed chunks
        chunks_unrelated = retriever.retrieve("What is photosynthesis?", topic="RAG", top_k=3)
        res_unrelated = answerer.generate_answer(chunks_unrelated, query="What is photosynthesis?")
        self.assertIn("Không tìm thấy thông tin đủ liên quan", res_unrelated["answer"])
        self.assertEqual(len(res_unrelated["citations"]), 0)

if __name__ == '__main__':
    unittest.main()
