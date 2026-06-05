"""~500 chatbot scenarios across in-scope (deep), near-scope, and out-of-scope.

Each scenario asserts the *category* of response, not the exact wording:

- ``chitchat``  → conversational_handler returned a canned reply (no citations)
- ``meta``      → meta-question handler returned the self-intro
- ``corpus``    → RAG pipeline returned an answer with at least one citation
- ``abstain``   → relevance gate / abstain message, no citations
- ``either``    → either acceptable (e.g. corpus genuinely contains an example
                  matching the question, or LLM may or may not bridge)

These tests run with the default ``LLM_PROVIDER=mock`` so behaviour is
deterministic and the LLM path stays untouched.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.template_answerer import NO_ANSWER_MESSAGE  # noqa: E402
from workflows.online_pipeline import get_local_qa_answer  # noqa: E402


def _category_of(res):
    answer = (res.get("answer") or "").strip()
    citations = res.get("citations") or []
    if res.get("provider") == "router":
        return "meta"
    if not answer:
        return "empty"
    if answer == NO_ANSWER_MESSAGE:
        return "abstain"
    if "ChronoRAG, trợ lý nghiên cứu timeline" in answer:
        return "meta"
    if citations:
        return "corpus"
    return "chitchat"


# ===========================================================================
# IN-SCOPE: GREETINGS / THANKS / GOODBYE / FEELING (many variants)
# ===========================================================================

GREETINGS: List[Tuple[str, str, str]] = [
    # Plain VI
    ("chào", "rag", "chitchat"),
    ("Chào", "rag", "chitchat"),
    ("chao", "rag", "chitchat"),
    ("xin chào", "rag", "chitchat"),
    ("xin chao", "rag", "chitchat"),
    ("chào bạn", "rag", "chitchat"),
    ("chao ban", "rag", "chitchat"),
    ("chào bot", "rag", "chitchat"),
    ("chao bot", "rag", "chitchat"),
    ("chào ai", "rag", "chitchat"),
    ("chao ai", "rag", "chitchat"),
    # Time-of-day VI
    ("chào buổi sáng", "rag", "chitchat"),
    ("chao buoi sang", "rag", "chitchat"),
    ("chào buổi chiều", "rag", "chitchat"),
    ("chao buoi chieu", "rag", "chitchat"),
    ("chào buổi tối", "rag", "chitchat"),
    ("chao buoi toi", "rag", "chitchat"),
    # Casual VI
    ("ê", "rag", "chitchat"),
    ("ê bot", "rag", "chitchat"),
    ("e bot", "rag", "chitchat"),
    ("alo", "rag", "chitchat"),
    ("hú", "rag", "chitchat"),
    ("ơi", "rag", "chitchat"),
    # EN plain
    ("hi", "rag", "chitchat"),
    ("Hi", "rag", "chitchat"),
    ("HI", "rag", "chitchat"),
    ("hi there", "rag", "chitchat"),
    ("hello", "rag", "chitchat"),
    ("Hello", "rag", "chitchat"),
    ("HELLO", "rag", "chitchat"),
    ("hellow", "rag", "chitchat"),
    ("helo", "rag", "chitchat"),
    ("hey", "ai_agent", "chitchat"),
    ("Hey", "rag", "chitchat"),
    ("yo", "rag", "chitchat"),
    ("sup", "rag", "chitchat"),
    ("howdy", "rag", "chitchat"),
    ("greetings", "rag", "chitchat"),
    # Time-of-day EN
    ("good morning", "rag", "chitchat"),
    ("good afternoon", "rag", "chitchat"),
    ("good evening", "rag", "chitchat"),
    ("morning", "rag", "chitchat"),
    ("evening", "rag", "chitchat"),
    ("afternoon", "rag", "chitchat"),
]

THANKS: List[Tuple[str, str, str]] = [
    # VI thanks
    ("cảm ơn", "rag", "chitchat"),
    ("cam on", "rag", "chitchat"),
    ("Cảm ơn", "rag", "chitchat"),
    ("CẢM ƠN", "rag", "chitchat"),
    ("cảm ơn nhé", "rag", "chitchat"),
    ("cam on nhe", "rag", "chitchat"),
    ("cảm ơn nha", "rag", "chitchat"),
    ("cam on nha", "rag", "chitchat"),
    # EN thanks
    ("thanks", "rag", "chitchat"),
    ("Thanks", "rag", "chitchat"),
    ("thank you", "rag", "chitchat"),
    ("Thank you", "rag", "chitchat"),
    ("thank u", "rag", "chitchat"),
    ("thx", "rag", "chitchat"),
    ("tks", "rag", "chitchat"),
    ("ty", "rag", "chitchat"),
    ("thank", "rag", "chitchat"),
    # Positive acknowledgement
    ("ok", "rag", "chitchat"),
    ("okay", "rag", "chitchat"),
    ("okie", "rag", "chitchat"),
    ("okela", "rag", "chitchat"),
    ("oki", "rag", "chitchat"),
    ("tốt", "rag", "chitchat"),
    ("tot", "rag", "chitchat"),
    ("good", "rag", "chitchat"),
    ("nice", "rag", "chitchat"),
    ("cool", "rag", "chitchat"),
    ("great", "rag", "chitchat"),
    ("perfect", "rag", "chitchat"),
    ("tuyệt", "rag", "chitchat"),
    ("tuyet", "rag", "chitchat"),
    ("tuyệt vời", "rag", "chitchat"),
    ("ngon", "rag", "chitchat"),
    ("đỉnh", "rag", "chitchat"),
    ("dinh", "rag", "chitchat"),
    ("xuất sắc", "rag", "chitchat"),
    ("xuat sac", "rag", "chitchat"),
]

GOODBYES: List[Tuple[str, str, str]] = [
    # EN
    ("bye", "rag", "chitchat"),
    ("Bye", "rag", "chitchat"),
    ("byebye", "rag", "chitchat"),
    ("goodbye", "rag", "chitchat"),
    ("good bye", "rag", "chitchat"),
    ("see you", "rag", "chitchat"),
    ("see ya", "rag", "chitchat"),
    ("later", "rag", "chitchat"),
    ("cya", "rag", "chitchat"),
    # VI
    ("tạm biệt", "rag", "chitchat"),
    ("tam biet", "rag", "chitchat"),
    ("hẹn gặp lại", "rag", "chitchat"),
    ("hen gap lai", "rag", "chitchat"),
    ("chào nhé", "rag", "chitchat"),
    ("chao nhe", "rag", "chitchat"),
]

FEELING: List[Tuple[str, str, str]] = [
    ("khỏe không", "rag", "chitchat"),
    ("khoe khong", "rag", "chitchat"),
    ("khỏe chứ", "rag", "chitchat"),
    ("khoe chu", "rag", "chitchat"),
    ("bạn khỏe không", "rag", "chitchat"),
    ("ban khoe khong", "rag", "chitchat"),
    ("how are you", "rag", "chitchat"),
    ("How are you?", "rag", "chitchat"),
    ("how r u", "rag", "chitchat"),
    ("how's it going", "rag", "chitchat"),
    ("you good", "rag", "chitchat"),
    ("you ok", "rag", "chitchat"),
]

TIME_DATE: List[Tuple[str, str, str]] = [
    ("mấy giờ rồi", "rag", "chitchat"),
    ("may gio roi", "rag", "chitchat"),
    ("bây giờ là mấy giờ", "rag", "chitchat"),
    ("bay gio la may gio", "rag", "chitchat"),
    ("what time is it", "rag", "chitchat"),
    ("What time is it?", "rag", "chitchat"),
    ("what time", "rag", "chitchat"),
    ("current time", "rag", "chitchat"),
    ("hôm nay là thứ mấy", "rag", "chitchat"),
    ("hom nay la thu may", "rag", "chitchat"),
    ("hôm nay ngày mấy", "rag", "chitchat"),
    ("hom nay ngay may", "rag", "chitchat"),
    ("what day is today", "rag", "chitchat"),
    ("today's date", "rag", "chitchat"),
    ("what's the date", "rag", "chitchat"),
]

# ===========================================================================
# META: questions ABOUT the bot itself
# ===========================================================================

META: List[Tuple[str, str, str]] = [
    # --- Identity (canonical) ---
    ("bạn là ai", "rag", "meta"),
    ("Bạn là ai?", "rag", "meta"),
    ("BẠN LÀ AI", "rag", "meta"),
    ("ban la ai", "rag", "meta"),
    ("Ban la ai?", "rag", "meta"),
    ("bạn tên gì", "rag", "meta"),
    ("ban ten gi", "rag", "meta"),
    ("tên bạn là gì", "rag", "meta"),
    ("bot là ai", "rag", "meta"),
    ("bot la ai", "rag", "meta"),
    ("bot là gì", "rag", "meta"),
    ("bot tên gì", "rag", "meta"),
    # --- Casual VI pronouns (m / mày / em / anh / tao) ---
    ("m là ai", "rag", "meta"),
    ("M là ai?", "rag", "meta"),
    ("m la ai", "rag", "meta"),
    ("mày là ai", "rag", "meta"),
    ("may la ai", "rag", "meta"),
    ("em là ai", "rag", "meta"),
    ("e la ai", "rag", "meta"),
    ("anh là ai", "rag", "meta"),
    ("tao là ai", "rag", "meta"),
    ("m tên gì", "rag", "meta"),
    ("m ten gi", "rag", "meta"),
    # --- "Who made you" ---
    ("ai tạo ra m", "rag", "meta"),
    ("ai tao ra m", "rag", "meta"),
    ("ai tạo ra bạn", "rag", "meta"),
    ("ai làm ra bạn", "rag", "meta"),
    ("ai tao ban", "rag", "meta"),
    ("who made you", "rag", "meta"),
    # --- Capability (canonical) ---
    ("bạn làm được gì", "rag", "meta"),
    ("bạn có thể làm gì", "rag", "meta"),
    ("bạn giúp gì", "rag", "meta"),
    ("bạn giúp được gì", "rag", "meta"),
    ("bạn biết gì", "rag", "meta"),
    # --- Capability (casual VI with shortcuts đc/dc/ddc/j) ---
    ("m làm gì", "rag", "meta"),
    ("m làm được gì", "rag", "meta"),
    ("m làm đc gì", "rag", "meta"),
    ("m làm ddc gì", "rag", "meta"),  # exact case from screenshot
    ("m làm dc gì", "rag", "meta"),
    ("m lam dc gi", "rag", "meta"),
    ("m lam duoc gi", "rag", "meta"),
    ("m có thể làm gì", "rag", "meta"),
    ("m có thể làm đc gì", "rag", "meta"),  # exact case from screenshot
    ("m có thể làm dc gì", "rag", "meta"),
    ("m co the lam dc gi", "rag", "meta"),
    ("mày làm gì được", "rag", "meta"),
    ("may lam gi duoc", "rag", "meta"),
    ("m biết gì", "rag", "meta"),
    ("m bik gì", "rag", "meta"),
    ("m bik j", "rag", "meta"),
    ("m biết làm gì", "rag", "meta"),
    ("m biết những gì", "rag", "meta"),
    ("m giúp dc gì", "rag", "meta"),
    ("m giúp gì được", "rag", "meta"),
    ("m giup dc gi", "rag", "meta"),
    ("bot giúp gì", "rag", "meta"),
    ("bot làm dc gì", "rag", "meta"),
    ("bot lam duoc gi", "rag", "meta"),
    # --- Usage / how-to ---
    ("dùng sao", "rag", "meta"),
    ("dung sao", "rag", "meta"),
    ("dùng thế nào", "rag", "meta"),
    ("dung the nao", "rag", "meta"),
    ("hướng dẫn", "rag", "meta"),
    ("huong dan", "rag", "meta"),
    ("hướng dẫn dùng", "rag", "meta"),
    ("how do i use you", "rag", "meta"),
    ("how does this work", "rag", "meta"),
    # --- English ---
    ("who are you", "rag", "meta"),
    ("Who are you?", "rag", "meta"),
    ("what are you", "rag", "meta"),
    ("what can you do", "rag", "meta"),
    ("What can you do?", "rag", "meta"),
    ("what do you know", "rag", "meta"),
    ("what are your capabilities", "rag", "meta"),
    ("introduce yourself", "rag", "meta"),
    ("tell me about yourself", "rag", "meta"),
    ("your name", "rag", "meta"),
    ("what's your name", "rag", "meta"),
    ("giới thiệu bản thân", "rag", "meta"),
    ("giới thiệu về bạn", "rag", "meta"),
    # --- Help shortcuts ---
    ("help me", "rag", "meta"),
    ("help please", "rag", "meta"),
    ("/help", "rag", "meta"),
    ("help", "rag", "meta"),
    ("?", "rag", "meta"),
    ("??", "rag", "meta"),
]

# ===========================================================================
# COMMON AI/ML/PROGRAMMING TERMS (outside corpus)
# Bot should give a canned brief definition, no citation.
# ===========================================================================

COMMON_TERMS: List[Tuple[str, str, str]] = [
    # Transformer
    ("transformer là gì", "rag", "chitchat"),
    ("transformer la gi", "rag", "chitchat"),
    ("Transformer là gì", "rag", "chitchat"),
    ("Transformer là gì?", "rag", "chitchat"),
    ("what is transformer", "rag", "chitchat"),
    ("what is a transformer", "rag", "chitchat"),
    ("What is the Transformer?", "rag", "chitchat"),
    ("define transformer", "rag", "chitchat"),
    ("explain transformer", "rag", "chitchat"),
    ("giải thích transformer", "rag", "chitchat"),
    ("transformer", "rag", "chitchat"),
    # LLM
    ("LLM là gì", "rag", "chitchat"),
    ("llm la gi", "rag", "chitchat"),
    ("LLM", "rag", "chitchat"),
    ("what is LLM", "rag", "chitchat"),
    ("what is an LLM", "rag", "chitchat"),
    ("Large Language Model là gì", "rag", "chitchat"),
    # GPT
    ("GPT là gì", "rag", "chitchat"),
    ("gpt la gi", "rag", "chitchat"),
    ("what is GPT", "rag", "chitchat"),
    ("GPT", "rag", "chitchat"),
    # BERT
    ("BERT là gì", "rag", "chitchat"),
    ("bert la gi", "rag", "chitchat"),
    ("what is BERT", "rag", "chitchat"),
    ("BERT", "rag", "chitchat"),
    # Fine-tuning
    ("fine-tune là gì", "rag", "chitchat"),
    ("fine tuning là gì", "rag", "chitchat"),
    ("finetune là gì", "rag", "chitchat"),
    ("what is fine-tuning", "rag", "chitchat"),
    ("what is finetuning", "rag", "chitchat"),
    ("explain fine-tuning", "rag", "chitchat"),
    # Prompt
    ("prompt là gì", "rag", "chitchat"),
    ("prompt la gi", "rag", "chitchat"),
    ("what is prompt", "rag", "chitchat"),
    ("what is a prompt", "rag", "chitchat"),
    # Embedding
    ("embedding là gì", "rag", "chitchat"),
    ("what is embedding", "rag", "chitchat"),
    ("what is an embedding", "rag", "chitchat"),
    # Attention
    ("attention là gì", "rag", "chitchat"),
    ("what is attention", "rag", "chitchat"),
    # Deep learning
    ("deep learning là gì", "rag", "chitchat"),
    ("deep learning la gi", "rag", "chitchat"),
    ("what is deep learning", "rag", "chitchat"),
    # Machine learning
    ("machine learning là gì", "rag", "chitchat"),
    ("machine learning la gi", "rag", "chitchat"),
    ("what is machine learning", "rag", "chitchat"),
    ("ML là gì", "rag", "chitchat"),
    ("ML", "rag", "chitchat"),
    # NLP
    ("nlp là gì", "rag", "chitchat"),
    ("NLP", "rag", "chitchat"),
    ("what is NLP", "rag", "chitchat"),
    # Vector DB
    ("vector database là gì", "rag", "chitchat"),
    ("vector db là gì", "rag", "chitchat"),
    ("what is a vector database", "rag", "chitchat"),
    ("what is vector store", "rag", "chitchat"),
    # BM25
    ("BM25 là gì", "rag", "chitchat"),
    ("what is BM25", "rag", "chitchat"),
    # Hallucination
    ("hallucination là gì", "rag", "chitchat"),
    ("what is hallucination", "rag", "chitchat"),
    # RL
    ("reinforcement learning là gì", "rag", "chitchat"),
    ("RLHF là gì", "rag", "chitchat"),
    ("what is RLHF", "rag", "chitchat"),
    ("RL là gì", "rag", "chitchat"),
    # Neural network
    ("neural network là gì", "rag", "chitchat"),
    ("mạng neural là gì", "rag", "chitchat"),
    ("what is a neural network", "rag", "chitchat"),
    # CNN/RNN/LSTM
    ("CNN là gì", "rag", "chitchat"),
    ("what is CNN", "rag", "chitchat"),
    ("RNN là gì", "rag", "chitchat"),
    ("what is RNN", "rag", "chitchat"),
    ("LSTM là gì", "rag", "chitchat"),
    ("what is LSTM", "rag", "chitchat"),
    # Backprop / gradient descent
    ("backpropagation là gì", "rag", "chitchat"),
    ("backprop là gì", "rag", "chitchat"),
    ("what is backpropagation", "rag", "chitchat"),
    ("gradient descent là gì", "rag", "chitchat"),
    ("what is gradient descent", "rag", "chitchat"),
    ("SGD là gì", "rag", "chitchat"),
    # Overfitting
    ("overfitting là gì", "rag", "chitchat"),
    ("what is overfitting", "rag", "chitchat"),
    # Regularization
    ("regularization là gì", "rag", "chitchat"),
    ("what is regularization", "rag", "chitchat"),
    # Classification / Regression
    ("classification là gì", "rag", "chitchat"),
    ("what is classification", "rag", "chitchat"),
    ("regression là gì", "rag", "chitchat"),
    ("what is regression", "rag", "chitchat"),
    # Precision / recall / F1
    ("precision là gì", "rag", "chitchat"),
    ("what is precision", "rag", "chitchat"),
    ("recall là gì", "rag", "chitchat"),
    ("what is recall", "rag", "chitchat"),
    ("F1 là gì", "rag", "chitchat"),
    ("what is F1", "rag", "chitchat"),
    # Tokenization
    ("tokenization là gì", "rag", "chitchat"),
    ("what is tokenization", "rag", "chitchat"),
    # Python / PyTorch / TF
    ("python là gì", "rag", "chitchat"),
    ("what is python", "rag", "chitchat"),
    ("pytorch là gì", "rag", "chitchat"),
    ("what is pytorch", "rag", "chitchat"),
    ("tensorflow là gì", "rag", "chitchat"),
    ("what is tensorflow", "rag", "chitchat"),
    # API / FastAPI / JSON / Git / Docker / SQL
    ("API là gì", "rag", "chitchat"),
    ("what is API", "rag", "chitchat"),
    ("REST API là gì", "rag", "chitchat"),
    ("FastAPI là gì", "rag", "chitchat"),
    ("what is FastAPI", "rag", "chitchat"),
    ("JSON là gì", "rag", "chitchat"),
    ("what is JSON", "rag", "chitchat"),
    ("git là gì", "rag", "chitchat"),
    ("what is git", "rag", "chitchat"),
    ("docker là gì", "rag", "chitchat"),
    ("what is docker", "rag", "chitchat"),
    ("SQL là gì", "rag", "chitchat"),
    ("what is SQL", "rag", "chitchat"),
]

# ===========================================================================
# IN-SCOPE CORPUS: RAG (deep coverage)
# ===========================================================================

CORPUS_RAG: List[Tuple[str, str, str]] = [
    # Core RAG
    ("What is RAG?", "rag", "corpus"),
    ("what is rag", "rag", "corpus"),
    ("RAG là gì", "rag", "corpus"),
    ("RAG la gi", "rag", "corpus"),
    ("RAG la gi?", "rag", "corpus"),
    ("Explain RAG", "rag", "corpus"),
    ("Define RAG", "rag", "corpus"),
    ("Tell me about RAG", "rag", "corpus"),
    ("How does RAG work?", "rag", "corpus"),
    ("RAG hoạt động thế nào", "rag", "corpus"),
    ("RAG architecture", "rag", "corpus"),
    ("Components of RAG", "rag", "corpus"),
    ("Use cases of RAG", "rag", "corpus"),
    ("Why use RAG", "rag", "corpus"),
    # Self-RAG
    ("What is Self-RAG?", "rag", "corpus"),
    ("Self-RAG là gì", "rag", "corpus"),
    ("How does Self-RAG decide when to retrieve?", "rag", "corpus"),
    ("Explain Self-RAG", "rag", "corpus"),
    # GraphRAG (corpus probably lacks; mark either)
    ("What is GraphRAG?", "rag", "either"),
    ("GraphRAG là gì", "rag", "either"),
    # REALM
    ("What is REALM?", "rag", "corpus"),
    ("REALM là gì", "rag", "corpus"),
    ("How does REALM pre-train?", "rag", "corpus"),
    # DPR
    ("What is dense passage retrieval?", "rag", "corpus"),
    ("Tell me about DPR", "rag", "corpus"),
    ("DPR là gì", "rag", "corpus"),
    ("Dense Passage Retriever?", "rag", "corpus"),
    # RETRO
    ("What is RETRO?", "rag", "corpus"),
    ("Explain RETRO", "rag", "corpus"),
    # FLARE / Active retrieval
    ("Explain active retrieval-augmented generation", "rag", "corpus"),
    ("What is FLARE?", "rag", "corpus"),
    # CRAG / Corrective
    ("What is corrective retrieval-augmented generation?", "rag", "corpus"),
    ("What is CRAG?", "rag", "corpus"),
    # Atlas
    ("What is Atlas language model?", "rag", "corpus"),
    ("Atlas language model là gì", "rag", "corpus"),
]

# ===========================================================================
# IN-SCOPE CORPUS: AI Agent (deep coverage)
# ===========================================================================

CORPUS_AGENT: List[Tuple[str, str, str]] = [
    # AutoGen
    ("What is AutoGen?", "ai_agent", "corpus"),
    ("AutoGen là gì", "ai_agent", "corpus"),
    ("Explain AutoGen", "ai_agent", "corpus"),
    ("Tell me about AutoGen", "ai_agent", "corpus"),
    ("How does AutoGen work?", "ai_agent", "corpus"),
    # AutoGPT
    ("What is AutoGPT?", "ai_agent", "corpus"),
    ("AutoGPT là gì", "ai_agent", "corpus"),
    ("Tell me about AutoGPT", "ai_agent", "corpus"),
    # ReAct
    ("What is ReAct?", "ai_agent", "corpus"),
    ("ReAct là gì", "ai_agent", "corpus"),
    ("How does ReAct work?", "ai_agent", "corpus"),
    ("Explain ReAct framework", "ai_agent", "corpus"),
    # Toolformer
    ("What is Toolformer?", "ai_agent", "corpus"),
    ("Toolformer là gì", "ai_agent", "corpus"),
    # LangGraph
    ("What is LangGraph?", "ai_agent", "corpus"),
    ("LangGraph là gì", "ai_agent", "corpus"),
    # Generic agent
    ("What is an LLM agent?", "ai_agent", "corpus"),
    ("Autonomous agents?", "ai_agent", "corpus"),
    ("Multi-agent frameworks?", "ai_agent", "corpus"),
    ("Tell me about autonomous agents", "ai_agent", "corpus"),
]

# ===========================================================================
# IN-SCOPE CORPUS: Knowledge Distillation (deep coverage)
# ===========================================================================

CORPUS_KD: List[Tuple[str, str, str]] = [
    # DistilBERT
    ("What is DistilBERT?", "knowledge_distillation", "corpus"),
    ("DistilBERT là gì", "knowledge_distillation", "corpus"),
    ("Explain DistilBERT", "knowledge_distillation", "corpus"),
    ("How does DistilBERT work?", "knowledge_distillation", "corpus"),
    # TinyBERT
    ("What is TinyBERT?", "knowledge_distillation", "corpus"),
    ("TinyBERT là gì", "knowledge_distillation", "corpus"),
    ("Explain TinyBERT", "knowledge_distillation", "corpus"),
    # MobileBERT
    ("What is MobileBERT?", "knowledge_distillation", "corpus"),
    ("MobileBERT là gì", "knowledge_distillation", "corpus"),
    # MiniLM
    ("What is MiniLM?", "knowledge_distillation", "corpus"),
    ("MiniLM là gì", "knowledge_distillation", "corpus"),
    # KD generic
    ("How does knowledge distillation work?", "knowledge_distillation", "corpus"),
    ("Tell me about teacher-student training", "knowledge_distillation", "corpus"),
    ("teacher-student là gì", "knowledge_distillation", "corpus"),
    ("knowledge distillation là gì", "knowledge_distillation", "corpus"),
]

# ===========================================================================
# CROSS-TOPIC: question mentions a topic different from the dropdown
# ===========================================================================

CROSS_TOPIC: List[Tuple[str, str, str]] = [
    ("What is AutoGen?", "rag", "corpus"),
    ("What is DistilBERT?", "rag", "corpus"),
    ("What is RAG?", "knowledge_distillation", "corpus"),
    ("RAG là gì", "knowledge_distillation", "corpus"),
    ("AutoGPT là gì", "knowledge_distillation", "corpus"),
    ("How does ReAct work?", "knowledge_distillation", "corpus"),
    ("What is DistilBERT?", "ai_agent", "corpus"),
    ("What is dense passage retrieval?", "knowledge_distillation", "corpus"),
    ("Explain Self-RAG", "ai_agent", "corpus"),
    ("Explain TinyBERT", "rag", "corpus"),
]

# ===========================================================================
# ADVERSARIAL / EDGE
# ===========================================================================

ADVERSARIAL: List[Tuple[str, str, str]] = [
    ("", "rag", "abstain"),                # empty
    ("   ", "rag", "abstain"),             # whitespace
    ("?", "rag", "meta"),                  # bare ? = help
    ("??", "rag", "meta"),
    ("...", "rag", "abstain"),
    ("!!!", "rag", "abstain"),
    ("@#$%", "rag", "abstain"),
    ("asdfghjk", "rag", "abstain"),
    ("xyzzy", "rag", "abstain"),
    ("12345", "rag", "abstain"),
    ("aaaaaaaaaa", "rag", "abstain"),
    # very long random
    ("a " * 200, "rag", "abstain"),
    # code injection style
    ("'; DROP TABLE users; --", "rag", "abstain"),
    ("<script>alert(1)</script>", "rag", "abstain"),
    # mixed unicode noise
    ("░▒▓█", "rag", "abstain"),
    ("🤔🤔🤔", "rag", "abstain"),
]

# ===========================================================================
# OUT-OF-SCOPE: bot must abstain (or LLM may abstain). Several of these
# accidentally appear as EXAMPLES in Self-RAG paper, so 'either' is the
# fair expectation -- the corpus genuinely contains those literal questions.
# ===========================================================================

OUT_OF_SCOPE: List[Tuple[str, str, str]] = [
    ("How do I cook pasta?", "rag", "abstain"),
    ("Recipe for pho", "rag", "abstain"),
    ("Cách nấu phở", "rag", "abstain"),
    ("What is the weather today", "rag", "abstain"),
    ("Thời tiết hôm nay", "rag", "either"),
    ("Giá Bitcoin hôm nay", "rag", "abstain"),
    ("Phim hay 2024", "rag", "abstain"),
    ("Best movie 2024", "rag", "either"),
    ("Top 10 anime", "rag", "abstain"),
    ("Stock price of NVDA", "rag", "abstain"),
    # Self-RAG paper has "Capital of France" / "World Cup" as example queries:
    ("Capital of France", "rag", "either"),
    ("Thủ đô của Pháp", "rag", "abstain"),
    ("Who won the world cup", "rag", "either"),
    ("How to learn Java", "rag", "either"),  # rag_007 mentions Java/Lucene
    ("Lập trình Python", "rag", "either"),   # python may match
    # Genuinely unrelated
    ("What is photosynthesis?", "rag", "abstain"),
    ("Lịch sử Việt Nam", "rag", "abstain"),
    ("Quantum physics for dummies", "rag", "abstain"),
    ("How to fix my car", "rag", "abstain"),
    ("Tôi nên ăn gì tối nay", "rag", "abstain"),
    ("What stocks should I buy", "rag", "abstain"),
    ("Recommend a movie", "rag", "abstain"),
    ("How tall is Mount Everest", "rag", "abstain"),
    ("Best programming language for ML", "rag", "either"),  # may match LLM/python
]

NEGATIVE: List[Tuple[str, str, str]] = [
    ("ngu", "rag", "chitchat"),
    ("tệ", "rag", "chitchat"),
    ("te", "rag", "chitchat"),
    ("stupid", "rag", "chitchat"),
    ("dumb", "rag", "chitchat"),
    ("shit", "rag", "chitchat"),
    ("rác", "rag", "chitchat"),
    ("rac", "rag", "chitchat"),
    ("kém", "rag", "chitchat"),
    ("kem", "rag", "chitchat"),
]

TYPOS: List[Tuple[str, str, str]] = [
    ("wat is rag", "rag", "corpus"),
    ("waht is RAG", "rag", "corpus"),
    ("rag la gi", "rag", "corpus"),
    ("rag la gi?", "rag", "corpus"),
    ("AutogPt", "ai_agent", "corpus"),
    ("autogen", "ai_agent", "corpus"),
    ("AUTOGEN", "ai_agent", "corpus"),
    ("distilbert", "knowledge_distillation", "corpus"),
    ("DISTILBERT", "knowledge_distillation", "corpus"),
    ("self rag", "rag", "corpus"),
    ("Self-Rag", "rag", "corpus"),
    ("reactor", "rag", "abstain"),  # NOT react agent
]


ALL_SCENARIOS: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    ("greetings", GREETINGS),
    ("thanks", THANKS),
    ("goodbyes", GOODBYES),
    ("feeling", FEELING),
    ("time_date", TIME_DATE),
    ("meta", META),
    ("common_terms", COMMON_TERMS),
    ("corpus_rag", CORPUS_RAG),
    ("corpus_agent", CORPUS_AGENT),
    ("corpus_kd", CORPUS_KD),
    ("cross_topic", CROSS_TOPIC),
    ("adversarial", ADVERSARIAL),
    ("out_of_scope", OUT_OF_SCOPE),
    ("negative", NEGATIVE),
    ("typos", TYPOS),
]


# ===========================================================================
# Test driver
# ===========================================================================

# Per-category leniency: ≤15% misses tolerated by default; bumped for
# categories where corpus quirks make exact assertions fragile.
_CATEGORY_LENIENCY = {
    "out_of_scope": 0.20,  # corpus has stray example questions
    "cross_topic": 0.20,
    "adversarial": 0.20,
}


def _category_of_safe(res):
    try:
        return _category_of(res)
    except Exception:
        return "empty"


def _safe_console(text: str) -> str:
    return str(text).encode("ascii", errors="backslashreplace").decode("ascii")


class TestChatbotScenarios(unittest.TestCase):
    """Run every scenario through ``get_local_qa_answer`` with mock LLM."""

    @classmethod
    def setUpClass(cls):
        cls._env_patcher = patch.dict("os.environ", {"LLM_PROVIDER": "mock"}, clear=False)
        cls._env_patcher.start()
        cls.totals = {}
        cls.passed = {}
        cls.failures = []

    @classmethod
    def tearDownClass(cls):
        cls._env_patcher.stop()
        total = sum(cls.totals.values())
        passed = sum(cls.passed.values())
        print()
        print("=" * 72)
        print(f"Chatbot scenario coverage: {passed}/{total}  ({passed * 100 // max(total,1)}%)")
        print("=" * 72)
        for cat in cls.totals:
            p, t = cls.passed.get(cat, 0), cls.totals[cat]
            mark = "OK  " if p == t else "WARN"
            print(f"  [{mark}] {cat:<22} {p}/{t}")
        if cls.failures:
            print(f"\nFailures ({len(cls.failures)} total, showing first 15):")
            for cat, q, want, got, ans in cls.failures[:15]:
                excerpt = (ans[:70] + "…") if len(ans) > 70 else ans
                excerpt = excerpt.replace("\n", " ")
                qshort = (q[:30] + "…") if len(q) > 30 else q
                print(f"  {cat:<18} | {_safe_console(qshort)!r:<35} want={want:<8} got={got:<8} ans={_safe_console(excerpt)!r}")
        print("=" * 72)

    def _run_category(self, category: str, scenarios: List[Tuple[str, str, str]]):
        cls = type(self)
        cls.totals[category] = cls.totals.get(category, 0) + len(scenarios)
        cls.passed.setdefault(category, 0)
        misses = []
        for question, topic, expected in scenarios:
            res = get_local_qa_answer(topic, question)
            got = _category_of_safe(res)
            if expected == "either":
                ok = got in {"corpus", "abstain", "chitchat"}
            else:
                ok = got == expected
            if ok:
                cls.passed[category] += 1
            else:
                fail = (category, question, expected, got, (res.get("answer") or "")[:200])
                cls.failures.append(fail)
                misses.append(fail)
        threshold = max(1, int(_CATEGORY_LENIENCY.get(category, 0.15) * len(scenarios)))
        self.assertLessEqual(
            len(misses),
            threshold,
            f"Too many misses in '{category}': "
            f"{cls.passed[category]}/{len(scenarios)} pass (threshold {threshold} misses). "
            f"First miss: {misses[0] if misses else 'n/a'}",
        )


def _make_test(name, scenarios):
    def test(self):
        self._run_category(name, scenarios)
    test.__name__ = f"test_{name}"
    return test


for _name, _scenarios in ALL_SCENARIOS:
    setattr(TestChatbotScenarios, f"test_{_name}", _make_test(_name, _scenarios))


if __name__ == "__main__":
    unittest.main()
