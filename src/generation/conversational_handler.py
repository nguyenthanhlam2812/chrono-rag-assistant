"""Lightweight chitchat / common-term handler.

Sits between the meta-question gate (``bạn là ai``) and the RAG pipeline.
Catches simple conversational input -- greetings, thanks, time, common
AI/ML terms -- so the bot responds pleasantly instead of either making the
RAG layer hallucinate from the corpus or abstaining on questions a user
reasonably expects an answer to.

Returns ``{"answer": str, "citations": []}`` on a hit, or ``None`` to let
the caller fall through to the regular RAG pipeline.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

# Vietnamese diacritic characters -- any presence flips us to Vietnamese.
_VI_DIACRITIC_RE = re.compile(
    "[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)

# Without diacritics we still want to detect Vietnamese ("ban la ai",
# "rag la gi"). These tokens are extremely common in VI but rare in EN
# sentences of the lengths we deal with here.
_VI_SIGNAL_TOKENS = frozenset({
    "la", "gi", "ko", "k", "nhe", "minh", "ban", "anh", "em", "co",
    "khong", "khac", "vay", "the", "nao", "duoc", "dc", "moi", "roi",
    "lam", "ai", "biet", "hieu", "noi", "voi", "cho", "cua", "thi",
    "boi", "tai", "sao", "tot", "te", "gioi",
})

_FILLER_CHARS = " .,!?…\"'`’"


def _detect_vietnamese(q: str) -> bool:
    if _VI_DIACRITIC_RE.search(q):
        return True
    tokens = re.findall(r"[a-z]+", q.lower())
    if any(t in _VI_SIGNAL_TOKENS for t in tokens):
        return True
    return False


def _norm(q: str) -> str:
    """Lowercase + strip surrounding punctuation. Used for exact-match keys."""
    return q.lower().strip().strip(_FILLER_CHARS)


# ---------------------------------------------------------------------------
# Canned conversational categories
# ---------------------------------------------------------------------------

_GREETINGS = (
    "chào", "xin chào", "chao", "xin chao", "chào bạn", "chao ban",
    "chào bot", "chao bot", "chào ai", "chao ai",
    "chào buổi sáng", "chao buoi sang", "chào buổi tối", "chao buoi toi",
    "chào buổi chiều", "chao buoi chieu",
    "ê", "ê bot", "e bot", "ơi", "alo", "hú",
    "hi", "hi there", "hello", "hellow", "helo", "hey", "yo", "sup",
    "howdy", "greetings", "good morning", "good afternoon", "good evening",
    "morning", "evening", "afternoon",
)

_THANKS = (
    "cảm ơn", "cam on", "cảm ơn nhé", "cam on nhe", "cảm ơn nha", "cam on nha",
    "thanks", "thank you", "thank u", "thx", "tks", "ty", "thank",
    "tuyệt", "tuyet", "ok", "okay", "okie", "okela", "oki", "good", "tốt", "tot",
    "nice", "cool", "great", "ngon", "đỉnh", "dinh", "xuất sắc", "xuat sac",
    "perfect", "tuyệt vời", "tuyet voi",
)

_GOODBYES = (
    "bye", "byebye", "goodbye", "good bye", "see you", "see ya", "later",
    "cya", "tạm biệt", "tam biet", "hẹn gặp lại", "hen gap lai", "chào nhé", "chao nhe",
)

_FEELING_QUESTIONS = (
    "khỏe không", "khoe khong", "khỏe chứ", "khoe chu",
    "bạn khỏe không", "ban khoe khong",
    "how are you", "how r u", "how's it going", "you good", "you ok",
)

_TIME_PATTERNS = (
    "mấy giờ", "may gio", "bây giờ là mấy giờ", "bay gio la may gio",
    "what time", "what time is it", "current time",
)
_DATE_PATTERNS = (
    "hôm nay là thứ mấy", "hom nay la thu may",
    "hôm nay ngày mấy", "hom nay ngay may",
    "what day", "today's date", "what's the date",
)

_NEGATIVE_PATTERNS = (
    "ngu", "tệ", "te", "stupid", "dumb", "shit", "rác", "rac",
    "tệ quá", "te qua", "kém", "kem",
)


def _greeting(q: str, is_vi: bool) -> str:
    if is_vi:
        return (
            "Chào đại ka 👋 Mình là ChronoRAG. Mình hỗ trợ 3 chủ đề: "
            "RAG, AI Agent, Knowledge Distillation. Cứ thoải mái hỏi nhé!"
        )
    return (
        "Hi there 👋 I'm ChronoRAG. I cover three topics: "
        "RAG, AI Agents, and Knowledge Distillation. Ask away!"
    )


def _thanks(is_vi: bool) -> str:
    return "Không có gì 😄 Nếu cần hỏi thêm gì cứ nhắn mình." if is_vi \
        else "You're welcome 😄 Ping me anytime."


def _goodbye(is_vi: bool) -> str:
    return "Tạm biệt đại ka, chúc đại ka làm việc hiệu quả 👋" if is_vi \
        else "Bye! Good luck with your research 👋"


def _feeling(is_vi: bool) -> str:
    return (
        "Mình là bot nên không có cảm xúc 😅 nhưng pipeline đang chạy ổn. "
        "Đại ka có muốn hỏi về RAG, AI Agent hay Knowledge Distillation không?"
        if is_vi
        else
        "I'm a bot so no feelings here 😅 pipeline is healthy though. "
        "Want to ask about RAG, AI Agents, or Knowledge Distillation?"
    )


def _time_now(is_vi: bool) -> str:
    now = datetime.now()
    return (
        f"Bây giờ là {now.strftime('%H:%M')} ngày {now.strftime('%d/%m/%Y')}."
        if is_vi
        else f"It's {now.strftime('%H:%M')} on {now.strftime('%Y-%m-%d')}."
    )


def _date_now(is_vi: bool) -> str:
    now = datetime.now()
    weekdays_vi = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    if is_vi:
        return f"Hôm nay là {weekdays_vi[now.weekday()]}, ngày {now.strftime('%d/%m/%Y')}."
    return f"Today is {now.strftime('%A, %Y-%m-%d')}."


def _negative_reply(is_vi: bool) -> str:
    return (
        "Mình hiểu là câu trả lời chưa tốt 🙏 đại ka thử hỏi cụ thể hơn về "
        "RAG, AI Agent hoặc Knowledge Distillation xem mình có giúp được không nhé."
        if is_vi
        else
        "Sorry the previous answer wasn't great 🙏 Try a more specific question "
        "about RAG, AI Agents, or Knowledge Distillation."
    )


# ---------------------------------------------------------------------------
# Common AI/ML terms outside the corpus
# Brief definitions so users aren't met with "abstain" on basic vocabulary.
# ---------------------------------------------------------------------------

_COMMON_TERMS: Dict[str, Dict[str, str]] = {
    "transformer": {
        "vi": "Transformer là kiến trúc mạng neural dùng cơ chế self-attention, "
              "do Vaswani et al. giới thiệu năm 2017 ('Attention Is All You Need'). "
              "Là nền tảng của BERT, GPT, LLaMA và hầu hết LLM hiện đại.",
        "en": "Transformer is a neural-network architecture built on self-attention, "
              "introduced by Vaswani et al. in 2017 ('Attention Is All You Need'). "
              "It underpins BERT, GPT, LLaMA, and most modern LLMs.",
    },
    "llm": {
        "vi": "Large Language Model (LLM) là mô hình ngôn ngữ lớn (hàng tỷ tham số) "
              "huấn luyện trên dữ liệu text khổng lồ. Ví dụ: GPT-4, Claude, Gemini, LLaMA. "
              "ChronoRAG cũng có thể kết nối LLM để trợ giúp diễn đạt câu trả lời.",
        "en": "A Large Language Model (LLM) is a billion-parameter language model "
              "trained on massive text corpora. Examples: GPT-4, Claude, Gemini, LLaMA. "
              "ChronoRAG can optionally wrap an LLM to polish its answers.",
    },
    "gpt": {
        "vi": "GPT (Generative Pre-trained Transformer) là dòng LLM của OpenAI, "
              "huấn luyện autoregressive trên text lớn. GPT-1 (2018), GPT-2 (2019), "
              "GPT-3 (2020), GPT-4 (2023).",
        "en": "GPT (Generative Pre-trained Transformer) is OpenAI's autoregressive "
              "LLM family. GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023).",
    },
    "bert": {
        "vi": "BERT (Bidirectional Encoder Representations from Transformers) là LM "
              "encoder hai chiều do Google giới thiệu 2018, dùng Masked Language Modeling. "
              "Mạnh ở các task hiểu ngôn ngữ. DistilBERT là phiên bản nén của BERT.",
        "en": "BERT (Bidirectional Encoder Representations from Transformers) is a "
              "bidirectional encoder LM from Google (2018) trained with Masked LM. "
              "Strong on understanding tasks. DistilBERT is its compressed cousin.",
    },
    "fine-tune": {  # also matches "fine-tuning", "finetune"
        "vi": "Fine-tuning là tiếp tục train một mô hình đã pre-trained trên dữ liệu "
              "task cụ thể (thường nhỏ hơn nhiều) để mô hình thích nghi với domain. "
              "Khác với prompt-engineering ở chỗ thực sự cập nhật trọng số.",
        "en": "Fine-tuning continues training a pre-trained model on a (usually small) "
              "task-specific dataset so it adapts to a domain. Unlike prompting, it "
              "updates model weights.",
    },
    "prompt": {
        "vi": "Prompt là chuỗi input đưa cho LLM để dẫn dắt output. "
              "Prompt engineering = nghệ thuật viết prompt rõ, có context, có ràng buộc.",
        "en": "A prompt is the input string steering an LLM's output. "
              "Prompt engineering = crafting clear, contextual, constrained prompts.",
    },
    "embedding": {
        "vi": "Embedding là vector đặc trưng của từ/câu/document trong không gian liên tục. "
              "Cosine similarity giữa các embedding đo độ tương đồng ngữ nghĩa — nền tảng "
              "của vector search trong RAG.",
        "en": "An embedding is a continuous-space vector representation of a word, "
              "sentence, or document. Cosine similarity between embeddings measures "
              "semantic closeness — the basis of vector search in RAG.",
    },
    "attention": {
        "vi": "Attention là cơ chế cho mô hình tự chọn phần input nào quan trọng cho output "
              "hiện tại. Self-attention so sánh mọi token với mọi token, là cốt lõi của Transformer.",
        "en": "Attention lets a model decide which input parts matter for the current "
              "output. Self-attention compares every token to every token — the core of "
              "Transformer.",
    },
    "deep learning": {
        "vi": "Deep learning là nhánh ML dùng mạng neural nhiều lớp. Học feature tự động "
              "từ data thô (ảnh, text, audio) thay vì cần human feature engineering.",
        "en": "Deep learning is the ML branch built on deep neural networks. Learns "
              "features automatically from raw data (images, text, audio) instead of "
              "manual feature engineering.",
    },
    "machine learning": {
        "vi": "Machine learning là lĩnh vực dạy máy học pattern từ data thay vì code rule "
              "tay. Có 3 nhánh chính: supervised (có nhãn), unsupervised (không nhãn), "
              "reinforcement learning (qua reward).",
        "en": "Machine learning is the field of teaching machines patterns from data "
              "instead of hand-coded rules. Three main branches: supervised (labeled), "
              "unsupervised (unlabeled), reinforcement (reward-driven).",
    },
    "nlp": {
        "vi": "NLP (Natural Language Processing) là nhánh AI xử lý ngôn ngữ tự nhiên: "
              "dịch, tóm tắt, hỏi đáp, phân loại, sinh text. ChronoRAG là 1 ứng dụng NLP.",
        "en": "NLP (Natural Language Processing) is the AI branch for human language: "
              "translation, summarisation, Q&A, classification, generation. ChronoRAG "
              "is an NLP application.",
    },
    "vector database": {
        "vi": "Vector database lưu embedding và truy vấn theo nearest-neighbor "
              "(cosine/L2). Ví dụ: FAISS, Pinecone, Milvus, Qdrant, Weaviate. "
              "ChronoRAG dùng BM25 + tùy chọn FAISS.",
        "en": "A vector database stores embeddings and queries by nearest-neighbor "
              "(cosine/L2). Examples: FAISS, Pinecone, Milvus, Qdrant, Weaviate. "
              "ChronoRAG uses BM25 with optional FAISS.",
    },
    "bm25": {
        "vi": "BM25 là thuật toán tính điểm relevance giữa query và document dựa trên "
              "term frequency + inverse document frequency, có chuẩn hoá độ dài. Là "
              "baseline mạnh cho retrieval, được ChronoRAG dùng làm sparse index.",
        "en": "BM25 is a relevance scoring algorithm based on term frequency + inverse "
              "document frequency with length normalisation. A strong retrieval baseline; "
              "ChronoRAG uses it as the sparse index.",
    },
    "hallucination": {
        "vi": "Hallucination là khi LLM bịa thông tin không có trong context hoặc training. "
              "RAG + citation giúp giảm hallucination bằng cách bám sentence thật.",
        "en": "Hallucination = an LLM inventing facts not in its context or training. "
              "RAG with citations mitigates it by grounding answers in real sentences.",
    },
    "reinforcement learning": {
        "vi": "Reinforcement Learning (RL) là dạy agent học bằng reward/penalty từ "
              "tương tác với môi trường. RLHF (RL from Human Feedback) là kỹ thuật fine-tune "
              "LLM theo phản hồi người dùng (ChatGPT, Claude dùng).",
        "en": "Reinforcement Learning (RL) trains an agent via reward/penalty from "
              "environment interaction. RLHF (RL from Human Feedback) fine-tunes LLMs "
              "with human ratings (used in ChatGPT, Claude).",
    },
    "neural network": {
        "vi": "Neural network là mô hình mô phỏng cấu trúc neuron, gồm các layer nối nhau, "
              "học tham số bằng backpropagation. Là nền tảng deep learning.",
        "en": "A neural network is a layered model loosely inspired by biological neurons, "
              "trained via backpropagation. Foundation of deep learning.",
    },
    "cnn": {
        "vi": "Convolutional Neural Network (CNN) là kiến trúc mạng neural với layer "
              "convolution + pooling, mạnh ở ảnh và signal. LeNet, AlexNet, ResNet là CNN.",
        "en": "Convolutional Neural Network (CNN) uses convolution + pooling layers; "
              "strong on images/signals. LeNet, AlexNet, ResNet are CNNs.",
    },
    "rnn": {
        "vi": "Recurrent Neural Network (RNN) xử lý dữ liệu tuần tự (text, time-series) "
              "bằng cách giữ state qua các bước. LSTM và GRU là 2 biến thể chính.",
        "en": "Recurrent Neural Network (RNN) processes sequential data (text, "
              "time-series) by carrying state across steps. LSTM and GRU are the two "
              "main variants.",
    },
    "lstm": {
        "vi": "Long Short-Term Memory (LSTM) là biến thể RNN có gating (forget, input, output) "
              "giúp giữ thông tin dài hạn, do Hochreiter & Schmidhuber giới thiệu 1997.",
        "en": "Long Short-Term Memory (LSTM) is an RNN variant with gating (forget, "
              "input, output) for long-range dependencies, introduced by Hochreiter & "
              "Schmidhuber in 1997.",
    },
    "backpropagation": {
        "vi": "Backpropagation là thuật toán tính gradient của loss theo từng tham số, "
              "lan truyền ngược từ output về input qua chain rule.",
        "en": "Backpropagation computes gradients of the loss w.r.t. every parameter "
              "by propagating errors backward through the chain rule.",
    },
    "gradient descent": {
        "vi": "Gradient descent là thuật toán tối ưu, cập nhật tham số theo hướng gradient âm "
              "để giảm loss. Variants: SGD, Adam, RMSProp.",
        "en": "Gradient descent is the optimisation routine that updates parameters in "
              "the negative-gradient direction to lower the loss. Variants: SGD, Adam, "
              "RMSProp.",
    },
    "overfitting": {
        "vi": "Overfitting là khi mô hình học quá khớp training data, làm tệ trên data mới. "
              "Chống bằng regularization, dropout, early stopping, thêm data.",
        "en": "Overfitting is when a model fits training data too tightly and fails on "
              "new data. Mitigations: regularisation, dropout, early stopping, more data.",
    },
    "regularization": {
        "vi": "Regularization là kỹ thuật chống overfitting: L1/L2 penalty, dropout, "
              "data augmentation, early stopping. Bias model về phía solution đơn giản hơn.",
        "en": "Regularisation curbs overfitting via L1/L2 penalties, dropout, data "
              "augmentation, or early stopping. Pushes the model toward simpler "
              "solutions.",
    },
    "classification": {
        "vi": "Classification là bài toán dự đoán nhãn rời rạc (vd: spam/ham, 0..9). "
              "Loss thường dùng cross-entropy. Metric: accuracy, F1, precision, recall.",
        "en": "Classification predicts a discrete label (e.g. spam/ham, 0..9). Typical "
              "loss: cross-entropy. Metrics: accuracy, F1, precision, recall.",
    },
    "regression": {
        "vi": "Regression là bài toán dự đoán giá trị liên tục (vd: giá nhà). "
              "Loss thường dùng MSE/MAE. Metric: RMSE, R².",
        "en": "Regression predicts a continuous value (e.g. house price). Typical loss: "
              "MSE/MAE. Metrics: RMSE, R².",
    },
    "precision": {
        "vi": "Precision = TP / (TP + FP). Đo tỷ lệ dự đoán positive đúng. "
              "Cao khi mô hình ít báo nhầm.",
        "en": "Precision = TP / (TP + FP). Fraction of positive predictions that are "
              "correct. High when the model rarely flags wrongly.",
    },
    "recall": {
        "vi": "Recall = TP / (TP + FN). Đo tỷ lệ positive thật bị bắt được. "
              "Cao khi mô hình ít bỏ sót.",
        "en": "Recall = TP / (TP + FN). Fraction of true positives the model catches. "
              "High when the model rarely misses cases.",
    },
    "f1": {
        "vi": "F1 = 2·P·R / (P + R), trung bình điều hòa của precision và recall. "
              "Cân bằng giữa false-positive và false-negative. ChronoRAG báo macro-F1.",
        "en": "F1 = 2·P·R / (P + R), the harmonic mean of precision and recall. "
              "Balances false positives and false negatives. ChronoRAG reports macro-F1.",
    },
    "tokenization": {
        "vi": "Tokenization là chia text thành token (từ, sub-word, hoặc ký tự) để feed vào model. "
              "BPE, WordPiece, SentencePiece là các thuật toán phổ biến.",
        "en": "Tokenization splits text into tokens (words, sub-words, or characters) "
              "for model input. BPE, WordPiece, SentencePiece are common algorithms.",
    },
    "python": {
        "vi": "Python là ngôn ngữ lập trình bậc cao, đa dụng, có cú pháp rõ ràng. "
              "Là ngôn ngữ chủ đạo cho AI/ML/data science nhờ ecosystem (PyTorch, "
              "TensorFlow, scikit-learn, NumPy, pandas).",
        "en": "Python is a high-level, general-purpose programming language with clear "
              "syntax. Dominant in AI/ML/data science thanks to its ecosystem (PyTorch, "
              "TensorFlow, scikit-learn, NumPy, pandas).",
    },
    "pytorch": {
        "vi": "PyTorch là framework deep learning của Meta, nổi tiếng vì dynamic computation graph "
              "(define-by-run), dễ debug, và phổ biến trong research.",
        "en": "PyTorch is Meta's deep-learning framework, known for its dynamic "
              "computation graph (define-by-run), ease of debugging, and dominance in "
              "research.",
    },
    "tensorflow": {
        "vi": "TensorFlow là framework deep learning của Google, mạnh ở deploy production "
              "(TF Serving, TFLite). Keras là high-level API đi kèm.",
        "en": "TensorFlow is Google's deep-learning framework, strong on production "
              "deployment (TF Serving, TFLite). Keras is its high-level API.",
    },
    "api": {
        "vi": "API (Application Programming Interface) là tập hợp endpoint cho app giao tiếp. "
              "REST API dùng HTTP + JSON. ChronoRAG có FastAPI backend ở port 8000.",
        "en": "An API (Application Programming Interface) is a contract for apps to "
              "talk to each other. REST APIs use HTTP + JSON. ChronoRAG's FastAPI "
              "backend runs on port 8000.",
    },
    "fastapi": {
        "vi": "FastAPI là Python web framework hiện đại, async, tự động generate OpenAPI docs. "
              "ChronoRAG dùng FastAPI cho backend REST.",
        "en": "FastAPI is a modern, async Python web framework that auto-generates "
              "OpenAPI docs. ChronoRAG uses FastAPI for the REST backend.",
    },
    "react": {
        "vi": "React (frontend) là thư viện JS của Meta để build UI dạng component. "
              "ChronoRAG dùng React + Vite cho dashboard.\n\n"
              "Lưu ý: 'ReAct' (in hoa R+A) còn là tên một AI Agent framework "
              "trong corpus ChronoRAG — hỏi rõ hơn để mình biết bạn muốn ý nào.",
        "en": "React (frontend) is Meta's JS library for component-based UIs. "
              "ChronoRAG uses React + Vite for the dashboard.\n\n"
              "Note: 'ReAct' (capital R+A) is also an AI Agent framework documented "
              "in the ChronoRAG corpus — ask more specifically if you meant that.",
    },
    "json": {
        "vi": "JSON (JavaScript Object Notation) là định dạng dữ liệu text dễ đọc, dùng "
              "phổ biến cho API. ChronoRAG dùng JSONL cho artifacts (1 JSON / dòng).",
        "en": "JSON (JavaScript Object Notation) is a human-readable text data format "
              "used widely for APIs. ChronoRAG stores artifacts as JSONL (one JSON per "
              "line).",
    },
    "git": {
        "vi": "Git là hệ thống quản lý version phân tán phổ biến nhất. "
              "GitHub/GitLab là dịch vụ host repo + collaboration trên Git.",
        "en": "Git is the most widely used distributed version-control system. "
              "GitHub/GitLab host Git repos and add collaboration features.",
    },
    "docker": {
        "vi": "Docker đóng gói ứng dụng + dependencies vào container chạy giống nhau ở mọi nơi. "
              "Image là blueprint, container là instance đang chạy.",
        "en": "Docker packages apps and their dependencies into containers that run "
              "identically anywhere. Images are blueprints; containers are running "
              "instances.",
    },
    "sql": {
        "vi": "SQL (Structured Query Language) là ngôn ngữ truy vấn relational database. "
              "Cơ bản: SELECT, INSERT, UPDATE, DELETE, JOIN, GROUP BY.",
        "en": "SQL (Structured Query Language) is the standard language for relational "
              "databases. Basics: SELECT, INSERT, UPDATE, DELETE, JOIN, GROUP BY.",
    },
    "ai": {
        "vi": "AI (Artificial Intelligence / Trí tuệ nhân tạo) là lĩnh vực nghiên cứu "
              "máy móc thực hiện những việc thường cần trí tuệ con người: nhận diện, "
              "suy luận, học từ dữ liệu, ra quyết định. Machine learning và deep "
              "learning là các nhánh chủ đạo của AI hiện nay.",
        "en": "AI (Artificial Intelligence) is the field of building machines that do "
              "things normally requiring human intelligence: perception, reasoning, "
              "learning from data, decision-making. Machine learning and deep learning "
              "are today's dominant branches.",
    },
}

# Aliases that should map to a canonical key in _COMMON_TERMS.
_TERM_ALIASES = {
    "large language model": "llm",
    "large language models": "llm",
    "fine tuning": "fine-tune",
    "fine-tuning": "fine-tune",
    "finetune": "fine-tune",
    "finetuning": "fine-tune",
    "deep learning": "deep learning",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "natural language processing": "nlp",
    "rl": "reinforcement learning",
    "rlhf": "reinforcement learning",
    "vector db": "vector database",
    "vector store": "vector database",
    "neural net": "neural network",
    "mạng neural": "neural network",
    "mang neural": "neural network",
    "mạng nơ-ron": "neural network",
    "convolutional neural network": "cnn",
    "recurrent neural network": "rnn",
    "long short-term memory": "lstm",
    "backprop": "backpropagation",
    "lan truyền ngược": "backpropagation",
    "lan truyen nguoc": "backpropagation",
    "gd": "gradient descent",
    "sgd": "gradient descent",
    "phân loại": "classification",
    "phan loai": "classification",
    "hồi quy": "regression",
    "hoi quy": "regression",
    "f1 score": "f1",
    "f1-score": "f1",
    "tokenizer": "tokenization",
    "tách từ": "tokenization",
    "tach tu": "tokenization",
    "py": "python",
    "torch": "pytorch",
    "tf": "tensorflow",
    "rest api": "api",
    "restful": "api",
    "react js": "react",
    "reactjs": "react",
}

# Question shapes a user might wrap a term in. Keys: (vi-template, en-template).
_TERM_QUESTION_PATTERNS = (
    r"(.*?)\s*là\s*g[iì]\??$",
    r"(.*?)\s*la\s*gi\??$",
    r"what\s+is\s+(?:an?\s+|the\s+)?(.*?)\??$",
    r"define\s+(.*?)\??$",
    r"explain\s+(.*?)\??$",
    r"(.*?)\s*means?\s*what\??$",
    r"giải\s+thích\s+(.*?)$",
    r"giai\s+thich\s+(.*?)$",
)


def _extract_term(q: str) -> Optional[str]:
    """If the question looks like a definition request, return the term."""
    for pattern in _TERM_QUESTION_PATTERNS:
        m = re.match(pattern, q, flags=re.IGNORECASE)
        if m:
            term = m.group(1).strip().strip(_FILLER_CHARS)
            if term:
                return term.lower()
    # Single bare word is also fair game ("transformer", "bert").
    if " " not in q and len(q) <= 30:
        return q.lower()
    return None


# Terms that name a ChronoRAG corpus entity (Self-RAG, ReAct, DistilBERT, ...)
# We intentionally do NOT answer these from the canned dictionary -- the user
# almost certainly wants the corpus-backed answer with a real citation.
_CORPUS_ENTITY_NAMES = frozenset({
    "rag", "self-rag", "self rag", "selfrag", "graphrag", "graph rag",
    "realm", "dpr", "dense passage retrieval", "retro", "atlas",
    "flare", "crag", "active retrieval", "corrective retrieval",
    "autogpt", "autogen", "react", "reactjs", "react js",
    "toolformer", "langgraph",
    "distilbert", "tinybert", "mobilebert", "minilm",
})


def _lookup_term(raw_term: str) -> Optional[str]:
    """Resolve a term against the common-term dictionary including aliases.

    Returns ``None`` (so RAG handles it) when the term is a corpus entity --
    those deserve a cited answer from the real paper, not a canned blurb.
    """
    if not raw_term:
        return None
    key = raw_term.strip().lower()
    if key in _CORPUS_ENTITY_NAMES:
        return None
    if key in _COMMON_TERMS:
        return key
    if key in _TERM_ALIASES:
        return _TERM_ALIASES[key]
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle_conversational(question: str) -> Optional[Dict[str, Any]]:
    """Return a canned response or None to let RAG handle it."""
    if not question:
        return None
    raw = question.strip()
    if not raw:
        return None
    q = _norm(raw)
    if not q:
        return None
    is_vi = _detect_vietnamese(raw)

    # 1. Exact-match short phrases (greetings/thanks/goodbye/feeling/negative).
    if q in _GREETINGS or any(q == g or q.startswith(g + " ") for g in _GREETINGS):
        return {"answer": _greeting(q, is_vi), "citations": []}
    if q in _THANKS or q.startswith(("cảm ơn ", "cam on ", "thanks ", "thank you ")):
        return {"answer": _thanks(is_vi), "citations": []}
    if q in _GOODBYES or any(q == g or q.startswith(g + " ") for g in _GOODBYES):
        return {"answer": _goodbye(is_vi), "citations": []}
    if any(p in q for p in _FEELING_QUESTIONS):
        return {"answer": _feeling(is_vi), "citations": []}

    # 2. Time / date.
    if any(p in q for p in _TIME_PATTERNS):
        return {"answer": _time_now(is_vi), "citations": []}
    if any(p in q for p in _DATE_PATTERNS):
        return {"answer": _date_now(is_vi), "citations": []}

    # 3. Acknowledgement of bad answer.
    if q in _NEGATIVE_PATTERNS:
        return {"answer": _negative_reply(is_vi), "citations": []}

    # 4. Common AI/ML term definitions outside the corpus.
    term_candidate = _extract_term(q)
    canonical = _lookup_term(term_candidate) if term_candidate else None
    if canonical:
        entry = _COMMON_TERMS[canonical]
        return {"answer": entry["vi" if is_vi else "en"], "citations": []}

    return None
