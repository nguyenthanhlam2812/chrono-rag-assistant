import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_mock_timeline(topic: str) -> List[Dict[str, Any]]:
    """Generate Sprint 0 mock timeline events based on the input topic."""
    topic_clean = topic.lower().strip()

    if "rag" in topic_clean:
        return [
            {
                "event_id": "rag_evt_001",
                "date": "May 2020",
                "year": 2020,
                "event_type": "method_proposed",
                "title": "Retrieval-Augmented Generation (RAG) proposed",
                "representative_sentence": "In 2020, Lewis et al. proposed Retrieval-Augmented Generation (RAG) to combine parametric and non-parametric memory.",
                "confidence": 0.95,
                "sources": [
                    {
                        "doc_id": "rag_001",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "source_url": "https://arxiv.org/abs/2005.11401",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "rag_evt_002",
                "date": "Late 2022",
                "year": 2022,
                "event_type": "release",
                "title": "Open-source RAG orchestrators",
                "representative_sentence": "In 2022, open-source frameworks like LangChain and LlamaIndex were released, making RAG implementation accessible.",
                "confidence": 0.88,
                "sources": [
                    {
                        "doc_id": "rag_001",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "source_url": "https://arxiv.org/abs/2005.11401",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "rag_evt_003",
                "date": "Early 2024",
                "year": 2024,
                "event_type": "method_proposed",
                "title": "Microsoft introduces GraphRAG",
                "representative_sentence": "In early 2024, Microsoft introduced GraphRAG, which leverages Knowledge Graphs rather than simple vector similarity for retrieval.",
                "confidence": 0.91,
                "sources": [
                    {
                        "doc_id": "rag_001",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "source_url": "https://arxiv.org/abs/2005.11401",
                    }
                ],
                "cluster_size": 1,
            },
        ]

    if "agent" in topic_clean:
        return [
            {
                "event_id": "agent_evt_001",
                "date": "March 2023",
                "year": 2023,
                "event_type": "release",
                "title": "AutoGPT release",
                "representative_sentence": "In March 2023, Toran Bruce Richards released AutoGPT, demonstrating self-directed task loops that execute complex multi-step objectives.",
                "confidence": 0.94,
                "sources": [
                    {
                        "doc_id": "agent_001",
                        "title": "LLM Powered Autonomous Agents",
                        "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "agent_evt_002",
                "date": "June 2023",
                "year": 2023,
                "event_type": "trend_application",
                "title": "LLM-powered autonomous agent framework",
                "representative_sentence": "In June 2023, Lilian Weng published a seminal blog post outlining LLM-powered autonomous agents.",
                "confidence": 0.96,
                "sources": [
                    {
                        "doc_id": "agent_001",
                        "title": "LLM Powered Autonomous Agents",
                        "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "agent_evt_003",
                "date": "2024",
                "year": 2024,
                "event_type": "release",
                "title": "Multi-agent frameworks gain popularity",
                "representative_sentence": "By 2024, multi-agent orchestrations like CrewAI and Microsoft's AutoGen emerged.",
                "confidence": 0.85,
                "sources": [
                    {
                        "doc_id": "agent_001",
                        "title": "LLM Powered Autonomous Agents",
                        "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    }
                ],
                "cluster_size": 1,
            },
        ]

    if "distill" in topic_clean:
        return [
            {
                "event_id": "kd_evt_001",
                "date": "2015",
                "year": 2015,
                "event_type": "method_proposed",
                "title": "Knowledge Distillation popularized by Hinton et al.",
                "representative_sentence": "In 2015, Geoffrey Hinton, Oriol Vinyals, and Jeff Dean popularized the concept of Knowledge Distillation.",
                "confidence": 0.97,
                "sources": [
                    {
                        "doc_id": "kd_001",
                        "title": "Distilling the Knowledge in a Neural Network",
                        "source_url": "https://arxiv.org/abs/1503.02531",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "kd_evt_002",
                "date": "2019",
                "year": 2019,
                "event_type": "release",
                "title": "DistilBERT release",
                "representative_sentence": "In 2019, Victor Sanh et al. released DistilBERT, compressing BERT by 40% while retaining 97% of its performance.",
                "confidence": 0.90,
                "sources": [
                    {
                        "doc_id": "kd_001",
                        "title": "Distilling the Knowledge in a Neural Network",
                        "source_url": "https://arxiv.org/abs/1503.02531",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "kd_evt_003",
                "date": "2020",
                "year": 2020,
                "event_type": "method_proposed",
                "title": "TinyBERT architecture introduced",
                "representative_sentence": "In 2020, Jiao et al. introduced TinyBERT, performing transformer-layer distillation.",
                "confidence": 0.89,
                "sources": [
                    {
                        "doc_id": "kd_001",
                        "title": "Distilling the Knowledge in a Neural Network",
                        "source_url": "https://arxiv.org/abs/1503.02531",
                    }
                ],
                "cluster_size": 1,
            },
        ]

    return []


def get_mock_sentence_predictions(topic: str) -> List[Dict[str, Any]]:
    """Return Sprint 0 mock event predictions using the planned event taxonomy."""
    return [
        {
            "sentence": "Retrieval-Augmented Generation (RAG) has become a key methodology in modern natural language processing.",
            "is_event": 0,
            "prob": 0.12,
            "type": "none",
        },
        {
            "sentence": "In 2020, Lewis et al. proposed RAG to combine parametric memory and non-parametric memory.",
            "is_event": 1,
            "prob": 0.95,
            "type": "method_proposed",
        },
        {
            "sentence": "By 2021, various researchers adapted RAG for question answering and open domain dialogues.",
            "is_event": 1,
            "prob": 0.74,
            "type": "trend_application",
        },
        {
            "sentence": "In 2022, open-source frameworks like LangChain and LlamaIndex were released.",
            "is_event": 1,
            "prob": 0.88,
            "type": "release",
        },
        {
            "sentence": "During 2023, Vector Databases like Pinecone, Milvus, and Qdrant saw massive adoption.",
            "is_event": 0,
            "prob": 0.45,
            "type": "none",
        },
        {
            "sentence": "In early 2024, Microsoft introduced GraphRAG, which leverages Knowledge Graphs for retrieval.",
            "is_event": 1,
            "prob": 0.91,
            "type": "method_proposed",
        },
    ]


def get_mock_answer(topic: str, question: str) -> Dict[str, Any]:
    """Generate Sprint 0 mock Q&A answer."""
    topic_clean = topic.lower().strip()

    if "rag" in topic_clean:
        return {
            "answer": "Retrieval-Augmented Generation (RAG) was introduced in 2020 by Lewis et al. [rag_001] as a hybrid approach that joins parametric memory with external source documents [rag_001]. Major framework packages like LangChain were released in 2022 [rag_001] to speed up deployment, and Microsoft's GraphRAG launched in 2024 [rag_001] to improve retrieval context using Knowledge Graphs.",
            "citations": [
                {
                    "doc_id": "rag_001",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "source_url": "https://arxiv.org/abs/2005.11401",
                }
            ],
        }

    if "agent" in topic_clean:
        return {
            "answer": "LLM powered autonomous agents are designed using planning, memory, and tool usage architectures [agent_001]. AutoGPT was released in March 2023 [agent_001] followed by Weng's foundational agent post in June 2023 [agent_001]. Multi-agent orchestrations like CrewAI and AutoGen gained popularity in 2024 [agent_001].",
            "citations": [
                {
                    "doc_id": "agent_001",
                    "title": "LLM Powered Autonomous Agents",
                    "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                }
            ],
        }

    return {
        "answer": "Knowledge Distillation transfers dark knowledge from a big teacher model to a smaller student [kd_001]. Popularized by Hinton in 2015 [kd_001], it led to compact models like DistilBERT in 2019 [kd_001] and TinyBERT in 2020 [kd_001].",
        "citations": [
            {
                "doc_id": "kd_001",
                "title": "Distilling the Knowledge in a Neural Network",
                "source_url": "https://arxiv.org/abs/1503.02531",
            }
        ],
    }


def get_mock_evaluation_metrics() -> Dict[str, Any]:
    """Return Sprint 0 placeholder metrics for the evaluation tab."""
    return {
        "summary": {
            "event_detection_f1": "84.2%",
            "event_type_macro_f1": "76.5%",
            "retrieval_recall_at_5": "91.0%",
            "timeline_date_accuracy": "88.9%",
        },
        "confusion_matrix_markdown": """
| Predicted \\ Actual | method_proposed | release | benchmark | trend_application | none |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **method_proposed** | **45** | 2 | 0 | 4 | 1 |
| **release** | 2 | **64** | 0 | 4 | 4 |
| **benchmark** | 2 | 0 | **22** | 4 | 0 |
| **trend_application** | 7 | 5 | 3 | **83** | 9 |
| **none** | 0 | 1 | 0 | 3 | **110** |
""",
        "experiment_markdown": """
| Model Architecture | Event Detection F1 | Event Type Macro-F1 | Inference Speed (ms/sent) |
| :--- | :---: | :---: | :---: |
| TF-IDF + Logistic Regression | 79.1% | 68.4% | **0.8 ms** |
| TF-IDF + SVM Baseline | 81.2% | 71.0% | 1.2 ms |
| **BiLSTM (PyTorch)** | **84.2%** | **76.5%** | 8.5 ms |
""",
    }


def get_local_qa_answer(topic: str, question: str) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate a template-based answer with real citations."""
    from src.retrieval.simple_retriever import SimpleRetriever
    from src.generation.template_answerer import TemplateAnswerer

    chunks_path = PROJECT_ROOT / 'data' / 'processed' / 'chunks.jsonl'
    
    if not chunks_path.exists():
        return get_mock_answer(topic, question)
        
    retriever = SimpleRetriever(chunks_path)
    if not retriever.chunks:
        return get_mock_answer(topic, question)
        
    # Retrieve top 3 matching chunks
    chunks = retriever.retrieve(question, topic=topic, top_k=3)
    
    answerer = TemplateAnswerer()
    return answerer.generate_answer(chunks, query=question)


