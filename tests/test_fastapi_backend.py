from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app  # noqa: E402


client = TestClient(app)


def test_health_and_topics_endpoints():
    health = client.get("/api/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "healthy"
    assert payload["documents"] >= 30
    assert payload["timelineEvents"] > 0

    topics = client.get("/api/topics")
    assert topics.status_code == 200
    topic_ids = {row["id"] for row in topics.json()["topics"]}
    assert {"rag", "ai_agent", "knowledge_distillation"} <= topic_ids


def test_cors_allows_local_dev_ports():
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://127.0.0.1:5174",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"


def test_overview_timeline_and_evaluation_contracts():
    overview = client.get("/api/overview", params={"topic": "rag"})
    assert overview.status_code == 200
    overview_payload = overview.json()
    assert overview_payload["summary"]["docsIngested"] >= 1
    assert overview_payload["summary"]["eventsDetected"] >= 1

    timeline = client.get("/api/timeline", params={"topic": "rag", "limit": 3})
    assert timeline.status_code == 200
    timeline_payload = timeline.json()
    assert timeline_payload["events"]
    assert len(timeline_payload["events"]) <= 3

    evaluation = client.get("/api/evaluation", params={"model": "sgd_log"})
    assert evaluation.status_code == 200
    evaluation_payload = evaluation.json()
    assert evaluation_payload["binary"]["confusionMatrix"]
    assert evaluation_payload["eventType"]["labels"]


def test_chat_returns_answer_with_citations():
    response = client.post(
        "/api/chat",
        json={"topic": "rag", "question": "What is RAG?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
