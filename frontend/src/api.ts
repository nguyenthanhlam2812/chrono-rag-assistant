import type {
  ChatResponse,
  EvaluationResponse,
  EventsResponse,
  GenerationStatus,
  OverviewResponse,
  RetrievalEvalResponse,
  SourcesResponse,
  TimelineResponse,
  Topic,
  TopicId
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (!response.ok) {
    throw new Error(`API ${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; documents: number; predictions: number; timelineEvents: number; generation?: GenerationStatus }>("/api/health"),
  topics: async () => {
    const data = await request<{ topics: Topic[] }>("/api/topics");
    return data.topics;
  },
  overview: (topic: TopicId) => request<OverviewResponse>(`/api/overview?topic=${topic}`),
  timeline: (topic: TopicId, limit = 30) => request<TimelineResponse>(`/api/timeline?topic=${topic}&limit=${limit}`),
  events: (topic: TopicId, limit = 40) => request<EventsResponse>(`/api/events?topic=${topic}&limit=${limit}`),
  sources: (topic: TopicId) => request<SourcesResponse>(`/api/sources?topic=${topic}`),
  evaluation: (model = "sgd_log") => request<EvaluationResponse>(`/api/evaluation?model=${model}`),
  retrievalEval: () => request<RetrievalEvalResponse>("/api/retrieval_eval"),
  chat: (
    topic: TopicId,
    question: string,
    history?: Array<{ role: "user" | "assistant"; content: string }>
  ) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ topic, question, history })
    })
};

export { API_BASE };
