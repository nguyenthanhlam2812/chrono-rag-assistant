import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  BarChart3,
  BrainCircuit,
  ChevronRight,
  Database,
  FileText,
  Grid2X2,
  Layers3,
  LineChart,
  Loader2,
  MessageSquare,
  SlidersHorizontal,
  Sparkles,
  SquareStack,
  Workflow
} from "lucide-react";
import { api } from "./api";
import type {
  ChatResponse,
  EvaluationResponse,
  EventPrediction,
  EventsResponse,
  GenerationStatus,
  OverviewResponse,
  RetrievalEvalResponse,
  RetrievalEvalSummary,
  SourceDocument,
  SourcesResponse,
  TimelineEvent,
  TimelineResponse,
  Topic,
  TopicId
} from "./types";

type ViewId = "overview" | "timeline" | "analysis" | "sources" | "chat" | "evaluation";
type Language = "vi" | "en";

// Must stay in sync with NO_ANSWER_MESSAGE in src/generation/template_answerer.py.
const ABSTAIN_PREFIX = "Không tìm thấy thông tin đủ liên quan";

// Backend embeds "[doc_id]" markers inline so they survive concatenation. We
// already render citations as chips below the message, so strip the inline
// duplicates from the visible text to keep the answer readable.
const CITATION_MARKER_RE = /\s*\[[a-z]+_\d+\]/gi;
function cleanAnswerText(text: string): string {
  return text.replace(CITATION_MARKER_RE, "").replace(/\s+/g, " ").trim();
}

// Sample questions shown when the chat is still empty. Kept in English so they
// land inside the corpus index reliably; users can still type Vietnamese.
const CHAT_SUGGESTIONS: Record<TopicId, string[]> = {
  rag: [
    "What is RAG?",
    "What is dense passage retrieval?",
    "How does Self-RAG decide when to retrieve?"
  ],
  ai_agent: [
    "What is AutoGen?",
    "How does the ReAct agent loop work?",
    "Why use multi-agent frameworks?"
  ],
  knowledge_distillation: [
    "What is DistilBERT?",
    "How does knowledge distillation work?",
    "DistilBERT vs TinyBERT — what is the difference?"
  ]
};

const fallbackTopics: Topic[] = [
  { id: "rag", label: "RAG" },
  { id: "ai_agent", label: "AI Agent" },
  { id: "knowledge_distillation", label: "Knowledge Distillation" }
];

const navItems: Array<{ id: ViewId; icon: typeof Grid2X2 }> = [
  { id: "overview", icon: Grid2X2 },
  { id: "timeline", icon: Workflow },
  { id: "analysis", icon: BarChart3 },
  { id: "sources", icon: Database },
  { id: "chat", icon: MessageSquare },
  { id: "evaluation", icon: SquareStack }
];

type Translations = {
  stable: string;
  newSession: string;
  topicScope: string;
  projectLead: string;
  systemHealthy: string;
  loading: string;
  languageToggle: string;
  healthSummary: (docs: number, events: number) => string;
  nav: Record<ViewId, string>;
  overviewTitle: string;
  overviewSubtitle: (topic: string) => string;
  projectSummary: string;
  eventsDetected: string;
  docsIngested: string;
  avgConfidence: string;
  yearRange: string;
  modelStatus: string;
  eventDetector: string;
  eventType: string;
  timeline: string;
  timelineStatus: (events: number, confidence: number) => string;
  chronologicalInsights: string;
  expandAll: string;
  oldestFirst: string;
  newestFirst: string;
  timelineTitle: string;
  timelineSubtitle: (total: number, topic: string) => string;
  analysisTitle: string;
  analysisSubtitle: (events: number, sentences: number, topic: string) => string;
  highConfidenceEvents: string;
  sourcesTitle: string;
  sourcesSubtitle: (count: number, topic: string) => string;
  words: (count: string) => string;
  openSource: string;
  chatTitle: string;
  chatSubtitle: string;
  chatWelcome: (topic: string) => string;
  you: string;
  askPlaceholder: (topic: string) => string;
  thinking: string;
  send: string;
  apiError: string;
  evaluationTitle: string;
  evaluationSubtitle: string;
  model: string;
  binaryF1: string;
  typeMacroF1: string;
  binaryAccuracy: string;
  binaryConfusionMatrix: string;
  eventTypeConfusionMatrix: string;
  experimentComparison: string;
  typeF1: string;
  binaryAcc: string;
  emptyTimeline: string;
  noMatrix: string;
  predictedLabel: string;
  trueLabel: string;
  confidence: string;
  viewCitations: string;
  loadingArtifacts: string;
  cannotConnect: string;
  showMore: string;
  showLess: string;
  suggestedQuestions: string;
  retrievalBenchmark: string;
  retrievalBenchmarkSubtitle: (n: number) => string;
  retrievalRecallAt: (k: number) => string;
  retrievalMrr: string;
  retrievalPerTopic: string;
  retrievalUnavailable: string;
  eventTypeLabels: Record<string, string>;
};

const translations: Record<Language, Translations> = {
  vi: {
    stable: "v2.4 ổn định",
    newSession: "Phiên nghiên cứu mới",
    topicScope: "Phạm vi chủ đề",
    projectLead: "Trưởng nhóm",
    systemHealthy: "Hệ thống ổn",
    loading: "Đang tải",
    languageToggle: "Đổi ngôn ngữ",
    healthSummary: (docs: number, events: number) => `${docs} tài liệu / ${events} sự kiện timeline`,
    nav: {
      overview: "Tổng quan",
      timeline: "Timeline",
      analysis: "Phân tích",
      sources: "Nguồn",
      chat: "Chat",
      evaluation: "Đánh giá"
    },
    overviewTitle: "Bảng tổng quan",
    overviewSubtitle: (topic: string) => `Phân tích corpus cho: ${topic.toLowerCase().replace(/\s+/g, "_")}_timeline.json`,
    projectSummary: "Tóm tắt project",
    eventsDetected: "Câu event phát hiện",
    docsIngested: "Tài liệu đã nạp",
    avgConfidence: "Độ tin cậy TB",
    yearRange: "Khoảng năm",
    modelStatus: "Trạng thái mô hình",
    eventDetector: "Event detector",
    eventType: "Loại event",
    timeline: "Timeline",
    timelineStatus: (events: number, confidence: number) => `${events} sự kiện @ ${confidence.toFixed(2)}`,
    chronologicalInsights: "Mốc thời gian nổi bật",
    expandAll: "Xem đầy đủ",
    oldestFirst: "Cũ trước",
    newestFirst: "Mới trước",
    timelineTitle: "Timeline theo thời gian",
    timelineSubtitle: (total: number, topic: string) => `${total} sự kiện timeline đã tạo cho ${topic}.`,
    analysisTitle: "Phân tích Event Detection",
    analysisSubtitle: (events: number, sentences: number, topic: string) =>
      `${events} câu event dự đoán từ ${sentences} câu trong ${topic}.`,
    highConfidenceEvents: "Các câu event độ tin cậy cao",
    sourcesTitle: "Nguồn tài liệu",
    sourcesSubtitle: (count: number, topic: string) => `${count} tài liệu có sẵn cho ${topic}.`,
    words: (count: string) => `${count} từ`,
    openSource: "Mở nguồn",
    chatTitle: "Chat RAG",
    chatSubtitle: "Hỏi tiếp dựa trên corpus local, câu trả lời có citation.",
    chatWelcome: (topic: string) => `Hỏi tôi về ${topic}. Tôi sẽ trả lời dựa trên corpus local và gắn citation.`,
    you: "Bạn",
    askPlaceholder: (topic: string) => `Hỏi về ${topic}...`,
    thinking: "Đang nghĩ",
    send: "Gửi",
    apiError: "Lỗi API",
    evaluationTitle: "Chỉ số đánh giá mô hình",
    evaluationSubtitle: "Đánh giá event detection và event type classification trên dataset ChronoRAG đã gán nhãn.",
    model: "Mô hình",
    binaryF1: "Binary F1",
    typeMacroF1: "Type Macro-F1",
    binaryAccuracy: "Binary Accuracy",
    binaryConfusionMatrix: "Confusion Matrix nhị phân",
    eventTypeConfusionMatrix: "Confusion Matrix loại event",
    experimentComparison: "So sánh thí nghiệm",
    typeF1: "Type F1",
    binaryAcc: "Binary Acc",
    emptyTimeline: "Chưa có sự kiện timeline.",
    noMatrix: "Chưa có confusion matrix.",
    predictedLabel: "Nhãn dự đoán",
    trueLabel: "Nhãn thật",
    confidence: "độ tin cậy",
    viewCitations: "Xem citation",
    loadingArtifacts: "Đang tải artifacts ChronoRAG...",
    cannotConnect: "Không kết nối được ChronoRAG API",
    showMore: "Xem thêm",
    showLess: "Thu gọn",
    suggestedQuestions: "Câu hỏi gợi ý",
    retrievalBenchmark: "Đánh giá retrieval",
    retrievalBenchmarkSubtitle: (n: number) => `${n} câu hỏi được gán nhãn để chấm Recall@k`,
    retrievalRecallAt: (k: number) => `Recall@${k}`,
    retrievalMrr: "MRR",
    retrievalPerTopic: "Theo chủ đề",
    retrievalUnavailable: "Chưa có kết quả retrieval benchmark. Chạy `python scripts/17_eval_retrieval.py`.",
    eventTypeLabels: {
      benchmark: "benchmark",
      method_proposed: "phương pháp",
      release: "phát hành",
      trend_application: "xu hướng / ứng dụng",
      none: "không event"
    }
  },
  en: {
    stable: "v2.4 Stable",
    newSession: "New Research Session",
    topicScope: "Topic Scope",
    projectLead: "Project Lead",
    systemHealthy: "System Healthy",
    loading: "Loading",
    languageToggle: "Change language",
    healthSummary: (docs: number, events: number) => `${docs} Docs / ${events} Timeline Events`,
    nav: {
      overview: "Overview",
      timeline: "Timeline",
      analysis: "Analysis",
      sources: "Sources",
      chat: "Chat",
      evaluation: "Evaluation"
    },
    overviewTitle: "Overview Dashboard",
    overviewSubtitle: (topic: string) => `Corpus analysis for: ${topic.toLowerCase().replace(/\s+/g, "_")}_timeline.json`,
    projectSummary: "Project Summary",
    eventsDetected: "Events Detected",
    docsIngested: "Docs Ingested",
    avgConfidence: "Avg Confidence",
    yearRange: "Year Range",
    modelStatus: "Model Status",
    eventDetector: "Event Detector",
    eventType: "Event Type",
    timeline: "Timeline",
    timelineStatus: (events: number, confidence: number) => `${events} events @ ${confidence.toFixed(2)}`,
    chronologicalInsights: "Chronological Insights",
    expandAll: "Expand All",
    oldestFirst: "Oldest First",
    newestFirst: "Newest First",
    timelineTitle: "Chronological Timeline",
    timelineSubtitle: (total: number, topic: string) => `${total} timeline events generated for ${topic}.`,
    analysisTitle: "Event Detection Analysis",
    analysisSubtitle: (events: number, sentences: number, topic: string) =>
      `${events} predicted events from ${sentences} sentences in ${topic}.`,
    highConfidenceEvents: "High Confidence Event Sentences",
    sourcesTitle: "Retrieved Sources",
    sourcesSubtitle: (count: number, topic: string) => `${count} documents available for ${topic}.`,
    words: (count: string) => `${count} words`,
    openSource: "Open source",
    chatTitle: "RAG Chat",
    chatSubtitle: "Ask follow-up questions with local retrieval and citation grounding.",
    chatWelcome: (topic: string) => `Ask me about ${topic}. I will answer using the local corpus and attach citations.`,
    you: "You",
    askPlaceholder: (topic: string) => `Ask about ${topic}...`,
    thinking: "Thinking",
    send: "Send",
    apiError: "API error",
    evaluationTitle: "Model Performance Metrics",
    evaluationSubtitle: "Evaluating event detection and event type classification across the labeled ChronoRAG dataset.",
    model: "Model",
    binaryF1: "Binary F1",
    typeMacroF1: "Type Macro-F1",
    binaryAccuracy: "Binary Accuracy",
    binaryConfusionMatrix: "Binary Confusion Matrix",
    eventTypeConfusionMatrix: "Event Type Confusion Matrix",
    experimentComparison: "Experiment Comparison",
    typeF1: "Type F1",
    binaryAcc: "Binary Acc",
    emptyTimeline: "No timeline events generated yet.",
    noMatrix: "No confusion matrix available.",
    predictedLabel: "Predicted Label",
    trueLabel: "True Label",
    confidence: "confidence",
    viewCitations: "View citations",
    loadingArtifacts: "Loading ChronoRAG artifacts...",
    cannotConnect: "Cannot connect to ChronoRAG API",
    showMore: "Show more",
    showLess: "Show less",
    suggestedQuestions: "Suggested questions",
    retrievalBenchmark: "Retrieval benchmark",
    retrievalBenchmarkSubtitle: (n: number) => `${n} hand-labeled questions scored against Recall@k`,
    retrievalRecallAt: (k: number) => `Recall@${k}`,
    retrievalMrr: "MRR",
    retrievalPerTopic: "Per topic",
    retrievalUnavailable: "Retrieval benchmark not run yet. Run `python scripts/17_eval_retrieval.py`.",
    eventTypeLabels: {
      benchmark: "benchmark",
      method_proposed: "method proposed",
      release: "release",
      trend_application: "trend / application",
      none: "none"
    }
  }
};

function App() {
  const [view, setView] = useState<ViewId>("overview");
  const [language, setLanguage] = useState<Language>("vi");
  const [topic, setTopic] = useState<TopicId>("rag");
  const [topics, setTopics] = useState<Topic[]>(fallbackTopics);
  const [model, setModel] = useState("sgd_log");
  const [health, setHealth] = useState<{
    status: string;
    documents: number;
    predictions: number;
    timelineEvents: number;
    generation?: GenerationStatus;
  } | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [events, setEvents] = useState<EventsResponse | null>(null);
  const [sources, setSources] = useState<SourcesResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [retrievalEval, setRetrievalEval] = useState<RetrievalEvalResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chatSessionId, setChatSessionId] = useState(0);

  const topicLabel = useMemo(
    () => topics.find((item) => item.id === topic)?.label ?? topic,
    [topics, topic]
  );
  const t = translations[language];

  useEffect(() => {
    let active = true;
    Promise.all([api.topics(), api.health(), api.retrievalEval()])
      .then(([topicRows, healthRow, retrievalRow]) => {
        if (!active) return;
        setTopics(topicRows);
        setHealth(healthRow);
        setRetrievalEval(retrievalRow);
      })
      .catch((err: Error) => {
        if (active) setError(err.message);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    Promise.all([
      api.overview(topic),
      api.timeline(topic),
      api.events(topic, 40),
      api.sources(topic)
    ])
      .then(([overviewRow, timelineRow, eventsRow, sourcesRow]) => {
        if (!active) return;
        setOverview(overviewRow);
        setTimeline(timelineRow);
        setEvents(eventsRow);
        setSources(sourcesRow);
      })
      .catch((err: Error) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [topic]);

  useEffect(() => {
    let active = true;
    api.evaluation(model)
      .then((evalRow) => {
        if (!active) return;
        setEvaluation(evalRow);
        if (evalRow.model !== model) {
          setModel(evalRow.model);
        }
      })
      .catch((err: Error) => {
        if (active) setError(err.message);
      });
    return () => {
      active = false;
    };
  }, [model]);

  return (
    <div className="app-shell">
      <Sidebar
        activeView={view}
        onViewChange={setView}
        onNewSession={() => {
          setView("chat");
          setChatSessionId((current) => current + 1);
        }}
        topics={topics}
        topic={topic}
        onTopicChange={setTopic}
        t={t}
      />

      <main className="main-shell">
        <Topbar health={health} language={language} onLanguageChange={setLanguage} t={t} />

        <section className="content-frame">
          {error ? <ErrorPanel message={error} t={t} /> : null}
          {loading && !overview ? <LoadingPanel t={t} /> : null}

          {!loading && !error && view === "overview" && overview && timeline ? (
            <OverviewView
              topicLabel={topicLabel}
              overview={overview}
              timelineEvents={timeline.events.slice(0, 6)}
              onViewChange={setView}
              language={language}
              t={t}
            />
          ) : null}
          {!loading && !error && view === "timeline" && timeline ? (
            <TimelineView topicLabel={topicLabel} timeline={timeline} language={language} t={t} />
          ) : null}
          {!loading && !error && view === "analysis" && events ? (
            <AnalysisView topicLabel={topicLabel} events={events} language={language} t={t} />
          ) : null}
          {!loading && !error && view === "sources" && sources ? (
            <SourcesView topicLabel={topicLabel} sources={sources.sources} t={t} />
          ) : null}
          {!loading && !error && view === "chat" ? (
            <ChatView key={chatSessionId} topic={topic} topicLabel={topicLabel} t={t} />
          ) : null}
          {!loading && !error && view === "evaluation" && evaluation ? (
            <EvaluationView evaluation={evaluation} model={model} onModelChange={setModel} language={language} retrievalEval={retrievalEval} t={t} />
          ) : null}
        </section>
      </main>
    </div>
  );
}

function Sidebar({
  activeView,
  onViewChange,
  onNewSession,
  topics,
  topic,
  onTopicChange,
  t
}: {
  activeView: ViewId;
  onViewChange: (view: ViewId) => void;
  onNewSession: () => void;
  topics: Topic[];
  topic: TopicId;
  onTopicChange: (topic: TopicId) => void;
  t: Translations;
}) {
  return (
    <aside className="sidebar">
      <div className="brand-row">
        <div className="brand-mark">
          <Workflow size={22} />
        </div>
        <div>
          <h1>ChronoRAG AI</h1>
          <p><span className="status-dot" /> {t.stable}</p>
        </div>
      </div>

      <button className="new-session" onClick={onNewSession}>
        <span>+</span>
        {t.newSession}
      </button>

      <label className="sidebar-label">{t.topicScope}</label>
      <select className="topic-select" value={topic} onChange={(event) => onTopicChange(event.target.value as TopicId)}>
        {topics.map((item) => (
          <option value={item.id} key={item.id}>
            {item.label}
          </option>
        ))}
      </select>

      <nav className="nav-list">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              className={`nav-item ${activeView === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => onViewChange(item.id)}
            >
              <Icon size={22} />
              <span>{t.nav[item.id]}</span>
            </button>
          );
        })}
      </nav>

      <div className="researcher-card">
        <div className="avatar">TL</div>
        <div>
          <strong>Tlam</strong>
          <span>{t.projectLead}</span>
        </div>
      </div>
    </aside>
  );
}

function Topbar({
  health,
  language,
  onLanguageChange,
  t
}: {
  health: {
    status: string;
    documents: number;
    predictions: number;
    timelineEvents: number;
    generation?: GenerationStatus;
  } | null;
  language: Language;
  onLanguageChange: (language: Language) => void;
  t: Translations;
}) {
  const generation = health?.generation;
  const generationLabel = generation?.mode === "llm"
    ? `LLM: ${generation.model}`
    : language === "vi"
      ? "Local RAG"
      : "Local RAG";
  return (
    <header className="topbar">
      <div className="top-status">
        <span className="healthy-pill">{t.systemHealthy}</span>
        <span className="mono">{health ? t.healthSummary(health.documents, health.timelineEvents) : t.loading}</span>
        <span className={`generation-pill ${generation?.mode === "llm" ? "llm" : "local"}`}>
          {generationLabel}
        </span>
        <div className="language-toggle" aria-label={t.languageToggle}>
          <button className={language === "vi" ? "active" : ""} onClick={() => onLanguageChange("vi")}>VI</button>
          <button className={language === "en" ? "active" : ""} onClick={() => onLanguageChange("en")}>EN</button>
        </div>
      </div>
      <div className="top-icons">
        <div className="mini-avatar">AI</div>
      </div>
    </header>
  );
}

function OverviewView({
  topicLabel,
  overview,
  timelineEvents,
  onViewChange,
  language,
  t
}: {
  topicLabel: string;
  overview: OverviewResponse;
  timelineEvents: TimelineEvent[];
  onViewChange: (view: ViewId) => void;
  language: Language;
  t: Translations;
}) {
  const { summary } = overview;
  const [timelineOrder, setTimelineOrder] = useState<"asc" | "desc">("asc");
  const sortedTimelineEvents = useMemo(
    () =>
      [...timelineEvents].sort((left, right) =>
        timelineOrder === "asc" ? left.year - right.year : right.year - left.year
      ),
    [timelineEvents, timelineOrder]
  );

  return (
    <>
      <PageHeader
        title={t.overviewTitle}
        subtitle={t.overviewSubtitle(topicLabel)}
      />

      <div className="overview-grid">
        <div className="left-stack">
          <Panel title={t.projectSummary} icon={<LineChart size={24} />}>
            <div className="summary-grid">
              <MetricTile label={t.eventsDetected} value={formatNumber(summary.eventsDetected)} />
              <MetricTile label={t.docsIngested} value={formatNumber(summary.docsIngested)} />
              <MetricTile label={t.avgConfidence} value={summary.avgConfidence.toFixed(2)} accent="green" />
              <MetricTile label={t.yearRange} value={summary.yearRange} />
            </div>
          </Panel>

          <Panel title={t.topicScope} icon={<SlidersHorizontal size={22} />}>
            <div className="topic-readout">{topicLabel}</div>
            <div className="terminal-card">
              <span className="prompt">$</span> &gt; query context --topic "{topicLabel}"
            </div>
          </Panel>

          <Panel title={t.modelStatus} icon={<BrainCircuit size={22} />}>
            <StatusLine label={t.eventDetector} value={overview.modelStatus.eventDetector} tone="green" />
            <StatusLine label={t.eventType} value={overview.modelStatus.eventTypeModel} tone="purple" />
            <StatusLine label={t.timeline} value={t.timelineStatus(summary.timelineEvents, summary.avgConfidence)} tone="orange" />
            <ProgressBar tone="orange" value={summary.avgConfidence} />
          </Panel>
        </div>

        <Panel
          className="timeline-panel"
          title={t.chronologicalInsights}
          icon={<Workflow size={25} />}
          actions={
            <PanelActions
              expandLabel={t.expandAll}
              sortLabel={timelineOrder === "asc" ? t.oldestFirst : t.newestFirst}
              onExpand={() => onViewChange("timeline")}
              onSort={() => setTimelineOrder((current) => (current === "asc" ? "desc" : "asc"))}
            />
          }
        >
          <TimelineCanvas events={sortedTimelineEvents} language={language} t={t} />
        </Panel>
      </div>
    </>
  );
}

function TimelineView({
  topicLabel,
  timeline,
  language,
  t
}: {
  topicLabel: string;
  timeline: TimelineResponse;
  language: Language;
  t: Translations;
}) {
  return (
    <>
      <PageHeader
        title={t.timelineTitle}
        subtitle={t.timelineSubtitle(Number(timeline.summary.total ?? 0), topicLabel)}
      />
      <div className="timeline-full">
        {timeline.events.map((event, index) => (
          <TimelineCard key={event.event_id} event={event} side={index % 2 === 0 ? "left" : "right"} language={language} t={t} />
        ))}
      </div>
    </>
  );
}

function AnalysisView({
  topicLabel,
  events,
  language,
  t
}: {
  topicLabel: string;
  events: EventsResponse;
  language: Language;
  t: Translations;
}) {
  return (
    <>
      <PageHeader
        title={t.analysisTitle}
        subtitle={t.analysisSubtitle(events.summary.predictedEvents, events.summary.totalSentences, topicLabel)}
      />
      <div className="analysis-layout">
        <Panel title={t.highConfidenceEvents} icon={<Sparkles size={22} />}>
          <div className="event-table">
            {events.events.slice(0, 18).map((event) => (
              <EventRow key={event.sentenceId} event={event} language={language} t={t} />
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}

function SourcesView({ topicLabel, sources, t }: { topicLabel: string; sources: SourceDocument[]; t: Translations }) {
  return (
    <>
      <PageHeader title={t.sourcesTitle} subtitle={t.sourcesSubtitle(sources.length, topicLabel)} />
      <div className="source-grid">
        {sources.map((source) => (
          <article className="source-card" key={source.docId}>
            <div className="source-topline">
              <span className={`badge ${source.sourceType}`}>{source.sourceType}</span>
              <span className="mono">{source.year}</span>
            </div>
            <h3>{source.title}</h3>
            <p>{source.preview}</p>
            <div className="source-footer">
              <span>{t.words(formatNumber(source.wordCount))}</span>
              <a href={source.sourceUrl} target="_blank" rel="noreferrer">{t.openSource} <ChevronRight size={16} /></a>
            </div>
          </article>
        ))}
      </div>
    </>
  );
}

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: ChatResponse["citations"];
  mode?: ChatResponse["mode"];
  provider?: ChatResponse["provider"];
  model?: ChatResponse["model"];
  // Marks the initial canned welcome so it always renders in the current
  // language even after the user has typed (otherwise it stays in whatever
  // language was active at mount).
  kind?: "welcome";
};

function ChatView({ topic, topicLabel, t }: { topic: TopicId; topicLabel: string; t: Translations }) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "", kind: "welcome" }
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  async function submit(text?: string) {
    const question = (text ?? input).trim();
    if (!question || sending) return;
    setInput("");
    // Only send the last 6 turns (3 user + 3 assistant) and drop the canned
    // welcome so the LLM can resolve follow-ups like "nó làm được gì".
    const history = messages
      .filter((m) => m.kind !== "welcome")
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));
    setMessages((rows) => [...rows, { role: "user", content: question }]);
    setSending(true);
    try {
      const answer = await api.chat(topic, question, history);
      setMessages((rows) => [
        ...rows,
        {
          role: "assistant",
          content: answer.answer,
          citations: answer.citations,
          mode: answer.mode,
          provider: answer.provider,
          model: answer.model
        }
      ]);
    } catch (err) {
      setMessages((rows) => [...rows, { role: "assistant", content: `${t.apiError}: ${(err as Error).message}` }]);
    } finally {
      setSending(false);
    }
  }

  const showSuggestions = messages.length === 1 && messages[0].kind === "welcome";
  const suggestions = CHAT_SUGGESTIONS[topic] ?? [];

  return (
    <>
      <PageHeader title={t.chatTitle} subtitle={t.chatSubtitle} />
      <div className="chat-shell">
        <div className="chat-messages">
          {messages.map((message, index) => {
            const displayedContent =
              message.kind === "welcome"
                ? t.chatWelcome(topicLabel)
                : message.role === "assistant"
                  ? cleanAnswerText(message.content)
                  : message.content;
            const isAbstain =
              message.role === "assistant" &&
              displayedContent.startsWith(ABSTAIN_PREFIX);
            const className = `chat-message ${message.role}${isAbstain ? " abstain" : ""}`;
            return (
              <div className={className} key={`${message.role}-${index}`}>
                <div className="message-role">
                  <span>{message.role === "user" ? t.you : "ChronoRAG"}</span>
                  {message.role === "assistant" && message.mode ? (
                    <span className={`message-mode ${message.mode === "llm" ? "llm" : "local"}`}>
                      {message.provider === "router"
                        ? "System"
                        : message.mode === "llm"
                          ? `LLM ${message.model ?? ""}`.trim()
                          : "Local RAG"}
                    </span>
                  ) : null}
                </div>
                <p>{displayedContent}</p>
                {message.citations?.length ? (
                  <div className="citation-list">
                    {message.citations.map((cite) => (
                      <a
                        key={cite.doc_id}
                        href={cite.source_url}
                        target="_blank"
                        rel="noreferrer"
                        title={cite.title}
                      >
                        <span className="citation-id">{cite.doc_id}</span>
                        <span className="citation-title">{cite.title}</span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            );
          })}
          {showSuggestions && suggestions.length ? (
            <div className="chat-suggestions">
              <span className="chat-suggestions-label">{t.suggestedQuestions}</span>
              <div className="chat-suggestions-chips">
                {suggestions.map((q) => (
                  <button
                    key={q}
                    type="button"
                    className="chat-suggestion-chip"
                    onClick={() => submit(q)}
                    disabled={sending}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
        <div className="chat-input-row">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submit();
            }}
            placeholder={t.askPlaceholder(topicLabel)}
          />
          <button onClick={() => submit()} disabled={sending}>{sending ? t.thinking : t.send}</button>
        </div>
      </div>
    </>
  );
}

function RetrievalBenchmarkPanel({
  data,
  t,
}: {
  data: RetrievalEvalResponse | null;
  t: Translations;
}) {
  if (!data || !data.available || !data.summary) {
    return (
      <Panel title={t.retrievalBenchmark} icon={<Database size={22} />}>
        <p className="panel-empty">{t.retrievalUnavailable}</p>
      </Panel>
    );
  }
  const s = data.summary;
  const n = s.count ?? data.config?.total_questions ?? 0;
  const kValues = (data.config?.k_values ?? [1, 3, 5, 10]) as number[];
  const topics = Object.entries(data.perTopic ?? {});

  return (
    <Panel
      title={t.retrievalBenchmark}
      icon={<Database size={22} />}
      subtitle={t.retrievalBenchmarkSubtitle(n)}
    >
      <div className="retrieval-summary">
        {kValues.map((k) => {
          const v = s[`recall@${k}` as keyof RetrievalEvalSummary] as number | undefined;
          return (
            <div className="retrieval-tile" key={`r${k}`}>
              <div className="retrieval-tile-label">{t.retrievalRecallAt(k)}</div>
              <div className="retrieval-tile-value">{v != null ? percent(v) : "—"}</div>
            </div>
          );
        })}
        <div className="retrieval-tile">
          <div className="retrieval-tile-label">{t.retrievalMrr}</div>
          <div className="retrieval-tile-value">{s.mrr != null ? s.mrr.toFixed(3) : "—"}</div>
        </div>
      </div>
      {topics.length ? (
        <div className="retrieval-per-topic">
          <div className="retrieval-per-topic-head">
            <span>{t.retrievalPerTopic}</span>
            <span>{t.retrievalRecallAt(1)}</span>
            <span>{t.retrievalRecallAt(5)}</span>
            <span>{t.retrievalMrr}</span>
          </div>
          {topics.map(([topic, metrics]) => (
            <div className="retrieval-per-topic-row" key={topic}>
              <span className="retrieval-per-topic-name">{topic}</span>
              <span>{metrics["recall@1"] != null ? percent(metrics["recall@1"]) : "—"}</span>
              <span>{metrics["recall@5"] != null ? percent(metrics["recall@5"]) : "—"}</span>
              <span>{metrics.mrr != null ? metrics.mrr.toFixed(3) : "—"}</span>
            </div>
          ))}
        </div>
      ) : null}
    </Panel>
  );
}

function EvaluationView({
  evaluation,
  model,
  onModelChange,
  language,
  retrievalEval,
  t
}: {
  evaluation: EvaluationResponse;
  model: string;
  onModelChange: (model: string) => void;
  language: Language;
  retrievalEval: RetrievalEvalResponse | null;
  t: Translations;
}) {
  return (
    <>
      <PageHeader
        title={t.evaluationTitle}
        subtitle={t.evaluationSubtitle}
      />
      <div className="eval-controls">
        <label>{t.model}</label>
        <select value={model} onChange={(event) => onModelChange(event.target.value)}>
          {evaluation.models.map((item) => (
            <option key={item} value={item}>{item}</option>
          ))}
        </select>
      </div>
      <div className="metric-row">
        <MetricPanel label={t.binaryF1} value={evaluation.summary.event_detection_f1} tone="purple" />
        <MetricPanel label={t.typeMacroF1} value={evaluation.summary.event_type_macro_f1} tone="orange" />
        <MetricPanel label={t.binaryAccuracy} value={percent(evaluation.binary.accuracy)} tone="green" />
      </div>
      <RetrievalBenchmarkPanel data={retrievalEval} t={t} />
      <div className="eval-grid">
        <Panel title={t.binaryConfusionMatrix} icon={<Layers3 size={22} />}>
          <ConfusionMatrix matrix={evaluation.binary.confusionMatrix} labels={evaluation.binary.labels.map(String)} language={language} t={t} />
        </Panel>
        <Panel title={t.eventTypeConfusionMatrix} icon={<BarChart3 size={22} />}>
          <ConfusionMatrix matrix={evaluation.eventType.confusionMatrix} labels={evaluation.eventType.labels} compact language={language} t={t} />
        </Panel>
      </div>
      <Panel title={t.experimentComparison} icon={<FileText size={22} />}>
        <div className="comparison-table">
          <div className="comparison-head">
            <span>{t.model}</span><span>{t.binaryF1}</span><span>{t.typeF1}</span><span>{t.binaryAcc}</span>
          </div>
          {evaluation.comparison.map((row) => (
            <div className="comparison-row" key={row.model}>
              <span>{row.model}</span>
              <span>{percent(row.binaryF1)}</span>
              <span>{percent(row.eventTypeF1)}</span>
              <span>{percent(row.binaryAccuracy)}</span>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}

function TimelineCanvas({ events, language, t }: { events: TimelineEvent[]; language: Language; t: Translations }) {
  if (!events.length) {
    return <EmptyState text={t.emptyTimeline} />;
  }
  return (
    <div className="timeline-canvas">
      <div className="timeline-axis" />
      {events.slice(0, 5).map((event, index) => (
        <TimelineCard key={event.event_id} event={event} side={index % 2 === 0 ? "right" : "left"} compact language={language} t={t} />
      ))}
    </div>
  );
}

function TimelineCard({
  event,
  side,
  compact = false,
  language,
  t
}: {
  event: TimelineEvent;
  side: "left" | "right";
  compact?: boolean;
  language: Language;
  t: Translations;
}) {
  const firstSource = event.sources?.[0];
  const docId = firstSource?.doc_id ?? firstSource?.docId ?? event.doc_ids?.[0] ?? "source";
  const sourceUrl = firstSource?.source_url ?? firstSource?.sourceUrl ?? "";
  const [expanded, setExpanded] = useState(false);
  // Only show toggle when the sentence is long enough to overflow the clamp;
  // ~340 chars maps roughly to >5 lines at our typography. Keeps short cards
  // clean (no useless "show more" button).
  const isLong = (event.representative_sentence?.length ?? 0) > 340;
  return (
    <article className={`timeline-card ${side} ${compact ? "compact" : ""}${expanded ? " expanded" : ""}`}>
      <div className={`timeline-dot ${event.event_type}`} />
      <div className="timeline-card-body">
        <div className="timeline-card-top">
          <span className="timeline-year">{event.date}</span>
          <span className={`badge ${event.event_type}`}>{labelEventType(event.event_type, language)}</span>
        </div>
        <p>{event.representative_sentence}</p>
        {!compact && isLong ? (
          <button type="button" className="timeline-card-toggle" onClick={() => setExpanded((v) => !v)}>
            {expanded ? t.showLess : t.showMore}
          </button>
        ) : null}
        <div className="timeline-card-footer">
          <span className="source-chip">{docId}</span>
          {sourceUrl ? <a href={sourceUrl} target="_blank" rel="noreferrer">{t.viewCitations} <ChevronRight size={15} /></a> : null}
        </div>
      </div>
    </article>
  );
}

function EventRow({ event, language, t }: { event: EventPrediction; language: Language; t: Translations }) {
  return (
    <div className="event-row">
      <div>
        <div className="event-row-top">
          <span className={`badge ${event.eventType}`}>{labelEventType(event.eventType, language)}</span>
          <span className="mono">{event.docId}</span>
          <span className="mono">{event.normalizedDate ?? event.year}</span>
        </div>
        <p>{event.sentence}</p>
      </div>
      <div className="confidence-cell">
        <strong>{percent(event.probability)}</strong>
        <span>{t.confidence}</span>
      </div>
    </div>
  );
}

function ConfusionMatrix({
  matrix,
  labels,
  compact = false,
  language,
  t
}: {
  matrix: number[][];
  labels: string[];
  compact?: boolean;
  language: Language;
  t: Translations;
}) {
  if (!matrix?.length) return <EmptyState text={t.noMatrix} />;
  const displayLabels = labels.map((label) => shortMatrixLabel(label, language));
  return (
    <div className={`matrix-table-wrap ${compact ? "compact" : ""}`}>
      <table className="matrix-table">
        <thead>
          <tr>
            <th>{`${t.trueLabel} / ${t.predictedLabel}`}</th>
            {displayLabels.map((label, index) => (
              <th key={labels[index]} title={labelEventType(labels[index], language)}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={labels[rowIndex] ?? rowIndex}>
              <th title={labelEventType(labels[rowIndex] ?? "", language)}>{displayLabels[rowIndex] ?? labels[rowIndex]}</th>
              {row.map((value, colIndex) => (
                <td className={rowIndex === colIndex ? "hit" : "miss"} key={`${rowIndex}-${colIndex}`}>
                  {formatNumber(value)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="matrix-caption">
        <span>{t.trueLabel}</span>
        <span>{t.predictedLabel}</span>
      </div>
    </div>
  );
}

function Panel({
  title,
  subtitle,
  icon,
  children,
  className = "",
  actions
}: {
  title: string;
  subtitle?: string;
  icon: ReactNode;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-header">
        <div className="panel-title">
          {icon}
          <div>
            <h2>{title}</h2>
            {subtitle ? <p className="panel-subtitle">{subtitle}</p> : null}
          </div>
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="page-header">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
  );
}

function PanelActions({
  expandLabel,
  sortLabel,
  onExpand,
  onSort
}: {
  expandLabel: string;
  sortLabel: string;
  onExpand: () => void;
  onSort: () => void;
}) {
  return (
    <div className="panel-actions">
      <button onClick={onExpand}>{expandLabel}</button>
      <button onClick={onSort}><SlidersHorizontal size={15} /> {sortLabel}</button>
    </div>
  );
}

function MetricTile({ label, value, accent }: { label: string; value: string; accent?: "green" }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong className={accent}>{value}</strong>
    </div>
  );
}

function MetricPanel({ label, value, tone }: { label: string; value: string; tone: "purple" | "orange" | "green" }) {
  return (
    <div className="metric-panel">
      <span>{label}</span>
      <strong>{value}</strong>
      <ProgressBar tone={tone} value={parseFloat(value) / 100 || 0.7} />
    </div>
  );
}

function StatusLine({ label, value, tone }: { label: string; value: string; tone: "green" | "purple" | "orange" }) {
  return (
    <div className="status-line">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}

function ProgressBar({ tone, value }: { tone: "green" | "purple" | "orange"; value: number }) {
  return (
    <div className="progress-track">
      <div className={`progress-fill ${tone}`} style={{ width: `${Math.min(Math.max(value, 0.08), 1) * 100}%` }} />
    </div>
  );
}

function LoadingPanel({ t }: { t: Translations }) {
  return (
    <div className="center-panel">
      <Loader2 className="spin" size={28} />
      <span>{t.loadingArtifacts}</span>
    </div>
  );
}

function ErrorPanel({ message, t }: { message: string; t: Translations }) {
  return (
    <div className="center-panel error">
      <strong>{t.cannotConnect}</strong>
      <span>{message}</span>
      <code>uvicorn backend.main:app --reload</code>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function percent(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

function labelEventType(type: string, language: Language) {
  return translations[language].eventTypeLabels[type] ?? type.replace(/_/g, " ");
}

function shortMatrixLabel(type: string, language: Language) {
  if (type === "0" || type === "1") return type;
  const labels: Record<Language, Record<string, string>> = {
    vi: {
      method_proposed: "phương pháp",
      release: "phát hành",
      benchmark: "benchmark",
      trend_application: "trend/app",
      none: "none"
    },
    en: {
      method_proposed: "method",
      release: "release",
      benchmark: "benchmark",
      trend_application: "trend/app",
      none: "none"
    }
  };
  return labels[language][type] ?? type.replace(/_/g, " ");
}

export default App;
