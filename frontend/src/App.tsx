import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import {
  BarChart3,
  Bell,
  BrainCircuit,
  ChevronRight,
  Database,
  FileText,
  Grid2X2,
  Layers3,
  LineChart,
  Loader2,
  MessageSquare,
  Search,
  Settings,
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
  OverviewResponse,
  SourceDocument,
  SourcesResponse,
  TimelineEvent,
  TimelineResponse,
  Topic,
  TopicId
} from "./types";

type ViewId = "overview" | "timeline" | "analysis" | "sources" | "chat" | "evaluation";

const fallbackTopics: Topic[] = [
  { id: "rag", label: "RAG" },
  { id: "ai_agent", label: "AI Agent" },
  { id: "knowledge_distillation", label: "Knowledge Distillation" }
];

const navItems: Array<{ id: ViewId; label: string; icon: typeof Grid2X2 }> = [
  { id: "overview", label: "Overview", icon: Grid2X2 },
  { id: "timeline", label: "Timeline", icon: Workflow },
  { id: "analysis", label: "Analysis", icon: BarChart3 },
  { id: "sources", label: "Sources", icon: Database },
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "evaluation", label: "Evaluation", icon: SquareStack }
];

function App() {
  const [view, setView] = useState<ViewId>("overview");
  const [topic, setTopic] = useState<TopicId>("rag");
  const [topics, setTopics] = useState<Topic[]>(fallbackTopics);
  const [model, setModel] = useState("sgd_log");
  const [health, setHealth] = useState<{ status: string; documents: number; predictions: number; timelineEvents: number } | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [events, setEvents] = useState<EventsResponse | null>(null);
  const [sources, setSources] = useState<SourcesResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const topicLabel = useMemo(
    () => topics.find((item) => item.id === topic)?.label ?? topic,
    [topics, topic]
  );

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    Promise.all([
      api.topics(),
      api.health(),
      api.overview(topic),
      api.timeline(topic),
      api.events(topic, 40),
      api.sources(topic),
      api.evaluation(model)
    ])
      .then(([topicRows, healthRow, overviewRow, timelineRow, eventsRow, sourcesRow, evalRow]) => {
        if (!active) return;
        setTopics(topicRows);
        setHealth(healthRow);
        setOverview(overviewRow);
        setTimeline(timelineRow);
        setEvents(eventsRow);
        setSources(sourcesRow);
        setEvaluation(evalRow);
        if (evalRow.model !== model) {
          setModel(evalRow.model);
        }
      })
      .catch((err: Error) => {
        if (!active) return;
        setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [topic, model]);

  return (
    <div className="app-shell">
      <Sidebar
        activeView={view}
        onViewChange={setView}
        topics={topics}
        topic={topic}
        onTopicChange={setTopic}
      />

      <main className="main-shell">
        <Topbar health={health} />

        <section className="content-frame">
          {error ? <ErrorPanel message={error} /> : null}
          {loading && !overview ? <LoadingPanel /> : null}

          {!loading && !error && view === "overview" && overview && timeline ? (
            <OverviewView
              topicLabel={topicLabel}
              overview={overview}
              evaluation={evaluation}
              timelineEvents={timeline.events.slice(0, 6)}
              onViewChange={setView}
            />
          ) : null}
          {!loading && !error && view === "timeline" && timeline ? (
            <TimelineView topicLabel={topicLabel} timeline={timeline} />
          ) : null}
          {!loading && !error && view === "analysis" && events ? (
            <AnalysisView topicLabel={topicLabel} events={events} />
          ) : null}
          {!loading && !error && view === "sources" && sources ? (
            <SourcesView topicLabel={topicLabel} sources={sources.sources} />
          ) : null}
          {!loading && !error && view === "chat" ? (
            <ChatView topic={topic} topicLabel={topicLabel} />
          ) : null}
          {!loading && !error && view === "evaluation" && evaluation ? (
            <EvaluationView evaluation={evaluation} model={model} onModelChange={setModel} />
          ) : null}
        </section>
      </main>
    </div>
  );
}

function Sidebar({
  activeView,
  onViewChange,
  topics,
  topic,
  onTopicChange
}: {
  activeView: ViewId;
  onViewChange: (view: ViewId) => void;
  topics: Topic[];
  topic: TopicId;
  onTopicChange: (topic: TopicId) => void;
}) {
  return (
    <aside className="sidebar">
      <div className="brand-row">
        <div className="brand-mark">
          <Workflow size={22} />
        </div>
        <div>
          <h1>ChronoRAG AI</h1>
          <p><span className="status-dot" /> v2.4 Stable</p>
        </div>
      </div>

      <button className="new-session" onClick={() => onViewChange("chat")}>
        <span>+</span>
        New Research Session
      </button>

      <label className="sidebar-label">Topic Scope</label>
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
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="researcher-card">
        <div className="avatar">TL</div>
        <div>
          <strong>Tlam</strong>
          <span>Project Lead</span>
        </div>
      </div>
    </aside>
  );
}

function Topbar({ health }: { health: { status: string; documents: number; predictions: number; timelineEvents: number } | null }) {
  return (
    <header className="topbar">
      <div className="search-box">
        <Search size={18} />
        <span>Search corpus or queries...</span>
      </div>
      <nav className="top-links">
        <span>Models</span>
        <span>Datasets</span>
        <span>Benchmarks</span>
      </nav>
      <div className="top-status">
        <span className="healthy-pill">System Healthy</span>
        <span className="mono">{health ? `${health.documents} Docs / ${health.timelineEvents} Timeline Events` : "Loading"}</span>
      </div>
      <div className="top-icons">
        <Bell size={22} />
        <Settings size={24} />
        <div className="mini-avatar">AI</div>
      </div>
    </header>
  );
}

function OverviewView({
  topicLabel,
  overview,
  evaluation,
  timelineEvents,
  onViewChange
}: {
  topicLabel: string;
  overview: OverviewResponse;
  evaluation: EvaluationResponse | null;
  timelineEvents: TimelineEvent[];
  onViewChange: (view: ViewId) => void;
}) {
  const { summary } = overview;
  const [timelineOrder, setTimelineOrder] = useState<"asc" | "desc">("asc");
  const eventF1 = evaluation?.binary.f1Macro ?? 0;
  const eventTypeF1 = evaluation?.eventType.f1Macro ?? 0;
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
        title="Overview Dashboard"
        subtitle={`Corpus analysis for: ${topicLabel.toLowerCase().replace(/\s+/g, "_")}_timeline.json`}
      />

      <div className="overview-grid">
        <div className="left-stack">
          <Panel title="Project Summary" icon={<LineChart size={24} />}>
            <div className="summary-grid">
              <MetricTile label="Events Detected" value={formatNumber(summary.eventsDetected)} />
              <MetricTile label="Docs Ingested" value={formatNumber(summary.docsIngested)} />
              <MetricTile label="Avg Confidence" value={summary.avgConfidence.toFixed(2)} accent="green" />
              <MetricTile label="Year Range" value={summary.yearRange} />
            </div>
          </Panel>

          <Panel title="Topic Scope" icon={<SlidersHorizontal size={22} />}>
            <div className="topic-readout">{topicLabel}</div>
            <div className="terminal-card">
              <span className="prompt">$</span> &gt; query context --topic "{topicLabel}"
            </div>
          </Panel>

          <Panel title="Model Status" icon={<BrainCircuit size={22} />}>
            <StatusLine label="Event Detector" value={`${overview.modelStatus.eventDetector} (${percent(eventF1)})`} tone="green" />
            <ProgressBar tone="green" value={eventF1} />
            <StatusLine label="Event Type" value={`${overview.modelStatus.eventTypeModel} (${percent(eventTypeF1)})`} tone="purple" />
            <ProgressBar tone="purple" value={eventTypeF1} />
            <StatusLine label="Timeline" value={`${summary.timelineEvents} events @ ${summary.avgConfidence.toFixed(2)}`} tone="orange" />
            <ProgressBar tone="orange" value={summary.avgConfidence} />
          </Panel>
        </div>

        <Panel
          className="timeline-panel"
          title="Chronological Insights"
          icon={<Workflow size={25} />}
          actions={
            <PanelActions
              sortLabel={timelineOrder === "asc" ? "Oldest First" : "Newest First"}
              onExpand={() => onViewChange("timeline")}
              onSort={() => setTimelineOrder((current) => (current === "asc" ? "desc" : "asc"))}
            />
          }
        >
          <TimelineCanvas events={sortedTimelineEvents} />
        </Panel>
      </div>
    </>
  );
}

function TimelineView({ topicLabel, timeline }: { topicLabel: string; timeline: TimelineResponse }) {
  return (
    <>
      <PageHeader
        title="Chronological Timeline"
        subtitle={`${timeline.summary.total} timeline events generated for ${topicLabel}.`}
      />
      <div className="timeline-full">
        {timeline.events.map((event, index) => (
          <TimelineCard key={event.event_id} event={event} side={index % 2 === 0 ? "left" : "right"} />
        ))}
      </div>
    </>
  );
}

function AnalysisView({ topicLabel, events }: { topicLabel: string; events: EventsResponse }) {
  return (
    <>
      <PageHeader
        title="Event Detection Analysis"
        subtitle={`${events.summary.predictedEvents} predicted events from ${events.summary.totalSentences} sentences in ${topicLabel}.`}
      />
      <div className="analysis-layout">
        <Panel title="High Confidence Event Sentences" icon={<Sparkles size={22} />}>
          <div className="event-table">
            {events.events.slice(0, 18).map((event) => (
              <EventRow key={event.sentenceId} event={event} />
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}

function SourcesView({ topicLabel, sources }: { topicLabel: string; sources: SourceDocument[] }) {
  return (
    <>
      <PageHeader title="Retrieved Sources" subtitle={`${sources.length} documents available for ${topicLabel}.`} />
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
              <span>{formatNumber(source.wordCount)} words</span>
              <a href={source.sourceUrl} target="_blank" rel="noreferrer">Open source <ChevronRight size={16} /></a>
            </div>
          </article>
        ))}
      </div>
    </>
  );
}

function ChatView({ topic, topicLabel }: { topic: TopicId; topicLabel: string }) {
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string; citations?: ChatResponse["citations"] }>>([
    {
      role: "assistant",
      content: `Ask me about ${topicLabel}. I will answer using the local corpus and attach citations.`
    }
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  async function submit() {
    const question = input.trim();
    if (!question || sending) return;
    setInput("");
    setMessages((rows) => [...rows, { role: "user", content: question }]);
    setSending(true);
    try {
      const answer = await api.chat(topic, question);
      setMessages((rows) => [...rows, { role: "assistant", content: answer.answer, citations: answer.citations }]);
    } catch (err) {
      setMessages((rows) => [...rows, { role: "assistant", content: `API error: ${(err as Error).message}` }]);
    } finally {
      setSending(false);
    }
  }

  return (
    <>
      <PageHeader title="RAG Chat" subtitle="Ask follow-up questions with local retrieval and citation grounding." />
      <div className="chat-shell">
        <div className="chat-messages">
          {messages.map((message, index) => (
            <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="message-role">{message.role === "user" ? "You" : "ChronoRAG"}</div>
              <p>{message.content}</p>
              {message.citations?.length ? (
                <div className="citation-list">
                  {message.citations.map((cite) => (
                    <a key={cite.doc_id} href={cite.source_url} target="_blank" rel="noreferrer">
                      [{cite.doc_id}] {cite.title}
                    </a>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
        </div>
        <div className="chat-input-row">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submit();
            }}
            placeholder={`Ask about ${topicLabel}...`}
          />
          <button onClick={submit} disabled={sending}>{sending ? "Thinking" : "Send"}</button>
        </div>
      </div>
    </>
  );
}

function EvaluationView({
  evaluation,
  model,
  onModelChange
}: {
  evaluation: EvaluationResponse;
  model: string;
  onModelChange: (model: string) => void;
}) {
  return (
    <>
      <PageHeader
        title="Model Performance Metrics"
        subtitle="Evaluating event detection and event type classification across the labeled ChronoRAG dataset."
      />
      <div className="eval-controls">
        <label>Model</label>
        <select value={model} onChange={(event) => onModelChange(event.target.value)}>
          {evaluation.models.map((item) => (
            <option key={item} value={item}>{item}</option>
          ))}
        </select>
      </div>
      <div className="metric-row">
        <MetricPanel label="Binary F1" value={evaluation.summary.event_detection_f1} tone="purple" />
        <MetricPanel label="Type Macro-F1" value={evaluation.summary.event_type_macro_f1} tone="orange" />
        <MetricPanel label="Binary Accuracy" value={percent(evaluation.binary.accuracy)} tone="green" />
      </div>
      <div className="eval-grid">
        <Panel title="Binary Confusion Matrix" icon={<Layers3 size={22} />}>
          <ConfusionMatrix matrix={evaluation.binary.confusionMatrix} labels={evaluation.binary.labels.map(String)} />
        </Panel>
        <Panel title="Event Type Confusion Matrix" icon={<BarChart3 size={22} />}>
          <ConfusionMatrix matrix={evaluation.eventType.confusionMatrix} labels={evaluation.eventType.labels} compact />
        </Panel>
      </div>
      <Panel title="Experiment Comparison" icon={<FileText size={22} />}>
        <div className="comparison-table">
          <div className="comparison-head">
            <span>Model</span><span>Binary F1</span><span>Type F1</span><span>Binary Acc</span>
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

function TimelineCanvas({ events }: { events: TimelineEvent[] }) {
  if (!events.length) {
    return <EmptyState text="No timeline events generated yet." />;
  }
  return (
    <div className="timeline-canvas">
      <div className="timeline-axis" />
      {events.slice(0, 5).map((event, index) => (
        <TimelineCard key={event.event_id} event={event} side={index % 2 === 0 ? "right" : "left"} compact />
      ))}
    </div>
  );
}

function TimelineCard({ event, side, compact = false }: { event: TimelineEvent; side: "left" | "right"; compact?: boolean }) {
  const firstSource = event.sources?.[0];
  const docId = firstSource?.doc_id ?? firstSource?.docId ?? event.doc_ids?.[0] ?? "source";
  const sourceUrl = firstSource?.source_url ?? firstSource?.sourceUrl ?? "";
  return (
    <article className={`timeline-card ${side} ${compact ? "compact" : ""}`}>
      <div className={`timeline-dot ${event.event_type}`} />
      <div className="timeline-card-body">
        <div className="timeline-card-top">
          <span className="timeline-year">{event.date}</span>
          <span className={`badge ${event.event_type}`}>{labelEventType(event.event_type)}</span>
        </div>
        <h3>{event.title}</h3>
        <p>{event.representative_sentence}</p>
        <div className="timeline-card-footer">
          <span className="source-chip">{docId}</span>
          {sourceUrl ? <a href={sourceUrl} target="_blank" rel="noreferrer">View citations <ChevronRight size={15} /></a> : null}
        </div>
      </div>
    </article>
  );
}

function EventRow({ event }: { event: EventPrediction }) {
  return (
    <div className="event-row">
      <div>
        <div className="event-row-top">
          <span className={`badge ${event.eventType}`}>{labelEventType(event.eventType)}</span>
          <span className="mono">{event.docId}</span>
          <span className="mono">{event.normalizedDate ?? event.year}</span>
        </div>
        <p>{event.sentence}</p>
      </div>
      <div className="confidence-cell">
        <strong>{percent(event.probability)}</strong>
        <span>confidence</span>
      </div>
    </div>
  );
}

function ConfusionMatrix({ matrix, labels, compact = false }: { matrix: number[][]; labels: string[]; compact?: boolean }) {
  if (!matrix?.length) return <EmptyState text="No confusion matrix available." />;
  return (
    <div className={`confusion ${compact ? "compact" : ""}`} style={{ "--matrix-size": labels.length } as CSSProperties}>
      <div className="matrix-label top">Predicted Label</div>
      <div className="matrix-label left">True Label</div>
      <div className="matrix-grid">
        {matrix.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => (
            <div className={`matrix-cell ${rowIndex === colIndex ? "hit" : "miss"}`} key={`${rowIndex}-${colIndex}`}>
              <strong>{formatNumber(value)}</strong>
              <span>{`${labels[rowIndex]} -> ${labels[colIndex]}`}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function Panel({
  title,
  icon,
  children,
  className = "",
  actions
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-header">
        <div className="panel-title">{icon}<h2>{title}</h2></div>
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
  sortLabel,
  onExpand,
  onSort
}: {
  sortLabel: string;
  onExpand: () => void;
  onSort: () => void;
}) {
  return (
    <div className="panel-actions">
      <button onClick={onExpand}>Expand All</button>
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

function LoadingPanel() {
  return (
    <div className="center-panel">
      <Loader2 className="spin" size={28} />
      <span>Loading ChronoRAG artifacts...</span>
    </div>
  );
}

function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="center-panel error">
      <strong>Cannot connect to ChronoRAG API</strong>
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

function labelEventType(type: string) {
  return type.replace(/_/g, " ");
}

export default App;
