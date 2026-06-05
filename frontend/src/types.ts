export type TopicId = "rag" | "ai_agent" | "knowledge_distillation";

export type Topic = {
  id: TopicId;
  label: string;
};

export type GenerationStatus = {
  mode: "local" | "llm" | string;
  provider: string;
  model: string;
  configured: boolean;
};

export type TimelineSource = {
  doc_id?: string;
  docId?: string;
  title: string;
  source_url?: string;
  sourceUrl?: string;
};

export type TimelineEvent = {
  event_id: string;
  topic: TopicId;
  date: string;
  normalized_date: string;
  year: number;
  event_type: string;
  title: string;
  representative_sentence: string;
  confidence: number;
  rank_score?: number;
  event_type_confidence?: number;
  date_confidence?: number;
  sources: TimelineSource[];
  cluster_size: number;
  doc_ids?: string[];
};

export type OverviewResponse = {
  topic: TopicId;
  summary: {
    eventsDetected: number;
    timelineEvents: number;
    docsIngested: number;
    totalDocs: number;
    avgConfidence: number;
    yearRange: string;
    corpusIngestedPercent: number;
  };
  topicBreakdown: Array<{
    topic: TopicId;
    label: string;
    documents: number;
    events: number;
    timelineEvents: number;
  }>;
  modelStatus: {
    eventDetector: string;
    eventTypeModel: string;
    retrieval: string;
    timeline: string;
    processedTopics: string[];
  };
  timelinePreview: TimelineEvent[];
};

export type TimelineResponse = {
  topic: TopicId;
  source: string;
  summary: Record<string, number | string>;
  events: TimelineEvent[];
};

export type EventPrediction = {
  sentenceId: string;
  docId: string;
  topic: TopicId;
  year: number;
  sourceUrl: string;
  sentence: string;
  isEvent: number;
  probability: number;
  eventType: string;
  eventTypeConfidence: number;
  normalizedDate?: string;
  dateConfidence?: number;
  dateSource?: string;
};

export type EventsResponse = {
  topic: TopicId;
  summary: {
    totalSentences: number;
    predictedEvents: number;
  };
  events: EventPrediction[];
};

export type SourceDocument = {
  docId: string;
  title: string;
  topic: TopicId;
  sourceType: string;
  sourceUrl: string;
  year: number;
  authors: string[] | string;
  wordCount: number;
  preview: string;
};

export type SourcesResponse = {
  topic: TopicId | null;
  sources: SourceDocument[];
};

export type EvaluationResponse = {
  model: string;
  models: string[];
  summary: {
    event_detection_f1: string;
    event_type_macro_f1: string;
    retrieval_recall_at_5: string;
    timeline_date_accuracy: string;
  };
  binary: {
    accuracy: number;
    f1Macro: number;
    precisionMacro: number;
    recallMacro: number;
    confusionMatrix: number[][];
    labels: Array<string | number>;
  };
  eventType: {
    accuracy: number;
    f1Macro: number;
    precisionMacro: number;
    recallMacro: number;
    confusionMatrix: number[][];
    labels: string[];
  };
  comparison: Array<{
    model: string;
    binaryF1: number;
    eventTypeF1: number;
    binaryAccuracy: number;
    eventTypeAccuracy: number;
  }>;
};

export type ChatResponse = {
  topic: TopicId;
  question: string;
  answer: string;
  citations: Array<{
    doc_id: string;
    title: string;
    source_url: string;
  }>;
  mode?: "local" | "llm" | string;
  provider?: string;
  model?: string;
};
