import sys
from pathlib import Path
import streamlit as st

# Add the project root to python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.online_pipeline import (
    get_mock_answer,
    get_mock_evaluation_metrics,
    get_mock_sentence_predictions,
    get_mock_timeline,
    get_local_qa_answer,
)

# Define beautiful CSS styles for modern, premium look
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* Typography & Base Styling */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
}

/* Glassmorphism Card Style */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
}

/* Timeline Vertical Line & Badges */
.timeline-item {
    position: relative;
    border-left: 3px solid #6366f1;
    padding-left: 24px;
    margin-bottom: 30px;
    margin-left: 10px;
}
.timeline-dot {
    position: absolute;
    left: -9px;
    top: 5px;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background-color: #6366f1;
    border: 3px solid #0e1117;
    box-shadow: 0 0 10px #6366f1;
}
.timeline-date {
    font-size: 1.15rem;
    font-weight: 700;
    color: #818cf8;
    margin-bottom: 6px;
}

/* Event Badges styling */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-method {
    background-color: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.badge-release {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-benchmark {
    background-color: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.badge-trend {
    background-color: rgba(236, 72, 153, 0.15);
    color: #f472b6;
    border: 1px solid rgba(236, 72, 153, 0.3);
}
.badge-none {
    background-color: rgba(156, 163, 175, 0.15);
    color: #9ca3af;
    border: 1px solid rgba(156, 163, 175, 0.3);
}

/* Metric Display card */
.metric-box {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #6366f1;
}
.metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-top: 4px;
}
</style>
"""

# App header config
st.set_page_config(
    page_title="ChronoRAG — ML-enhanced Timeline-Aware Assistant",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #818cf8;'>⏱️ ChronoRAG</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #9ca3af;'>ML-enhanced Timeline-Aware Research Assistant</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### Settings")
    topic_select = st.selectbox(
        "Choose Topic",
        ["RAG", "AI Agent", "Knowledge Distillation"]
    )
    
    retrieval_k = st.slider("Retrieval Top-K Chunks", min_value=1, max_value=10, value=5)
    model_mode = st.radio("Classifier Mode", ["TF-IDF + LogReg (Baseline)", "BiLSTM (Deep Learning)"])
    
    st.markdown("---")
    st.markdown("### Environment Info")
    st.warning("⚠️ LLM Provider is offline (mock). Change LLM_PROVIDER in .env to use OpenAI/Gemini.")

# Main app title area
st.markdown("<h1 style='margin-bottom: 5px;'>ChronoRAG Timeline Assistant</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #9ca3af; margin-bottom: 25px;'>Synthesizing progress for topic: <strong>{topic_select}</strong></p>", unsafe_allow_html=True)

# Fetch timeline events mock
timeline_events = get_mock_timeline(topic_select)

# Tabs
tab_timeline, tab_event_det, tab_chatbot, tab_evaluation = st.tabs([
    "📅 Interactive Timeline",
    "🔍 Event Detection (ML)",
    "💬 RAG Chatbot",
    "📈 Evaluation Dashboard"
])

# -----------------
# TAB 1: INTERACTIVE TIMELINE
# -----------------
with tab_timeline:
    st.markdown("### Research Progress Timeline")
    st.markdown("Here is the chronological evolution of the selected topic, synthesized from peer-reviewed publications and documentation.")
    
    if not timeline_events:
        st.info("No timeline events found for this topic.")
    else:
        for item in timeline_events:
            event_type = item["event_type"]
            badge_class = "badge-none"
            if event_type == "method_proposed":
                badge_class = "badge-method"
            elif event_type == "release":
                badge_class = "badge-release"
            elif event_type == "benchmark":
                badge_class = "badge-benchmark"
            elif event_type == "trend_application":
                badge_class = "badge-trend"
                
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-date">{item['date']} ({item['year']})</div>
                <span class="badge {badge_class}">{event_type.replace('_', ' ')}</span>
                <div class="glass-card">
                    <h4 style="margin-top:0px; margin-bottom:10px; color:#ffffff;">{item['title']}</h4>
                    <p style="color:#e2e8f0; font-size:0.95rem; line-height:1.5;">"{item['representative_sentence']}"</p>
                    <div style="margin-top: 15px; font-size: 0.85rem; color: #a1a1aa;">
                        <strong>Sources:</strong> 
                        {", ".join([f"<a href='{src['source_url']}' target='_blank' style='color:#818cf8; text-decoration:none;'>[{src['doc_id']}] {src['title']}</a>" for src in item['sources']])}
                    </div>
                    <div style="margin-top: 5px; font-size: 0.8rem; color: #71717a;">
                        Confidence Score: <code>{item['confidence'] * 100:.1f}%</code> &bull; Cluster Size: <code>{item['cluster_size']}</code>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------
# TAB 2: EVENT DETECTION (ML)
# -----------------
with tab_event_det:
    st.markdown("### Sentence-Level Classification")
    st.markdown("This tab displays how the sentence classifier classifies event vs non-event sentences, along with event type categorization and timeline confidence scores.")
    
    for row in get_mock_sentence_predictions(topic_select):
        is_evt = row["is_event"]
        prob = row["prob"]
        text_color = "#34d399" if is_evt else "#9ca3af"
        label_text = "EVENT" if is_evt else "NON-EVENT"
        
        st.markdown(f"""
        <div style="padding:15px; background:rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.04); border-radius:8px; margin-bottom:12px;">
            <p style="margin-bottom:8px; font-size:0.95rem; font-weight:500;">"{row['sentence']}"</p>
            <div style="display:flex; gap:20px; align-items:center; font-size:0.85rem;">
                <span style="color:{text_color}; font-weight:bold;">{label_text}</span>
                <span style="color:#a1a1aa;">Event Probability: <code>{prob*100:.1f}%</code></span>
                <span style="color:#a1a1aa;">Predicted Type: <code>{row['type']}</code></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------
# TAB 3: CHATBOT
# -----------------
with tab_chatbot:
    st.markdown("### Ask follow-up questions")
    st.markdown("Ask anything about this topic. The system will retrieve related chunks and formulate a response citing sources.")
    
    # Initialize message list
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Form input
    if prompt := st.chat_input("Enter your question (e.g., When was RAG proposed?)"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response using local QA retriever
        response_data = get_local_qa_answer(topic_select, prompt)
        answer = response_data["answer"]
        citations = response_data["citations"]
        
        # Format answer with citations list at the end
        full_response = answer + "\n\n**Sources & Citations:**\n"
        for i, cite in enumerate(citations, 1):
            full_response += f"- [{cite['doc_id']}] **{cite['title']}** - [Source Link]({cite['source_url']})\n"
            
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# -----------------
# TAB 4: EVALUATION DASHBOARD
# -----------------
with tab_evaluation:
    st.markdown("### Quantitative Model Evaluation")
    st.markdown("Evaluation metrics calculated on the validation/test document split (stratified by topic).")
    metrics = get_mock_evaluation_metrics()
    summary = metrics["summary"]
    
    # Layout metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{summary['event_detection_f1']}</div>
            <div class="metric-label">Event Detection F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{summary['event_type_macro_f1']}</div>
            <div class="metric-label">Event Type Macro-F1</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{summary['retrieval_recall_at_5']}</div>
            <div class="metric-label">Retrieval Recall@5</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{summary['timeline_date_accuracy']}</div>
            <div class="metric-label">Timeline Date Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><h4>Confusion Matrix (Event Type Classification)</h4>", unsafe_allow_html=True)
    st.markdown(metrics["confusion_matrix_markdown"], unsafe_allow_html=True)
    
    st.markdown("<br><h4>Experiment Comparison (ML vs DL)</h4>", unsafe_allow_html=True)
    st.markdown(metrics["experiment_markdown"], unsafe_allow_html=True)
