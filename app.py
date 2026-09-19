"""
AI Video Meeting Assistant — Streamlit UI
Wraps the existing CLI pipeline (utils.audio_processor, core.transcriber,
core.summarize, core.extractor, core.rag_engine) in a polished web UI.

Run with:
    streamlit run streamlit_app.py
"""

import os
import tempfile
import traceback
import logging

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Meeting Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Design system (CSS only — no behavioural changes)
# --------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
        /* ---------- Design tokens ---------- */
        :root {
            --brand:        #4f46e5;
            --brand-hover:  #4338ca;
            --brand-soft:   rgba(79, 70, 229, 0.10);
            --brand-ring:   rgba(79, 70, 229, 0.28);

            --bg:           #f7f8fb;
            --surface:      #ffffff;
            --surface-2:    #fbfbfd;
            --sidebar-bg:   #ffffff;

            --text:         #0f172a;
            --text-muted:   #64748b;
            --text-faint:   #94a3b8;

            --border:       #e5e7eb;
            --border-soft:  #eef0f4;

            --radius-sm:    10px;
            --radius:       14px;
            --radius-lg:    18px;

            --shadow-xs:    0 1px 2px rgba(16, 24, 40, 0.05);
            --shadow-sm:    0 1px 3px rgba(16, 24, 40, 0.07), 0 1px 2px rgba(16, 24, 40, 0.04);
            --shadow-md:    0 4px 14px rgba(16, 24, 40, 0.08);

            --ease:         cubic-bezier(0.4, 0, 0.2, 1);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --brand:        #818cf8;
                --brand-hover:  #a5b4fc;
                --brand-soft:   rgba(129, 140, 248, 0.14);
                --brand-ring:   rgba(129, 140, 248, 0.32);

                --bg:           #0b0f1a;
                --surface:      #131826;
                --surface-2:    #171d2e;
                --sidebar-bg:   #0e1320;

                --text:         #e9edf5;
                --text-muted:   #98a2b3;
                --text-faint:   #6b7688;

                --border:       #242b3d;
                --border-soft:  #1c2232;

                --shadow-xs:    0 1px 2px rgba(0, 0, 0, 0.35);
                --shadow-sm:    0 1px 3px rgba(0, 0, 0, 0.4);
                --shadow-md:    0 4px 14px rgba(0, 0, 0, 0.45);
            }
        }

        /* ---------- Base ---------- */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .stApp { background: var(--bg); color: var(--text); }

        .main .block-container {
            padding-top: 2.2rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }

        h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; font-weight: 700; }

        /* Cleaner product chrome */
        #MainMenu, footer { visibility: hidden; }
        div[data-testid="stStatusWidget"] { display: none; }
        header[data-testid="stHeader"] { background: transparent; }

        /* ---------- Hero ---------- */
        .hero {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.35rem 1.6rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.6rem;
        }
        .hero-mark {
            flex: 0 0 auto;
            width: 48px; height: 48px;
            display: grid; place-items: center;
            font-size: 1.5rem;
            border-radius: var(--radius);
            background: var(--brand-soft);
            border: 1px solid var(--brand-ring);
        }
        .hero-title {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.2;
            margin: 0;
            color: var(--text);
        }
        .hero-subtitle {
            color: var(--text-muted);
            font-size: 0.92rem;
            margin: 0.2rem 0 0 0;
        }

        /* ---------- Section heading ---------- */
        .doc-title {
            display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
            font-size: 1.45rem; font-weight: 750;
            letter-spacing: -0.025em;
            margin: 0.4rem 0 1.1rem 0;
            color: var(--text);
        }
        .doc-title .pill {
            font-size: 0.68rem; font-weight: 650;
            text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--brand);
            background: var(--brand-soft);
            border: 1px solid var(--brand-ring);
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
        }

        /* ---------- Stat cards ---------- */
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1rem 1.15rem;
            box-shadow: var(--shadow-xs);
            transition: transform 0.18s var(--ease), box-shadow 0.18s var(--ease), border-color 0.18s var(--ease);
            height: 100%;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            border-color: var(--brand-ring);
        }
        .stat-head {
            display: flex; align-items: center; gap: 0.45rem;
            font-size: 0.74rem; font-weight: 650;
            text-transform: uppercase; letter-spacing: 0.07em;
            color: var(--text-muted);
            margin-bottom: 0.45rem;
        }
        .stat-value {
            font-size: 1.9rem; font-weight: 750;
            line-height: 1; letter-spacing: -0.03em;
            color: var(--text);
        }

        /* ---------- Content cards ---------- */
        .section-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.35rem 1.5rem;
            box-shadow: var(--shadow-xs);
            margin-bottom: 0.4rem;
            line-height: 1.65;
        }
        .section-card ul { margin: 0; padding-left: 1.15rem; }
        .section-card li { margin-bottom: 0.45rem; }
        .empty-state {
            text-align: center;
            padding: 2rem 1rem;
            color: var(--text-faint);
            font-size: 0.9rem;
        }
        .empty-state .glyph { font-size: 1.8rem; display: block; margin-bottom: 0.5rem; opacity: 0.8; }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.35rem;
            box-shadow: var(--shadow-xs);
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 0.5rem 0.95rem;
            border-radius: var(--radius-sm);
            font-size: 0.88rem;
            font-weight: 550;
            color: var(--text-muted);
            background: transparent;
            border: none;
            transition: all 0.16s var(--ease);
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text);
            background: var(--brand-soft);
        }
        .stTabs [aria-selected="true"] {
            background: var(--surface) !important;
            color: var(--brand) !important;
            font-weight: 650;
            box-shadow: var(--shadow-sm);
        }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] { display: none; }
        .stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] > div { padding-top: 1.6rem; }
        .side-brand {
            display: flex; align-items: center; gap: 0.55rem;
            font-size: 1.02rem; font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 0.15rem;
        }
        .side-label {
            font-size: 0.7rem; font-weight: 650;
            text-transform: uppercase; letter-spacing: 0.09em;
            color: var(--text-faint);
            margin: 1.15rem 0 0.35rem 0;
        }
        section[data-testid="stSidebar"] hr {
            margin: 1.1rem 0;
            border-color: var(--border-soft);
        }

        /* ---------- Inputs ---------- */
        .stTextInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text) !important;
            font-size: 0.9rem !important;
            transition: border-color 0.16s var(--ease), box-shadow 0.16s var(--ease);
        }
        .stTextInput input:focus,
        .stTextArea textarea:focus,
        div[data-baseweb="select"] > div:focus-within {
            border-color: var(--brand) !important;
            box-shadow: 0 0 0 3px var(--brand-ring) !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder { color: var(--text-faint) !important; }

        label, .stRadio label, .stSelectbox label, .stTextInput label, .stFileUploader label {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: var(--text-muted) !important;
        }

        /* Radio group styled as a soft segmented control */
        .stRadio [role="radiogroup"] { gap: 0.3rem; }
        .stRadio [role="radiogroup"] label {
            padding: 0.45rem 0.6rem;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            background: var(--surface);
            transition: all 0.16s var(--ease);
        }
        .stRadio [role="radiogroup"] label:hover {
            border-color: var(--brand-ring);
            background: var(--brand-soft);
        }

        /* File uploader dropzone */
        section[data-testid="stFileUploaderDropzone"] {
            background: var(--surface-2);
            border: 1.5px dashed var(--border);
            border-radius: var(--radius);
            transition: border-color 0.16s var(--ease), background 0.16s var(--ease);
        }
        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--brand);
            background: var(--brand-soft);
        }

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {
            border-radius: var(--radius-sm);
            font-weight: 600;
            font-size: 0.89rem;
            padding: 0.55rem 1.05rem;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text);
            box-shadow: var(--shadow-xs);
            transition: all 0.16s var(--ease);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--brand);
            color: var(--brand);
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }
        .stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
            box-shadow: 0 0 0 3px var(--brand-ring);
            outline: none;
        }
        .stButton > button[kind="primary"] {
            background: var(--brand);
            border-color: var(--brand);
            color: #ffffff;
            box-shadow: 0 2px 8px var(--brand-ring);
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--brand-hover);
            border-color: var(--brand-hover);
            color: #ffffff;
            box-shadow: 0 6px 16px var(--brand-ring);
        }

        /* ---------- Alerts ---------- */
        div[data-testid="stAlert"] {
            border-radius: var(--radius);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-xs);
            padding: 0.85rem 1rem;
        }

        /* ---------- Progress ---------- */
        .stProgress > div > div > div { background: var(--border-soft); border-radius: 999px; height: 7px; }
        .stProgress > div > div > div > div { background: var(--brand); border-radius: 999px; }

        /* ---------- Expander ---------- */
        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--surface);
            box-shadow: var(--shadow-xs);
            overflow: hidden;
        }

        /* ---------- Monospace surfaces ---------- */
        code, pre, .stCodeBlock { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
        .stTextArea textarea {
            font-family: 'JetBrains Mono', ui-monospace, monospace !important;
            font-size: 0.82rem !important;
            line-height: 1.7 !important;
        }

        /* ---------- Chat bubbles ---------- */
        .chat-row { display: flex; margin: 0.5rem 0; }
        .chat-row.user { justify-content: flex-end; }
        .chat-bubble-user, .chat-bubble-bot {
            max-width: 82%;
            padding: 0.7rem 1rem;
            font-size: 0.9rem;
            line-height: 1.6;
            box-shadow: var(--shadow-xs);
            animation: bubbleIn 0.22s var(--ease);
        }
        .chat-bubble-user {
            background: var(--brand);
            color: #ffffff;
            border-radius: var(--radius) var(--radius) 4px var(--radius);
        }
        .chat-bubble-bot {
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: var(--radius) var(--radius) var(--radius) 4px;
        }
        .chat-speaker {
            display: block;
            font-size: 0.7rem; font-weight: 650;
            text-transform: uppercase; letter-spacing: 0.06em;
            opacity: 0.75;
            margin-bottom: 0.25rem;
        }
        @keyframes bubbleIn {
            from { opacity: 0; transform: translateY(4px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] > div {
            border-radius: var(--radius) !important;
            border-color: var(--border) !important;
        }

        /* ---------- Responsive ---------- */
        @media (max-width: 768px) {
            .main .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.4rem; }
            .hero { flex-direction: column; align-items: flex-start; gap: 0.75rem; padding: 1.1rem; }
            .hero-title { font-size: 1.3rem; }
            .stat-value { font-size: 1.5rem; }
            .stTabs [data-baseweb="tab"] { padding: 0.45rem 0.7rem; font-size: 0.8rem; }
            .chat-bubble-user, .chat-bubble-bot { max-width: 95%; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
defaults = {
    "result": None,
    "chat_history": [],
    "processing": False,
    "error": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Pipeline runner (mirrors run_pipeline from the CLI script)
# --------------------------------------------------------------------------
def run_pipeline(source: str, language: str, progress_cb=None) -> dict:
    def step(msg, pct):
        LOGGER.info(msg)
        if progress_cb:
            progress_cb(msg, pct)

    step("Download/conversion and chunking started", 10)
    chunks = process_input(source)
    step(f"Chunking completed: {len(chunks)} chunk(s)", 25)

    step("Transcribing audio...", 30)
    transcript = transcribe_all(chunks, language=language, progress_cb=progress_cb)
    LOGGER.info("Transcript length: %s", len(transcript))

    step("Generating title...", 50)
    title = generate_title(transcript)

    step("Summarizing meeting...", 62)
    summary = summarize(transcript)

    step("Extracting action items...", 74)
    action_items = extract_action_items(transcript)

    step("Extracting key decisions...", 84)
    decisions = extract_key_decisions(transcript)

    step("Extracting open questions...", 92)
    questions = extract_questions(transcript)

    step("Building chat index (RAG)...", 98)
    rag_chain = build_rag_chain(transcript)

    step("Done!", 100)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# --------------------------------------------------------------------------
# Presentational helpers (display only — no logic)
# --------------------------------------------------------------------------
def stat_card(icon: str, label: str, value) -> str:
    return (
        f'<div class="stat-card">'
        f'  <div class="stat-head">{icon} {label}</div>'
        f'  <div class="stat-value">{value}</div>'
        f'</div>'
    )


def count_or_dash(value):
    return len(value) if isinstance(value, (list, tuple)) else "—"


# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-brand">🎥 Meeting Assistant</div>', unsafe_allow_html=True)
    st.caption("Feed it a YouTube link or upload a local recording.")

    st.markdown('<div class="side-label">Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Source type",
        ["YouTube URL", "Upload file"],
        horizontal=False,
        label_visibility="collapsed",
    )

    source_value = None
    uploaded_file = None

    if input_mode == "YouTube URL":
        source_value = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video file",
            type=["mp4", "mov", "mkv", "mp3", "wav", "m4a", "webm"],
        )

    st.markdown('<div class="side-label">Options</div>', unsafe_allow_html=True)
    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.markdown("---")
    run_clicked = st.button("🚀 Run Assistant", use_container_width=True, type="primary")

    if st.session_state.result:
        st.markdown("---")
        if st.button("🗑️ Clear session", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.error = None
            st.rerun()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-mark">🎥</div>
        <div>
            <div class="hero-title">AI Video Meeting Assistant</div>
            <p class="hero-subtitle">Transcribe, summarize, and chat with any meeting recording.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Handle "Run" click
# --------------------------------------------------------------------------
if run_clicked:
    resolved_source = None
    tmp_path = None

    if input_mode == "YouTube URL":
        if not source_value or not source_value.strip():
            st.error("Please paste a YouTube URL first.")
        else:
            resolved_source = source_value.strip()
    else:
        if uploaded_file is None:
            st.error("Please upload a file first.")
        else:
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp_dir = tempfile.mkdtemp(prefix="meeting_assistant_")
            tmp_path = os.path.join(tmp_dir, f"upload{suffix}")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            resolved_source = tmp_path

    if resolved_source:
        st.session_state.error = None
        st.session_state.chat_history = []
        progress_bar = st.progress(0, text="Starting up...")
        st.session_state.processing = True
        try:
            def progress_cb(msg, pct):
                progress_bar.progress(pct, text=msg)

            with st.spinner("Working through the meeting — this can take a few minutes for long recordings..."):
                st.session_state.result = run_pipeline(resolved_source, language, progress_cb)
            progress_bar.empty()
            st.success("Meeting processed successfully!")
        except Exception as e:
            LOGGER.exception("Pipeline failed")
            st.session_state.result = None
            progress_bar.empty()
            st.session_state.error = f"{e}\n\n{traceback.format_exc()}"
        finally:
            st.session_state.processing = False
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


# --------------------------------------------------------------------------
# Error display
# --------------------------------------------------------------------------
if st.session_state.error:
    with st.expander("⚠️ Something went wrong — details", expanded=True):
        st.code(st.session_state.error)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.result

if result is None and not st.session_state.error:
    st.info("👈 Add a YouTube URL or upload a file in the sidebar, then click **Run Assistant** to get started.")

if result:
    st.markdown(
        f'<div class="doc-title"><span class="pill">Meeting</span> 📌 {result["title"]}</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown(stat_card("✅", "Action items", count_or_dash(result["action_items"])), unsafe_allow_html=True)
    with col2:
        st.markdown(stat_card("🔑", "Key decisions", count_or_dash(result["key_decisions"])), unsafe_allow_html=True)
    with col3:
        st.markdown(stat_card("❓", "Open questions", count_or_dash(result["open_questions"])), unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Decisions", "❓ Questions", "📝 Transcript", "💬 Chat"]
    )

    def render_list_or_text(content):
        """Same branching as before: list/tuple -> bullets, anything else -> st.write."""
        if isinstance(content, (list, tuple)):
            if not content:
                st.markdown(
                    '<div class="section-card"><div class="empty-state">'
                    '<span class="glyph">🗒️</span>Nothing found.</div></div>',
                    unsafe_allow_html=True,
                )
                return
            items = "".join(f"<li>{item}</li>" for item in content)
            st.markdown(f'<div class="section-card"><ul>{items}</ul></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.write(content)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_summary:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write(result["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_actions:
        render_list_or_text(result["action_items"])

    with tab_decisions:
        render_list_or_text(result["key_decisions"])

    with tab_questions:
        render_list_or_text(result["open_questions"])

    with tab_transcript:
        st.text_area("Full transcript", result["transcript"], height=400)
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["transcript"],
            file_name=f"{result['title'].replace(' ', '_')}_transcript.txt",
            mime="text/plain",
        )

    with tab_chat:
        st.caption("Ask questions about this meeting — answered using retrieval over the transcript.")

        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    '<div class="section-card"><div class="empty-state">'
                    '<span class="glyph">💬</span>No messages yet — ask anything about this meeting below.'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            for role, msg in st.session_state.chat_history:
                is_user = role == "user"
                bubble_class = "chat-bubble-user" if is_user else "chat-bubble-bot"
                row_class = "chat-row user" if is_user else "chat-row"
                speaker = "🧑 You" if is_user else "🤖 Assistant"
                st.markdown(
                    f'<div class="{row_class}"><div class="{bubble_class}">'
                    f'<span class="chat-speaker">{speaker}</span>{msg}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        question = st.chat_input("Ask something about the meeting...")
        if question:
            st.session_state.chat_history.append(("user", question))
            try:
                with st.spinner("Thinking..."):
                    answer = ask_question(result["rag_chain"], question)
            except Exception as e:
                answer = f"⚠️ Error answering question: {e}"
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()
