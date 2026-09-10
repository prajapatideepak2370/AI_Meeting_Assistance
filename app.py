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

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# --------------------------------------------------------------------------
# Page config & light styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Meeting Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; max-width: 1100px; }
        .hero-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0.1rem; }
        .hero-subtitle { color: #8a8f98; margin-bottom: 1.4rem; }
        .stat-card {
            background: rgba(127,127,127,0.08);
            border: 1px solid rgba(127,127,127,0.18);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
        }
        .section-card {
            background: rgba(127,127,127,0.05);
            border: 1px solid rgba(127,127,127,0.15);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }
        .chat-bubble-user {
            background: #2563eb22;
            border-radius: 12px;
            padding: 0.6rem 0.9rem;
            margin: 0.35rem 0;
        }
        .chat-bubble-bot {
            background: rgba(127,127,127,0.10);
            border-radius: 12px;
            padding: 0.6rem 0.9rem;
            margin: 0.35rem 0;
        }
        div[data-testid="stStatusWidget"] { display: none; }
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
        if progress_cb:
            progress_cb(msg, pct)

    step("Loading & chunking audio...", 10)
    chunks = process_input(source)

    step("Transcribing audio...", 30)
    transcript = transcribe_all(chunks, language=language)

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
# Sidebar — inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎥 New Meeting")
    st.caption("Feed it a YouTube link or upload a local recording.")

    input_mode = st.radio("Source type", ["YouTube URL", "Upload file"], horizontal=False)

    source_value = None
    uploaded_file = None

    if input_mode == "YouTube URL":
        source_value = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video file",
            type=["mp4", "mov", "mkv", "mp3", "wav", "m4a", "webm"],
        )

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
st.markdown('<div class="hero-title">🎥 AI Video Meeting Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Transcribe, summarize, and chat with any meeting recording.</div>',
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
        try:
            def progress_cb(msg, pct):
                progress_bar.progress(pct, text=msg)

            with st.spinner("Working through the meeting — this can take a few minutes for long recordings..."):
                st.session_state.result = run_pipeline(resolved_source, language, progress_cb)
            progress_bar.empty()
            st.success("Meeting processed successfully!")
        except Exception as e:
            progress_bar.empty()
            st.session_state.error = f"{e}\n\n{traceback.format_exc()}"
        finally:
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
    st.markdown(f"## 📌 {result['title']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="stat-card">✅ <b>{len(result["action_items"]) if isinstance(result["action_items"], (list, tuple)) else "—"}</b><br>Action items</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card">🔑 <b>{len(result["key_decisions"]) if isinstance(result["key_decisions"], (list, tuple)) else "—"}</b><br>Key decisions</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card">❓ <b>{len(result["open_questions"]) if isinstance(result["open_questions"], (list, tuple)) else "—"}</b><br>Open questions</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Decisions", "❓ Questions", "📝 Transcript", "💬 Chat"]
    )

    def render_list_or_text(content):
        if isinstance(content, (list, tuple)):
            if not content:
                st.caption("Nothing found.")
            for item in content:
                st.markdown(f"- {item}")
        else:
            st.write(content)

    with tab_summary:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write(result["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_actions:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_list_or_text(result["action_items"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_decisions:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_list_or_text(result["key_decisions"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_questions:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_list_or_text(result["open_questions"])
        st.markdown("</div>", unsafe_allow_html=True)

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
            for role, msg in st.session_state.chat_history:
                bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
                speaker = "🧑 You" if role == "user" else "🤖 Assistant"
                st.markdown(
                    f'<div class="{bubble_class}"><b>{speaker}:</b> {msg}</div>',
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