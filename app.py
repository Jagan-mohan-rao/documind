import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from rag import load_and_split, build_vectorstore, answer_question_stream

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# GLASSMORPHISM THEME
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-void: #0B0F19;
    --glow-indigo: #6C63FF;
    --glow-magenta: #C147E9;
    --glow-cyan: #22D3EE;
    --glass-fill: rgba(255,255,255,0.06);
    --glass-border: rgba(255,255,255,0.14);
    --text-primary: #F1F0FF;
    --text-muted: #9CA3C4;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: var(--bg-void); overflow: hidden; }
.stApp::before, .stApp::after {
    content: ""; position: fixed; width: 50vw; height: 50vw; border-radius: 50%;
    filter: blur(110px); z-index: 0; opacity: 0.5; pointer-events: none;
}
.stApp::before {
    background: radial-gradient(circle, var(--glow-indigo), transparent 70%);
    top: -18%; left: -10%; animation: driftA 22s ease-in-out infinite alternate;
}
.stApp::after {
    background: radial-gradient(circle, var(--glow-magenta), transparent 70%);
    bottom: -20%; right: -8%; animation: driftB 26s ease-in-out infinite alternate;
}
@keyframes driftA { from { transform: translate(0,0); } to { transform: translate(5vw, 3vw); } }
@keyframes driftB { from { transform: translate(0,0); } to { transform: translate(-4vw, -4vw); } }
@media (prefers-reduced-motion: reduce) { .stApp::before, .stApp::after { animation: none; } }

.block-container { position: relative; z-index: 1; padding-top: 2rem; max-width: 1400px; }

.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;margin-top:20px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--glow-cyan);
    background: rgba(34,211,238,0.08); border: 1px solid rgba(34,211,238,0.25);
    border-radius: 999px; padding: 5px 14px; margin-bottom: 14px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.4rem;
    line-height: 1.1; margin: 0 0 6px 0;
    background: linear-gradient(120deg, #FFFFFF 0%, var(--glow-indigo) 55%, var(--glow-cyan) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 14px; }
.scan-line {
    height: 2px; width: 100%; border-radius: 2px; margin-bottom: 28px;
    background: linear-gradient(90deg, transparent, var(--glow-cyan), var(--glow-indigo), var(--glow-magenta), transparent);
    background-size: 200% 100%; animation: scan 3.5s linear infinite;
}
@keyframes scan { from { background-position: 200% 0; } to { background-position: -200% 0; } }
@media (prefers-reduced-motion: reduce) { .scan-line { animation: none; } }

.panel-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--glow-cyan); margin-bottom: 10px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass-fill) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(18px) saturate(160%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    padding: 6px;
    height: 100%;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed var(--glass-border) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text-muted) !important; }

.stButton > button {
    background: linear-gradient(120deg, var(--glow-indigo), var(--glow-magenta)) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease; width: 100%;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 26px rgba(193,71,233,0.45); }

[data-testid="stChatMessage"] {
    background: var(--glass-fill) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(14px);
    margin-bottom: 10px;
}
[data-testid="stChatMessage"] p { color: var(--text-primary) !important; }

[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;padding:20px;
    color: var(--text-primary) !important;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: var(--glow-cyan) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
[data-testid="stExpander"] p { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; color: var(--text-muted) !important; }

[data-testid="stAlert"] {
    background: var(--glass-fill) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px);
}
[data-testid="stAlert"] p { color: var(--text-primary) !important; }

h1, h2, h3, label, .stMarkdown p { color: var(--text-primary); }

@media (max-width: 900px) {
    .hero-title { font-size: 1.8rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-eyebrow">◆ RAG · Retrieval-Augmented Generation</div>
<h1 class="hero-title">DocuMind</h1>
<p class="hero-sub">Upload any document and Ask anything. Answers grounded in what it actually says.</p>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


def render_message(role: str, text: str):
    """User bubble on the right, assistant bubble on the left — WhatsApp/ChatGPT style."""
    align = "flex-end" if role == "user" else "flex-start"
    bubble_bg = "rgba(108,99,255,0.28)" if role == "user" else "rgba(255,255,255,0.07)"
    st.markdown(f"""
    <div style="display:flex; justify-content:{align}; margin-bottom:10px;">
        <div style="max-width:78%; background:{bubble_bg}; border:1px solid var(--glass-border);
                    border-radius:14px; padding:10px 14px; color:var(--text-primary);
                    font-size:0.95rem; line-height:1.4;">
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TWO-COLUMN LAYOUT — LEFT: upload · RIGHT: chat
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.6], gap="medium")

with left:
    st.markdown('<div class="panel-label">01 — Document</div>', unsafe_allow_html=True)
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
        if uploaded_file and st.button("Process document"):
            with st.spinner("Chunking and embedding document..."):
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                chunks = load_and_split(tmp_path)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.chat_history = []
                st.session_state.doc_name = uploaded_file.name
            st.success(f"'{uploaded_file.name}' → {len(chunks)} chunks. Ready in the chat.")

        if st.session_state.doc_name:
            st.caption(f"Active document: **{st.session_state.doc_name}**")

with right:
    st.markdown('<div class="panel-label">02 — Chat</div>', unsafe_allow_html=True)

    if st.session_state.vectorstore:
        messages_box = st.container(height=480, border=True)
        with messages_box:
            for role, message in st.session_state.chat_history:
                render_message(role, message)
    else:
        with st.container(border=True):
            st.info("Upload and process a document on the left to start chatting.")

# ---------------------------------------------------------------------------
# CHAT INPUT — placed OUTSIDE both columns so Streamlit docks it to the
# bottom of the whole page, instead of rendering it inline inside a panel.
# ---------------------------------------------------------------------------
if st.session_state.vectorstore:
    question = st.chat_input("Ask a question about the document...")

    if question:
        st.session_state.chat_history.append(("user", question))
        with messages_box:
            render_message("user", question)
            with st.spinner("Searching the document..."):
                stream, sources = answer_question_stream(
                    st.session_state.vectorstore,
                    question,
                    chat_history=st.session_state.chat_history,
                )
            answer = st.write_stream(stream)
            with st.expander("Sources used"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"[{i}] {doc.page_content[:300]}...")
        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()