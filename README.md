# 📄 DocuMind — AI-Native Document Q&A Agent (RAG)

DocuMind is a Retrieval-Augmented Generation (RAG) chat agent that lets you upload a document and ask questions about it in natural language. Answers are grounded strictly in the uploaded document's content — no hallucinated facts, no outside knowledge — with the exact source chunks shown alongside every answer.

Built as a hands-on exploration of the modern AI-native development stack: LLM APIs, embeddings, vector search, and agent-style prompting, wired together into a working product rather than a notebook demo.

---

## ✨ Features

- **Upload & chat** — PDF, DOCX, or TXT files, answered conversationally
- **Grounded answers** — retrieval-augmented generation means the model answers from the document, not its training data, and says "I don't know" when the document doesn't cover something
- **Source transparency** — every answer shows exactly which chunks of the document it was generated from
- **Follow-up aware** — recent conversation history is fed back into the prompt so questions like "what about the second one?" resolve correctly
- **Streamed responses** — answers appear token-by-token instead of after a long wait
- **Fresh context per document** — each upload gets its own isolated, in-memory vector store, so answers never mix content from a previous document
- **Two-panel UI** — document upload on the left, chat on the right, in a custom glassmorphism interface

---

## 🏗️ Architecture

```
 Upload (PDF/DOCX/TXT)
        │
        ▼
  Load & Chunk            (LangChain document loaders + RecursiveCharacterTextSplitter)
        │
        ▼
  Embed chunks             (HuggingFace sentence-transformer, local — no API cost)
        │
        ▼
  Store in ChromaDB        (in-memory, per-document vector store)
        │
        ▼
  User asks a question
        │
        ▼
  Similarity search        (retrieve top-k most relevant chunks)
        │
        ▼
  Prompt + context + recent history → Groq LLM API (Llama 3.3)
        │
        ▼
  Streamed, grounded answer + cited source chunks
```

**Why this design:**
- **Local embeddings, cloud generation** — indexing documents costs nothing (runs on your machine), while generation uses a fast, free-tier cloud API (Groq) so the app works the same locally and when deployed publicly.
- **Fresh vector store per upload** — avoids the common RAG bug where old documents silently pollute new answers.
- **Streaming output** — perceived speed matters as much as actual speed for a chat UI.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (custom glassmorphism CSS) |
| Orchestration | LangChain |
| Embeddings | HuggingFace Sentence Transformers (local) |
| Vector store | ChromaDB (in-memory) |
| LLM | Groq API (Llama 3.3 70B) — OpenAI-compatible endpoint |
| Document parsing | PyPDF, Docx2txt |
| Language | Python 3.12 |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12
- A free [Groq API key](https://console.groq.com) (no credit card required)

### Setup

```bash
git clone https://github.com/<your-username>/documind.git
cd documind

python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your-key-here
```

### Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), upload a document, and start asking questions.

---

## 📁 Project Structure

```
documind/
├── app.py              # Streamlit UI
├── rag.py              # Core RAG pipeline (load, chunk, embed, retrieve, generate)
├── requirements.txt
├── .env                # API key (not committed)
└── .gitignore
```

---

## 🔍 Key Design Decisions

- **Chunk size 800 / overlap 120** — balances retrieval precision against having enough context per chunk; overlap prevents meaning from being lost at chunk boundaries.
- **k=6 retrieved chunks** — enough context for most questions without overloading the prompt.
- **Temperature 0.2** — keeps answers literal and grounded rather than creative, appropriate for a fact-retrieval tool.
- **No persisted vector store** — each session/document is isolated by design; this is a document Q&A tool, not a permanent knowledge base, so persistence would only introduce bugs (stale/mixed context) without adding value.

---

## 🗺️ Possible Extensions

- Swap Groq for a fully local LLM (Ollama) for an offline-only mode
- Multi-document support with per-document filtering
- Persistent chat history across sessions
- Support for URLs / web pages as a document source

---

## 🎯 Challenges & What I Learned

Building this surfaced a few real engineering problems that don't show up in tutorials:

- **Stale context bug** — early on, answers were mixing content from previously tested documents. The cause: the vector store was persisted to disk with a fixed collection name, so every new upload was *added* to old data instead of replacing it. Fixed by making each document get its own fresh, in-memory, uniquely-named vector store — a good example of why "it works on the first try" doesn't mean the design is correct.
- **Perceived vs. actual latency** — the biggest UX complaint during development wasn't that answers were slow, it was that they *felt* slow because nothing appeared until the full response was ready. Switching to token-by-token streaming (`st.write_stream`) fixed the complaint without changing the actual generation time at all — a reminder that perceived performance and real performance aren't the same problem.
- **Cost/privacy vs. speed tradeoff** — initially ran the LLM fully locally via Ollama for zero cost and full privacy, but inference speed on consumer hardware was a bottleneck. Moved generation to Groq's free tier (still $0, still no card required) while keeping embeddings local — a deliberate hybrid: free where speed doesn't matter (indexing), fast where it does (answering).
- **Streamlit's layout model** — components like `st.chat_input` only behave a certain way (auto-docking to the bottom of the page) when placed at the top level of the script, not nested inside columns or containers. Getting a clean two-panel layout meant understanding Streamlit's rendering model, not just stacking widgets.

---

## 📄 License

MIT — feel free to fork and adapt.
