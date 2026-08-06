import os
import uuid
from pathlib import Path
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_split(file_path: str):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(docs)


@lru_cache(maxsize=1)
def get_embeddings():
    # This model load (import torch + download/load weights) is the single
    # biggest source of "processing" lag. Previously it ran on every single
    # "Process document" click; caching means only the FIRST document in a
    # session pays that cost, every one after is instant.
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(chunks):
    """Build a fresh, in-memory vector store scoped to THIS document only.

    No persist_directory (so nothing is written to disk / reused later) and a
    random collection_name every call (so this document's chunks can never mix
    with a previous upload's chunks, even within the same run).
    """
    embeddings = get_embeddings()
    collection_name = f"documind-{uuid.uuid4().hex[:8]}"
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )


PROMPT_TEMPLATE = """You are a precise assistant answering questions about a document.
Answer using ONLY the context below — never use outside knowledge and never guess.
Be specific: cite exact figures, names, dates, or clauses from the context where relevant.
If the answer isn't in the context, say plainly that the document doesn't cover it.
Use the conversation so far only to resolve follow-up questions (e.g. "the second one",
"what about after that") — the answer itself must still come only from the context.

Conversation so far:
{history}

Context:
{context}

Question: {question}

Answer:"""


def _format_history(chat_history, max_turns=3):
    """Keep the last few Q/A turns so follow-ups have context, without
    bloating every prompt with the full conversation."""
    if not chat_history:
        return "(none yet)"
    recent = chat_history[-(max_turns * 2):]
    lines = [f"{'User' if role == 'user' else 'Assistant'}: {msg}" for role, msg in recent]
    return "\n".join(lines)


def _stream_llm(prompt: str):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        temperature=0.2,  # lower temperature = more literal, less "creative" drift from the source text
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def answer_question_stream(vectorstore, question: str, chat_history=None, k: int = 4):
    """Returns (token_generator, source_docs). The generator is meant for
    st.write_stream — the UI shows the answer as it's generated instead of
    waiting for the whole thing, which is what actually feels slow today."""
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    history = _format_history(chat_history or [])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question, history=history)
    return _stream_llm(prompt), retrieved_docs