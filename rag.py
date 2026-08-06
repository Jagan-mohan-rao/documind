import os
import uuid
from pathlib import Path
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


def get_embeddings():
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


PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a document.
Use ONLY the context below to answer. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def _call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def answer_question(vectorstore, question: str, k: int = 4):
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = _call_llm(prompt)
    return answer, retrieved_docs