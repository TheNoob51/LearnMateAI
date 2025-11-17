import os
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


# ==========================
# Config & constants
# ==========================

load_dotenv()  # load GROQ_API_KEY from .env

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

PROMPT_TEMPLATE = """
You are LearnMate, an AI tutor. Use ONLY the context below to answer the question.

Context:
{context}

---

Question: {question}

Answer in a clear, helpful way. If the context does not contain the answer, say you don't know.
"""


# ==========================
# Logging helpers (show what's happening)
# ==========================

if "logs" not in st.session_state:
    st.session_state.logs = []  # list of strings


def log(msg: str):
    """Append a log line and also print to console."""
    st.session_state.logs.append(msg)
    print(msg)


# ==========================
# Document ingestion
# ==========================

def load_documents(data_path: str = DATA_PATH):
    """
    Load .md, .txt, .pdf, .docx files from DATA_PATH.
    No unstructured, no nltk.
    """
    docs = []
    base = Path(data_path)
    base.mkdir(parents=True, exist_ok=True)

    log(f"[INGEST] Looking for documents in: {base.resolve()}")

    # Markdown + TXT (.md, .txt)
    log("[INGEST] Loading .md and .txt files...")
    md_txt_loader = DirectoryLoader(
        data_path,
        glob="**/*.[mt][dx]t",  # matches .md and .txt
        loader_cls=TextLoader,
        show_progress=True,
    )
    md_txt_docs = md_txt_loader.load()
    log(f"[INGEST] Loaded {len(md_txt_docs)} .md/.txt documents.")
    docs.extend(md_txt_docs)

    # PDFs
    pdf_files = list(base.rglob("*.pdf"))
    log(f"[INGEST] Found {len(pdf_files)} .pdf files.")
    for pdf in pdf_files:
        log(f"[INGEST] Loading PDF: {pdf.name}")
        pdf_loader = PyPDFLoader(str(pdf))
        docs.extend(pdf_loader.load())

    # DOCX
    docx_files = list(base.rglob("*.docx"))
    log(f"[INGEST] Found {len(docx_files)} .docx files.")
    for doc in docx_files:
        log(f"[INGEST] Loading DOCX: {doc.name}")
        doc_loader = Docx2txtLoader(str(doc))
        docs.extend(doc_loader.load())

    log(f"[INGEST] Total loaded documents: {len(docs)}")
    return docs


def split_documents(docs, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Split Documents into smaller chunks for better retrieval.
    """
    log(f"[SPLIT] Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    log(f"[SPLIT] Created {len(chunks)} chunks.")
    return chunks


def generate_data_store() -> str:
    """
    Build the Chroma vector store using local FastEmbed embeddings.
    """
    log("[INDEX] Starting index build...")
    docs = load_documents(DATA_PATH)
    if not docs:
        msg = (
            "No documents found. Put .md/.txt/.pdf/.docx files into "
            f"{DATA_PATH}/ and try again."
        )
        log(f"[INDEX] {msg}")
        return msg

    chunks = split_documents(docs)

    log("[INDEX] Initializing FastEmbed embeddings (first time may download the model)...")
    embedding = FastEmbedEmbeddings()

    log(f"[INDEX] Building Chroma index at '{CHROMA_PATH}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_PATH,
    )

    # Chroma 0.4+ auto-persists; this is harmless but may warn.
    try:
        vectorstore.persist()
    except Exception as e:
        log(f"[WARN] persist() warning/exception: {e}")

    msg = (
        f"Indexed {len(docs)} document(s) into {len(chunks)} chunks. "
        f"Chroma persisted at '{CHROMA_PATH}'."
    )
    log(f"[INDEX] {msg}")
    return msg


# ==========================
# RAG + LLM
# ==========================

def get_db() -> Optional[Chroma]:
    """
    Lazy-load and cache Chroma DB in session_state.
    """
    if "db" not in st.session_state:
        if not Path(CHROMA_PATH).exists():
            log(f"[DB] No Chroma index found at '{CHROMA_PATH}'.")
            return None

        log(f"[DB] Loading Chroma index from '{CHROMA_PATH}'...")
        embedding = FastEmbedEmbeddings()
        st.session_state.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding,
        )
        log("[DB] Chroma index loaded and cached.")
    return st.session_state.db


def get_llm() -> Optional[ChatGroq]:
    """
    Groq-backed chat model. Requires GROQ_API_KEY.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        log("[LLM] GROQ_API_KEY not set.")
        return None

    log("[LLM] Initializing ChatGroq (llama-3.1-70b-versatile)...")
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0.2,
        api_key=groq_key,
    )
    return llm


def run_rag_query(query_text: str, k: int = 3) -> Tuple[str, List[str]]:
    """
    Use the Chroma index + Groq LLM to answer a query.
    Returns (answer_text, list_of_sources).
    """
    log(f"[RAG] New query: {query_text!r}")

    db = get_db()
    if db is None:
        return (
            "Database not available. Please build/rebuild it from the sidebar.",
            [],
        )

    log(f"[RAG] Running similarity search (k={k})...")
    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    log(f"[RAG] Retrieved {len(results)} results.")

    if not results:
        return "No results found in the vector store.", []

    top_score = results[0][1]
    log(f"[RAG] Top relevance score: {top_score:.3f}")

    # Build context from all retrieved chunks (even if score is low)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    log("[RAG] Built context from retrieved chunks.")

    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    llm = get_llm()
    if llm is None:
        return "Groq LLM not available (check GROQ_API_KEY in .env).", []

    log("[RAG] Calling Groq LLM...")
    try:
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        log("[RAG] Received response from Groq.")
    except Exception as e:
        err = f"Error calling Groq: {e}"
        log(f"[RAG] {err}")
        return (err, [])

    # Collect sources
    sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
    unique_sources = sorted(set(sources))
    log(f"[RAG] Sources used: {unique_sources}")

    return response_text, unique_sources


# ==========================
# File handling
# ==========================

def save_uploaded_files(files) -> int:
    """
    Save uploaded .md/.txt/.pdf/.docx files into DATA_PATH.
    """
    if not files:
        log("[UPLOAD] No files to save.")
        return 0

    base = Path(DATA_PATH)
    base.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        name = f.name
        lower = name.lower()
        if lower.endswith((".md", ".txt", ".pdf", ".docx")):
            dest = base / name
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            saved += 1
            log(f"[UPLOAD] Saved file: {dest}")
        else:
            log(f"[UPLOAD] Skipped unsupported file type: {name}")
    return saved


# ==========================
# Streamlit UI
# ==========================

st.set_page_config(
    page_title="LearnMate AI - RAG Tutor (Groq + Local Embeddings)",
    page_icon="📚",
    layout="wide",
)

st.title("📚 LearnMate AI — RAG Tutor")
st.markdown(
    "Ask questions about your documents.\n\n"
    "- **Embeddings:** local & free via FastEmbed\n"
    "- **LLM:** Groq `llama-3.1-70b-versatile`\n"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----- Sidebar -----

with st.sidebar:
    st.header("📂 Knowledge Base")

    st.caption(f"**Data folder:** `{DATA_PATH}`")
    st.caption(f"**Chroma path:** `{CHROMA_PATH}`")

    st.markdown("---")

    st.subheader("1️⃣ Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload `.md`, `.txt`, `.pdf`, `.docx`",
        type=["md", "txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

    if st.button("💾 Save Files"):
        if uploaded_files:
            with st.spinner("Saving files..."):
                saved_count = save_uploaded_files(uploaded_files)
            if saved_count > 0:
                st.success(f"Saved {saved_count} file(s) to `{DATA_PATH}`.")
            else:
                st.warning("No valid files were saved.")
        else:
            st.warning("Please select files before clicking save.")

    st.markdown("---")

    st.subheader("2️⃣ Build / Rebuild Index")
    if st.button("🔄 Build / Rebuild Database"):
        with st.spinner("Building index... (this may take a moment)"):
            msg = generate_data_store()
        st.info(msg)
        # Clear cached DB so it reloads next time
        if "db" in st.session_state:
            del st.session_state["db"]

    st.markdown("---")
    st.subheader("🧾 Logs (what's happening)")
    with st.expander("Show logs", expanded=False):
        if st.session_state.logs:
            for line in st.session_state.logs[-200:]:
                st.text(line)
        else:
            st.caption("No logs yet. Actions (upload, build, ask) will show here.")


# ----- Chat area -----

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

user_input = st.chat_input("Ask something about the documents...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking with Groq + RAG..."):
            answer, sources = run_rag_query(user_input)
            st.markdown(answer)
            if sources:
                with st.expander("📄 Sources"):
                    for src in sources:
                        st.markdown(f"- `{src}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
