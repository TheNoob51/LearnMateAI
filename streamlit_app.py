import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from inspiration.create_database import generate_data_store, CHROMA_PATH, DATA_PATH
import shutil

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LearnMate AI - RAG Tutor",
    page_icon="📚",
    layout="wide",
)

# Constants
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_db():
    """Lazily create/load the Chroma DB and cache in session state."""
    if "db" not in st.session_state:
        if not os.path.exists(CHROMA_PATH):
            st.error(f"Database not found at {CHROMA_PATH}. Please rebuild the database first.")
            return None
        
        embedding_function = OpenAIEmbeddings()
        st.session_state.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    return st.session_state.db

def rebuild_database():
    """Rebuild the database and clear cached DB."""
    with st.spinner("Rebuilding database..."):
        generate_data_store()
        # Clear cached DB so it reloads
        if "db" in st.session_state:
            del st.session_state.db
        st.success("Database rebuilt successfully!")

def save_uploaded_files(uploaded_files):
    """Save uploaded files to the data directory."""
    if not uploaded_files:
        return False
    
    # Ensure data directory exists
    os.makedirs(DATA_PATH, exist_ok=True)
    
    saved_count = 0
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.md'):
            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1
    
    return saved_count > 0

def run_rag_query(query_text: str):
    """Run RAG query and return answer with sources."""
    db = get_db()
    if db is None:
        return "Database not available. Please rebuild the database first.", []
    
    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 or results[0][1] < 0.7:
        return "I couldn't find relevant information in the documents to answer your question. Please try rephrasing or ask about topics covered in the uploaded documents.", []
    
    # Build context from retrieved chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Groq API key not found. Please set GROQ_API_KEY in your .env file.", []
    
    # Use Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0.2,
        api_key=groq_api_key
    )
    
    try:
        response_text = llm.predict(prompt)
    except Exception as e:
        return f"Error generating response: {str(e)}", []
    
    # Collect sources
    sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
    unique_sources = list(set(sources))  # Remove duplicates
    
    return response_text, unique_sources

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
st.title("📚 LearnMate AI - RAG Tutor")
st.markdown("*A RAG-powered tutor over your documents, powered by Groq*")

# Sidebar
with st.sidebar:
    st.header("📂 Database Management")
    
    # Current paths info
    st.caption(f"**Data folder:** `{DATA_PATH}`")
    st.caption(f"**Chroma path:** `{CHROMA_PATH}`")
    
    st.markdown("---")
    
    # Rebuild database button
    if st.button("🔄 Rebuild Database", help="Rebuild from existing files in data/books/"):
        rebuild_database()
    
    st.markdown("---")
    
    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose .md files",
        type=['md'],
        accept_multiple_files=True,
        help="Upload markdown files to add to your knowledge base"
    )
    
    if st.button("💾 Save Files & Rebuild DB"):
        if uploaded_files:
            if save_uploaded_files(uploaded_files):
                st.success(f"Saved {len(uploaded_files)} files!")
                rebuild_database()
            else:
                st.error("Failed to save files.")
        else:
            st.warning("Please select files to upload first.")

# Main chat area
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"• `{source}`")

# Chat input
if prompt := st.chat_input("Ask something about the documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = run_rag_query(prompt)
        
        st.markdown(response)
        
        # Show sources
        if sources:
            with st.expander("📄 Sources"):
                for source in sources:
                    st.markdown(f"• `{source}`")
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })