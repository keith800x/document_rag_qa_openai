"""Streamlit front end for the Document RAG Q&A Assistant."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from database.db_manager import (
    create_session,
    get_document_for_session,
    get_messages_for_session,
    get_recent_sessions,
    init_db,
    save_message,
    save_session_document,
)

from rag_utils import (
    answer_question,
    build_vector_store,
    load_document_from_path,
    split_documents,
)



APP_TITLE = "Document RAG Q&A Assistant"


def initialise_session_state() -> None:
    """Creates Streamlit session state variables used by the app."""
    if "document_file_id" not in st.session_state:
        st.session_state.document_file_id = None


    if "indexed_session_id" not in st.session_state:
        st.session_state.indexed_session_id = None    

    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    if "chunks" not in st.session_state:
        st.session_state.chunks = []


def clear_document_state() -> None:
    """Clears the current runtime document, vector store, chunks, and chat history."""
    st.session_state.vector_store = None
    st.session_state.document_name = None
    st.session_state.document_file_id = None

    st.session_state.chunk_count = 0
    st.session_state.chunks = []
    st.session_state.messages = []
    st.session_state.indexed_session_id = None

def save_uploaded_file_for_session(uploaded_file, session_id: int) -> Path:
    """Saves the uploaded file into a session-specific folder."""
    session_folder = Path("data/uploads") / f"session_{session_id}"
    session_folder.mkdir(parents=True, exist_ok=True)

    safe_filename = uploaded_file.name.replace("/", "_").replace("\\", "_")
    stored_path = session_folder / safe_filename

    with open(stored_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return stored_path

def load_saved_session(session_id: int) -> None:
    """Loads a saved session, its messages, and rebuilds its RAG index if a document exists."""
    clear_document_state()

    st.session_state.current_session_id = session_id
    st.session_state.messages = get_messages_for_session(session_id)

    saved_document = get_document_for_session(session_id)

    if saved_document is None:
        return

    stored_path = Path(saved_document["stored_path"])

    if not stored_path.exists():
        st.warning("Saved document file is missing. Chat history was loaded, but RAG cannot be rebuilt.")
        return

    documents = load_document_from_path(stored_path)
    chunks = split_documents(documents)

    for chunk in chunks:
        chunk.metadata["document_name"] = saved_document["original_filename"]
        chunk.metadata["session_id"] = session_id

    vector_store = build_vector_store(chunks)

    st.session_state.vector_store = vector_store
    st.session_state.chunks = chunks
    st.session_state.document_name = saved_document["original_filename"]
    st.session_state.chunk_count = len(chunks)
    st.session_state.document_file_id = f"{saved_document['original_filename']}-{stored_path.stat().st_size}"

    st.session_state.indexed_session_id = session_id

def start_new_session() -> None:
    """Creates a new saved Q&A session and clears the current runtime state."""
    session_id = create_session()
    clear_document_state()
    st.session_state.current_session_id = session_id
    

def process_uploaded_document(uploaded_file) -> None:
    """Saves, loads, chunks, embeds, and stores the uploaded document for the current session."""

    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = create_session()

    session_id = st.session_state.current_session_id

    stored_path = save_uploaded_file_for_session(uploaded_file, session_id)

    documents = load_document_from_path(stored_path)
    chunks = split_documents(documents)

    for chunk in chunks:
        chunk.metadata["document_name"] = uploaded_file.name
        chunk.metadata["session_id"] = session_id

    vector_store = build_vector_store(chunks)

    save_session_document(
        session_id=session_id,
        original_filename=uploaded_file.name,
        stored_path=str(stored_path),
        chunk_count=len(chunks),
    )


    st.session_state.vector_store = vector_store
    st.session_state.chunks = chunks
    st.session_state.document_name = uploaded_file.name
    st.session_state.chunk_count = len(chunks)
    st.session_state.messages = []
    st.session_state.indexed_session_id = session_id


def render_sidebar() -> None:
    """Shows session controls, document upload, and project information in the sidebar."""
    with st.sidebar:
        st.header("Sessions")

        if st.button("➕ New Session", type="primary"):
            start_new_session()
            st.rerun()

        # Create a default session if the app starts with no active session.
        if st.session_state.current_session_id is None:
            start_new_session()

        st.caption(f"Current session ID: {st.session_state.current_session_id}")

        st.divider()
        st.header("1. Upload document")

        uploaded_file = st.file_uploader(
            "Upload a PDF, TXT, or Markdown file",
            type=["pdf", "txt", "md"],
            key=f"file_uploader_{st.session_state.current_session_id}",
        )

        if uploaded_file is not None:
            current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            file_changed = current_file_id != st.session_state.document_file_id

            if file_changed:
                session_id = st.session_state.current_session_id

                try:
                    # Clear old runtime RAG state, but keep the active session ID.
                    clear_document_state()
                    st.session_state.current_session_id = session_id

                    with st.spinner("Building RAG index for this session..."):
                        process_uploaded_document(uploaded_file)

                    st.session_state.document_file_id = current_file_id
                    st.success("RAG index built automatically for this session.")

                except Exception as error:
                    clear_document_state()
                    st.session_state.current_session_id = session_id
                    st.error(f"Could not process the document: {error}")

        if st.session_state.vector_store is not None:
            st.success(f"Current document: {st.session_state.document_name}")
            st.write(f"Chunks created: {st.session_state.chunk_count}")
        else:
            st.info("No document indexed for this session yet.")

        st.divider()
        st.subheader("Recent Sessions")

        recent_sessions = get_recent_sessions(limit=8)

        if not recent_sessions:
            st.caption("No saved sessions yet.")
        else:
            for session_id, title, updated_at, document_name, chunk_count, message_count in recent_sessions:
                label = f"{title} ({message_count} messages)"

                if st.button(label, key=f"load_session_{session_id}"):
                    with st.spinner("Loading session and rebuilding RAG index..."):
                        load_saved_session(session_id)
                    st.rerun()

        st.divider()
        st.header("How it works")
        st.write(
            "Upload → split into chunks → embed with HuggingFace → store in ChromaDB "
            "→ retrieve relevant chunks → answer with OpenAI."
        )

        st.header("Limitations")
        st.write("- Best for text-based PDFs, TXT, and Markdown files.")
        st.write("- Scanned image PDFs may not extract text correctly.")
        st.write("- Answers depend on retrieved chunks, so missing retrieval can affect quality.")

        st.header("OpenAI model")
        st.code(os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), language="text")


def render_chat_history() -> None:
    """Displays the chat messages stored in Streamlit session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_question() -> None:
    """Reads the user's question, runs RAG, and displays the assistant answer."""
    question = st.chat_input("Ask a question about the uploaded document")

    if not question:
        return

    if st.session_state.vector_store is None:
        st.warning("Please upload a document")
        return
    
    if st.session_state.indexed_session_id != st.session_state.current_session_id:
        st.warning("This session does not have its own document index yet. Please upload a document.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    if st.session_state.current_session_id is not None:
        save_message(
        session_id=st.session_state.current_session_id,
        role="user",
        content=question,
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant chunks and asking OpenAI..."):
                answer, docs, context = answer_question(
                vector_store=st.session_state.vector_store,
                question=question,
                all_chunks=st.session_state.chunks,
            )

            st.markdown(answer)

            with st.expander("Retrieved context used for this answer"):
                st.text(context)

            with st.expander("Source chunk metadata"):
                for i, doc in enumerate(docs, start=1):
                    st.write(f"Chunk {i}: {doc.metadata}")

        except Exception as error:
            answer = (
                "Sorry, something went wrong while answering the question.\n\n"
                f"Error: {error}"
            )
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.current_session_id is not None:
        save_message(
        session_id=st.session_state.current_session_id,
        role="assistant",
        content=answer,
        retrieved_context=context,
    )


def main() -> None:
    """Runs the Streamlit app."""
    load_dotenv()
    init_db()
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    initialise_session_state()

    st.title("📄 Document RAG Q&A Assistant")
    st.write(
        "Upload a document, build a RAG index, then ask questions. "
        "The app retrieves relevant chunks and asks OpenAI to answer only from those chunks."
    )

    render_sidebar()
    render_chat_history()
    handle_user_question()


if __name__ == "__main__":
    main()
