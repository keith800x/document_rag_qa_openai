"""Streamlit front end for the OpenAI Document RAG Q&A project."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag_utils import (
    answer_question,
    build_vector_store,
    load_document_from_path,
    save_uploaded_file_to_temp,
    split_documents,
)


APP_TITLE = "Document RAG Q&A - OpenAI"


def initialise_session_state() -> None:
    """Creates Streamlit session state variables used by the app."""
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
    """Clears the current document, vector store, and chat history."""
    st.session_state.vector_store = None
    st.session_state.document_name = None
    st.session_state.chunk_count = 0
    st.session_state.chunks = []
    st.session_state.messages = []


def process_uploaded_document(uploaded_file) -> None:
    """Loads, chunks, embeds, and stores the uploaded document in ChromaDB."""
    temp_path = save_uploaded_file_to_temp(uploaded_file)
    documents = load_document_from_path(temp_path)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks)

    st.session_state.vector_store = vector_store
    st.session_state.chunks = chunks
    st.session_state.document_name = uploaded_file.name
    st.session_state.chunk_count = len(chunks)
    st.session_state.messages = []


def render_sidebar() -> None:
    """Shows document upload controls and project information in the sidebar."""
    with st.sidebar:
        st.header("1. Upload document")
        uploaded_file = st.file_uploader(
            "Upload a PDF, TXT, or Markdown file",
            type=["pdf", "txt", "md"],
        )

        if uploaded_file is not None:
            file_changed = uploaded_file.name != st.session_state.document_name

            if file_changed and st.button("Build RAG index", type="primary"):
                try:
                    with st.spinner("Loading document, splitting chunks, and creating embeddings..."):
                        process_uploaded_document(uploaded_file)
                    st.success("RAG index built successfully.")
                except Exception as error:
                    clear_document_state()
                    st.error(f"Could not process the document: {error}")

        if st.session_state.vector_store is not None:
            st.success(f"Current document: {st.session_state.document_name}")
            st.write(f"Chunks created: {st.session_state.chunk_count}")

        if st.button("Clear document and chat"):
            clear_document_state()
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
        st.warning("Please upload a document and click 'Build RAG index' first.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

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


def main() -> None:
    """Runs the Streamlit app."""
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    initialise_session_state()

    st.title("📄 Document RAG Q&A - OpenAI")
    st.write(
        "Upload a document, build a RAG index, then ask questions. "
        "The app retrieves relevant chunks and asks OpenAI to answer only from those chunks."
    )

    render_sidebar()
    render_chat_history()
    handle_user_question()


if __name__ == "__main__":
    main()
