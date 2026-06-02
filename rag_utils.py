"""Helper functions for loading documents, building the vector store, and asking OpenAI.

This file follows a standard RAG flow:
1. Load document text.
2. Split document into smaller chunks.
3. Create embeddings for each chunk.
4. Store the chunks in ChromaDB.
5. Retrieve the most relevant chunks for a user question.
6. Send only the retrieved context to OpenAI for answer generation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from prompts import SYSTEM_PROMPT


# These constants make the RAG settings easy to find and tune.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

DEFAULT_RETRIEVAL_K = 4
SPECIFIC_RETRIEVAL_K = 3
BROAD_RETRIEVAL_K = 6

NEIGHBOR_WINDOW = 1
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def save_uploaded_file_to_temp(uploaded_file) -> str:
    """Saves a Streamlit uploaded file to a temporary path and returns that path."""
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def load_document_from_path(file_path: str) -> List[Document]:
    """Loads a PDF, TXT, or Markdown file into LangChain Document objects."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    if suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    raise ValueError("Unsupported file type. Please upload a PDF, TXT, or Markdown file.")


def split_documents(documents: Iterable[Document]) -> List[Document]:
    """Splits loaded document pages into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Creates the local HuggingFace embedding model used for document retrieval."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(chunks: List[Document]) -> Chroma:
    """Creates an in-memory ChromaDB vector store from document chunks."""
    if not chunks:
        raise ValueError("No text chunks were created from the uploaded document.")

    embeddings = get_embedding_model()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="uploaded_document_collection",
    )


def retrieve_relevant_chunks(vector_store: Chroma, question: str, k: int = DEFAULT_RETRIEVAL_K) -> List[Document]:
    """Retrieves the most relevant chunks from ChromaDB for the user's question."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def format_context(docs: List[Document]) -> str:
    """Formats retrieved chunks into readable context for the LLM prompt."""
    formatted_parts = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        page = metadata.get("page")

        if isinstance(page, int):
            source_label = f"page {page + 1}"
        else:
            source_label = "unknown page"

        formatted_parts.append(
            f"[Chunk {index} | Source: {source_label}]\n{doc.page_content.strip()}"
        )

    return "\n\n".join(formatted_parts)


def get_openai_client() -> OpenAI:
    """Creates an OpenAI client using OPENAI_API_KEY from the environment."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Please add it to your .env file.")

    return OpenAI(api_key=api_key)


def ask_openai_with_context(question: str, context: str) -> str:
    """Asks OpenAI to answer a question using only the retrieved RAG context."""
    client = get_openai_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

    user_prompt = f"""
Use the retrieved context below to answer the user's question.

Retrieved context:
{context}

User question:
{question}

Remember: answer only from the retrieved context. If the answer is not present,
say: "I cannot find this in the document."
""".strip()

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        max_output_tokens=900,
    )

    answer = getattr(response, "output_text", "") or ""
    if not answer.strip():
        return "The AI returned an empty response. Please try again."

    return answer.strip()


def answer_question(vector_store: Chroma, question: str, all_chunks) -> Tuple[str, List[Document], str]:
    """Runs retrieval and generation, then returns the answer, source chunks, and context."""
    k = choose_retrieval_k(question)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    retrieved_docs = retriever.invoke(question)
    
    context_docs = add_neighbor_chunks(
        retrieved_docs=retrieved_docs,
        all_chunks=all_chunks,
        window=NEIGHBOR_WINDOW,
    )

    # docs = retrieve_relevant_chunks(vector_store, question, k=k)
    context = format_context(context_docs)
    answer = ask_openai_with_context(question=question, context=context)
    return answer, context_docs, context

def add_neighbor_chunks(
    retrieved_docs: List[Document],
    all_chunks: List[Document],
    window: int = NEIGHBOR_WINDOW,
) -> List[Document]:
    """Adds nearby chunks so answers are not cut off at chunk boundaries."""
    if not retrieved_docs:
        return []

    if not all_chunks:
        return retrieved_docs

    selected_indexes = set()

    for doc in retrieved_docs:
        chunk_index = doc.metadata.get("chunk_index")

        if not isinstance(chunk_index, int):
            continue

        start = max(0, chunk_index - window)
        end = min(len(all_chunks), chunk_index + window + 1)

        for index in range(start, end):
            selected_indexes.add(index)

    if not selected_indexes:
        return retrieved_docs

    return [all_chunks[index] for index in sorted(selected_indexes)]

def choose_retrieval_k(question: str) -> int:
    """Chooses how many chunks to retrieve based on the question type."""
    q = question.lower()

    broad_keywords = [
        "structure",
        "summary",
        "summarise",
        "summarize",
        "main points",
        "key points",
        "list",
        "all",
        "sections",
        "requirements",
        "overview",
        "explain",
        "compare",
    ]

    specific_keywords = [
        "who",
        "when",
        "where",
        "what is",
        "define",
        "how many",
    ]

    if any(keyword in q for keyword in broad_keywords):
        return BROAD_RETRIEVAL_K

    if any(keyword in q for keyword in specific_keywords):
        return SPECIFIC_RETRIEVAL_K

    return 3