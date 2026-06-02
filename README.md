# Document RAG Q&A - OpenAI

## 1. Project Title and Description

**Document RAG Q&A - OpenAI** is a Streamlit application that lets a user upload a PDF, TXT, or Markdown document and ask questions about it.

The app uses Retrieval-Augmented Generation (RAG): it retrieves relevant chunks from the uploaded document, then asks OpenAI to answer using only those chunks.

## 2. Problem Statement

Long documents can be difficult to search manually. This tool helps users ask natural-language questions and receive grounded answers based on the uploaded document.

It is useful for students, interns, and junior developers who need to understand lecture notes, requirements documents, reports, or technical handouts quickly.

## 3. Technology Stack

- Python
- Streamlit
- OpenAI Python SDK
- OpenAI Responses API
- LangChain
- ChromaDB
- HuggingFace sentence-transformer embeddings
- PyPDF
- python-dotenv

## 4. Setup Instructions

### Step 1: Clone or open the project folder

```bash
cd document_rag_qa_openai
```

### Step 2: Create a virtual environment

Windows PowerShell:

```bash
py -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create your `.env` file

Copy `.env.example` to `.env`.

Windows PowerShell:

```bash
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Then edit `.env` and add your real OpenAI API key:

```env
OPENAI_API_KEY=your_real_openai_api_key_here
OPENAI_MODEL=gpt-5.4-mini
```

### Step 5: Run the app

```bash
streamlit run app.py
```

## 5. Usage Examples

### Example 1

**User question:**

```text
What is considered completed for error handling in the capstone project?
```

**Expected output:**

```text
For error handling, the project is considered complete if it handles errors by showing a clear message when an API call fails or returns nothing, and it does not crash with an unhandled error or exception. This is stated in Chunk 5 / page 3.
```

### Example 2

**User question:**

```text
What is the presentation slide structure?
```

**Expected output:**

```text
The presentation slide structure is:

1. Title — Project name, your name, and one sentence describing what the application does.
2. The Problem — What problem your application addresses, who would use it, and why AI is the right tool for it.
3. How It Works — A simple diagram or a few bullet points showing the flow: user input → your code → AI API → output.
4. Live Demo — No slide is needed; switch directly to your application, show at least two different inputs, and explain what is happening as you go.
5. What You Learned — One thing that was harder than expected and how you worked through it, one thing you would do differently, and one idea for what comes next.

Source: page 5, Chunk 5.
```

### Example 3

**User question:**

```text
What is the weather in Singapore today?
```

**Expected output:**

```text
I cannot find this in the document.
```

## 6. Known Limitations

1. Scanned image PDFs may not extract text correctly because this project does not include OCR.
2. The app retrieves only a few chunks per question, so an answer may be incomplete if the relevant chunk is not retrieved.
3. The app uses an online OpenAI API, so it requires internet access and a valid API key.

## 7. Future Improvements

1. Add OCR support for scanned PDFs.
2. Add source citations with clickable page references.
3. Add support for multiple uploaded documents.
4. Add a local Ollama mode for private/offline document Q&A.
5. Add evaluation tests to check whether answers are faithful to retrieved context.

## How the RAG Pipeline Works

```text
Upload document
→ load PDF/TXT/MD
→ split into chunks
→ create local HuggingFace embeddings
→ store chunks in ChromaDB
→ retrieve relevant chunks for the question
→ send retrieved context to OpenAI
→ display answer and retrieved source chunks
```

## Important Security Note

Do not commit your real `.env` file to GitHub. This project includes `.env.example` for placeholders only.
