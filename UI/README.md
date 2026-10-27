# Sakthi Finance RAG Assistant — Web UI

A clean, professional Streamlit web interface with **permanent SQLite chat history** for the Sakthi Finance Financial Document RAG system.

---

## Features

1. **Clean Chat Interface:**
   - Chronological message history with distinct user and assistant styling.
   - Dynamic answer generation with loading spinners.
   - Expandable source citation drawers displaying Document Name, Page Number, Section/Table, and Financial Period.
   - Follow-up conversation context awareness.
2. **Permanent SQLite Storage:**
   - Stored in `UI/data/chat_history.db`.
   - **Survives restarts and Mac shutdowns.**
   - Stores Conversation ID, Title, Creation/Update times, Message ID, Role, Content, Timestamp, and JSON Citations.
   - Parameterized queries with transactions and cascade deletion.
3. **History Sidebar:**
   - Lists all past conversations with timestamps.
   - Reopen and continue any past conversation.
   - Create a **New Chat** at any time.
   - Delete conversations with confirmation (never deletes PDF documents, chunks, or pgvector embeddings).
4. **Zero Backend Changes:**
   - Fully connects to the existing `HybridRetriever` and `AnswerGenerator` without altering the existing terminal CLI (`ragcli ask`, `ragcli chat`).

---

## Quick Start

### 1. Prerequisites
Ensure Ollama and PostgreSQL are running:
```bash
ollama serve
# or verify model:
ollama list
```

### 2. Launch the Web UI
From the project root:
```bash
source .venv/bin/activate
streamlit run UI/app.py
```

Or run directly from the `UI/` folder:
```bash
cd UI
source ../.venv/bin/activate
streamlit run app.py
```

The application will open in your browser at:
`http://localhost:8501`

---

## Directory Structure

```
UI/
├── app.py                 # Streamlit application UI
├── chat_history.py        # SQLite persistent storage manager
├── rag_adapter.py         # RAG pipeline bridge
├── README.md              # Setup and usage guide
└── data/
    └── chat_history.db    # Persistent SQLite database (auto-created)
```
