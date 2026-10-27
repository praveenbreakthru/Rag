"""
Streamlit Web UI for Sakthi Finance RAG Assistant.
Features:
- Clean, professional chat interface with chronological messages.
- Permanent SQLite conversation history (survives restarts & Mac reboots).
- Direct connection to existing HybridRetriever & AnswerGenerator without modifying backend.
- Citations, latency, confidence display, and follow-up conversation context.
- History sidebar with conversation switching, creation, and deletion.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root and src to path
UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in (str(PROJECT_ROOT), str(SRC_DIR), str(UI_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
# pyrefly: ignore[missing-import]
from chat_history import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation_messages,
    init_chat_history_db,
    list_conversations,
    update_conversation_title,
)
# pyrefly: ignore[missing-import]
from rag_adapter import RAGAdapter

# Page Configuration
st.set_page_config(
    page_title="Sakthi Finance RAG Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, professional look
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .citation-box {
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 8px 12px;
        margin-top: 8px;
        font-size: 0.85rem;
        border-radius: 4px;
    }
    .stChatMessage {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize database
init_chat_history_db()

# Load RAG backend instance (cached across reruns)
@st.cache_resource(show_spinner=False)
def load_rag_backend():
    return RAGAdapter.get_instance()

rag_engine = load_rag_backend()

# Manage active conversation in session state
if "active_conv_id" not in st.session_state:
    conversations = list_conversations()
    if conversations:
        st.session_state.active_conv_id = conversations[0]["id"]
    else:
        st.session_state.active_conv_id = create_conversation("New Financial Chat")

if "confirm_delete_id" not in st.session_state:
    st.session_state.confirm_delete_id = None


# --- SIDEBAR: CONVERSATION HISTORY ---
with st.sidebar:
    st.markdown("### 💬 Conversations")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = create_conversation("New Financial Chat")
        st.session_state.active_conv_id = new_id
        st.session_state.confirm_delete_id = None
        st.rerun()

    st.markdown("---")

    # List all conversations from SQLite
    conv_list = list_conversations()

    if not conv_list:
        st.caption("No previous conversations.")
    else:
        for conv in conv_list:
            c_id = conv["id"]
            c_title = conv["title"] or "Financial Inquiry"
            is_active = (c_id == st.session_state.active_conv_id)

            # Format timestamp
            try:
                dt = datetime.fromisoformat(conv["updated_at"])
                date_str = dt.strftime("%b %d, %I:%M %p")
            except Exception:
                date_str = conv["updated_at"][:16]

            # Render each conversation card
            col_conv, col_del = st.columns([5, 1])

            with col_conv:
                btn_type = "secondary"
                label = f"{'▶ ' if is_active else ''}{c_title}\n\n_{date_str}_"
                if st.button(
                    label,
                    key=f"conv_{c_id}",
                    use_container_width=True,
                    help=f"Open: {c_title}",
                ):
                    st.session_state.active_conv_id = c_id
                    st.session_state.confirm_delete_id = None
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{c_id}", help="Delete conversation"):
                    st.session_state.confirm_delete_id = c_id

        # Delete Confirmation Modal / Alert
        if st.session_state.confirm_delete_id:
            del_id = st.session_state.confirm_delete_id
            st.warning("⚠️ Delete this conversation and its messages?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Yes, Delete", type="primary", use_container_width=True):
                    delete_conversation(del_id)
                    st.session_state.confirm_delete_id = None
                    # Update active conversation
                    remaining = list_conversations()
                    st.session_state.active_conv_id = remaining[0]["id"] if remaining else create_conversation("New Financial Chat")
                    st.rerun()
            with c_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_delete_id = None
                    st.rerun()


# --- MAIN AREA: CHAT INTERFACE ---
st.markdown('<div class="main-header">Sakthi Finance RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Source-Grounded Financial Intelligence & Reporting</div>', unsafe_allow_html=True)

# Fetch messages for current active conversation
messages = get_conversation_messages(st.session_state.active_conv_id)

# Display existing messages in chronological order
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Display citations if present for assistant messages
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📄 Source Citations & References", expanded=False):
                for cit in msg["citations"]:
                    doc_p = f"**{cit['document_name']}** (Page {cit['page_number']})"
                    meta_parts = []
                    if cit.get("section_title") and cit["section_title"] != "-":
                        meta_parts.append(f"Section: {cit['section_title']}")
                    if cit.get("financial_period") and cit["financial_period"] != "-":
                        meta_parts.append(f"Period: {cit['financial_period']}")
                    if cit.get("unit") and cit["unit"] != "-":
                        meta_parts.append(f"Unit: {cit['unit']}")
                    
                    sub = f" — {', '.join(meta_parts)}" if meta_parts else ""
                    st.markdown(f"- {doc_p}{sub}")


# Check if the active conversation has a pending user question that needs an answer
pending_prompt = None
if messages and messages[-1]["role"] == "user":
    pending_prompt = messages[-1]["content"]

# Chat Input Box at bottom
user_input = st.chat_input("Ask a financial question (e.g. EBITDA for 2024 to 2025, Note 4 details, Borrowings)...")

if user_input:
    # 1. Immediately save user message to SQLite
    add_message(st.session_state.active_conv_id, "user", user_input)

    # 2. If this is the first question in the conversation, update title
    if len(messages) == 0:
        new_title = user_input.strip()[:50]
        update_conversation_title(st.session_state.active_conv_id, new_title)

    st.rerun()

elif pending_prompt:
    # Prepare conversation context for follow-up questions (excluding the pending question itself)
    conv_history_for_rag = [
        {"role": m["role"], "content": m["content"]}
        for m in messages[:-1]
    ]

    # Generate answer with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Searching annual reports & computing audited answer..."):
            try:
                result = rag_engine.ask(
                    question=pending_prompt,
                    conversation_history=conv_history_for_rag,
                )
                answer_text = result["answer"]
                citations = result["citations"]

                # Render answer
                st.markdown(answer_text)

                # Render citations
                if citations:
                    with st.expander("📄 Source Citations & References", expanded=False):
                        for cit in citations:
                            doc_p = f"**{cit['document_name']}** (Page {cit['page_number']})"
                            meta_parts = []
                            if cit.get("section_title") and cit["section_title"] != "-":
                                meta_parts.append(f"Section: {cit['section_title']}")
                            if cit.get("financial_period") and cit["financial_period"] != "-":
                                meta_parts.append(f"Period: {cit['financial_period']}")
                            if cit.get("unit") and cit["unit"] != "-":
                                meta_parts.append(f"Unit: {cit['unit']}")
                            sub = f" — {', '.join(meta_parts)}" if meta_parts else ""
                            st.markdown(f"- {doc_p}{sub}")

                # Save assistant response to SQLite
                add_message(
                    conversation_id=st.session_state.active_conv_id,
                    role="assistant",
                    content=answer_text,
                    citations=citations,
                )

                st.rerun()

            except Exception as e:
                error_msg = f"An error occurred while generating the financial answer: {str(e)}"
                st.error(error_msg)

