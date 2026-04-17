"""
Streamlit user interface for the ARA agent.

This module renders the chat interface, handles session state initialization,
manages conversation threads, and coordinates streaming responses from the
LangGraph backend. It also supports PDF uploads for RAG functionality.
"""

import uuid
import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

from src.graph import chatbot
from src.rag import ingest_pdf, has_document, document_metadata
from src.database import (
    retrieve_all_threads,
    save_thread_name,
    get_thread_name,
    generate_thread_name,
    load_thread_messages,
)
from src.config import langsmith_enabled


st.set_page_config(
    page_title="ARA - Agentic Research Assistant",
    page_icon="🤖",
    layout="wide"
)


def init_state():
    """
    Initialize required Streamlit session state variables.

    Ensures 'current' thread ID and 'conversations' cache exist in
    st.session_state before rendering the UI.
    """
    if "current" not in st.session_state:
        st.session_state.current = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}


def new_chat():
    """
    Create a new conversation thread and switch to it.

    Generates a UUID, creates a new conversation entry in session state,
    and sets it as the current active thread.

    Returns:
        str: The new thread UUID.
    """
    tid = str(uuid.uuid4())
    st.session_state.current = tid
    st.session_state.conversations[tid] = {
        "title": "New Chat",
        "messages": [],
        "named": False,
        "last_ts": time.time(),
    }
    return tid


def sidebar():
    """
    Render the sidebar for thread navigation and actions.

    Displays controls to start a new chat and lists existing conversation
    threads retrieved from the database. Allows switching between threads.
    """
    with st.sidebar:
        st.title("ARA")
        st.caption("Agentic Research Assistant")

        if langsmith_enabled:
            st.info("📊 LangSmith Enabled")

        st.divider()

        if st.button("➕ New Chat", use_container_width=True):
            new_chat()
            st.rerun()

        st.divider()
        st.subheader("Chats")

        threads = retrieve_all_threads()
        if not threads:
            st.caption("No chats yet.")
            return

        # newest chats first
        threads = list(reversed(threads))

        for tid in threads:
            name = get_thread_name(tid)
            conv = st.session_state.conversations.get(tid)
            if conv and conv.get("named", False):
                name = conv["title"]

            btn_type = "primary" if tid == st.session_state.current else "secondary"
            if st.button(name, key=f"thr_{tid}", type=btn_type, use_container_width=True):
                st.session_state.current = tid
                st.rerun()


@traceable(name="stream_reply")
def stream_reply(tid: str, text: str):
    """
    Stream partial assistant responses from the backend graph.

    Args:
        tid (str): The thread ID for LangGraph state persistence.
        text (str): The user's input message.

    Yields:
        str: Accumulated response text chunks as they become available.

    Returns:
        str | None: The final complete response string, or None if empty.

    Side Effects:
        Emits LangSmith trace data with basic metrics.
    """
    start = time.time()

    config = {
        "configurable": {"thread_id": tid},
        "metadata": {
            "thread_id": tid,
            "run_name": "chat_turn",
            "category": "inference",
            "component": "chat",
            "unit": "message",
            "started_at": start,
        }
    }

    state = {"messages": [HumanMessage(content=text)]}

    final = None
    for evt in chatbot.stream(state, config, stream_mode="values"):
        if "messages" in evt and evt["messages"]:
            msg = evt["messages"][-1]
            if isinstance(msg, AIMessage) and msg.content:
                final = msg.content
                yield final

    # post-metrics
    config["metadata"]["elapsed"] = time.time() - start
    config["metadata"]["completed_at"] = time.time()

    return final


def ui():
    """
    Render the main chat interface for the active thread.

    Handles:
    - Message history display
    - PDF upload widget
    - User input capture
    - Message streaming and rendering
    - Thread auto-naming on first interaction
    """
    tid = st.session_state.current
    if tid is None:
        st.info("Start a new chat or select one from the left")
        return

    conv = st.session_state.conversations.get(tid)
    if conv is None:
        # restore from DB
        conv = {
            "title": get_thread_name(tid),
            "messages": [],
            "named": True,
            "last_ts": time.time(),
        }
        restored = load_thread_messages(tid)
        for m in restored:
            role = "assistant" if isinstance(m, AIMessage) else "user"
            conv["messages"].append({"role": role, "content": m.content})
        st.session_state.conversations[tid] = conv

    st.header(conv["title"])

    pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
    if pdf:
        res = ingest_pdf(pdf.read(), tid, pdf.name)
        if res.get("success"):
            st.success(f"Indexed {res['pages']} pages")
        else:
            st.error(res.get("error", "PDF ingestion failed"))

    if has_document(tid):
        meta = document_metadata(tid)
        st.caption(f"📄 {meta.get('filename', 'Document loaded')} ready for RAG")

    st.divider()

    # replay history
    for m in conv["messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user = st.chat_input("Ask...")
    if user:
        if not conv.get("named", False):
            name = generate_thread_name(user)
            conv["title"] = name
            conv["named"] = True
            save_thread_name(tid, name)

        conv["messages"].append({"role": "user", "content": user})
        conv["last_ts"] = time.time()

        with st.chat_message("user"):
            st.write(user)

        with st.chat_message("assistant"):
            final = ""
            for chunk in stream_reply(tid, user):
                final = chunk
                st.write(chunk)

            conv["messages"].append({"role": "assistant", "content": final})
            conv["last_ts"] = time.time()


def main():
    """Main entry point for the Streamlit application."""
    init_state()
    sidebar()
    ui()


if __name__ == "__main__":
    main()
