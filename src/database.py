"""
Database + checkpoint support for ARA.
Persists thread names and conversation checkpoints.
"""

import sqlite3
import logging
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

from src.config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)


# --- SQLite checkpoint database for LangGraph state ---
conn = sqlite3.connect(database=SQLITE_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# --- Thread naming table (human friendly chat titles) ---
conn.execute("""
CREATE TABLE IF NOT EXISTS thread_names (
    thread_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

logger.info(f"Database initialized: {SQLITE_DB_PATH}")


@traceable(name="retrieve_all_threads")
def retrieve_all_threads() -> list:
    """Return all thread IDs stored in checkpoint DB."""
    threads = set()
    try:
        for cp in checkpointer.list(None):
            cfg = cp.config
            if cfg and "configurable" in cfg:
                tid = cfg["configurable"].get("thread_id")
                if tid:
                    threads.add(tid)
    except Exception as e:
        logger.error(f"retrieve_all_threads failed: {e}")
    return list(threads)


def save_thread_name(thread_id: str, name: str):
    try:
        conn.execute(
            "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
            (thread_id, name),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"save_thread_name failed: {e}")


def get_thread_name(thread_id: str) -> str:
    try:
        cur = conn.execute(
            "SELECT name FROM thread_names WHERE thread_id = ?", (thread_id,)
        )
        row = cur.fetchone()
        if row:
            return row[0]
    except Exception as e:
        logger.error(f"get_thread_name failed: {e}")
    return f"Chat {thread_id[:8]}"


def generate_thread_name(first_msg: str) -> str:
    name = first_msg.strip().replace("\n", " ")
    return name[:40] + ("..." if len(name) > 40 else "")


def load_thread_messages(thread_id: str) -> list:
    """
    Rebuild conversation history for a thread from LangGraph checkpoints.
    Returns list[BaseMessage].
    """
    msgs = []
    try:
        for cp in checkpointer.list({"configurable": {"thread_id": thread_id}}):
            state = cp.state
            if isinstance(state, dict) and "messages" in state:
                msgs.extend(state["messages"])
    except Exception as e:
        logger.error(f"load_thread_messages failed: {e}")
        return []

    # dedupe while preserving order
    out = []
    seen = set()
    for m in msgs:
        key = (m.__class__.__name__, getattr(m, "content", None))
        if key not in seen:
            out.append(m)
            seen.add(key)
    return out
