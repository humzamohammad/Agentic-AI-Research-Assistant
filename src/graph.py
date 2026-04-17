"""
LangGraph workflow for the ARA agent.

Defines the conversational state machine with tool-calling capability.
Uses LangGraph's StateGraph to orchestrate:
1. Chat node: LLM generates response or requests tool calls
2. Tool node: Executes requested tools
3. Conditional routing: Directs flow based on tool_call presence

Graph Structure:
    START → chat → (tools_condition) → tools → chat → ... → END

Tools Bound:
    - web_search: General web search
    - news_search: Recent news lookup
    - tavily_search: Semantic research search
    - rag_search: PDF document retrieval

Persistence:
    Compiled graph uses SqliteSaver checkpointer for state persistence.
"""

import logging
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, BaseMessage
from langsmith import traceable

from src.llm import create_llm
from src.tools import TOOLS
from src.rag import rag_search
from src.database import checkpointer

logger = logging.getLogger(__name__)


# Keywords that indicate user query requires real-time data
TEMPORAL = ["current", "today", "news", "latest", "recent", "this week", "this month", "price", "market"]


class ChatState(TypedDict):
    """
    State schema for the LangGraph conversation workflow.

    Attributes:
        messages: Accumulated conversation messages. Uses add_messages reducer
            which appends new messages rather than replacing the list.
    """
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """
You are ARA, a research assistant with controlled tool usage.

Rules:
- Never guess dates, weather, prices, news or recent events.
- If query contains temporal triggers (current, today, latest, recent, price, market, news):
    ALWAYS call a web/news tool first.
- For deep research queries, use tavily_search.
- For news queries, use news_search.
- For general factual real-time queries, use web_search.
- For PDF context, use rag_search(thread_id).
- After tool results, summarize clearly.
"""


@traceable(name="chat_node")
def chat_node(state: ChatState, config=None):
    """
    LLM node that processes messages and may request tool calls.

    Extracts thread_id from config, binds tools to LLM, constructs
    system prompt with thread context, and invokes the model.

    Args:
        state (ChatState): Current conversation state with messages.
        config (dict | None): LangGraph config containing thread_id in
            configurable.thread_id.

    Returns:
        dict: Updated state with LLM response appended to messages.

    Side Effects:
        - Calls Groq LLM via create_llm()
        - Traced by LangSmith if enabled
    """
    tid = config.get("configurable", {}).get("thread_id", "") if config else ""
    llm = create_llm(temperature=0.25).bind_tools(TOOLS + [rag_search])

    system = SystemMessage(
        content=f"""{SYSTEM_PROMPT}

Current thread_id="{tid}"
When calling rag_search, always pass thread_id="{tid}".
"""
    )

    msgs = [system, *state["messages"]]
    resp = llm.invoke(msgs, config=config)
    return {"messages": [resp]}


# Pre-built ToolNode executes tool calls from LLM responses
tool_node = ToolNode(TOOLS + [rag_search])

# Build the state graph
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges(
    "chat",
    tools_condition,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "chat")

# Compile with checkpointer for conversation persistence
chatbot = graph.compile(checkpointer=checkpointer)
