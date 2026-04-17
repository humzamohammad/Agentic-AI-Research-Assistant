# ARA — Agentic Research Assistant

> A production-grade, tool-augmented conversational AI system built on LangGraph, featuring persistent memory, PDF RAG, real-time web search, and full observability via LangSmith.

![ARA Chat Interface](image%20copy.png)

---

## Project Overview

**ARA** is an agentic research assistant that demonstrates how modern LLM applications move beyond simple prompt→response patterns into stateful, tool-calling workflows. Unlike basic chatbots, ARA maintains conversation context across sessions, dynamically invokes external tools (web search, news, semantic research), and retrieves information from uploaded PDF documents using vector similarity search.

This project exists to solve a specific architectural challenge: **how do you build an AI assistant that remembers, reasons, acts, and explains—while remaining observable and debuggable?** The answer lies in treating the LLM as a decision-maker within a graph-based state machine, rather than a monolithic black box.

---

## Architecture Overview

![Architecture Diagram](image%20copy%202.png)

ARA follows a **LangGraph-native architecture** where the LLM operates as a node within a directed graph. The system boundaries are:

| Layer | Responsibility |
|-------|---------------|
| **Frontend** | Streamlit chat UI with streaming responses |
| **Orchestration** | LangGraph StateGraph with conditional tool routing |
| **LLM Backend** | Groq (Llama 3.3 70B) with automatic fallback |
| **Tools** | Web search (Serper), News (NewsAPI), Research (Tavily), RAG |
| **Persistence** | SQLite checkpointer + ChromaDB vector store |
| **Observability** | LangSmith traces with thread grouping |

The key insight is that the LLM never directly "calls" tools—it emits structured tool requests that the graph runtime intercepts, executes, and feeds back into the conversation state.

---

## LangGraph Model

ARA implements a **chat → tool → chat feedback loop** using LangGraph's `StateGraph`:

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐     needs_tool?     ┌─────────┐
│  chat   │────────────────────▶│  tools  │
│  node   │◀────────────────────│   node  │
└────┬────┘                     └─────────┘
     │ no tools needed
     ▼
┌─────────┐
│   END   │
└─────────┘
```

### State Definition

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The `add_messages` reducer ensures that tool outputs and LLM responses accumulate correctly without overwriting history.

### Conditional Routing

```python
graph.add_conditional_edges(
    "chat",
    tools_condition,  # Built-in LangGraph router
    {"tools": "tools", END: END}
)
```

When the LLM emits a `tool_call`, `tools_condition` routes to the `ToolNode`. When it emits plain text, the graph terminates.

---

## Tooling

ARA treats tools as **first-class citizens** using LangGraph's `ToolNode` pattern:

| Tool | Provider | Use Case |
|------|----------|----------|
| `web_search` | Serper API | General factual queries |
| `news_search` | NewsAPI | Time-sensitive current events |
| `tavily_search` | Tavily | Deep semantic research |
| `rag_search` | ChromaDB | PDF document retrieval |

### Temporal Tool Enforcement

The system prompt includes **temporal triggers** that force tool usage:

```python
TEMPORAL = ["current", "today", "news", "latest", "recent", "price", "market"]
```

If the user query contains these keywords, the LLM is instructed to always invoke a search tool before responding—preventing hallucinated dates, prices, or events.

### RAG as a Tool (Not a Pipeline)

Unlike traditional RAG architectures where retrieval happens before LLM invocation, ARA exposes RAG as a callable tool:

```python
@tool
def rag_search(query: str, thread_id: str) -> str:
    """Search uploaded PDF documents for relevant excerpts."""
```

This means the LLM **decides** when document context is needed, rather than blindly injecting context on every turn.

---

## Memory & Persistence

![Persistence Flow](image%20copy%204.png)

### Conversation Checkpointing

```python
conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
chatbot = graph.compile(checkpointer=checkpointer)
```

Every message exchange is persisted to SQLite. When a user returns to a previous thread, the full conversation state is restored—including tool call history.

### Thread Isolation

Each conversation receives a unique `thread_id`:

```python
config = {"configurable": {"thread_id": "abc-123"}}
chatbot.stream(state, config)
```

This enables:
- Per-thread conversation memory
- Per-thread vector store collections
- LangSmith trace grouping

### Vector Store

PDF embeddings are stored in ChromaDB with thread-scoped collections:

```python
Chroma.from_documents(
    chunks,
    embedding=embeddings,
    collection_name=f"thread_{thread_id}",
    persist_directory=CHROMA_PERSIST_DIR,
)
```

---

## Observability

### The Black Box Problem

LLM applications are notoriously difficult to debug. A user asks "What's the stock price of Apple?" and receives incorrect output. Was it:
- A hallucination?
- A tool failure?
- A parsing error?
- Rate limiting?

### LangSmith Integration

ARA addresses this with comprehensive tracing:

```python
@traceable(name="chat_node")
def chat_node(state, config):
    ...
```

Every component—LLM calls, tool invocations, embedding requests—is traced with:
- **Latency metrics**
- **Token counts**
- **Input/output payloads**
- **Thread grouping for multi-turn analysis**

### Trace Hierarchy

```
Run: "chat_turn"
├── chat_node (LLM invocation)
│   └── bind_tools
├── tools (ToolNode)
│   └── web_search
│       └── HTTP request
└── chat_node (final response)
```

---

## Streaming UX

Traditional request-response patterns create dead air while the LLM thinks. ARA uses LangGraph's native streaming:

```python
for event in chatbot.stream(state, config, stream_mode="values"):
    if "messages" in event:
        yield event["messages"][-1].content
```

This enables:
- Token-by-token response rendering
- Visible tool execution feedback
- Perception of faster response times

---

## Workflow Graphs

![LangGraph Workflow](workflow_graph.png)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (Llama 3.3 70B Versatile) |
| Orchestration | LangGraph |
| Embeddings | HuggingFace Inference API (BGE-small) |
| Vector Store | ChromaDB |
| Checkpointing | SQLite via SqliteSaver |
| Web Search | Serper API |
| News Search | NewsAPI |
| Research Search | Tavily |
| Observability | LangSmith |
| Frontend | Streamlit |
| Language | Python 3.11+ |

---

## Engineering Challenges

- **HuggingFace Inference Timeouts**: The hosted inference API returns 504s under load. Solved with batching (16 texts/batch), retry logic (3 attempts), and exponential backoff.

- **Groq Tool Call Format**: Groq's function calling occasionally outputs malformed JSON. Lowered temperature to 0.25 and simplified tool signatures.

- **Thread State Restoration**: Rebuilding conversation UI from checkpoint data required deduplication logic to handle reducer accumulation.

- **Embedding Shape Variance**: HuggingFace returns inconsistent tensor shapes (`[[[vec]]]` vs `[[vec]]`). Built a normalization layer to handle all cases.

- **Chroma Collection Naming**: Thread UUIDs contain hyphens, which Chroma rejects. Implemented sanitization: `thread_abc-123` → `thread_abc_123`.

---

## Limitations

- **No multi-modal support**: Images and audio are not processed.
- **Single-user design**: No authentication or multi-tenancy.
- **Cold start latency**: First embedding request after idle can take 5-10 seconds.
- **Context window limits**: Very long PDFs may exceed chunk capacity.
- **No citation linking**: Tool results are summarized but sources aren't hyperlinked in responses.

---

## Future Scope

- **Agent memory abstraction**: Replace per-thread isolation with semantic memory (remembering across threads).
- **Parallel tool execution**: Run web_search and news_search concurrently for faster responses.
- **Multi-agent delegation**: Spawn specialist sub-agents for research, writing, analysis.
- **Self-correction**: Implement reflection loops where the agent critiques and revises its own output.
- **Deployment**: Containerize for production with PostgresSaver and Redis caching.

---

## Relevant Impact

This project demonstrates competency in:

- **LangGraph** — State machines, conditional routing, tool binding, checkpointers
- **LangChain** — Tool definitions, message types, embedding interfaces
- **LLMOps** — LangSmith tracing, latency optimization, error handling
- **RAG Engineering** — Chunking strategies, vector stores, retrieval-as-tool pattern
- **Production Python** — Async patterns, retry logic, configuration management
- **System Design** — Separating orchestration from execution, thread-scoped state

---

## Getting Started

### Prerequisites

- Python 3.11+
- API keys: Groq, HuggingFace, Serper, NewsAPI, Tavily (optional), LangSmith (optional)

### Installation

```bash
git clone https://github.com/ayushsyntax/Agentic-Research-Assistant.git
cd Agentic-Research-Assistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create `.env` from the example:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
streamlit run src/app.py
```

Open http://localhost:8501

---

## Repository Structure

```
ara/
├── src/
│   ├── app.py           # Streamlit frontend
│   ├── config.py        # Environment + LangSmith setup
│   ├── database.py      # SQLite checkpointer + thread names
│   ├── embeddings.py    # HuggingFace embedding client
│   ├── graph.py         # LangGraph workflow
│   ├── llm.py           # Groq LLM factory
│   ├── rag.py           # PDF ingestion + retrieval tool
│   └── tools.py         # Web/news/research tools
├── tests/
│   └── test_*.py        # Unit tests
├── data/
│   ├── sqlite/          # Checkpoint persistence
│   └── chroma/          # Vector store
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI
├── requirements.txt
├── .env.example
└── README.md
```

---

## License
[MIT](LICENSE)

---

**Built with curiosity and caffeine.**
