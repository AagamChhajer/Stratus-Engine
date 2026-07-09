# Stratus Engine

Stratus Engine is a pragmatic memory layer for LLM applications. The MVP keeps
active conversations in MongoDB, promotes only useful facts into long-term
memory in Chroma, and warms relevant context when a session is reopened.

The point is simple: do not make vector search behave like a chat log. Keep the
active session close, then send only valuable extracted memories to the long-term
vector store.

## Current MVP

- MongoDB session layer for active conversations
- Structured memory extraction from user messages
- Chroma vector DB adapter for long-term memories
- Optional OpenAI embeddings and live answer generation
- Session reopen cache warming with predefined queries
- Context assembly that prefers active session state before ad-hoc recall
- Metrics for estimated context tokens, retrieved memories, and relevance
- Benchmark script for vector-search reduction
- Runnable local demo and tests

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python examples/demo.py
python examples/benchmark.py
pytest
```

## Docker Services

Start MongoDB and Chroma:

```powershell
docker compose up -d mongo chroma
```

## Chroma Vector Demo

This uses MongoDB for sessions and Chroma for long-term memory. It uses local
hash embeddings, so it does not need an OpenAI key.

```powershell
pip install -e ".[mongo,vector,dev]"
python examples/chroma_demo.py
```

## Live OpenAI Demo

This uses MongoDB, Chroma, OpenAI embeddings, and an OpenAI chat completion. It
prints context metrics and API token usage.

Create a local `.env` file with `OPENAI_API_KEY` set to your key. The live demo
loads that file automatically, or you can set the variable in the current
PowerShell session before running it.

```powershell
pip install -e ".[demo]"
$env:OPENAI_API_KEY = "your_api_key_here"
python examples/live_openai_demo.py
```

## Benchmark

The benchmark compares naive retrieval, where every prompt would run vector
search, against Stratus's gated recall.

```powershell
python examples/benchmark.py
```

It prints vector-search reduction, average estimated context tokens, average
memory relevance, and a prompt-level trace.

## Programmatic Usage

```python
from stratus_engine import (
    ChromaLongTermMemoryStore,
    MongoSessionStore,
    OpenAIEmbeddingProvider,
    StratusEngine,
    analyze_context,
)

engine = StratusEngine(
    session_store=MongoSessionStore("mongodb://localhost:27017"),
    memory_store=ChromaLongTermMemoryStore(
        host="localhost",
        port=8000,
        embedding_provider=OpenAIEmbeddingProvider(),
    ),
)

session = engine.create_session("user_123", title="Demo")
engine.append_user_message(session.id, "I prefer React over Angular.")
engine.extract_memories(session.id)
engine.reopen_session(session.id)

context = engine.build_context(session.id, "What frontend preference do I have?")
metrics = analyze_context("What frontend preference do I have?", context)
print(context.as_prompt_sections())
print(metrics)
```

## Demo Story

The demo shows five important behaviors:

1. Active conversation context is read from the session layer.
2. Useful facts are extracted into structured memories.
3. Chroma retrieves long-term memories.
4. Reopening a session warms relevant context before the user asks anything.
5. Metrics show token footprint and retrieval relevance.

## Design Direction

Next steps are intentionally narrow: replace the heuristic extractor with an
LLM-backed extractor, add background extraction, and add a small evaluation set
that proves latency, recall quality, and reduced vector-search usage.
