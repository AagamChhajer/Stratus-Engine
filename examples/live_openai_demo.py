import os
from pathlib import Path
from uuid import uuid4

from stratus_engine import (
    ChromaLongTermMemoryStore,
    MongoSessionStore,
    OpenAIConversationClient,
    OpenAIEmbeddingProvider,
    OpenAIMemoryExtractor,
    StratusEngine,
    analyze_context,
)


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def main() -> None:
    _load_local_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or the current shell before running the live demo.")

    run_id = uuid4().hex[:8]
    collection_name = f"stratus_memories_openai_demo_{run_id}"

    engine = StratusEngine(
        session_store=MongoSessionStore("mongodb://localhost:27017"),
        memory_store=ChromaLongTermMemoryStore(
            host="localhost",
            port=8000,
            collection_name=collection_name,
            embedding_provider=OpenAIEmbeddingProvider(),
        ),
        extractor=OpenAIMemoryExtractor(),
    )
    llm = OpenAIConversationClient()

    session = engine.create_session("live_demo_user", title="Live memory demo")
    conversation: list[tuple[str, str]] = []
    seed_turns = [
        "I want this memory layer to feel practical, not like a toy demo.",
        "I prefer React over Angular for dashboard-style internal tools.",
        "The project is Stratus Engine, which keeps active chat in MongoDB and stores durable facts in Chroma.",
    ]
    for turn in seed_turns:
        engine.append_user_message(session.id, turn)
        turn_context = engine.build_context(session.id, turn)
        response = llm.respond(prompt=turn, context=turn_context)
        engine.append_assistant_message(session.id, response.answer)
        conversation.append((turn, response.answer))

    bridge_prompt = "So how do you keep the memory layer useful without turning it into a chat-log search engine?"
    engine.append_user_message(session.id, bridge_prompt)
    bridge_context = engine.build_context(session.id, bridge_prompt)
    bridge_response = llm.respond(prompt=bridge_prompt, context=bridge_context)
    engine.append_assistant_message(session.id, bridge_response.answer)
    conversation.append((bridge_prompt, bridge_response.answer))

    promoted_ids = engine.extract_memories(session.id)
    warmed_ids = engine.reopen_session(session.id)

    contrast_prompt = (
        "What frontend preference do I have, what project am I building, and what "
        "is the practical difference between naive every-message vector search and "
        "Stratus's gated recall approach? Be specific and conversational."
    )

    context = engine.build_context(session.id, contrast_prompt)
    metrics = analyze_context(contrast_prompt, context)

    follow_up_prompt = (
        "Based only on the conversation and the metrics below, explain the practical "
        "difference between naive every-message vector search and Stratus's gated "
        "recall approach. Be concrete, conversational, and specific. Mention the "
        "actual memory and retrieval numbers shown here. Do not invent any new "
        "metrics or claims.\n\n"
        f"Promoted memories: {len(promoted_ids)}\n"
        f"Warmed memories on reopen: {len(warmed_ids)}\n"
        f"Retrieval reason: {metrics.retrieval_reason}\n"
        f"Estimated context tokens before API call: {metrics.estimated_context_tokens}\n"
        f"Estimated memory tokens: {metrics.estimated_memory_tokens}\n"
        f"Context relevance score: {metrics.relevance_score}\n"
        "\nKeep it in 3 short paragraphs or 5 bullets max."
    )

    result = llm.respond(prompt=follow_up_prompt, context=context)
    engine.append_user_message(session.id, follow_up_prompt)
    engine.append_assistant_message(session.id, result.answer)

    promoted_memories = _unique_memories(engine.memory_store.get_many(promoted_ids))
    warmed_memories = _unique_memories(engine.memory_store.get_many(warmed_ids))
    recalled_memories = _unique_memories(context.recalled_memories)
    recent_messages = context.as_prompt_sections()["recent_messages"]
    naive_vector_searches = len(seed_turns) + 1
    actual_vector_searches = 1 if context.retrieval_reason == "ad_hoc_recall" else 0
    avoided_searches = naive_vector_searches - actual_vector_searches

    print("Live Stratus Engine demo")
    print("========================")
    print(f"Session: {session.id}")
    print(f"Run collection: {collection_name}")
    print()
    print("Conversation")
    print("------------")
    for index, (user_turn, assistant_turn) in enumerate(conversation, start=1):
        print(f"{index}. user: {user_turn}")
        print(f"   assistant: {assistant_turn}")
    print(f"{len(conversation) + 1}. user: {follow_up_prompt}")
    print()

    print("Memory extraction")
    print("-----------------")
    print(f"Promoted memories: {len(promoted_memories)}")
    for memory in promoted_memories:
        print(f"- [{memory.type.value}] {memory.text}")
    print()

    print("Session reopen / warmup")
    print("-----------------------")
    print(f"Warmed memories on reopen: {len(warmed_memories)}")
    for memory in warmed_memories:
        print(f"- [{memory.type.value}] {memory.text}")
    print(f"Retrieval reason: {metrics.retrieval_reason}")
    print(f"Estimated context tokens before API call: {metrics.estimated_context_tokens}")
    print(f"Estimated memory tokens: {metrics.estimated_memory_tokens}")
    print(f"Context relevance score: {metrics.relevance_score}")
    print()

    print("Comparison snapshot")
    print("-------------------")
    print(f"Naive vector searches: {naive_vector_searches}")
    print(f"Stratus ad-hoc vector searches: {actual_vector_searches}")
    print(f"Vector searches avoided: {avoided_searches}")
    print(f"Conversation turns kept active: {len(recent_messages)}")
    print()

    print("Retrieval context for the contrast question")
    print("-------------------------------------------")
    print("Recent session messages:")
    for message in recent_messages:
        print(f"- {message}")
    print("Warmed memories:")
    for memory in context.warmed_memories:
        print(f"- [{memory.type.value}] {memory.text}")
    print("Recalled memories:")
    if recalled_memories:
        for memory in recalled_memories:
            print(f"- [{memory.type.value}] {memory.text}")
    else:
        print("- (none)")
    print()

    print("OpenAI usage")
    print("------------")
    print(f"OpenAI prompt tokens: {result.prompt_tokens}")
    print(f"OpenAI completion tokens: {result.completion_tokens}")
    print(f"OpenAI total tokens: {result.total_tokens}")
    print()

    print("Assistant reflection")
    print("-------------------")
    print(result.answer)


def _unique_memories(memories):
    unique = []
    seen = set()
    for memory in memories:
        key = (memory.type.value, memory.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(memory)
    return unique


if __name__ == "__main__":
    main()

