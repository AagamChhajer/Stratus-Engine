from stratus_engine import StratusEngine, analyze_context


def test_recent_session_context_does_not_trigger_recall() -> None:
    engine = StratusEngine.local()
    session = engine.create_session("u1")

    engine.append_user_message(session.id, "Explain Docker.")
    engine.append_assistant_message(session.id, "Docker packages apps into containers.")

    context = engine.build_context(session.id, "Continue explaining it.")

    assert context.retrieval_reason == "session_context_only"
    assert [message.content for message in context.recent_messages] == [
        "Explain Docker.",
        "Docker packages apps into containers.",
    ]
    assert context.recalled_memories == []


def test_extracts_memory_and_recalls_it_later() -> None:
    engine = StratusEngine.local()
    session = engine.create_session("u1")

    engine.append_user_message(session.id, "I prefer React over Angular.")
    promoted_ids = engine.extract_memories(session.id)

    context = engine.build_context(session.id, "What frontend preference do I have?")

    assert len(set(promoted_ids)) == 1
    assert context.retrieval_reason == "ad_hoc_recall"
    assert context.recalled_memories[0].text == "User prefers React over Angular."


def test_reopen_session_warms_predefined_context() -> None:
    engine = StratusEngine.local()
    session = engine.create_session("u1")

    engine.append_user_message(session.id, "My project is Stratus Engine, a memory layer for LLM apps.")
    engine.append_user_message(session.id, "I prefer MongoDB for active session storage.")
    engine.extract_memories(session.id)

    warmed_ids = engine.reopen_session(session.id)
    context = engine.build_context(session.id, "Continue where we left off.")

    assert len(warmed_ids) == 2
    assert len(context.warmed_memories) == 2
    assert context.retrieval_reason == "session_context_only"


def test_deduplicates_repeated_memories() -> None:
    engine = StratusEngine.local()
    session = engine.create_session("u1")

    engine.append_user_message(session.id, "I like Barcelona FC.")
    engine.append_user_message(session.id, "I love Barcelona FC.")
    engine.extract_memories(session.id)

    memories = list(engine.memory_store.memories.values())

    assert len(memories) == 1
    assert memories[0].text == "User prefers Barcelona FC."


def test_context_metrics_measure_retrieval_payload() -> None:
    engine = StratusEngine.local()
    session = engine.create_session("u1")

    engine.append_user_message(session.id, "I prefer PostgreSQL for analytics projects.")
    engine.extract_memories(session.id)
    engine.reopen_session(session.id)

    prompt = "What database preference do I have?"
    context = engine.build_context(session.id, prompt)
    metrics = analyze_context(prompt, context)

    assert metrics.warmed_memory_count == 1
    assert metrics.recalled_memory_count == 1
    assert metrics.estimated_context_tokens > metrics.estimated_session_only_tokens
    assert metrics.estimated_memory_tokens > 0
    assert metrics.relevance_score > 0
