from __future__ import annotations

from dataclasses import dataclass

from stratus_engine.models import ContextBundle


@dataclass(frozen=True)
class ContextMetrics:
    recent_message_count: int
    warmed_memory_count: int
    recalled_memory_count: int
    estimated_context_tokens: int
    estimated_session_only_tokens: int
    estimated_memory_tokens: int
    relevance_score: float
    retrieval_reason: str

    @property
    def context_expansion_ratio(self) -> float:
        if self.estimated_session_only_tokens == 0:
            return 1.0
        return round(self.estimated_context_tokens / self.estimated_session_only_tokens, 3)


def analyze_context(prompt: str, context: ContextBundle) -> ContextMetrics:
    recent_text = "\n".join(
        f"{message.role.value}: {message.content}"
        for message in context.recent_messages
    )
    memory_text = "\n".join(
        memory.text for memory in [*context.warmed_memories, *context.recalled_memories]
    )
    summary_text = context.session.summary

    session_tokens = estimate_tokens(f"{summary_text}\n{recent_text}")
    memory_tokens = estimate_tokens(memory_text)
    relevance = _relevance(prompt, memory_text)

    return ContextMetrics(
        recent_message_count=len(context.recent_messages),
        warmed_memory_count=len(context.warmed_memories),
        recalled_memory_count=len(context.recalled_memories),
        estimated_context_tokens=session_tokens + memory_tokens,
        estimated_session_only_tokens=session_tokens,
        estimated_memory_tokens=memory_tokens,
        relevance_score=relevance,
        retrieval_reason=context.retrieval_reason,
    )


def estimate_tokens(text: str, *, model: str = "gpt-4o-mini") -> int:
    try:
        import tiktoken
    except ImportError:
        return max(1, len(text.split()) + (len(text) // 24))

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _relevance(prompt: str, memory_text: str) -> float:
    prompt_terms = _terms(prompt)
    memory_terms = _terms(memory_text)
    if not prompt_terms or not memory_terms:
        return 0.0
    return round(len(prompt_terms & memory_terms) / len(prompt_terms), 3)


def _terms(text: str) -> set[str]:
    aliases = {
        "preferences": "preference",
        "prefers": "preference",
        "prefer": "preference",
        "projects": "project",
        "goals": "goal",
        "tasks": "todo",
    }
    terms = set()
    for raw in text.split():
        token = raw.strip(".,!?;:()[]{}'\"").lower()
        token = aliases.get(token, token)
        if len(token) > 2:
            terms.add(token)
    return terms

