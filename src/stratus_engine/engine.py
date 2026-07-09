from __future__ import annotations

from stratus_engine.extractor import HeuristicMemoryExtractor
from stratus_engine.memory_store import InMemoryLongTermMemoryStore, LongTermMemoryStore
from stratus_engine.models import ContextBundle, Message, Role
from stratus_engine.session_store import InMemorySessionStore, SessionStore


DEFAULT_WARMUP_QUERIES = [
    "user profile and durable facts",
    "preferences",
    "current projects",
    "goals",
    "open todos and tasks",
    "recent events and deadlines",
]


class StratusEngine:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        memory_store: LongTermMemoryStore,
        extractor: HeuristicMemoryExtractor | None = None,
        recent_message_limit: int = 12,
        recall_limit: int = 5,
    ) -> None:
        self.session_store = session_store
        self.memory_store = memory_store
        self.extractor = extractor or HeuristicMemoryExtractor()
        self.recent_message_limit = recent_message_limit
        self.recall_limit = recall_limit

    @classmethod
    def local(cls) -> "StratusEngine":
        return cls(
            session_store=InMemorySessionStore(),
            memory_store=InMemoryLongTermMemoryStore(),
        )

    def create_session(self, user_id: str, *, title: str = ""):
        return self.session_store.create_session(user_id=user_id, title=title)

    def append_user_message(self, session_id: str, content: str) -> Message:
        return self._append_message(session_id, Role.USER, content)

    def append_assistant_message(self, session_id: str, content: str) -> Message:
        return self._append_message(session_id, Role.ASSISTANT, content)

    def extract_memories(self, session_id: str) -> list[str]:
        session = self.session_store.get_session(session_id)
        messages = self.session_store.list_messages(session_id)
        memories = self.extractor.extract(user_id=session.user_id, messages=messages)
        promoted_ids = []
        for memory in memories:
            promoted = self.memory_store.upsert(memory)
            promoted_ids.append(promoted.id)
        return promoted_ids

    def reopen_session(
        self,
        session_id: str,
        *,
        warmup_queries: list[str] | None = None,
    ) -> list[str]:
        session = self.session_store.get_session(session_id)
        warmed_ids: list[str] = []
        for query in warmup_queries or DEFAULT_WARMUP_QUERIES:
            memories = self.memory_store.search(
                user_id=session.user_id,
                query=query,
                limit=self.recall_limit,
            )
            for memory in memories:
                if memory.id not in warmed_ids:
                    warmed_ids.append(memory.id)
        self.session_store.set_warmed_memory_ids(session_id, warmed_ids)
        return warmed_ids

    def build_context(self, session_id: str, prompt: str) -> ContextBundle:
        session = self.session_store.get_session(session_id)
        recent_messages = self.session_store.list_messages(
            session_id,
            limit=self.recent_message_limit,
        )
        warmed_memories = self.memory_store.get_many(session.warmed_memory_ids)

        if self._should_recall(prompt):
            recalled_memories = self.memory_store.search(
                user_id=session.user_id,
                query=prompt,
                limit=self.recall_limit,
            )
            reason = "ad_hoc_recall"
        else:
            recalled_memories = []
            reason = "session_context_only"

        return ContextBundle(
            session=session,
            recent_messages=recent_messages,
            warmed_memories=warmed_memories,
            recalled_memories=recalled_memories,
            retrieval_reason=reason,
        )

    def summarize_session(self, session_id: str) -> str:
        messages = self.session_store.list_messages(session_id)
        recent_user_turns = [
            message.content
            for message in messages
            if message.role == Role.USER
        ][-3:]
        summary = " ".join(recent_user_turns)
        self.session_store.update_summary(session_id, summary)
        return summary

    def _append_message(self, session_id: str, role: Role, content: str) -> Message:
        message = Message(session_id=session_id, role=role, content=content)
        self.session_store.append_message(message)
        return message

    def _should_recall(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(
            marker in lowered
            for marker in (
                "remember",
                "previous",
                "before",
                "preference",
                "what do i",
                "what did i",
                "my project",
                "my goal",
            )
        )

