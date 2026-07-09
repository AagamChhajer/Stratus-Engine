from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    PROJECT = "project"
    GOAL = "goal"
    TODO = "todo"
    EVENT = "event"
    PROFILE = "profile"
    SUMMARY = "summary"


@dataclass(frozen=True)
class Message:
    session_id: str
    role: Role
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            role=Role(data["role"]),
            content=data["content"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    id: str
    user_id: str
    title: str = ""
    summary: str = ""
    warmed_memory_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    user_id: str
    text: str
    type: MemoryType
    source_session_id: str
    importance: float
    confidence: float
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_accessed_at: datetime | None = None
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_score(self) -> float:
        return round((self.importance * 0.65) + (self.confidence * 0.35), 4)

    def mark_accessed(self) -> None:
        self.access_count += 1
        self.last_accessed_at = utc_now()


@dataclass(frozen=True)
class ContextBundle:
    session: Session
    recent_messages: list[Message]
    warmed_memories: list[Memory]
    recalled_memories: list[Memory]
    retrieval_reason: str

    def as_prompt_sections(self) -> dict[str, list[str] | str]:
        return {
            "session_summary": self.session.summary,
            "recent_messages": [
                f"{message.role.value}: {message.content}"
                for message in self.recent_messages
            ],
            "memories": [
                f"[{memory.type.value}] {memory.text}"
                for memory in [*self.warmed_memories, *self.recalled_memories]
            ],
        }

