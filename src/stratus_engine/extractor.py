from __future__ import annotations

import re

from pydantic import BaseModel, Field

from stratus_engine.models import Memory, MemoryType, Message, Role


class MemoryItem(BaseModel):
    type: str
    text: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class MemoryExtractionResult(BaseModel):
    memories: list[MemoryItem] = Field(default_factory=list)


class HeuristicMemoryExtractor:
    """Fast MVP extractor. Replace with an LLM extractor when quality matters."""

    def extract(self, *, user_id: str, messages: list[Message]) -> list[Memory]:
        memories: list[Memory] = []
        for message in messages:
            if message.role != Role.USER:
                continue
            memories.extend(self._extract_from_user_message(user_id=user_id, message=message))
        return memories

    def _extract_from_user_message(self, *, user_id: str, message: Message) -> list[Memory]:
        text = " ".join(message.content.split())
        lowered = text.lower()
        candidates: list[tuple[MemoryType, str, float]] = []

        preference = re.search(
            r"\b(?:i prefer|i like|i love|my favorite|my favourite)\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if preference:
            candidates.append(
                (
                    MemoryType.PREFERENCE,
                    f"User prefers {preference.group(1).rstrip('.')}.",
                    0.82,
                )
            )

        if any(marker in lowered for marker in ("my project", "we are building", "i am building")):
            candidates.append((MemoryType.PROJECT, text, 0.78))

        if any(marker in lowered for marker in ("my goal", "i want to", "i need to")):
            candidates.append((MemoryType.GOAL, text, 0.74))

        if any(marker in lowered for marker in ("todo", "remind me", "i should", "we should")):
            candidates.append((MemoryType.TODO, text, 0.7))

        if any(marker in lowered for marker in ("meeting", "interview", "deadline", "tomorrow")):
            candidates.append((MemoryType.EVENT, text, 0.68))

        return [
            Memory(
                user_id=user_id,
                text=memory_text,
                type=memory_type,
                source_session_id=message.session_id,
                importance=importance,
                confidence=0.72,
                metadata={"source_message_id": message.id},
            )
            for memory_type, memory_text, importance in candidates
        ]


class OpenAIMemoryExtractor:
    """LLM-backed extractor for higher-quality memory promotion."""

    def __init__(self, *, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIMemoryExtractor requires `pip install -e \".[openai]\"`."
            ) from exc

        self.client = OpenAI()
        self.model = model

    def extract(self, *, user_id: str, messages: list[Message]) -> list[Memory]:
        user_messages = [message for message in messages if message.role == Role.USER]
        if not user_messages:
            return []

        transcript = "\n".join(
            f"{message.role.value}: {message.content}" for message in messages
        )
        response = self.client.responses.parse(
            model=self.model,
            instructions=(
                "You extract durable memory facts from conversations. Return valid JSON only."
            ),
            input=(
                "Extract concise durable memories from the conversation below. "
                "Return JSON with a top-level 'memories' array. Each item must include "
                "'type', 'text', 'importance', and 'confidence'. Use only these types: "
                "preference, project, goal, todo, event, profile, summary. Only include "
                "durable facts that would be useful later.\n\n"
                f"Conversation:\n{transcript}"
            ),
            text_format=MemoryExtractionResult,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            return []

        memories: list[Memory] = []
        for item in parsed.memories:
            text = item.text.strip()
            if not text:
                continue
            memories.append(
                Memory(
                    user_id=user_id,
                    text=text,
                    type=_parse_memory_type(item.type),
                    source_session_id=user_messages[-1].session_id,
                    importance=item.importance,
                    confidence=item.confidence,
                    metadata={"source": "openai_extractor"},
                )
            )

        return memories


def _parse_memory_type(value: str) -> MemoryType:
    try:
        return MemoryType(value.strip().lower())
    except ValueError:
        return MemoryType.PROFILE

