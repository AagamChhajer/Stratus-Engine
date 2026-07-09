from __future__ import annotations

from typing import Protocol

from stratus_engine.models import Memory, MemoryType


class LongTermMemoryStore(Protocol):
    def upsert(self, memory: Memory) -> Memory: ...
    def search(self, *, user_id: str, query: str, limit: int = 5) -> list[Memory]: ...
    def get_many(self, memory_ids: list[str]) -> list[Memory]: ...


class InMemoryLongTermMemoryStore:
    """Keyword-ranked placeholder for the later vector DB adapter."""

    def __init__(self) -> None:
        self.memories: dict[str, Memory] = {}

    def upsert(self, memory: Memory) -> Memory:
        duplicate = self._find_duplicate(memory)
        if duplicate:
            duplicate.importance = max(duplicate.importance, memory.importance)
            duplicate.confidence = max(duplicate.confidence, memory.confidence)
            duplicate.metadata = {**duplicate.metadata, **memory.metadata}
            return duplicate
        self.memories[memory.id] = memory
        return memory

    def search(self, *, user_id: str, query: str, limit: int = 5) -> list[Memory]:
        query_terms = _terms(query)
        ranked: list[tuple[float, Memory]] = []
        for memory in self.memories.values():
            if memory.user_id != user_id:
                continue
            haystack = f"{memory.type.value} {memory.text}"
            overlap = len(query_terms & _terms(haystack))
            if overlap == 0:
                continue
            score = overlap + memory.rank_score + (memory.access_count * 0.03)
            ranked.append((score, memory))

        results = [memory for _, memory in sorted(ranked, reverse=True, key=lambda item: item[0])[:limit]]
        for memory in results:
            memory.mark_accessed()
        return results

    def get_many(self, memory_ids: list[str]) -> list[Memory]:
        return [self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories]

    def _find_duplicate(self, candidate: Memory) -> Memory | None:
        candidate_terms = _terms(candidate.text)
        for memory in self.memories.values():
            if memory.user_id != candidate.user_id or memory.type != candidate.type:
                continue
            union = candidate_terms | _terms(memory.text)
            if not union:
                continue
            similarity = len(candidate_terms & _terms(memory.text)) / len(union)
            if similarity >= 0.72:
                return memory
        return None


def _terms(text: str) -> set[str]:
    aliases = {
        "preferences": "preference",
        "prefers": "preference",
        "prefer": "preference",
        "projects": "project",
        "goals": "goal",
        "tasks": "todo",
        "todos": "todo",
        "events": "event",
    }
    terms = set()
    for raw in text.split():
        token = raw.strip(".,!?;:()[]{}'\"").lower()
        token = aliases.get(token, token)
        if len(token) > 2:
            terms.add(token)
    return terms

