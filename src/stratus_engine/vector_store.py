from __future__ import annotations

from typing import Any

from stratus_engine.embeddings import EmbeddingProvider, HashEmbeddingProvider
from stratus_engine.models import Memory, MemoryType


class ChromaLongTermMemoryStore:
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "stratus_memories",
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaLongTermMemoryStore requires `pip install -e \".[vector]\"`."
            ) from exc

        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Stratus Engine long-term memories"},
        )

    def upsert(self, memory: Memory) -> Memory:
        embedding = self.embedding_provider.embed_texts([_memory_document(memory)])[0]
        self.collection.upsert(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.text],
            metadatas=[_memory_metadata(memory)],
        )
        return memory

    def search(self, *, user_id: str, query: str, limit: int = 5) -> list[Memory]:
        embedding = self.embedding_provider.embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        memories = [
            _memory_from_chroma(
                memory_id=memory_id,
                document=document,
                metadata=metadata,
                distance=distance,
            )
            for memory_id, document, metadata, distance in zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
                strict=False,
            )
        ]
        for memory in memories:
            memory.mark_accessed()
        return memories

    def get_many(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        results = self.collection.get(ids=memory_ids, include=["documents", "metadatas"])
        return [
            _memory_from_chroma(memory_id=memory_id, document=document, metadata=metadata)
            for memory_id, document, metadata in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
                strict=False,
            )
        ]


def _memory_document(memory: Memory) -> str:
    return f"{memory.type.value}: {memory.text}"


def _memory_metadata(memory: Memory) -> dict[str, Any]:
    return {
        "user_id": memory.user_id,
        "type": memory.type.value,
        "source_session_id": memory.source_session_id,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "created_at": memory.created_at.isoformat(),
    }


def _memory_from_chroma(
    *,
    memory_id: str,
    document: str,
    metadata: dict[str, Any],
    distance: float | None = None,
) -> Memory:
    confidence = float(metadata.get("confidence", 0.7))
    if distance is not None:
        confidence = max(0.0, min(1.0, 1.0 - float(distance)))

    return Memory(
        id=memory_id,
        user_id=str(metadata["user_id"]),
        text=document,
        type=MemoryType(str(metadata["type"])),
        source_session_id=str(metadata["source_session_id"]),
        importance=float(metadata.get("importance", 0.7)),
        confidence=confidence,
    )

