from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from stratus_engine.models import Message, Role, Session, utc_now


class SessionStore(Protocol):
    def create_session(self, user_id: str, *, title: str = "") -> Session: ...
    def get_session(self, session_id: str) -> Session: ...
    def append_message(self, message: Message) -> None: ...
    def list_messages(self, session_id: str, *, limit: int | None = None) -> list[Message]: ...
    def update_summary(self, session_id: str, summary: str) -> None: ...
    def set_warmed_memory_ids(self, session_id: str, memory_ids: list[str]) -> None: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.messages: dict[str, list[Message]] = defaultdict(list)

    def create_session(self, user_id: str, *, title: str = "") -> Session:
        session = Session(id=_new_id("sess"), user_id=user_id, title=title)
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        return self.sessions[session_id]

    def append_message(self, message: Message) -> None:
        self.messages[message.session_id].append(message)
        self.sessions[message.session_id].updated_at = utc_now()

    def list_messages(self, session_id: str, *, limit: int | None = None) -> list[Message]:
        messages = self.messages[session_id]
        return messages[-limit:] if limit else list(messages)

    def update_summary(self, session_id: str, summary: str) -> None:
        session = self.get_session(session_id)
        session.summary = summary
        session.updated_at = utc_now()

    def set_warmed_memory_ids(self, session_id: str, memory_ids: list[str]) -> None:
        session = self.get_session(session_id)
        session.warmed_memory_ids = memory_ids
        session.updated_at = utc_now()


class MongoSessionStore:
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        *,
        database: str = "stratus_engine",
    ) -> None:
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError(
                "MongoSessionStore requires pymongo. Install with `pip install -e .[mongo]`."
            ) from exc

        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.sessions = self.db["sessions"]
        self.messages = self.db["messages"]
        self._ensure_indexes()

    def create_session(self, user_id: str, *, title: str = "") -> Session:
        session = Session(id=_new_id("sess"), user_id=user_id, title=title)
        self.sessions.insert_one(_session_to_doc(session))
        return session

    def get_session(self, session_id: str) -> Session:
        doc = self.sessions.find_one({"_id": session_id})
        if not doc:
            raise KeyError(f"Session not found: {session_id}")
        return _session_from_doc(doc)

    def append_message(self, message: Message) -> None:
        self.messages.insert_one({**message.to_dict(), "_id": message.id})
        self.sessions.update_one(
            {"_id": message.session_id},
            {"$set": {"updated_at": utc_now()}},
        )

    def list_messages(self, session_id: str, *, limit: int | None = None) -> list[Message]:
        cursor = self.messages.find({"session_id": session_id}).sort("created_at", 1)
        if limit:
            cursor = self.messages.find({"session_id": session_id}).sort("created_at", -1).limit(limit)
            return [Message.from_dict(doc) for doc in reversed(list(cursor))]
        return [Message.from_dict(doc) for doc in cursor]

    def update_summary(self, session_id: str, summary: str) -> None:
        self.sessions.update_one(
            {"_id": session_id},
            {"$set": {"summary": summary, "updated_at": utc_now()}},
        )

    def set_warmed_memory_ids(self, session_id: str, memory_ids: list[str]) -> None:
        self.sessions.update_one(
            {"_id": session_id},
            {"$set": {"warmed_memory_ids": memory_ids, "updated_at": utc_now()}},
        )

    def _ensure_indexes(self) -> None:
        self.sessions.create_index([("user_id", 1), ("updated_at", -1)])
        self.messages.create_index([("session_id", 1), ("created_at", 1)])
        self.messages.create_index([("session_id", 1), ("role", 1)])


def _new_id(prefix: str) -> str:
    from uuid import uuid4

    return f"{prefix}_{uuid4().hex}"


def _session_to_doc(session: Session) -> dict:
    return {
        "_id": session.id,
        "user_id": session.user_id,
        "title": session.title,
        "summary": session.summary,
        "warmed_memory_ids": session.warmed_memory_ids,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "metadata": session.metadata,
    }


def _session_from_doc(doc: dict) -> Session:
    return Session(
        id=doc["_id"],
        user_id=doc["user_id"],
        title=doc.get("title", ""),
        summary=doc.get("summary", ""),
        warmed_memory_ids=doc.get("warmed_memory_ids", []),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        metadata=doc.get("metadata", {}),
    )

