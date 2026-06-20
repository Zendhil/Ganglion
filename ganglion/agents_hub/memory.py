"""
Memory interface and stub implementation.

Description: Defines abstract memory interface for agent persistence.
    MemoryStub is a no-op placeholder — swap implementation later
    (Chroma/Qdrant/pgvector) without touching agent code.
"""

# Standard library imports
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ── Abstract Memory Interface ─────────────────────────────────────────────────

class AbstractMemory(ABC):
    """
    Abstract interface for agent memory.

    Description: Defines read/write/search contract. All agents use this
        interface — swap backend implementation without code changes.
    """

    @abstractmethod
    def write(self, key: str, value: Any) -> None:
        """
        Store a value by key.

        :param key: Storage key (typically task_id).
        :param value: Value to store (typically TaskResult).
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.

        :param key: Storage key.
        :return: Stored value or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """
        Semantic search across stored values.

        :param query: Search query string.
        :param top_k: Maximum results to return.
        :return: List of matching values.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored values.
        """
        raise NotImplementedError


# ── Memory Stub Implementation ────────────────────────────────────────────────

class MemoryStub(AbstractMemory):
    """
    No-op memory placeholder.

    Description: Implements AbstractMemory interface with no-op methods.
        Use during development until real memory backend is ready.
        Swap to Chroma/Qdrant/pgvector without changing agent code.
    """

    def __init__(self):
        logger.info("In class MemoryStub, function __init__: Entered")
        logger.warning(
            "In class MemoryStub, function __init__: Using stub memory - "
            "data will not be persisted"
        )

    def write(self, key: str, value: Any) -> None:
        """
        No-op write.

        :param key: Storage key (ignored).
        :param value: Value to store (ignored).
        """
        pass

    def read(self, key: str) -> Optional[Any]:
        """
        No-op read.

        :param key: Storage key (ignored).
        :return: Always None.
        """
        return None

    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """
        No-op search.

        :param query: Search query (ignored).
        :param top_k: Max results (ignored).
        :return: Always empty list.
        """
        return []

    def clear(self) -> None:
        """
        No-op clear.
        """
        pass


# ── In-Memory Implementation (for testing) ────────────────────────────────────

class InMemoryStore(AbstractMemory):
    """
    Simple in-memory storage for testing.

    Description: Stores values in a dictionary. No semantic search —
        search returns all values containing query substring.
    """

    def __init__(self):
        logger.info("In class InMemoryStore, function __init__: Entered")
        self._store: dict = {}

    def write(self, key: str, value: Any) -> None:
        """
        Store value by key.

        :param key: Storage key.
        :param value: Value to store.
        """
        self._store[key] = value
        logger.info(f"In class InMemoryStore, function write: Stored key={key}")

    def read(self, key: str) -> Optional[Any]:
        """
        Retrieve value by key.

        :param key: Storage key.
        :return: Stored value or None.
        """
        value = self._store.get(key)
        logger.info(f"In class InMemoryStore, function read: key={key}, found={value is not None}")
        return value

    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """
        Simple substring search (not semantic).

        :param query: Search query.
        :param top_k: Max results.
        :return: Values with keys containing query.
        """
        results = []
        for key, value in self._store.items():
            if query.lower() in key.lower():
                results.append(value)
                if len(results) >= top_k:
                    break
        logger.info(f"In class InMemoryStore, function search: query={query}, found={len(results)}")
        return results

    def clear(self) -> None:
        """
        Clear all stored values.
        """
        count = len(self._store)
        self._store.clear()
        logger.info(f"In class InMemoryStore, function clear: Cleared {count} entries")
