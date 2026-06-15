"""
Semantic cache for query routing results.

Description: Abstract cache interface with in-memory implementation.
    Caches RouteResult by query embedding similarity to avoid
    redundant routing computations.
"""

# Standard library imports
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
# from  sklearn.metrics.pairwise import cosine_similarity

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Third party imports
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TTL_SECONDS = 3600  # 1 hour
DEFAULT_SIMILARITY_THRESHOLD = 0.95
DEFAULT_MAX_ENTRIES = 10000


# ── Cache Entry Schema ────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """
    Single cache entry with embedding and metadata.

    :param query: Original query string.
    :param embedding: Query embedding vector.
    :param route: Cached route name.
    :param confidence: Original routing confidence.
    :param created_at: Unix timestamp of entry creation.
    :param ttl_seconds: Time-to-live in seconds.
    :param hit_count: Number of times this entry was hit.
    """

    query: str
    embedding: np.ndarray
    route: str
    confidence: float
    created_at: float
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    hit_count: int = 0

    def is_expired(self) -> bool:
        """
        Check if entry has exceeded its TTL.

        :return: True if expired, False otherwise.
        """
        return time.time() > (self.created_at + self.ttl_seconds)


@dataclass
class CacheStats:
    """
    Cache performance statistics.

    :param total_queries: Total queries processed.
    :param cache_hits: Number of cache hits.
    :param cache_misses: Number of cache misses.
    :param entries_count: Current number of entries.
    :param evictions: Number of entries evicted.
    :param avg_similarity: Average similarity score on hits.
    """

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    entries_count: int = 0
    evictions: int = 0
    avg_similarity: float = 0.0

    @property
    def hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        :return: Hit rate as float (0.0-1.0).
        """
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries


# ── Abstract Cache Interface ──────────────────────────────────────────────────

class AbstractSemanticCache(ABC):
    """
    Abstract interface for semantic caching.

    Description: Defines the contract for semantic cache implementations.
        Subclasses must implement get, put, and clear methods.
    """

    @abstractmethod
    def get(
        self,
        query: str,
        embedding: np.ndarray,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Look up query in cache by embedding similarity.

        Description: Finds the most similar cached entry above the
            similarity threshold.

        :param query: Query string (for logging/debugging).
        :param embedding: Query embedding vector.
        :return: Tuple of (route, confidence, similarity) if hit, None if miss.
        """
        raise NotImplementedError

    @abstractmethod
    def put(
        self,
        query: str,
        embedding: np.ndarray,
        route: str,
        confidence: float,
    ) -> None:
        """
        Store query result in cache.

        :param query: Query string.
        :param embedding: Query embedding vector.
        :param route: Route result to cache.
        :param confidence: Routing confidence score.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """
        Get cache performance statistics.

        :return: CacheStats object with current metrics.
        """
        raise NotImplementedError


# ── In-Memory Implementation ──────────────────────────────────────────────────

class InMemorySemanticCache(AbstractSemanticCache):
    """
    In-memory semantic cache implementation.

    Description: Stores cache entries in memory with cosine similarity
        lookup. Supports TTL expiration and LRU-style eviction.

    :param similarity_threshold: Minimum similarity for cache hit (0.0-1.0).
    :param ttl_seconds: Default TTL for cache entries.
    :param max_entries: Maximum number of entries before eviction.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        logger.info("In class InMemorySemanticCache, function __init__: Entered")

        self._similarity_threshold = similarity_threshold
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: List[CacheEntry] = []
        self._stats = CacheStats()

        logger.info(
            f"In class InMemorySemanticCache, function __init__: Initialized with "
            f"threshold={similarity_threshold}, ttl={ttl_seconds}s, max_entries={max_entries}"
        )

    def get(
        self,
        query: str,
        embedding: np.ndarray,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Look up query in cache by embedding similarity.

        Description: Computes cosine similarity against all cached embeddings,
            returns best match above threshold. Evicts expired entries.

        :param query: Query string (for logging).
        :param embedding: Query embedding vector.
        :return: Tuple of (route, confidence, similarity) if hit, None if miss.
        """
        logger.info(f"In class InMemorySemanticCache, function get: Entered")

        self._stats.total_queries += 1

        # Evict expired entries
        self._evict_expired()

        if not self._entries:
            self._stats.cache_misses += 1
            logger.info("In class InMemorySemanticCache, function get: Cache empty, miss")
            return None

        # Compute similarities
        best_entry: Optional[CacheEntry] = None
        best_similarity = 0.0

        for entry in self._entries:
            similarity = self._cosine_similarity(embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        # Check threshold
        if best_entry is not None and best_similarity >= self._similarity_threshold:
            best_entry.hit_count += 1
            self._stats.cache_hits += 1
            self._update_avg_similarity(best_similarity)

            logger.info(
                f"In class InMemorySemanticCache, function get: Cache hit "
                f"(similarity={best_similarity:.3f}, route={best_entry.route})"
            )
            return (best_entry.route, best_entry.confidence, best_similarity)

        self._stats.cache_misses += 1
        logger.info(
            f"In class InMemorySemanticCache, function get: Cache miss "
            f"(best_similarity={best_similarity:.3f} < threshold={self._similarity_threshold})"
        )
        return None

    def put(
        self,
        query: str,
        embedding: np.ndarray,
        route: str,
        confidence: float,
    ) -> None:
        """
        Store query result in cache.

        Description: Creates new cache entry. Evicts oldest entries
            if max_entries exceeded.

        :param query: Query string.
        :param embedding: Query embedding vector.
        :param route: Route result to cache.
        :param confidence: Routing confidence score.
        """
        logger.info(f"In class InMemorySemanticCache, function put: Entered for route={route}")

        # Evict if at capacity
        if len(self._entries) >= self._max_entries:
            self._evict_lru()

        entry = CacheEntry(
            query=query,
            embedding=embedding,
            route=route,
            confidence=confidence,
            created_at=time.time(),
            ttl_seconds=self._ttl_seconds,
        )
        self._entries.append(entry)
        self._stats.entries_count = len(self._entries)

        logger.info(
            f"In class InMemorySemanticCache, function put: Added entry "
            f"(total entries={len(self._entries)})"
        )

    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        logger.info("In class InMemorySemanticCache, function clear: Entered")

        count = len(self._entries)
        self._entries.clear()
        self._stats.entries_count = 0

        logger.info(f"In class InMemorySemanticCache, function clear: Cleared {count} entries")

    def get_stats(self) -> CacheStats:
        """
        Get cache performance statistics.

        :return: CacheStats object with current metrics.
        """
        self._stats.entries_count = len(self._entries)
        return self._stats

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        :param a: First vector.
        :param b: Second vector.
        :return: Cosine similarity (0.0-1.0).
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _evict_expired(self) -> None:
        """
        Remove expired entries from cache.
        """
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_expired()]
        evicted = before - len(self._entries)

        if evicted > 0:
            self._stats.evictions += evicted
            logger.info(
                f"In class InMemorySemanticCache, function _evict_expired: "
                f"Evicted {evicted} expired entries"
            )

    def _evict_lru(self) -> None:
        """
        Evict least recently used entries (lowest hit count, oldest).
        """
        if not self._entries:
            return

        # Sort by hit_count (ascending), then created_at (ascending)
        self._entries.sort(key=lambda e: (e.hit_count, e.created_at))

        # Remove oldest 10% or at least 1
        evict_count = max(1, len(self._entries) // 10)
        self._entries = self._entries[evict_count:]
        self._stats.evictions += evict_count

        logger.info(
            f"In class InMemorySemanticCache, function _evict_lru: "
            f"Evicted {evict_count} LRU entries"
        )

    def _update_avg_similarity(self, similarity: float) -> None:
        """
        Update running average similarity score.

        :param similarity: New similarity value to incorporate.
        """
        hits = self._stats.cache_hits
        if hits == 1:
            self._stats.avg_similarity = similarity
        else:
            self._stats.avg_similarity = (
                (self._stats.avg_similarity * (hits - 1) + similarity) / hits
            )


# ── Factory Function ──────────────────────────────────────────────────────────

def create_cache(
    cache_type: str = "inmemory",
    **kwargs: Any,
) -> AbstractSemanticCache:
    """
    Factory function to create cache instances.

    Description: Creates cache instance based on type string.
        Currently supports 'inmemory'. Redis support planned.

    :param cache_type: Type of cache ('inmemory', 'redis' future).
    :param kwargs: Additional arguments passed to cache constructor.
    :return: Cache instance.
    :raises ValueError: If cache_type is not supported.
    """
    logger.info(f"In function create_cache: Entered with type={cache_type}")

    if cache_type == "inmemory":
        return InMemorySemanticCache(**kwargs)

    # Future: Redis implementation
    # if cache_type == "redis":
    #     return RedisSemanticCache(**kwargs)

    raise ValueError(f"Unsupported cache type: {cache_type}")
