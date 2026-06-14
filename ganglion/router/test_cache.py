"""
Test script for semantic cache functionality.

Description: Tests cache hit/miss behavior and statistics.

Usage:
    python -m ganglion.router.test_cache
"""

# Standard library imports
import logging

# App imports
from ganglion.router.routes import QueryRouter
from ganglion.router.cache import CacheStats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_cache_tests() -> None:
    """
    Run cache functionality tests.

    Description: Tests cache with repeated and similar queries,
        verifies hit/miss behavior and statistics.
    """
    logger.info("Initializing QueryRouter with cache enabled...")
    router = QueryRouter(enable_cache=True)

    print("\n" + "=" * 70)
    print("SEMANTIC CACHE TEST RESULTS")
    print("=" * 70 + "\n")

    # Test queries
    queries = [
        "Write a Python function to validate email addresses",
        "Write a Python function to validate email addresses",  # Exact duplicate
        "Write a Python function that validates email",  # Similar
        "Debug this async function",
        "Debug this async function",  # Exact duplicate
        "Find documentation for requests library",
    ]

    for i, query in enumerate(queries):
        result = router.route(query)
        print(f"Query {i + 1}: {query[:45]}...")
        print(f"  Route: {result.route}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Cache hit: {result.cache_hit}")
        print()

    # Print cache statistics
    stats = router.get_cache_stats()
    if stats:
        print("=" * 70)
        print("CACHE STATISTICS")
        print("=" * 70)
        print(f"  Total queries: {stats.total_queries}")
        print(f"  Cache hits: {stats.cache_hits}")
        print(f"  Cache misses: {stats.cache_misses}")
        print(f"  Hit rate: {stats.hit_rate:.1%}")
        print(f"  Entries count: {stats.entries_count}")
        print(f"  Avg similarity: {stats.avg_similarity:.3f}")
        print("=" * 70)


if __name__ == "__main__":
    run_cache_tests()
