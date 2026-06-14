"""
Test script for semantic routing.

Description: Tests route matching on sample queries and reports
    confidence scores and routing decisions.

Usage:
    python -m ganglion.router.test_routes
"""

# Standard library imports
import logging
import sys

# App imports
from ganglion.router.routes import QueryRouter, RouteResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Test Queries ──────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # Expected: code_agent
    ("Write a Python function to validate email addresses", "code_agent"),
    # Expected: search_agent
    ("Find documentation for the requests library", "search_agent"),
    # Expected: data_agent
    ("Write a SQL query to get all users created last month", "data_agent"),
    # Expected: code_agent
    ("Debug this async function that's not awaiting properly", "code_agent"),
    # Expected: search_agent (or head_agent if ambiguous)
    ("What are the best practices for error handling in Python", "search_agent"),
]


def run_tests() -> None:
    """
    Run routing tests on sample queries.

    Description: Initializes router, tests each query, and reports
        results with pass/fail status.
    """
    logger.info("Initializing QueryRouter...")
    router = QueryRouter()

    print("\n" + "=" * 70)
    print("SEMANTIC ROUTING TEST RESULTS")
    print("=" * 70 + "\n")

    passed = 0
    total = len(TEST_QUERIES)

    for query, expected in TEST_QUERIES:
        result = router.route(query)

        # Check if routed correctly (or to head_agent for ambiguous)
        is_correct = (
            result.route == expected
            or (result.route == "head_agent" and result.confidence < 0.75)
        )

        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            passed += 1

        print(f"Query: {query[:50]}...")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result.route} (confidence: {result.confidence:.3f})")
        print(f"  Status:   {status}")
        print()

    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} passed")
    print("=" * 70)

    # Return exit code for CI
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_tests()
