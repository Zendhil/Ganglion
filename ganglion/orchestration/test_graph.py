"""
Test script for LangGraph orchestration.

Description: Tests the orchestration graph with mocked agent calls.

⚠️ TODO: These tests need to be updated for the new architecture:
- router.py has been deleted (routing moved to QueryRouter + HeadAgent)
- nodes.py doesn't exist (agents are nodes directly)
- Orchestrator now requires session_id parameter
- New state fields: specialist_available, handled_by, warning, next_agent

Usage:
    python -m ganglion.orchestration.test_graph
"""

# Standard library imports
import logging
from unittest.mock import patch, MagicMock

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# NOTE: Tests below are currently broken and need updating


def test_state_creation():
    """Test initial state creation."""
    print("\n--- Test: State Creation ---")

    from ganglion.orchestration.state import create_initial_state

    state = create_initial_state(
        query="Write a Python function",
        session_id="test_session",
        route="code_agent",
        confidence=0.85,
    )

    assert state["query"] == "Write a Python function"
    assert state["session_id"] == "test_session"
    assert state["route"] == "code_agent"
    assert state["confidence"] == 0.85
    assert state["subtasks"] == []
    assert state["results"] == {}

    print("PASS: State created correctly")
    return True


def test_edge_routing():
    """Test edge routing logic."""
    print("\n--- Test: Edge Routing ---")

    from ganglion.orchestration.state import create_initial_state
    from ganglion.orchestration.router import route_to_agent, should_continue_review

    # High confidence code route
    state1 = create_initial_state(
        query="test",
        session_id="test",
        route="code_agent",
        confidence=0.85,
    )
    assert route_to_agent(state1) == "code_agent"

    # Low confidence - decompose
    state2 = create_initial_state(
        query="test",
        session_id="test",
        route="code_agent",
        confidence=0.3,
    )
    assert route_to_agent(state2) == "decompose"

    # Review passed - aggregate
    state3 = create_initial_state(query="test", session_id="test")
    state3["code_review"]["review_passed"] = True
    assert should_continue_review(state3) == "aggregate"

    # Review failed, can retry
    state4 = create_initial_state(query="test", session_id="test")
    state4["code_review"]["review_passed"] = False
    state4["code_review"]["retry_count"] = 0
    state4["code_review"]["max_retries"] = 2
    assert should_continue_review(state4) == "code_agent"

    # Review failed, max retries reached
    state5 = create_initial_state(query="test", session_id="test")
    state5["code_review"]["review_passed"] = False
    state5["code_review"]["retry_count"] = 2
    state5["code_review"]["max_retries"] = 2
    assert should_continue_review(state5) == "aggregate"

    print("PASS: Edge routing works correctly")
    return True


def test_decompose_node():
    """Test decomposition node."""
    print("\n--- Test: Decompose Node ---")

    from ganglion.orchestration.state import create_initial_state
    from ganglion.orchestration.nodes import decompose_node

    # High confidence direct route
    state1 = create_initial_state(
        query="Write a function",
        session_id="test",
        route="code_agent",
        confidence=0.85,
    )
    result1 = decompose_node(state1)
    assert len(result1["subtasks"]) == 1
    assert result1["subtasks"][0]["agent"] == "code_agent"

    # Code task gets review step
    state2 = create_initial_state(
        query="Write a function",
        session_id="test",
        route="code_agent",
        confidence=0.5,
    )
    result2 = decompose_node(state2)
    assert len(result2["subtasks"]) == 2
    assert result2["subtasks"][0]["agent"] == "code_agent"
    assert result2["subtasks"][1]["agent"] == "review_agent"

    print("PASS: Decompose node works correctly")
    return True


def test_graph_compilation():
    """Test that graph compiles without errors."""
    print("\n--- Test: Graph Compilation ---")

    from ganglion.orchestration.orchestrator import build_orchestration_graph, build_parallel_graph

    # Test main graph
    graph1 = build_orchestration_graph()
    assert graph1 is not None

    # Test parallel graph
    graph2 = build_parallel_graph()
    assert graph2 is not None

    print("PASS: Graphs compile successfully")
    return True


def test_orchestrator_with_mock():
    """Test orchestrator with mocked LLM calls."""
    print("\n--- Test: Orchestrator (Mocked) ---")

    from ganglion.orchestration.orchestrator import Orchestrator
    from ganglion.orchestration.state import create_initial_state

    # Mock litellm
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """```python
def hello():
    return "Hello, World!"
```"""

    mock_review_response = MagicMock()
    mock_review_response.choices = [MagicMock()]
    mock_review_response.choices[0].message.content = """{
        "passed": true,
        "score": 0.9,
        "summary": "Code looks good",
        "issues": [],
        "suggestions": []
    }"""

    call_count = [0]

    def mock_completion(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return mock_response
        return mock_review_response

    with patch("ganglion.agents_hub.core.litellm") as mock_litellm:
        mock_litellm.completion.side_effect = mock_completion
        mock_litellm.completion_cost.return_value = 0.001

        orchestrator = Orchestrator()

        # Run with high confidence code route
        result = orchestrator.run(
            query="Write a hello world function",
            session_id="test_session",
            route="code_agent",
            confidence=0.85,
        )

        assert result is not None
        assert "final_output" in result
        assert result.get("total_cost_usd", 0) > 0

    print("PASS: Orchestrator runs correctly with mocks")
    return True


def test_code_review_loop():
    """Test code → review retry loop."""
    print("\n--- Test: Code Review Loop ---")

    from ganglion.orchestration.router import should_continue_review
    from ganglion.orchestration.state import create_initial_state, CodeReviewState

    # Simulate retry loop states
    states = []

    # Initial state - should retry
    state1 = create_initial_state(query="test", session_id="test")
    state1["code_review"] = CodeReviewState(
        code_output="bad code",
        review_result="rejected",
        review_passed=False,
        retry_count=0,
        max_retries=2,
        feedback_history=["Fix indentation"],
    )
    assert should_continue_review(state1) == "code_agent"

    # After first retry - should retry again
    state2 = create_initial_state(query="test", session_id="test")
    state2["code_review"] = CodeReviewState(
        code_output="better code",
        review_result="still issues",
        review_passed=False,
        retry_count=1,
        max_retries=2,
        feedback_history=["Fix indentation", "Add docs"],
    )
    assert should_continue_review(state2) == "code_agent"

    # After max retries - should aggregate
    state3 = create_initial_state(query="test", session_id="test")
    state3["code_review"] = CodeReviewState(
        code_output="final code",
        review_result="still not perfect",
        review_passed=False,
        retry_count=2,
        max_retries=2,
        feedback_history=["Fix indentation", "Add docs"],
    )
    assert should_continue_review(state3) == "aggregate"

    # Review passed - should aggregate
    state4 = create_initial_state(query="test", session_id="test")
    state4["code_review"] = CodeReviewState(
        code_output="good code",
        review_result="approved",
        review_passed=True,
        retry_count=1,
        max_retries=2,
        feedback_history=[],
    )
    assert should_continue_review(state4) == "aggregate"

    print("PASS: Code review loop logic works correctly")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LANGGRAPH ORCHESTRATION TEST RESULTS")
    print("=" * 70)

    tests = [
        test_state_creation,
        test_edge_routing,
        test_decompose_node,
        test_graph_compilation,
        test_code_review_loop,
        test_orchestrator_with_mock,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} raised {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
