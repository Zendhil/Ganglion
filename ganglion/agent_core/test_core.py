"""
Test script for agent-core functionality.

Description: Tests AgentCore base class with a simple test agent.
    Verifies metrics, escalation, and basic execution flow.

Usage:
    python -m ganglion.agent_core.test_core
"""

# Standard library imports
import logging
from unittest.mock import patch, MagicMock

# App imports
from ganglion.agent_core.core import AgentCore
from ganglion.agent_core.models import Task, TaskResult
from ganglion.agent_core.memory import InMemoryStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Test Agent Implementation ─────────────────────────────────────────────────

class TestAgent(AgentCore):
    """
    Simple test agent for verification.
    """

    @property
    def system_prompt(self) -> str:
        return "You are a helpful test agent."

    @property
    def tools(self) -> list:
        return []

    def score_output(self, task: Task, output: str) -> float:
        # Simple scoring: check if output contains expected keyword
        if "success" in output.lower():
            return 0.9
        elif "partial" in output.lower():
            return 0.5
        return 0.3


# ── Test Functions ────────────────────────────────────────────────────────────

def test_agent_initialization() -> bool:
    """
    Test agent initialization.
    """
    print("\n--- Test: Agent Initialization ---")

    agent = TestAgent(
        agent_id="test_agent",
        session_id="test_session",
        interleaved_thinking=False,
    )

    assert agent.agent_id == "test_agent"
    assert agent.session_id == "test_session"
    assert agent.interleaved_thinking is False
    assert agent.metrics.tasks_total == 0

    print("PASS: Agent initialized correctly")
    return True


def test_model_selection() -> bool:
    """
    Test model tier selection logic.
    """
    print("\n--- Test: Model Selection ---")

    agent = TestAgent(agent_id="test", session_id="sess")

    # Short task -> local
    task_short = Task(id="1", content="short task")
    assert agent.select_model(task_short) == "local"

    # Medium task -> mid
    task_medium = Task(id="2", content=" ".join(["word"] * 200))
    assert agent.select_model(task_medium) == "mid"

    # Long task -> cloud
    task_long = Task(id="3", content=" ".join(["word"] * 500))
    assert agent.select_model(task_long) == "cloud"

    # Escalation
    task_escalate = Task(id="4", content="test", _escalate=True, _last_model="local")
    assert agent.select_model(task_escalate) == "mid"

    task_escalate._last_model = "mid"
    assert agent.select_model(task_escalate) == "cloud"

    print("PASS: Model selection works correctly")
    return True


def test_metrics_update() -> bool:
    """
    Test metrics tracking.
    """
    print("\n--- Test: Metrics Update ---")

    agent = TestAgent(agent_id="test", session_id="sess")

    # Simulate successful result
    result1 = TaskResult(
        task_id="1",
        output="success",
        success=True,
        score=0.9,
        cost_usd=0.001,
        latency_ms=100,
        model_used="local",
    )
    agent._update_metrics(result1)

    assert agent.metrics.tasks_total == 1
    assert agent.metrics.tasks_passed == 1
    assert agent.metrics.tasks_failed == 0
    assert agent.metrics.model_usage.get("local") == 1

    # Simulate failed result
    result2 = TaskResult(
        task_id="2",
        output="fail",
        success=False,
        score=0.3,
        cost_usd=0.002,
        latency_ms=200,
        model_used="mid",
    )
    agent._update_metrics(result2)

    assert agent.metrics.tasks_total == 2
    assert agent.metrics.tasks_passed == 1
    assert agent.metrics.tasks_failed == 1
    assert agent.metrics.pass_rate == 0.5

    print("PASS: Metrics tracking works correctly")
    return True


def test_run_task_with_mock() -> bool:
    """
    Test run_task with mocked LiteLLM.
    """
    print("\n--- Test: Run Task (Mocked) ---")

    agent = TestAgent(
        agent_id="test",
        session_id="sess",
        memory=InMemoryStore(),
    )

    # Mock litellm.completion
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a success response"

    with patch("ganglion.agent_core.core.litellm") as mock_litellm:
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.001

        task = Task(id="test_task", content="Test the agent")
        result = agent.run_task(task)

        assert result.success is True
        assert result.score == 0.9
        assert "success" in result.output.lower()
        assert agent.metrics.tasks_passed == 1

    print("PASS: Task execution works correctly")
    return True


def test_context_management() -> bool:
    """
    Test conversation context management.
    """
    print("\n--- Test: Context Management ---")

    agent = TestAgent(agent_id="test", session_id="sess")

    # Add context
    task = Task(id="1", content="Hello")
    agent._update_context(task, "Hi there!")

    assert len(agent.context) == 2
    assert agent.context[0]["role"] == "user"
    assert agent.context[1]["role"] == "assistant"

    # Clear context
    agent.clear_context()
    assert len(agent.context) == 0

    print("PASS: Context management works correctly")
    return True


def run_all_tests() -> None:
    """
    Run all test functions.
    """
    print("\n" + "=" * 70)
    print("AGENT-CORE TEST RESULTS")
    print("=" * 70)

    tests = [
        test_agent_initialization,
        test_model_selection,
        test_metrics_update,
        test_run_task_with_mock,
        test_context_management,
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
            failed += 1

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
