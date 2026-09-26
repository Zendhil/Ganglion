"""
Unit tests for task decomposition.
"""
import json
# import pytest
from ganglion.agents_hub.agents import HeadAgent, CodeAgent, ReviewAgent, TailAgent
from ganglion.agents_hub.core import AgentCore
from ganglion.agents_hub.models import Task, TaskResult


def test_decompose_query_valid_output():
    """Test _decompose_query with valid LLM output."""
    session_id = "test_decompose"
    head = HeadAgent(session_id=session_id)

    # Mock run_task to return valid JSON
    original_run_task = head.run_task

    def mock_run_task(task):
        if task.id == "decompose":
            output = json.dumps({
                "subtasks": [
                    {
                        "id": "task1",
                        "content": "Write Python code",
                        "agent": "code_agent",
                        "depends_on": []
                    }
                ]
            })
            return TaskResult(
                task_id=task.id,
                output=output,
                success=True,
                score=0.9,
                cost_usd=0.001,
                latency_ms=500,
                model_used="ollama/llama3.1:latest"
            )
        return original_run_task(task)

    head.run_task = mock_run_task

    result = head._decompose_query("Write a function")

    assert len(result) == 1
    assert isinstance(result[0], Task)
    assert result[0].id == "task1"
    assert result[0].metadata["agent"] == "code_agent"
    assert result[0].metadata["status"] == "pending"
    assert result[0].metadata["depends_on"] == []


def test_validate_subtasks_invalid_agent():
    """Test that invalid agents fall back to head_agent."""
    session_id = "test_validate"
    head = HeadAgent(session_id=session_id)

    subtasks = [
        {
            "id": "task1",
            "content": "Do something",
            "agent": "nonexistent_agent",
            "depends_on": []
        }
    ]

    validated = head._validate_subtasks(subtasks)

    assert len(validated) == 1
    assert isinstance(validated[0], Task)
    assert validated[0].metadata["agent"] == "head_agent"  # Fallback
    assert validated[0].metadata["status"] == "pending"


def test_validate_subtasks_missing_fields():
    """Test that missing fields are filled with defaults."""
    session_id = "test_missing"
    head = HeadAgent(session_id=session_id)

    subtasks = [
        {
            "content": "Task without id or agent"
        }
    ]

    validated = head._validate_subtasks(subtasks)

    assert len(validated) == 1
    assert isinstance(validated[0], Task)
    assert validated[0].id.startswith("task_")
    assert validated[0].metadata["agent"] == "head_agent"
    assert validated[0].metadata["depends_on"] == []
    assert validated[0].metadata["status"] == "pending"


def test_validate_subtasks_empty_content():
    """Test that tasks with empty content are skipped."""
    session_id = "test_empty"
    head = HeadAgent(session_id=session_id)

    subtasks = [
        {
            "id": "task1",
            "content": "",
            "agent": "code_agent",
            "depends_on": []
        },
        {
            "id": "task2",
            "content": "Valid task",
            "agent": "code_agent",
            "depends_on": []
        }
    ]

    validated = head._validate_subtasks(subtasks)

    assert len(validated) == 1
    assert validated[0].id == "task2"


def test_decompose_query_markdown_wrapped():
    """Test handling of markdown-wrapped JSON."""
    session_id = "test_markdown"
    head = HeadAgent(session_id=session_id)

    original_run_task = head.run_task

    def mock_run_task(task):
        if task.id == "decompose":
            output = '''```json
{
    "subtasks": [
        {"id": "t1", "content": "Test task", "agent": "code_agent", "depends_on": []}
    ]
}
```'''
            return TaskResult(
                task_id=task.id,
                output=output,
                success=True,
                score=0.9,
                cost_usd=0.001,
                latency_ms=500,
                model_used="ollama/llama3.1:latest"
            )
        return original_run_task(task)

    head.run_task = mock_run_task

    result = head._decompose_query("Test query")

    assert len(result) == 1
    assert isinstance(result[0], Task)
    assert result[0].id == "t1"
    assert result[0].metadata["status"] == "pending"


def test_decompose_query_invalid_json():
    """Test that invalid JSON returns empty list."""
    session_id = "test_invalid"
    head = HeadAgent(session_id=session_id)

    original_run_task = head.run_task

    def mock_run_task(task):
        if task.id == "decompose":
            return TaskResult(
                task_id=task.id,
                output="This is not valid JSON",
                success=True,
                score=0.5,
                cost_usd=0.001,
                latency_ms=500,
                model_used="ollama/llama3.1:latest"
            )
        return original_run_task(task)

    head.run_task = mock_run_task

    result = head._decompose_query("Test query")

    assert result == []


def test_validate_subtasks_with_valid_agents():
    """Test that valid agent names are preserved."""
    session_id = "test_valid_agents"
    head = HeadAgent(session_id=session_id)
    code = CodeAgent(session_id=session_id)
    review = ReviewAgent(session_id=session_id)

    subtasks = [
        {
            "id": "code_task",
            "content": "Write code",
            "agent": "code_agent",
            "depends_on": []
        },
        {
            "id": "review_task",
            "content": "Review code",
            "agent": "review_agent",
            "depends_on": ["code_task"]
        }
    ]

    validated = head._validate_subtasks(subtasks)

    assert len(validated) == 2
    assert isinstance(validated[0], Task)
    assert isinstance(validated[1], Task)
    assert validated[0].metadata["agent"] == "code_agent"
    assert validated[1].metadata["agent"] == "review_agent"
    assert validated[0].metadata["status"] == "pending"
    assert validated[1].metadata["status"] == "pending"
    assert validated[1].metadata["depends_on"] == ["code_task"]


def test_default_model_metadata():
    """Test that decompose task uses default_model metadata."""
    session_id = "test_default_model"
    head = HeadAgent(session_id=session_id)

    # Track the task passed to run_task
    captured_task = None

    original_run_task = head.run_task

    def mock_run_task(task):
        nonlocal captured_task
        if task.id == "decompose":
            captured_task = task
            return TaskResult(
                task_id=task.id,
                output=json.dumps({"subtasks": []}),
                success=True,
                score=0.9,
                cost_usd=0.001,
                latency_ms=500,
                model_used="ollama/llama3.1:latest"
            )
        return original_run_task(task)

    head.run_task = mock_run_task

    head._decompose_query("wrute a pyton code to add 2 numbers")

    assert captured_task is not None
    assert "default_model" in captured_task.metadata
    assert captured_task.metadata["default_model"] == "ollama/llama3.1:latest"

def test_real_llm_decomposition():
    """Test with real LLM to verify prompt assigns correct agents."""
    session_id = "test_real_llm"
    head = HeadAgent(session_id=session_id)
    code = CodeAgent(session_id=session_id)
    review = ReviewAgent(session_id=session_id)

    print("\n=== Testing Real LLM Decomposition ===")

    # Give it a coding task that should decompose into code + review
    query = "Write a Python function to calculate the factorial of a number and review it"
    print(f"Query: {query}")

    subtasks = head._decompose_query(query)

    print(f"\nGenerated {len(subtasks)} subtasks:")
    for i, task in enumerate(subtasks, 1):
        print(f"\nSubtask {i}:")
        print(f"  ID: {task.id}")
        print(f"  Content: {task.content}")
        print(f"  Agent: {task.metadata.get('agent', 'UNKNOWN')}")
        print(f"  Status: {task.metadata.get('status', 'UNKNOWN')}")
        print(f"  Depends on: {task.metadata.get('depends_on', [])}")

    # Verify we got subtasks
    assert len(subtasks) > 0, "Should generate at least one subtask"

    # Check if code_agent is assigned
    has_code_agent = any(t.metadata.get('agent') == 'code_agent' for t in subtasks)
    print(f"\n✓ Has code_agent: {has_code_agent}")

    # Check if review_agent is assigned (optional but good to have)
    has_review_agent = any(t.metadata.get('agent') == 'review_agent' for t in subtasks)
    print(f"✓ Has review_agent: {has_review_agent}")

    # At minimum, should assign code_agent for a coding task
    if not has_code_agent:
        print("\n⚠️  WARNING: LLM did not assign code_agent!")
        print("Agent assignments:")
        for task in subtasks:
            print(f"  - {task.metadata.get('agent')}")
        print("\n💡 PROMPT MAY NEED ADJUSTMENT")
        return False

    print("\n✅ LLM correctly assigned code_agent!")
    return True


if __name__ =='__main__':
    # Run the real LLM test
    test_real_llm_decomposition()