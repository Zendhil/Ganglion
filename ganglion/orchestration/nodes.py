"""
LangGraph node wrappers for agents.

Description: Wraps AgentCore instances as LangGraph-compatible node
    functions. Each node takes state, executes agent, returns updated state.
"""

# Standard library imports
import json
import logging
from typing import Any, Dict, List

# App imports
from ganglion.agent_core import Task
from ganglion.orchestration.state import OrchestratorState, SubTask, TaskResultDict, CodeReviewState
from ganglion.orchestration.agents import CodeAgent, ReviewAgent, SearchAgent, DataAgent, get_agent

logger = logging.getLogger(__name__)


# ── Head Agent Node ───────────────────────────────────────────────────────────

def decompose_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Decompose user query into subtasks.

    Description: Uses head agent logic to break down complex queries
        into atomic subtasks with dependencies.

    :param state: Current orchestration state.
    :return: Updated state with subtasks.
    """
    logger.info("In function decompose_node: Entered")

    query = state["query"]
    session_id = state["session_id"]
    route = state.get("route", "")
    confidence = state.get("confidence", 0.0)

    # High confidence direct route - single task
    if confidence >= 0.75 and route:
        subtask = SubTask(
            id="task_1",
            content=query,
            agent=route,
            depends_on=[],
            status="pending",
        )
        logger.info(f"In function decompose_node: Direct route to {route}")
        return {
            "subtasks": [subtask],
            "current_task_id": "task_1",
        }

    # Complex query - decompose into subtasks
    # For code tasks, add review step
    if "code" in route or _is_code_task(query):
        subtasks = [
            SubTask(
                id="code_task",
                content=query,
                agent="code_agent",
                depends_on=[],
                status="pending",
            ),
            SubTask(
                id="review_task",
                content="Review the generated code for quality and correctness",
                agent="review_agent",
                depends_on=["code_task"],
                status="pending",
            ),
        ]
        logger.info("In function decompose_node: Code task with review")
    else:
        # Single task for non-code queries
        subtasks = [
            SubTask(
                id="task_1",
                content=query,
                agent=route or "search_agent",
                depends_on=[],
                status="pending",
            ),
        ]

    return {
        "subtasks": subtasks,
        "current_task_id": subtasks[0]["id"] if subtasks else "",
    }


def _is_code_task(query: str) -> bool:
    """Check if query is a coding task."""
    code_keywords = [
        "write", "code", "implement", "function", "class",
        "debug", "fix", "refactor", "test", "api", "script",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in code_keywords)


# ── Code Agent Node ───────────────────────────────────────────────────────────

def code_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Execute coding task.

    Description: Wraps CodeAgent execution as a LangGraph node.
        Updates state with code output and results.

    :param state: Current orchestration state.
    :return: Updated state with code output.
    """
    logger.info("In function code_agent_node: Entered")

    session_id = state["session_id"]
    query = state["query"]
    code_review = state.get("code_review", {})
    retry_count = code_review.get("retry_count", 0)
    feedback_history = code_review.get("feedback_history", [])

    # Build task content with feedback if retrying
    if retry_count > 0 and feedback_history:
        content = f"""Original task: {query}

Previous code was rejected. Please revise based on this feedback:
{feedback_history[-1]}

Generate improved code that addresses the issues."""
    else:
        content = query

    # Create and run agent
    agent = CodeAgent(session_id=session_id)
    task = Task(id="code_task", content=content)
    result = agent.run_task(task)

    # Build result dict
    result_dict = TaskResultDict(
        task_id=result.task_id,
        output=result.output,
        success=result.success,
        score=result.score,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )

    # Update code review state
    new_code_review = CodeReviewState(
        code_output=result.output,
        review_result=code_review.get("review_result", ""),
        review_passed=False,
        retry_count=retry_count,
        max_retries=code_review.get("max_retries", 2),
        feedback_history=feedback_history,
    )

    logger.info(
        f"In function code_agent_node: Completed with score={result.score:.2f}"
    )

    return {
        "results": {"code_task": result_dict},
        "code_review": new_code_review,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + result.cost_usd,
        "messages": [{"role": "assistant", "content": result.output}],
    }


# ── Review Agent Node ─────────────────────────────────────────────────────────

def review_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Execute code review.

    Description: Wraps ReviewAgent execution as a LangGraph node.
        Reviews code output and returns pass/fail decision.

    :param state: Current orchestration state.
    :return: Updated state with review result.
    """
    logger.info("In function review_agent_node: Entered")

    session_id = state["session_id"]
    code_review = state.get("code_review", {})
    code_output = code_review.get("code_output", "")

    # Build review task
    content = f"""Please review the following code:

{code_output}

Evaluate for correctness, quality, security, and best practices."""

    # Create and run agent
    agent = ReviewAgent(session_id=session_id)
    task = Task(id="review_task", content=content)
    result = agent.run_task(task)

    # Parse review result
    review_data = agent.parse_review_result(result.output)
    review_passed = review_data.get("passed", False)
    review_score = review_data.get("score", 0.0)

    # Build result dict
    result_dict = TaskResultDict(
        task_id=result.task_id,
        output=result.output,
        success=result.success,
        score=result.score,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )

    # Update feedback history if rejected
    feedback_history = list(code_review.get("feedback_history", []))
    if not review_passed:
        feedback = review_data.get("summary", "")
        issues = review_data.get("issues", [])
        if issues:
            feedback += "\nIssues:\n- " + "\n- ".join(issues)
        suggestions = review_data.get("suggestions", [])
        if suggestions:
            feedback += "\nSuggestions:\n- " + "\n- ".join(suggestions)
        feedback_history.append(feedback)

    # Update code review state
    new_code_review = CodeReviewState(
        code_output=code_output,
        review_result=result.output,
        review_passed=review_passed,
        retry_count=code_review.get("retry_count", 0) + (0 if review_passed else 1),
        max_retries=code_review.get("max_retries", 2),
        feedback_history=feedback_history,
    )

    logger.info(
        f"In function review_agent_node: Completed - passed={review_passed}, "
        f"score={review_score:.2f}"
    )

    return {
        "results": {"review_task": result_dict},
        "code_review": new_code_review,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + result.cost_usd,
        "messages": [{"role": "assistant", "content": result.output}],
    }


# ── Search Agent Node ─────────────────────────────────────────────────────────

def search_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Execute search task.

    Description: Wraps SearchAgent execution as a LangGraph node.

    :param state: Current orchestration state.
    :return: Updated state with search results.
    """
    logger.info("In function search_agent_node: Entered")

    session_id = state["session_id"]
    query = state["query"]

    agent = SearchAgent(session_id=session_id)
    task = Task(id="search_task", content=query)
    result = agent.run_task(task)

    result_dict = TaskResultDict(
        task_id=result.task_id,
        output=result.output,
        success=result.success,
        score=result.score,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )

    logger.info(f"In function search_agent_node: Completed with score={result.score:.2f}")

    return {
        "results": {"search_task": result_dict},
        "final_output": result.output,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + result.cost_usd,
        "messages": [{"role": "assistant", "content": result.output}],
    }


# ── Data Agent Node ───────────────────────────────────────────────────────────

def data_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Execute data task.

    Description: Wraps DataAgent execution as a LangGraph node.

    :param state: Current orchestration state.
    :return: Updated state with data results.
    """
    logger.info("In function data_agent_node: Entered")

    session_id = state["session_id"]
    query = state["query"]

    agent = DataAgent(session_id=session_id)
    task = Task(id="data_task", content=query)
    result = agent.run_task(task)

    result_dict = TaskResultDict(
        task_id=result.task_id,
        output=result.output,
        success=result.success,
        score=result.score,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )

    logger.info(f"In function data_agent_node: Completed with score={result.score:.2f}")

    return {
        "results": {"data_task": result_dict},
        "final_output": result.output,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + result.cost_usd,
        "messages": [{"role": "assistant", "content": result.output}],
    }


# ── Aggregation Node ──────────────────────────────────────────────────────────

def aggregate_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Aggregate results into final output.

    Description: Combines results from all executed tasks into
        coherent final response.

    :param state: Current orchestration state.
    :return: Updated state with final output.
    """
    logger.info("In function aggregate_node: Entered")

    results = state.get("results", {})
    code_review = state.get("code_review", {})

    # If code task, return the approved code
    if "code_task" in results:
        code_output = code_review.get("code_output", "")
        review_passed = code_review.get("review_passed", False)

        if review_passed:
            final_output = f"Code approved by review:\n\n{code_output}"
        else:
            final_output = f"Code (max retries reached):\n\n{code_output}"

        return {"final_output": final_output}

    # For other tasks, combine outputs
    outputs = []
    for task_id, result in results.items():
        if result.get("output"):
            outputs.append(f"## {task_id}\n{result['output']}")

    final_output = "\n\n".join(outputs) if outputs else "No results generated."

    logger.info(f"In function aggregate_node: Aggregated {len(results)} results")

    return {"final_output": final_output}
