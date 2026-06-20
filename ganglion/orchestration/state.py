"""
State definitions for LangGraph orchestration.

Description: Defines the state schema that flows through the graph.
    Uses TypedDict for type safety and LangGraph compatibility.
"""

# Standard library imports
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
import operator


# ── Subtask Schema ────────────────────────────────────────────────────────────

class SubTask(TypedDict):
    """
    Single subtask in a decomposed workflow.

    :param id: Unique subtask identifier.
    :param content: Task description/prompt.
    :param agent: Target agent (code_agent, search_agent, data_agent, review_agent).
    :param depends_on: List of subtask IDs this task depends on.
    :param status: Current status (pending, in_progress, completed, failed).
    """

    id: str
    content: str
    agent: str
    depends_on: List[str]
    status: Literal["pending", "in_progress", "completed", "failed"]


# ── Task Result Schema ────────────────────────────────────────────────────────

class TaskResultDict(TypedDict):
    """
    Result of a single task execution.

    :param task_id: ID of the executed task.
    :param output: Model output.
    :param success: Whether task met quality threshold.
    :param score: Quality score (0.0-1.0).
    :param cost_usd: API cost.
    :param latency_ms: Execution time.
    :param model_used: Model tier used.
    """

    task_id: str
    output: Any
    success: bool
    score: float
    cost_usd: float
    latency_ms: float
    model_used: str


# ── Code Review State ─────────────────────────────────────────────────────────

class CodeReviewState(TypedDict):
    """
    State for code → review loop.

    :param code_output: Generated code from coding agent.
    :param review_result: Review feedback and decision.
    :param review_passed: Whether review approved the code.
    :param retry_count: Number of code revision attempts.
    :param max_retries: Maximum allowed retries.
    :param feedback_history: List of review feedback for iterations.
    """

    code_output: str
    review_result: str
    review_passed: bool
    retry_count: int
    max_retries: int
    feedback_history: List[str]


# ── Orchestration State ───────────────────────────────────────────────────────

def merge_results(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two result dictionaries.

    Description: Used as reducer for parallel branch results.

    :param left: First result dict.
    :param right: Second result dict.
    :return: Merged dictionary.
    """
    merged = left.copy()
    merged.update(right)
    return merged


def append_messages(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """
    Append messages to conversation history.

    :param left: Existing messages.
    :param right: New messages to append.
    :return: Combined message list.
    """
    return left + right


class OrchestratorState(TypedDict):
    """
    Main orchestration state that flows through the graph.

    :param query: Original user query.
    :param route: Routing decision from semantic router.
    :param confidence: Routing confidence score.
    :param specialist_available: Whether a specialist agent is available for this query.
    :param handled_by: Which agent handled the query (for tracking).
    :param warning: Warning message (e.g., fallback used).
    :param next_agent: Next agent to execute (set by head_agent).
    :param needs_decomposition: Whether query needs decomposition.
    :param subtasks: Decomposed subtasks from head agent.
    :param current_task_id: Currently executing task ID.
    :param results: Map of task_id → TaskResultDict.
    :param messages: Conversation history for context.
    :param output: Output from individual agent (before aggregation).
    :param final_output: Aggregated final response.
    :param error: Error message if workflow failed.
    :param session_id: Session identifier for metrics.
    :param total_cost_usd: Cumulative cost.
    :param code_review: Nested state for code/review loop.
    """

    query: str
    route: str
    confidence: float
    specialist_available: bool
    handled_by: str
    warning: Optional[str]
    next_agent: str
    needs_decomposition: bool
    subtasks: List[SubTask]
    current_task_id: str
    results: Annotated[Dict[str, TaskResultDict], merge_results]
    messages: Annotated[List[Dict[str, Any]], append_messages]
    output: str
    final_output: str
    error: Optional[str]
    session_id: str
    total_cost_usd: float
    code_review: CodeReviewState


# ── State Factory ─────────────────────────────────────────────────────────────

def create_initial_state(
    query: str,
    session_id: str,
    route: str = "",
    confidence: float = 0.0,
) -> OrchestratorState:
    """
    Create initial orchestration state.

    :param query: User query string.
    :param session_id: Session identifier.
    :param route: Initial route (optional).
    :param confidence: Routing confidence (optional).
    :return: Initialized OrchestratorState.
    """
    return OrchestratorState(
        query=query,
        route=route,
        confidence=confidence,
        specialist_available=True,
        handled_by="",
        warning=None,
        next_agent="",
        needs_decomposition=False,
        subtasks=[],
        current_task_id="",
        results={},
        messages=[],
        output="",
        final_output="",
        error=None,
        session_id=session_id,
        total_cost_usd=0.0,
        code_review=CodeReviewState(
            code_output="",
            review_result="",
            review_passed=False,
            retry_count=0,
            max_retries=2,
            feedback_history=[],
        ),
    )
