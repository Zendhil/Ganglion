"""
Conditional edge functions for LangGraph.

Description: Defines routing logic for conditional edges in the
    orchestration graph. Used to determine next node based on state.
"""

# Standard library imports
import logging
from typing import Literal

# App imports
from ganglion.orchestration.state import OrchestratorState
from ganglion.agent_core import AgentCore

logger = logging.getLogger(__name__)


# ── Route Selection Edge ──────────────────────────────────────────────────────

def route_to_agent(
    state: OrchestratorState,
) -> AgentCore:
    """
    Route to appropriate agent based on route field.

    Description: Examines route and confidence in state to determine
        which agent node to execute next.

    :param state: Current orchestration state.
    :return: Name of next node.
    """
    route = state.get("route", "")
    confidence = state.get("confidence", 0.0)

    logger.info(
        f"In function route_to_agent: route={route}, confidence={confidence:.2f}"
    )

    # Low confidence - need decomposition
    if confidence < 0.45:
        return "decompose"

    # Map route to agent node
    route_map = {
        "code_agent": "code_agent",
        "search_agent": "search_agent",
        "data_agent": "data_agent",
    }

    agent = route_map.get(route)
    if agent:
        return agent

    # Unknown or head_agent route - decompose
    return "decompose"


# ── Code Review Loop Edge ─────────────────────────────────────────────────────

def should_continue_review(
    state: OrchestratorState,
) -> Literal["code_agent", "aggregate"]:
    """
    Decide whether to retry code generation or proceed to aggregation.

    Description: Implements the code → review loop logic:
        - If review passed: proceed to aggregate
        - If review failed and retries < max: loop back to code_agent
        - If max retries reached: proceed to aggregate anyway

    :param state: Current orchestration state.
    :return: Name of next node.
    """
    code_review = state.get("code_review", {})
    review_passed = code_review.get("review_passed", False)
    retry_count = code_review.get("retry_count", 0)
    max_retries = code_review.get("max_retries", 2)

    logger.info(
        f"In function should_continue_review: passed={review_passed}, "
        f"retry_count={retry_count}, max_retries={max_retries}"
    )

    if review_passed:
        logger.info("In function should_continue_review: Review passed, aggregating")
        return "aggregate"

    if retry_count < max_retries:
        logger.info(
            f"In function should_continue_review: Retry {retry_count + 1}/{max_retries}"
        )
        return "code_agent"

    logger.info("In function should_continue_review: Max retries reached, aggregating")
    return "aggregate"


# ── Task Type Edge ────────────────────────────────────────────────────────────

def route_by_task_type(
    state: OrchestratorState,
) -> Literal["code_flow", "search_agent", "data_agent"]:
    """
    Route based on decomposed task type.

    Description: Examines subtasks to determine which workflow to use.
        Code tasks go through code → review flow, others go direct.

    :param state: Current orchestration state.
    :return: Name of next node or subgraph.
    """
    subtasks = state.get("subtasks", [])

    if not subtasks:
        return "search_agent"  # Default

    first_task = subtasks[0]
    agent = first_task.get("agent", "")

    logger.info(f"In function route_by_task_type: first task agent={agent}")

    if agent == "code_agent":
        return "code_flow"
    elif agent == "data_agent":
        return "data_agent"
    else:
        return "search_agent"


# ── Parallel Branch Completion Edge ───────────────────────────────────────────

def check_parallel_completion(
    state: OrchestratorState,
) -> Literal["aggregate", "wait"]:
    """
    Check if parallel branches have completed.

    Description: Used when running multiple agents in parallel.
        Proceeds to aggregation when all expected results are present.

    :param state: Current orchestration state.
    :return: Next node name.
    """
    subtasks = state.get("subtasks", [])
    results = state.get("results", {})

    expected_tasks = {t["id"] for t in subtasks}
    completed_tasks = set(results.keys())

    # Check if all non-review tasks are done
    non_review_tasks = {t["id"] for t in subtasks if t["agent"] != "review_agent"}

    if non_review_tasks.issubset(completed_tasks):
        return "aggregate"

    return "wait"
