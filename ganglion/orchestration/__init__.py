"""
Orchestration package using LangGraph.

Description: Provides LangGraph-based orchestration layer that wraps
    AgentCore instances as nodes and handles workflow execution.

Features:
    - AgentCore wrapped as LangGraph nodes (no modifications to core)
    - State graph for head agent dispatch logic
    - Code → Review conditional loop with max 2 retries
    - Parallel execution support via LangGraph branches
"""

from ganglion.orchestration.state import (
    OrchestratorState,
    SubTask,
    TaskResultDict,
    CodeReviewState,
    create_initial_state,
)
from agents_hub.agents import (
    CodeAgent,
    ReviewAgent,
    SearchAgent,
    DataAgent,
    get_agent,
)
from ganglion.orchestration.nodes import (
    decompose_node,
    code_agent_node,
    review_agent_node,
    search_agent_node,
    data_agent_node,
    aggregate_node,
)
from ganglion.orchestration.router import (
    route_to_agent,
    should_continue_review,
    route_by_task_type,
)
from ganglion.orchestration.orchestrator import (
    build_orchestration_graph,
    build_parallel_graph,
    Orchestrator,
)

__all__ = [
    # State
    "OrchestratorState",
    "SubTask",
    "TaskResultDict",
    "CodeReviewState",
    "create_initial_state",
    # Agents
    "CodeAgent",
    "ReviewAgent",
    "SearchAgent",
    "DataAgent",
    "get_agent",
    # Nodes
    "decompose_node",
    "code_agent_node",
    "review_agent_node",
    "search_agent_node",
    "data_agent_node",
    "aggregate_node",
    # Edges
    "route_to_agent",
    "should_continue_review",
    "route_by_task_type",
    # Graph
    "build_orchestration_graph",
    "build_parallel_graph",
    "Orchestrator",
]
