"""
Orchestration package using LangGraph.

Description: Provides LangGraph-based orchestration layer that wraps
    AgentCore instances as nodes and handles workflow execution.

Features:
    - AgentCore wrapped as LangGraph nodes (no modifications to core)
    - Dynamic agent discovery from AgentCore.live_agents registry
    - QueryRouter integration for semantic routing
    - Fallback handling when no specialist available
    - Future: Parallel execution support via LangGraph Send API
"""

from ganglion.orchestration.state import (
    OrchestratorState,
    SubTask,
    TaskResultDict,
    CodeReviewState,
    create_initial_state,
)
from ganglion.agents_hub.agents import (
    HeadAgent,
    TailAgent,
    CodeAgent,
    ReviewAgent,
    SearchAgent,
    DataAgent,
)
from ganglion.orchestration.orchestrator import Orchestrator

__all__ = [
    # State
    "OrchestratorState",
    "SubTask",
    "TaskResultDict",
    "CodeReviewState",
    "create_initial_state",
    # Agents
    "HeadAgent",
    "TailAgent",
    "CodeAgent",
    "ReviewAgent",
    "SearchAgent",
    "DataAgent",
    # Orchestrator
    "Orchestrator",
]
