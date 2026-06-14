"""
LangGraph state graph for orchestration.

Description: Builds the main orchestration graph with:
    - Decomposition node for complex queries
    - Agent nodes wrapped from AgentCore
    - Code → Review conditional loop with max 2 retries
    - Parallel execution support via LangGraph branches
"""

# Standard library imports
import logging
from typing import Any, Dict, Literal

# Third party imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# App imports
from ganglion.orchestration.state import OrchestratorState, create_initial_state
from ganglion.orchestration.nodes import (
    decompose_node,
    code_agent_node,
    review_agent_node,
    search_agent_node,
    data_agent_node,
    aggregate_node,
)
from ganglion.orchestration.edges import (
    route_to_agent,
    should_continue_review,
    route_by_task_type,
)

logger = logging.getLogger(__name__)


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_orchestration_graph() -> CompiledStateGraph:
    """
    Build the main orchestration state graph.

    Description: Creates LangGraph with:
        - Entry routing based on semantic router confidence
        - Decomposition for complex/ambiguous queries
        - Code → Review loop with conditional retry
        - Direct paths for search and data agents
        - Aggregation node for final output

    Graph structure:
        START
          │
          ▼
        [route_to_agent] ──────────────────────┐
          │                                    │
          ├─→ code_agent ─→ review_agent ──────┤
          │       ▲              │             │
          │       └──── (retry) ─┘             │
          │                                    │
          ├─→ search_agent ────────────────────┤
          │                                    │
          ├─→ data_agent ──────────────────────┤
          │                                    │
          └─→ decompose ─→ [route_by_task] ────┘
                                               │
                                               ▼
                                          aggregate
                                               │
                                               ▼
                                              END

    :return: Compiled LangGraph state graph.
    """
    logger.info("In function build_orchestration_graph: Building graph")

    # Create state graph
    graph = StateGraph(OrchestratorState)

    # ── Add nodes ─────────────────────────────────────────────────────────────

    graph.add_node("decompose", decompose_node)
    graph.add_node("code_agent", code_agent_node)
    graph.add_node("review_agent", review_agent_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("aggregate", aggregate_node)

    # ── Add edges ─────────────────────────────────────────────────────────────

    # Entry point: route based on semantic router result
    graph.add_conditional_edges(
        START,
        route_to_agent,
        {
            "code_agent": "code_agent",
            "search_agent": "search_agent",
            "data_agent": "data_agent",
            "decompose": "decompose",
        },
    )

    # Decompose routes to appropriate agent flow
    graph.add_conditional_edges(
        "decompose",
        route_by_task_type,
        {
            "code_flow": "code_agent",
            "search_agent": "search_agent",
            "data_agent": "data_agent",
        },
    )

    # Code agent always goes to review
    graph.add_edge("code_agent", "review_agent")

    # Review agent: conditional loop or aggregate
    graph.add_conditional_edges(
        "review_agent",
        should_continue_review,
        {
            "code_agent": "code_agent",
            "aggregate": "aggregate",
        },
    )

    # Search and data agents go directly to aggregate
    graph.add_edge("search_agent", "aggregate")
    graph.add_edge("data_agent", "aggregate")

    # Aggregate goes to end
    graph.add_edge("aggregate", END)

    # ── Compile ───────────────────────────────────────────────────────────────

    compiled = graph.compile()
    logger.info("In function build_orchestration_graph: Graph compiled successfully")

    return compiled


# ── Parallel Execution Graph ──────────────────────────────────────────────────

def build_parallel_graph() -> CompiledStateGraph:
    """
    Build graph with parallel execution branches.

    Description: For queries that can benefit from parallel agent
        execution (e.g., search + code simultaneously). Uses LangGraph's
        native parallel branches for concurrent execution.

    Graph structure:
        START
          │
          ▼
        decompose
          │
          ├─→ code_agent ─→ review_agent ─┐
          │       ▲              │        │
          │       └──── (retry) ─┘        │
          │                               │
          ├─→ search_agent ───────────────┤
          │                               │
          └─→ data_agent ─────────────────┤
                                          │
                                          ▼
                                      aggregate
                                          │
                                          ▼
                                         END

    :return: Compiled LangGraph with parallel branches.
    """
    logger.info("In function build_parallel_graph: Building parallel graph")

    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("decompose", decompose_node)
    graph.add_node("code_agent", code_agent_node)
    graph.add_node("review_agent", review_agent_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("aggregate", aggregate_node)

    # Start with decomposition
    graph.add_edge(START, "decompose")

    # Decompose routes to the appropriate agent
    graph.add_conditional_edges(
        "decompose",
        _parallel_route,
        {
            "code_agent": "code_agent",
            "search_agent": "search_agent",
            "data_agent": "data_agent",
        },
    )

    # Code flow with review loop
    graph.add_edge("code_agent", "review_agent")
    graph.add_conditional_edges(
        "review_agent",
        should_continue_review,
        {
            "code_agent": "code_agent",
            "aggregate": "aggregate",
        },
    )

    # Direct agents to aggregate
    graph.add_edge("search_agent", "aggregate")
    graph.add_edge("data_agent", "aggregate")

    # End
    graph.add_edge("aggregate", END)

    compiled = graph.compile()
    logger.info("In function build_parallel_graph: Parallel graph compiled")

    return compiled


def _parallel_route(
    state: OrchestratorState,
) -> Literal["code_agent", "search_agent", "data_agent"]:
    """
    Determine which agent to route to based on subtasks.

    :param state: Current state.
    :return: Agent node name.
    """
    subtasks = state.get("subtasks", [])

    if not subtasks:
        return "search_agent"

    first_task = subtasks[0]
    agent = first_task.get("agent", "search_agent")

    if agent in ["code_agent", "search_agent", "data_agent"]:
        return agent

    return "search_agent"


# ── Orchestrator Class ────────────────────────────────────────────────────────

class Orchestrator:
    """
    Main orchestrator using LangGraph.

    Description: Wraps the compiled state graph and provides
        a simple interface for query execution.

    :param use_parallel: Whether to use parallel execution graph.
    """

    def __init__(self, use_parallel: bool = False):
        logger.info(f"In class Orchestrator, function __init__: parallel={use_parallel}")

        if use_parallel:
            self._graph = build_parallel_graph()
        else:
            self._graph = build_orchestration_graph()

    def run(
        self,
        query: str,
        session_id: str,
        route: str = "",
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute orchestration for a query.

        :param query: User query string.
        :param session_id: Session identifier.
        :param route: Pre-determined route (optional).
        :param confidence: Routing confidence (optional).
        :return: Final state with results.
        """
        logger.info(f"In class Orchestrator, function run: query_len={len(query)}")

        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            route=route,
            confidence=confidence,
        )

        # Run the graph
        final_state = self._graph.invoke(initial_state)

        logger.info(
            f"In class Orchestrator, function run: Completed, "
            f"cost=${final_state.get('total_cost_usd', 0):.4f}"
        )

        return final_state

    async def arun(
        self,
        query: str,
        session_id: str,
        route: str = "",
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute orchestration asynchronously.

        :param query: User query string.
        :param session_id: Session identifier.
        :param route: Pre-determined route (optional).
        :param confidence: Routing confidence (optional).
        :return: Final state with results.
        """
        logger.info(f"In class Orchestrator, function arun: query_len={len(query)}")

        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            route=route,
            confidence=confidence,
        )

        # Run the graph asynchronously
        final_state = await self._graph.ainvoke(initial_state)

        return final_state

    def stream(
        self,
        query: str,
        session_id: str,
        route: str = "",
        confidence: float = 0.0,
    ):
        """
        Stream orchestration execution for a query.

        :param query: User query string.
        :param session_id: Session identifier.
        :param route: Pre-determined route (optional).
        :param confidence: Routing confidence (optional).
        :yields: State updates as graph executes.
        """
        logger.info(f"In class Orchestrator, function stream: query_len={len(query)}")

        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            route=route,
            confidence=confidence,
        )

        # Stream graph execution
        for state_update in self._graph.stream(initial_state):
            yield state_update

    @property
    def graph(self) -> CompiledStateGraph:
        """Get the underlying compiled graph."""
        return self._graph
