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

from agents_hub import AgentCore
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
        self._graph = self.build_orchestration_graph(use_parallel)

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

    @staticmethod
    def _parallel_route(
            state: OrchestratorState,
    ) -> AgentCore:
        """
        Determine which agent to route to based on subtasks.

        :param state: Current state.
        :return: Agent node name.
        """
        subtasks = state.get("subtasks", [])

        if not subtasks:
            return "head_agent"

        first_task = subtasks[0]
        agent = first_task.get("agent", "head_agent")

        return agent

    # ── Graph Builder ─────────────────────────────────────────────────────────────
    def build_orchestration_graph(use_parallel=None) -> CompiledStateGraph:
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

        # fixed structural nodes — always present
        graph.add_node("head_agent", AgentCore.get_agent("head_agent"))
        graph.add_node("tail_agent", AgentCore.get_agent("tail_agent"))

        # dynamic specialist nodes — everything else
        specialist_ids = {
            k for k in AgentCore.live_agents
            if k not in ("head_agent", "tail_agent")
        }
        for agent_id in specialist_ids:
            graph.add_node(agent_id, AgentCore.get_agent(agent_id))

        # edges — fully dynamic, zero hardcoding
        graph.add_edge(START, "head_agent")
        call_router = _parallel_route if use_parallel else route_to_agent
        graph.add_conditional_edges(
            "head_agent",
            call_router,
            {k: k for k in specialist_ids},
        )
        for agent_id in specialist_ids:
            graph.add_edge(agent_id, "tail_agent")

        graph.add_edge("tail_agent", END)

        # ── Compile ───────────────────────────────────────────────────────────────

        compiled = graph.compile()
        logger.info("In function build_orchestration_graph: Graph compiled successfully")

        return compiled

