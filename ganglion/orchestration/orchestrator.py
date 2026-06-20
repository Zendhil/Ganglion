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
from ganglion.agents_hub import AgentCore
from ganglion.orchestration.state import OrchestratorState, create_initial_state
from ganglion.router.routes import route_query

logger = logging.getLogger(__name__)

# ── Orchestrator Class ────────────────────────────────────────────────────────

class Orchestrator:
    """
    Main orchestrator using LangGraph.

    Description: Wraps the compiled state graph and provides
        a simple interface for query execution.

    :param session_id: Session identifier for all agents.
    :param use_parallel: Whether to use parallel execution (future implementation).
    """

    def __init__(self, session_id: str, use_parallel: bool = False):
        logger.info(f"In class Orchestrator, function __init__: session_id={session_id}, parallel={use_parallel}")
        self.session_id = session_id
        self.use_parallel = use_parallel
        self._graph = self.build_orchestration_graph()

    def run(
        self,
        query: str,
        route: str = "",
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute orchestration for a query.

        :param query: User query string.
        :param route: Pre-determined route (optional, will call QueryRouter if not provided).
        :param confidence: Routing confidence (optional).
        :return: Final state with results.
        """
        logger.info(f"In class Orchestrator, function run: query_len={len(query)}")

        # Get routing decision from QueryRouter if not provided
        if not route:
            route_result = route_query(query)
            route = route_result.route
            confidence = route_result.confidence
            logger.info(
                f"In class Orchestrator, function run: QueryRouter decision: "
                f"route={route}, confidence={confidence:.2f}"
            )

        initial_state = create_initial_state(
            query=query,
            session_id=self.session_id,
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
        route: str = "",
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute orchestration asynchronously.

        :param query: User query string.
        :param route: Pre-determined route (optional).
        :param confidence: Routing confidence (optional).
        :return: Final state with results.
        """
        logger.info(f"In class Orchestrator, function arun: query_len={len(query)}")

        # Get routing decision from QueryRouter if not provided
        if not route:
            route_result = route_query(query)
            route = route_result.route
            confidence = route_result.confidence

        initial_state = create_initial_state(
            query=query,
            session_id=self.session_id,
            route=route,
            confidence=confidence,
        )

        # Run the graph asynchronously
        final_state = await self._graph.ainvoke(initial_state)

        return final_state

    def stream(
        self,
        query: str,
        route: str = "",
        confidence: float = 0.0,
    ):
        """
        Stream orchestration execution for a query.

        :param query: User query string.
        :param route: Pre-determined route (optional).
        :param confidence: Routing confidence (optional).
        :yields: State updates as graph executes.
        """
        logger.info(f"In class Orchestrator, function stream: query_len={len(query)}")

        # Get routing decision from QueryRouter if not provided
        if not route:
            route_result = route_query(query)
            route = route_result.route
            confidence = route_result.confidence

        initial_state = create_initial_state(
            query=query,
            session_id=self.session_id,
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

    # ── Graph Builder ─────────────────────────────────────────────────────────────
    def build_orchestration_graph(self) -> CompiledStateGraph:
        """
        Build the main orchestration state graph.

        Description: Creates LangGraph with:
            - START → head_agent (receives route from QueryRouter)
            - head_agent validates route and handles fallback
            - Conditional routing to specialists based on head_agent decision
            - All specialists → tail_agent for aggregation
            - tail_agent → END

        Graph structure:
            START
              │
              ▼
            head_agent ─────────┐
              │                 │
              ├─→ specialist ───┤
              │                 │
              └─→ tail_agent ───┘
                      │
                      ▼
                     END

        :return: Compiled LangGraph state graph.
        """
        logger.info("In function build_orchestration_graph: Building graph")

        # Create state graph
        graph = StateGraph(OrchestratorState)

        # Get all registered agents dynamically
        for agent_id in AgentCore.live_agents.keys():
            agent_instance = AgentCore.get_agent(agent_id)
            graph.add_node(agent_id, agent_instance)
            logger.info(f"In function build_orchestration_graph: Added node {agent_id}")

        # Define conditional routing function from head_agent
        def route_from_head(state: OrchestratorState) -> str:
            """Routes based on HeadAgent's decision."""
            # If specialist not available, head already handled it → go to tail
            if not state.get("specialist_available", True):
                logger.info("In route_from_head: No specialist, routing to tail_agent")
                return "tail_agent"

            # If needs decomposition, head will handle subtasks → go to tail
            if state.get("needs_decomposition", False):
                logger.info("In route_from_head: Needs decomposition, routing to tail_agent")
                return "tail_agent"

            # Otherwise route to specialist
            next_agent = state.get("next_agent", "tail_agent")
            logger.info(f"In route_from_head: Routing to specialist: {next_agent}")
            return next_agent

        # Edges: START → head_agent
        graph.add_edge(START, "head_agent")

        # Dynamic routing from head_agent
        specialist_ids = {
            k for k in AgentCore.live_agents
            if k not in ("head_agent", "tail_agent")
        }
        route_map = {agent_id: agent_id for agent_id in specialist_ids}
        route_map["tail_agent"] = "tail_agent"  # Fallback path

        graph.add_conditional_edges("head_agent", route_from_head, route_map)

        # All specialists route to tail_agent
        for agent_id in specialist_ids:
            graph.add_edge(agent_id, "tail_agent")

        graph.add_edge("tail_agent", END)

        # ── Compile ───────────────────────────────────────────────────────────────

        compiled = graph.compile()
        logger.info("In function build_orchestration_graph: Graph compiled successfully")

        return compiled

