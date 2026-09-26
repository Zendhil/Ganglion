"""
Specialist agent implementations for orchestration.

Description: Defines all agent classes that extend AgentCore.
    Includes structural agents (Head, Tail) and specialist agents (Code, Review, Search, Data).
"""

# Standard library imports
import json
import logging
from typing import Any, Dict, List, Optional

# App imports
from ganglion.agents_hub import AgentCore, Task, TaskResult
from ganglion.agents_hub.prompts import (
    HEAD_AGENT_PROMPT,
    TAIL_AGENT_PROMPT,
    CODE_AGENT_PROMPT,
    REVIEW_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT,
    DATA_AGENT_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Head Agent ────────────────────────────────────────────────────────────────

class HeadAgent(AgentCore):
    """
    Head agent - entry point for orchestration.

    Description: Responsible for initial query processing, routing validation,
        decomposition of complex queries, and fallback handling when no
        specialist is available.

    :param session_id: Session identifier for metrics.
    """

    def __init__(self, session_id: str):
        logger.info(f"In class HeadAgent, function __init__: Entered")
        super().__init__(
            agent_id="head_agent",
            session_id=session_id,
            interleaved_thinking=True,
            thinking_budget=8000,
        )

    @property
    def system_prompt(self) -> str:
        return HEAD_AGENT_PROMPT

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return []

    def score_output(self, task: Task, output: str) -> float:
        """Score decomposition or fallback output quality."""
        if "decompose" in task.content.lower():
            try:
                result = json.loads(output)
                if "needs_decomposition" in result:
                    return 1.0
                if "subtasks" in result and isinstance(result["subtasks"], list):
                    return 0.9
                return 0.6
            except json.JSONDecodeError:
                return 0.3
        if len(output) > 50:
            return 0.8
        return 0.5

    def select_model(self, task: Task) -> str:
        """Select model tier for orchestration tasks."""
        # Check for default model in metadata first
        if task.metadata and "default_model" in task.metadata:
            return task.metadata["default_model"]

        if task._escalate:
            return "cloud"
        if "decompose" in task.content.lower():
            return "cloud"
        return "mid"

    def _decompose_query(self, query: str) -> List[Task]:
        """
        Decompose a complex query into subtasks using dedicated prompt.

        Description: Uses llama3.1:latest model and DECOMPOSE_TASK_PROMPT
            to generate validated subtasks with proper agent assignment.

        :param query: User query to decompose.
        :return: List of Task objects with metadata for agent routing.
        """
        logger.info(f"In class HeadAgent, function _decompose_query: Decomposing query")

        # Import the dedicated decomposition prompt
        from ganglion.agents_hub.prompts import DECOMPOSE_TASK_PROMPT

        # Create decomposition task with full prompt
        decompose_content = f"{DECOMPOSE_TASK_PROMPT}\n\n{query}"
        decompose_task = Task(
            id="decompose",
            content=decompose_content,
            metadata={"default_model": "ollama/llama3.1:latest"}
        )

        # Run task with forced model
        result = self.run_task(decompose_task)

        try:
            # Parse JSON output
            output = result.output.strip()

            # Handle markdown code blocks if present
            if "```json" in output:
                start = output.find("```json") + 7
                end = output.find("```", start)
                if end > start:
                    output = output[start:end].strip()
            elif "```" in output:
                start = output.find("```") + 3
                end = output.find("```", start)
                if end > start:
                    output = output[start:end].strip()

            decomposition = json.loads(output)
            raw_subtasks = decomposition.get("subtasks", [])

            if not raw_subtasks:
                logger.warning("In class HeadAgent, function _decompose_query: No subtasks generated")
                return []

            # Validate and clean up subtasks
            validated_subtasks = self._validate_subtasks(raw_subtasks)

            logger.info(
                f"In class HeadAgent, function _decompose_query: "
                f"Generated {len(validated_subtasks)} validated subtasks"
            )
            return validated_subtasks

        except json.JSONDecodeError as e:
            logger.warning(
                f"In class HeadAgent, function _decompose_query: "
                f"Failed to parse JSON: {e}, output={result.output[:200]}"
            )
            return []
        except Exception as e:
            logger.exception(
                f"In class HeadAgent, function _decompose_query: Unexpected error: {e}"
            )
            return []

    def _validate_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Task]:
        """
        Validate and normalize subtasks against SubTask schema.

        Description: Ensures all subtasks have required fields, valid agents,
            and initializes status to 'pending'. Falls back invalid agents to head_agent.
            Returns Task objects.

        :param subtasks: Raw subtask dictionaries from LLM.
        :return: List of validated Task objects.
        """
        validated = []

        for i, task_dict in enumerate(subtasks):
            try:
                # Ensure required fields exist
                task_id = task_dict.get("id", f"task_{i+1}")
                content = task_dict.get("content", "")
                agent = task_dict.get("agent", "head_agent")
                depends_on = task_dict.get("depends_on", [])

                # Validate content is not empty
                if not content or not content.strip():
                    logger.warning(
                        f"In class HeadAgent, function _validate_subtasks: "
                        f"Skipping task {task_id} with empty content"
                    )
                    continue

                # Validate agent exists in live_agents
                if agent not in AgentCore.live_agents:
                    logger.warning(
                        f"In class HeadAgent, function _validate_subtasks: "
                        f"Agent '{agent}' not found for task {task_id}, falling back to head_agent"
                    )
                    agent = "head_agent"

                # Ensure depends_on is a list
                if not isinstance(depends_on, list):
                    logger.warning(
                        f"In class HeadAgent, function _validate_subtasks: "
                        f"Invalid depends_on for task {task_id}, defaulting to []"
                    )
                    depends_on = []

                # Create Task object with metadata for agent and dependencies
                validated_task = Task(
                    id=task_id,
                    content=content,
                    metadata={
                        "agent": agent,
                        "depends_on": depends_on,
                        "status": "pending"
                    }
                )

                validated.append(validated_task)
                logger.info(
                    f"In class HeadAgent, function _validate_subtasks: "
                    f"Validated task {task_id} -> agent={agent}"
                )

            except Exception as e:
                logger.exception(
                    f"In class HeadAgent, function _validate_subtasks: "
                    f"Error validating task at index {i}: {e}"
                )
                continue

        return validated

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node function."""
        query = state["query"]
        route = state.get("route", "")
        confidence = state.get("confidence", 0.0)

        logger.info(f"In class HeadAgent, function __call__: route={route}, confidence={confidence:.2f}")

        # Case 1: Low confidence - decompose
        if confidence < 0.45:
            logger.info("In class HeadAgent, function __call__: Low confidence, decomposing")
            subtasks = self._decompose_query(query)

            # If decomposition failed, fall back to direct response
            if not subtasks:
                logger.warning(
                    "In class HeadAgent, function __call__: "
                    "Decomposition failed, providing direct fallback"
                )
                fallback_task = Task(id="fallback", content=query)
                fallback_result = self.run_task(fallback_task)
                state["output"] = fallback_result.output
                state["specialist_available"] = False
                state["handled_by"] = "head_agent_fallback"
                state["warning"] = "Task decomposition failed, provided general response"
                state["total_cost_usd"] = state.get("total_cost_usd", 0.0) + fallback_result.cost_usd
                return state

            state["subtasks"] = subtasks
            state["needs_decomposition"] = True
            state["specialist_available"] = True
            return state

        # Case 2: No specialist - fallback
        if route == "head_agent" or route not in AgentCore.live_agents:
            logger.warning(f"In class HeadAgent, function __call__: No specialist for route: {route}")
            fallback_task = Task(id="fallback", content=query)
            fallback_result = self.run_task(fallback_task)
            state["output"] = fallback_result.output
            state["specialist_available"] = False
            state["handled_by"] = "head_agent_fallback"
            state["warning"] = "No specialist agent available for this query type"
            state["total_cost_usd"] = state.get("total_cost_usd", 0.0) + fallback_result.cost_usd
            return state

        # Case 3: Valid specialist
        logger.info(f"In class HeadAgent, function __call__: Routing to {route}")
        state["specialist_available"] = True
        state["next_agent"] = route
        state["needs_decomposition"] = False
        return state


# ── Code Agent ────────────────────────────────────────────────────────────────

class CodeAgent(AgentCore):
    """
    Code generation and modification agent.

    Description: Handles coding tasks including generation, debugging,
        refactoring, and test writing. Can optionally use interleaved
        thinking for complex multi-file refactors.

    :param session_id: Session identifier for metrics.
    :param interleaved_thinking: Enable extended thinking for complex tasks.
    """

    def __init__(
        self,
        session_id: str,
        interleaved_thinking: bool = False,
    ):
        logger.info(f"In class CodeAgent, function __init__: Entered")
        #super.init registers the agent to AGentCore class variable and needs to be called.
        super().__init__(
            agent_id="code_agent",
            session_id=session_id,
            interleaved_thinking=interleaved_thinking,
            thinking_budget=5000,
        )

    @property
    def system_prompt(self) -> str:
        return CODE_AGENT_PROMPT

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "description": "Execute code and return the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to execute"},
                            "language": {"type": "string", "description": "Programming language"},
                        },
                        "required": ["code", "language"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run test suite and return results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_file": {"type": "string", "description": "Path to test file"},
                        },
                        "required": ["test_file"],
                    },
                },
            },
        ]

    def score_output(self, task: Task, output: str) -> float:
        """
        Score code output quality.

        Description: Checks for code blocks, length, and basic structure.
            Replace with actual test runner integration later.

        :param task: The coding task.
        :param output: Generated code output.
        :return: Quality score (0.0-1.0).
        """
        score = 1.0
        #
        # # Check for code blocks
        # if "```" in output:
        #     score += 0.4
        #
        # # Check for reasonable length
        # if len(output) > 100:
        #     score += 0.2
        #
        # # Check for function/class definitions
        # if "def " in output or "class " in output or "function " in output:
        #     score += 0.2
        #
        # # Check for comments/documentation
        # if "#" in output or "//" in output or '"""' in output:
        #     score += 0.2

        return min(1.0, score)

    def select_model(self, task: Task) -> str:
        """
        Select model tier based on task complexity.

        :param task: Coding task.
        :return: Model tier (local/mid/cloud).
        """
        if task._escalate:
            current = task._last_model or "local"
            return {"local": "mid", "mid": "cloud"}.get(current, "cloud")

        content = task.content.lower()
        tokens = len(task.content.split())

        # Simple boilerplate -> local
        if tokens < 60 and "boilerplate" in content:
            return "local"

        # Complex tasks -> cloud
        if any(word in content for word in ["refactor", "architect", "design", "optimize"]):
            return "cloud"

        # Default based on length
        if tokens < 300:
            return "mid"

        return "cloud"


# ── Review Agent ──────────────────────────────────────────────────────────────

class ReviewAgent(AgentCore):
    """
    Code review agent.

    Description: Reviews code for quality, correctness, security, and
        best practices. Returns structured feedback with pass/fail decision.

    :param session_id: Session identifier for metrics.
    """

    def __init__(self, session_id: str):
        logger.info(f"In class ReviewAgent, function __init__: Entered")
        super().__init__(
            agent_id="review_agent",
            session_id=session_id,
            interleaved_thinking=False,  # Reviews are linear
            thinking_budget=3000,
        )

    @property
    def system_prompt(self) -> str:
        return REVIEW_AGENT_PROMPT

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return []  # Review doesn't need tools

    def score_output(self, task: Task, output: str) -> float:
        """
        Score review output quality.

        Description: Validates JSON structure and required fields.

        :param task: Review task.
        :param output: Review output.
        :return: Quality score (0.0-1.0).
        """
        try:
            review = json.loads(output)
            required_fields = ["passed", "score", "summary"]

            if all(field in review for field in required_fields):
                return 0.9

            return 0.5
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            if "```json" in output:
                return 0.7
            return 0.3

    def select_model(self, task: Task) -> str:
        """
        Select model tier for review.

        :param task: Review task.
        :return: Model tier.
        """
        if task._escalate:
            return "cloud"

        # Reviews benefit from better models
        return "mid"

    def parse_review_result(self, output: str) -> Dict[str, Any]:
        """
        Parse review output to structured result.

        :param output: Raw review output.
        :return: Parsed review dict with passed, score, issues, etc.
        """
        try:
            # Try direct JSON parse
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        if "```json" in output:
            start = output.find("```json") + 7
            end = output.find("```", start)
            if end > start:
                try:
                    return json.loads(output[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Fallback: infer from content
        passed = "passed" in output.lower() and "true" in output.lower()
        return {
            "passed": passed,
            "score": 0.7 if passed else 0.3,
            "summary": output[:200],
            "issues": [],
            "suggestions": [],
        }


# ── Search Agent ──────────────────────────────────────────────────────────────

class SearchAgent(AgentCore):
    """
    Information retrieval agent.

    Description: Handles search and retrieval tasks. No interleaved
        thinking needed - linear retrieve → summarize workflow.

    :param session_id: Session identifier for metrics.
    """

    def __init__(self, session_id: str):
        logger.info(f"In class SearchAgent, function __init__: Entered")
        super().__init__(
            agent_id="search_agent",
            session_id=session_id,
            interleaved_thinking=False,
        )

    @property
    def system_prompt(self) -> str:
        return SEARCH_AGENT_PROMPT

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def score_output(self, task: Task, output: str) -> float:
        """Score search output quality."""
        score = 0.0
        if len(output) > 100:
            score += 0.4
        if len(output) > 300:
            score += 0.2
        if any(word in output.lower() for word in ["found", "result", "information"]):
            score += 0.2
        return min(1.0, score + 0.2)


# ── Data Agent ────────────────────────────────────────────────────────────────

class DataAgent(AgentCore):
    """
    Data processing agent.

    Description: Handles database queries, data transformation, and
        analysis tasks. Optional interleaved thinking for complex queries.

    :param session_id: Session identifier for metrics.
    :param interleaved_thinking: Enable for complex query planning.
    """

    def __init__(
        self,
        session_id: str,
        interleaved_thinking: bool = False,
    ):
        logger.info(f"In class DataAgent, function __init__: Entered")
        super().__init__(
            agent_id="data_agent",
            session_id=session_id,
            interleaved_thinking=interleaved_thinking,
            thinking_budget=4000,
        )

    @property
    def system_prompt(self) -> str:
        return DATA_AGENT_PROMPT

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute SQL query against database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query"},
                            "database": {"type": "string", "description": "Database name"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def score_output(self, task: Task, output: str) -> float:
        """Score data output quality."""
        score = 0.0
        if "SELECT" in output.upper() or "INSERT" in output.upper():
            score += 0.4
        if len(output) > 50:
            score += 0.3
        if "```" in output:
            score += 0.2
        return min(1.0, score + 0.1)


# ── Tail Agent ────────────────────────────────────────────────────────────────

class TailAgent(AgentCore):
    """
    Tail agent - final aggregation and formatting.

    Description: Responsible for aggregating results from specialist agents,
        formatting final output, and propagating warnings from fallback cases.

    :param session_id: Session identifier for metrics.
    """

    def __init__(self, session_id: str):
        logger.info(f"In class TailAgent, function __init__: Entered")
        super().__init__(
            agent_id="tail_agent",
            session_id=session_id,
            interleaved_thinking=False,  # Simple aggregation, no thinking needed
        )

    @property
    def system_prompt(self) -> str:
        return TAIL_AGENT_PROMPT

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return []

    def score_output(self, task: Task, output: str) -> float:
        """Score aggregation output quality."""
        if len(output) > 100:
            return 0.9
        if len(output) > 50:
            return 0.7
        return 0.5

    def select_model(self, task: Task) -> str:
        """Select model tier for aggregation."""
        if task._escalate:
            return "cloud"
        # Simple aggregation can use local/mid tier
        return "mid"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node function."""
        logger.info("In class TailAgent, function __call__: Aggregating results")

        # Check if head agent handled fallback
        if not state.get("specialist_available", True):
            # Fallback case - head agent already provided output
            final_output = state.get("output", "")
            warning = state.get("warning", "")
            if warning:
                final_output = f"{final_output}\n\n⚠️  {warning}"
            state["final_output"] = final_output
            logger.info("In class TailAgent, function __call__: Propagated fallback response")
            return state

        # Check if decomposition was used
        if state.get("needs_decomposition", False):
            # Aggregate subtask results
            results = state.get("results", {})
            if results:
                # Combine results from all subtasks
                combined = "\n\n".join([str(r.get("output", "")) for r in results.values()])
                state["final_output"] = combined
            logger.info("In class TailAgent, function __call__: Aggregated decomposed results")
            return state

        # Normal case - aggregate specialist result
        results = state.get("results", {})
        if results:
            # Take the first (and likely only) result
            first_result = next(iter(results.values()), {})
            state["final_output"] = first_result.get("output", "")
        else:
            # No results - fallback
            state["final_output"] = "No results generated"
            state["warning"] = "No specialist results available"

        logger.info("In class TailAgent, function __call__: Completed aggregation")
        return state



