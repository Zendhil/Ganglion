"""
Agent core loop implementation.

Description: Base class for all agents. Defines the execution loop with
    measurement, escalation, retry logic, and optional interleaved thinking.
    Specialist agents extend this — they never modify the loop.
"""

# Standard library imports
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

# Third party imports
import litellm

# App imports
from ganglion.agents_hub.models import Task, TaskResult, AgentMetrics
from ganglion.agents_hub.memory import MemoryStub, AbstractMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_THINKING_BUDGET = 5000  # Deprecated - kept for compatibility
DEFAULT_MAX_RETRIES = 3
DEFAULT_SUCCESS_THRESHOLD = 0.6
DEFAULT_MAX_CONTEXT_MESSAGES = 20

# Model tier constants
MODEL_LOCAL = "local"
MODEL_MID = "mid"
MODEL_CLOUD = "cloud"


# ── Agent Core Class ──────────────────────────────────────────────────────────

class AgentCore:
    """
    Base class for all agents. Extend this — do not modify the loop.

    Description: Provides the core execution loop with LLM calls via LiteLLM,
        metrics, retries, and escalation. Specialist agents override:
        system_prompt, tools, score_output(), select_model().

    :param agent_id: Unique name for this agent instance.
    :param session_id: Session identifier for grouping metrics.
    :param interleaved_thinking: DEPRECATED - kept for compatibility, no effect.
    :param thinking_budget: DEPRECATED - kept for compatibility, no effect.
    :param memory: Memory backend instance. Defaults to MemoryStub.
    :param max_retries: Maximum retry attempts per task. Default: 3.
    :param success_threshold: Minimum score for task success. Default: 0.6.
    """
    live_agents: dict[str, "AgentCore"] = {}
    def __init__(
        self,
        agent_id: str,
        session_id: str,
        interleaved_thinking: bool = False,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        memory: Optional[AbstractMemory] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    ):
        logger.info(f"In class AgentCore, function __init__: Entered for agent_id={agent_id}")

        self.agent_id = agent_id
        AgentCore.live_agents[agent_id] = self
        self.session_id = session_id
        self.interleaved_thinking = interleaved_thinking
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.success_threshold = success_threshold

        # Initialize metrics
        self.metrics = AgentMetrics(agent_id=agent_id, session_id=session_id)

        # Initialize memory
        self.memory = memory if memory is not None else MemoryStub()

        # Conversation context (sliding window)
        self.context: List[Dict[str, Any]] = []

        # Note: interleaved_thinking parameter kept for compatibility but not used
        # All LLM calls go through LiteLLM

        logger.info(
            f"In class AgentCore, function __init__: Initialized agent_id={agent_id}, "
            f"thinking={interleaved_thinking}, max_retries={max_retries}"
        )

    # ── Override in subclass ──────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        """
        System prompt for the agent.

        Description: Override in subclass to define agent behavior.

        :return: System prompt string.
        :raises NotImplementedError: If not overridden.
        """
        raise NotImplementedError("Subclass must implement system_prompt property")

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """
        Tool definitions for the agent.

        Description: Override in subclass to define available tools.
            Return empty list if no tools needed.

        :return: List of tool definition dicts.
        """
        return []

    def score_output(self, task: Task, output: str) -> float:
        """
        Score the quality of task output.

        Description: Override in subclass for domain-specific scoring.
            Default implementation uses crude length heuristic.

        :param task: The executed task.
        :param output: Model output string.
        :return: Quality score (0.0-1.0).
        """
        # Default: crude length heuristic — replace in subclass
        return min(1.0, len(output) / 500)

    def select_model(self, task: Task) -> str:
        """
        Select model tier for task execution.

        Description: Override in subclass for domain-specific complexity
            signals. Default uses token count heuristic with escalation.
            Supports forced model via task.metadata["force_model"].

        :param task: Task to execute.
        :return: Model tier string (local/mid/cloud) or direct model name.
        """
        # Check for default model in metadata
        if task.metadata and "default_model" in task.metadata:
            default_model = task.metadata["default_model"]
            logger.info(
                f"In class AgentCore, function select_model: "
                f"Using default model: {default_model}"
            )
            return default_model

        # Todo: Model selection shuld be on task complexity on not on token heuristic
        # Handle escalation from previous attempt
        if task._escalate:
            current = task._last_model or MODEL_LOCAL
            escalation_map = {MODEL_LOCAL: MODEL_MID, MODEL_MID: MODEL_CLOUD}
            return escalation_map.get(current, MODEL_CLOUD)

        # Default: token count heuristic
        tokens = len(task.content.split())
        if tokens < 100:
            return MODEL_LOCAL
        elif tokens < 400:
            return MODEL_MID
        return MODEL_CLOUD

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool call.

        Description: Override in subclass to implement actual tool execution.
            Default returns empty string (stub).

        :param tool_name: Name of the tool to execute.
        :param tool_input: Tool input parameters.
        :return: Tool execution result string.
        """
        logger.warning(
            f"In class AgentCore, function execute_tool: Stub called for {tool_name}"
        )
        return ""

    # ── Core loop — do not override ───────────────────────────────────────────

    def run_task(self, task: Task) -> TaskResult:
        """
        Execute a single task with measurement, escalation, and retry.

        Description: Core execution loop. Selects model, calls LLM,
            scores output, handles retries and escalation. Do not override.

        :param task: Task to execute.
        :return: TaskResult with output and metrics.
        """
        logger.info(f"In class AgentCore, function run_task: Entered for task_id={task.id}")

        attempt = 0
        last_error: Optional[str] = None

        while attempt < self.max_retries:
            attempt += 1
            t0 = time.monotonic()

            try:
                # Select model tier
                model = self.select_model(task)
                task._last_model = model

                # Build messages
                messages = self._build_messages(task)

                # Call LLM via LiteLLM
                output, cost = self._call_litellm(model, messages)
                thinking_used = False

                latency = (time.monotonic() - t0) * 1000
                score = self.score_output(task, output)

                result = TaskResult(
                    task_id=task.id,
                    output=output,
                    success=score >= self.success_threshold,
                    score=score,
                    cost_usd=cost,
                    latency_ms=latency,
                    model_used=model,
                    thinking_used=thinking_used,
                    retries=attempt - 1,
                )

                self._update_metrics(result)
                self._update_context(task, output)
                self.memory.write(task.id, result)

                if result.success:
                    logger.info(
                        f"In class AgentCore, function run_task: Success "
                        f"task_id={task.id}, score={score:.2f}, model={model}"
                    )
                    return result

                # Score below threshold — escalate and retry
                task._escalate = True
                logger.info(
                    f"In class AgentCore, function run_task: Escalating "
                    f"task_id={task.id}, score={score:.2f} < {self.success_threshold}"
                )

            except Exception as e:
                last_error = str(e)
                latency = (time.monotonic() - t0) * 1000
                logger.exception(
                    f"In class AgentCore, function run_task: Error on attempt {attempt}"
                )

                # Record failed attempt in metrics
                self._update_metrics(TaskResult(
                    task_id=task.id,
                    output="",
                    success=False,
                    score=0.0,
                    cost_usd=0.0,
                    latency_ms=latency,
                    model_used=task._last_model or "unknown",
                    error=last_error,
                    retries=attempt,
                ))

        # All retries exhausted
        logger.error(
            f"In class AgentCore, function run_task: All retries exhausted "
            f"task_id={task.id}, last_error={last_error}"
        )

        return TaskResult(
            task_id=task.id,
            output="",
            success=False,
            score=0.0,
            cost_usd=0.0,
            latency_ms=0.0,
            model_used="unknown",
            error=last_error,
            retries=self.max_retries,
        )

    def run_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """
        Execute multiple tasks sequentially.

        Description: Runs tasks one by one. For parallel execution,
            wrap in asyncio.gather() or use thread pool.

        :param tasks: List of tasks to execute.
        :return: List of TaskResult objects.
        """
        logger.info(f"In class AgentCore, function run_tasks: Processing {len(tasks)} tasks")
        return [self.run_task(task) for task in tasks]

    # ── LLM call backends ─────────────────────────────────────────────────────

    def _call_litellm(
        self,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, float]:
        """
        Standard LLM call via LiteLLM.

        Description: Works with any model tier (local/mid/cloud).
            Uses LiteLLM for unified API across providers.

        :param model: Model tier name.
        :param messages: Message list for completion.
        :return: Tuple of (output_text, cost_usd).
        """
        logger.info(f"In class AgentCore, function _call_litellm: Calling model={model}")

        response = litellm.completion(
            model=model,
            messages=messages,
            tools=self.tools if self.tools else None,
        )

        output = response.choices[0].message.content or ""
        cost = litellm.completion_cost(response)

        logger.info(
            f"In class AgentCore, function _call_litellm: Completed "
            f"output_len={len(output)}, cost=${cost:.6f}"
        )

        return output, cost

    # NOTE: Anthropic SDK thinking mode removed - all LLM calls go through LiteLLM

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(self, task: Task) -> List[Dict[str, Any]]:
        """
        Build message list for LLM call.

        Description: Combines system prompt, recent context, and current task.

        :param task: Current task.
        :return: Message list for completion API.
        """
        messages = []

        # Add system prompt for non-thinking calls
        if not self.interleaved_thinking:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add recent context (sliding window)
        messages.extend(self.context[-DEFAULT_MAX_CONTEXT_MESSAGES:])

        # Add current task
        messages.append({"role": "user", "content": task.content})

        return messages

    def _update_context(self, task: Task, output: str) -> None:
        """
        Update conversation context with task and output.

        :param task: Executed task.
        :param output: Model output.
        """
        self.context.append({"role": "user", "content": task.content})
        self.context.append({"role": "assistant", "content": output})

        # Trim to max size
        if len(self.context) > DEFAULT_MAX_CONTEXT_MESSAGES:
            self.context = self.context[-DEFAULT_MAX_CONTEXT_MESSAGES:]

    def _update_metrics(self, result: TaskResult) -> None:
        """
        Update agent metrics with task result.

        :param result: Task execution result.
        """
        m = self.metrics
        m.tasks_total += 1

        if result.success:
            m.tasks_passed += 1
        else:
            m.tasks_failed += 1

        m.total_cost_usd += result.cost_usd

        # Running average for latency
        m.avg_latency_ms = (
            (m.avg_latency_ms * (m.tasks_total - 1) + result.latency_ms)
            / m.tasks_total
        )

        # Running average for score
        m.avg_score = (
            (m.avg_score * (m.tasks_total - 1) + result.score)
            / m.tasks_total
        )

        # Model usage count
        m.model_usage[result.model_used] = m.model_usage.get(result.model_used, 0) + 1

        # Thinking usage count
        if result.thinking_used:
            m.thinking_calls += 1

    def clear_context(self) -> None:
        """
        Clear conversation context.
        """
        self.context.clear()
        logger.info("In class AgentCore, function clear_context: Context cleared")

    def get_metrics(self) -> AgentMetrics:
        """
        Get current agent metrics.

        :return: AgentMetrics object.
        """
        return self.metrics

    @classmethod
    def get_agent(cls, agent_name: str, session_id: str = "") -> "AgentCore":
        """
        Get agent by name from the live_agents registry.

        :param agent_name: Name of the agent (code_agent, review_agent, etc).
        :param session_id: Session identifier (unused, kept for compatibility).
        :return: Agent instance.
        :raises ValueError: If agent name is unknown.
        """
        instance = cls.live_agents.get(agent_name)
        if instance is None:
            raise ValueError(f"Unknown agent: {agent_name}")

        return instance
