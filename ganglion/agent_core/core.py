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
import anthropic
import litellm

# App imports
from ganglion.agent_core.models import Task, TaskResult, AgentMetrics
from ganglion.agent_core.memory import MemoryStub, AbstractMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_THINKING_BUDGET = 5000
DEFAULT_MAX_RETRIES = 3
DEFAULT_SUCCESS_THRESHOLD = 0.6
DEFAULT_MAX_CONTEXT_MESSAGES = 20

# Model tier constants
MODEL_LOCAL = "local"
MODEL_MID = "mid"
MODEL_CLOUD = "cloud"

# Claude model for thinking mode
CLAUDE_THINKING_MODEL = "claude-sonnet-4-6"

# Cost per million tokens (Claude Sonnet 4)
CLAUDE_INPUT_COST_PER_M = 3.0
CLAUDE_OUTPUT_COST_PER_M = 15.0


# ── Agent Core Class ──────────────────────────────────────────────────────────

class AgentCore:
    """
    Base class for all agents. Extend this — do not modify the loop.

    Description: Provides the core execution loop with LLM calls, metrics,
        retries, and optional interleaved thinking. Specialist agents
        override: system_prompt, tools, score_output(), select_model().

    :param agent_id: Unique name for this agent instance.
    :param session_id: Session identifier for grouping metrics.
    :param interleaved_thinking: If True, use Anthropic SDK with thinking
        enabled. Requires cloud model to be Claude. Default: False.
    :param thinking_budget: Max tokens for thinking blocks per call.
        Only used when interleaved_thinking=True. Default: 5000.
    :param memory: Memory backend instance. Defaults to MemoryStub.
    :param max_retries: Maximum retry attempts per task. Default: 3.
    :param success_threshold: Minimum score for task success. Default: 0.6.
    """

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

        # Initialize Anthropic client for thinking mode
        if interleaved_thinking:
            self._anthropic = anthropic.Anthropic()
            logger.info(
                f"In class AgentCore, function __init__: Interleaved thinking enabled "
                f"with budget={thinking_budget}"
            )
        else:
            self._anthropic = None

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

        :param task: Task to execute.
        :return: Model tier string (local/mid/cloud).
        """
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

                # Call LLM
                if self.interleaved_thinking and model == MODEL_CLOUD:
                    output, cost = self._call_with_thinking(messages)
                    thinking_used = True
                else:
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

    def _call_with_thinking(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, float]:
        """
        LLM call with interleaved thinking via Anthropic SDK.

        Description: Uses Claude's extended thinking feature. Model reasons
            between tool calls. Loop continues until stop_reason is 'end_turn'.

        :param messages: Message list for completion.
        :return: Tuple of (output_text, cost_usd).
        """
        logger.info("In class AgentCore, function _call_with_thinking: Entered")

        output_text = ""
        total_input = 0
        total_output = 0
        msgs = list(messages)

        while True:
            response = self._anthropic.messages.create(
                model=CLAUDE_THINKING_MODEL,
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                },
                tools=self._convert_tools_for_anthropic(),
                system=self.system_prompt,
                messages=msgs,
            )

            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            # Process response content blocks
            for block in response.content:
                if block.type == "text":
                    output_text += block.text
                elif block.type == "thinking":
                    # Log thinking but don't include in output
                    logger.debug(f"Thinking block: {block.thinking[:100]}...")

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_results = self._execute_tools(response.content)
                msgs.append({"role": "assistant", "content": response.content})
                msgs.append({"role": "user", "content": tool_results})

        # Calculate cost (Claude Sonnet 4: $3/M input, $15/M output)
        cost = (total_input * CLAUDE_INPUT_COST_PER_M + total_output * CLAUDE_OUTPUT_COST_PER_M) / 1_000_000

        logger.info(
            f"In class AgentCore, function _call_with_thinking: Completed "
            f"output_len={len(output_text)}, cost=${cost:.6f}"
        )

        return output_text, cost

    def _convert_tools_for_anthropic(self) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Anthropic format.

        :return: Tools in Anthropic API format.
        """
        if not self.tools:
            return []

        anthropic_tools = []
        for tool in self.tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })

        return anthropic_tools

    def _execute_tools(
        self,
        content_blocks: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute tool_use blocks and return tool_result list.

        Description: Calls execute_tool() for each tool_use block.
            Override execute_tool() in subclass for actual implementation.

        :param content_blocks: Response content blocks from Claude.
        :return: List of tool_result dicts.
        """
        results = []
        for block in content_blocks:
            if hasattr(block, "type") and block.type == "tool_use":
                tool_output = self.execute_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_output,
                })

        return results

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
