"""
Data models for agent-core.

Description: Defines TaskResult, AgentMetrics, and Task dataclasses
    used throughout the agent system for consistent data structures.
"""

# Standard library imports
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Task Schema ───────────────────────────────────────────────────────────────

@dataclass
class Task:
    """
    Task input for agent processing.

    :param id: Unique task identifier.
    :param content: Task content/prompt string.
    :param metadata: Optional additional metadata.
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal fields for escalation tracking
    _escalate: bool = field(default=False, repr=False)
    _last_model: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary for message building.

        :return: Dictionary representation.
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "_escalate": self._escalate,
            "_last_model": self._last_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """
        Create Task from dictionary.

        :param data: Dictionary with task fields.
        :return: Task instance.
        """
        return cls(
            id=data.get("id", "unknown"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            _escalate=data.get("_escalate", False),
            _last_model=data.get("_last_model"),
        )


# ── Task Result Schema ────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """
    Result of task execution with metrics.

    :param task_id: ID of the executed task.
    :param output: Model output string or structured data.
    :param success: Whether task met quality threshold.
    :param score: Quality score (0.0-1.0).
    :param cost_usd: API cost in USD.
    :param latency_ms: Execution time in milliseconds.
    :param model_used: Model tier used (local/mid/cloud).
    :param thinking_used: Whether interleaved thinking was used.
    :param error: Error message if task failed.
    :param retries: Number of retry attempts.
    """

    task_id: str
    output: Any
    success: bool
    score: float
    cost_usd: float
    latency_ms: float
    model_used: str
    thinking_used: bool = False
    error: Optional[str] = None
    retries: int = 0


# ── Agent Metrics Schema ──────────────────────────────────────────────────────

@dataclass
class AgentMetrics:
    """
    Aggregate metrics for an agent session.

    :param agent_id: Unique agent identifier.
    :param session_id: Session identifier for grouping.
    :param tasks_total: Total tasks processed.
    :param tasks_passed: Tasks that met quality threshold.
    :param tasks_failed: Tasks that failed or didn't meet threshold.
    :param total_cost_usd: Cumulative API cost.
    :param avg_latency_ms: Running average latency.
    :param avg_score: Running average quality score.
    :param model_usage: Count of tasks per model tier.
    :param thinking_calls: Number of times thinking was used.
    """

    agent_id: str
    session_id: str
    tasks_total: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    avg_score: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)
    thinking_calls: int = 0

    @property
    def pass_rate(self) -> float:
        """
        Calculate task pass rate.

        :return: Pass rate as float (0.0-1.0).
        """
        if self.tasks_total == 0:
            return 0.0
        return self.tasks_passed / self.tasks_total

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary for logging/export.

        :return: Dictionary representation.
        """
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tasks_total": self.tasks_total,
            "tasks_passed": self.tasks_passed,
            "tasks_failed": self.tasks_failed,
            "pass_rate": self.pass_rate,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_score": self.avg_score,
            "model_usage": self.model_usage,
            "thinking_calls": self.thinking_calls,
        }
