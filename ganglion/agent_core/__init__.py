"""
Agent core package for Ganglion multi-agent system.

Provides the base agent loop that all specialist agents inherit.
Handles LLM calls, metrics, retries, and optional interleaved thinking.
"""

from ganglion.agent_core.core import AgentCore
from ganglion.agent_core.models import TaskResult, AgentMetrics, Task
from ganglion.agent_core.memory import MemoryStub, AbstractMemory, InMemoryStore

__all__ = [
    "AgentCore",
    "TaskResult",
    "AgentMetrics",
    "Task",
    "MemoryStub",
    "AbstractMemory",
    "InMemoryStore",
]
