"""
Agent core package for Ganglion multi-agent system.

Provides the base agent loop that all specialist agents inherit.
Handles LLM calls, metrics, retries, and optional interleaved thinking.
"""

from ganglion.agents_hub.core import AgentCore
from ganglion.agents_hub.models import TaskResult, AgentMetrics, Task
from ganglion.agents_hub.memory import MemoryStub, AbstractMemory, InMemoryStore

__all__ = [
    "AgentCore",
    "TaskResult",
    "AgentMetrics",
    "Task",
    "MemoryStub",
    "AbstractMemory",
    "InMemoryStore",
]
