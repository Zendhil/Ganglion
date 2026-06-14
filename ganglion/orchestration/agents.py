"""
Specialist agent implementations for orchestration.

Description: Defines CodeAgent and ReviewAgent that extend AgentCore.
    These are wrapped as LangGraph nodes in nodes.py.
"""

# Standard library imports
import json
import logging
from typing import Any, Dict, List, Optional

# App imports
from ganglion.agent_core import AgentCore, Task, TaskResult

logger = logging.getLogger(__name__)


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
        super().__init__(
            agent_id="code_agent",
            session_id=session_id,
            interleaved_thinking=interleaved_thinking,
            thinking_budget=5000,
        )

    @property
    def system_prompt(self) -> str:
        return """You are an expert software engineer. Your task is to write clean,
efficient, and well-documented code.

When given a coding task:
1. Understand the requirements clearly
2. Plan your approach before coding
3. Write clean, readable code with appropriate comments
4. Follow best practices and design patterns
5. Consider edge cases and error handling

Return your code in markdown code blocks with the appropriate language tag.
Include brief explanations of your implementation choices."""

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
        score = 0.0

        # Check for code blocks
        if "```" in output:
            score += 0.4

        # Check for reasonable length
        if len(output) > 100:
            score += 0.2

        # Check for function/class definitions
        if "def " in output or "class " in output or "function " in output:
            score += 0.2

        # Check for comments/documentation
        if "#" in output or "//" in output or '"""' in output:
            score += 0.2

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
        return """You are an expert code reviewer. Review the provided code for:

1. **Correctness**: Does the code do what it's supposed to?
2. **Code Quality**: Is it readable, maintainable, and well-structured?
3. **Best Practices**: Does it follow language idioms and patterns?
4. **Security**: Are there any security vulnerabilities?
5. **Edge Cases**: Are edge cases handled properly?

Respond with a JSON object:
{
    "passed": true/false,
    "score": 0.0-1.0,
    "summary": "Brief overall assessment",
    "issues": ["list of specific issues found"],
    "suggestions": ["list of improvement suggestions"]
}

Be constructive but thorough. Only pass code that meets quality standards."""

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
        return """You are an expert research assistant. Your task is to find and
synthesize information based on user queries.

When given a search task:
1. Identify key concepts and search terms
2. Retrieve relevant information
3. Synthesize findings into a clear summary
4. Cite sources when available

Provide comprehensive but concise answers."""

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
        return """You are an expert data engineer and analyst. Your task is to
work with databases, transform data, and perform analysis.

When given a data task:
1. Understand the data schema and requirements
2. Write efficient queries or transformations
3. Validate results and handle edge cases
4. Explain your approach clearly

Return SQL queries, transformation code, or analysis results as appropriate."""

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


# ── Agent Factory ─────────────────────────────────────────────────────────────

def get_agent(agent_name: str, session_id: str) -> AgentCore:
    """
    Factory function to get agent by name.

    :param agent_name: Name of the agent (code_agent, review_agent, etc).
    :param session_id: Session identifier.
    :return: Agent instance.
    :raises ValueError: If agent name is unknown.
    """
    agents = {
        "code_agent": lambda: CodeAgent(session_id),
        "review_agent": lambda: ReviewAgent(session_id),
        "search_agent": lambda: SearchAgent(session_id),
        "data_agent": lambda: DataAgent(session_id),
    }

    factory = agents.get(agent_name)
    if factory is None:
        raise ValueError(f"Unknown agent: {agent_name}")

    return factory()
