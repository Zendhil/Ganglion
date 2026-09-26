# Ganglion

A multi-agent AI system with intelligent routing, automatic quality-based escalation, and cost-optimized model selection.

## Overview

Ganglion is a production-ready multi-agent framework that combines semantic routing, LLM model orchestration, and workflow execution into a single cohesive system. It automatically routes queries to specialist agents, escalates to more powerful (and expensive) models only when needed, and provides comprehensive metrics for cost and quality optimization.

**Key Design Principles:**
- **Route to agents, not models** - Semantic router dispatches queries to specialist agents based on task type
- **Agents route to models** - Each agent intelligently selects model tier (local/mid/cloud) based on task complexity
- **Cost discipline from day one** - Start with cheap local models, escalate only on quality signals
- **Measure everything** - Every task execution records cost, latency, quality score, and model usage

## Features

- 🎯 **Semantic Routing** - Natural language query classification to specialist agents (code, search, data, review)
- 🔄 **Automatic Escalation** - Quality-based retry with model tier escalation (local → mid → cloud)
- 💰 **Cost Optimization** - LiteLLM integration for unified multi-provider LLM access with cost tracking
- 📊 **Built-in Metrics** - Per-task and per-agent metrics (cost, latency, quality score, model usage)
- 🔌 **Extensible Agent System** - Self-registering agents with dynamic graph construction
- 🧠 **Semantic Caching** - Optional similarity-based response caching to reduce redundant LLM calls
- 🌊 **LangGraph Orchestration** - State graph workflow with support for complex multi-agent coordination
- 🎭 **Fallback Handling** - Graceful degradation when no specialist agent is available

## Installation

### Prerequisites

- Python 3.11+
- (Optional) Local Ollama instance for `local` model tier
- API keys for cloud providers (Anthropic, DeepSeek, etc.)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Ganglion
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install semantic-router litellm langgraph anthropic pyyaml numpy
```

4. Set up environment variables:
```bash
export ANTHROPIC_API_KEY="your-key-here"
export DEEPSEEK_KEY="your-key-here"  # Optional
export REDIS_URL="redis://localhost:6379"  # Optional, for caching
```

## Quick Start

```python
import logging
from ganglion.orchestration import Orchestrator
from ganglion.agents_hub.agents import HeadAgent, TailAgent, CodeAgent, SearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize agents (they auto-register)
session_id = "demo_session"
head = HeadAgent(session_id=session_id)
tail = TailAgent(session_id=session_id)
code = CodeAgent(session_id=session_id)
search = SearchAgent(session_id=session_id)

# Create orchestrator
orchestrator = Orchestrator(session_id=session_id)

# Run a query
result = orchestrator.run(query="Write a Python function to calculate fibonacci numbers")

# Access results
print(result["final_output"])
print(f"Total cost: ${result['total_cost_usd']:.4f}")
```

### Streaming Results

```python
# Stream execution updates
for state_update in orchestrator.stream(query="Explain how transformers work"):
    print(state_update)
```

### Async Execution

```python
import asyncio

async def main():
    result = await orchestrator.arun(query="Query the user database for active accounts")
    print(result["final_output"])

asyncio.run(main())
```

## Configuration

### Model Tiers

Edit `ganglion/router/litellm_config.yaml` to configure model tiers:

```yaml
model_list:
  - model_name: local
    litellm_params:
      model: ollama/codestral
      api_base: http://localhost:11434

  - model_name: mid
    litellm_params:
      model: deepseek/deepseek-coder
      api_key: os.environ/DEEPSEEK_KEY

  - model_name: cloud
    litellm_params:
      model: claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
```

### Semantic Routes

Edit `ganglion/router/routes.yaml` to tune semantic routing:

```yaml
routes:
  - name: code_agent
    description: Handles code generation, debugging, refactoring
    utterances:
      - "write a function that"
      - "debug this code"
      - "refactor this class"
      # Add 10-20 diverse examples per route

thresholds:
  high: 0.75    # Route directly to specialist
  medium: 0.45  # Head agent decides
```

**Tuning tips:**
- Add 10-20 diverse utterances per route for best coverage
- Use natural phrasing, not keywords
- Test with `python -m ganglion.router.test_routes` after changes

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Router (Semantic Dispatch + Model Gateway)   │
│  - semantic-router: Query → Agent routing               │
│  - LiteLLM: Model tier → API endpoint mapping          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Agent Core (Execution Loop)                   │
│  - AgentCore base class: metrics, retries, escalation   │
│  - All specialist agents inherit from AgentCore         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Orchestration (LangGraph Workflow)            │
│  - State graph: START → head → specialist → tail → END  │
│  - Dynamic agent discovery via registry pattern         │
└─────────────────────────────────────────────────────────┘
```

### Agent Types

**Structural Agents:**
- `HeadAgent` - Entry point, routing validation, decomposition, fallback handling
- `TailAgent` - Aggregates results, formats final output

**Specialist Agents:**
- `CodeAgent` - Code generation, debugging, refactoring, testing
- `SearchAgent` - Information retrieval, documentation lookup, research
- `DataAgent` - Database queries, data transformation, analysis
- `ReviewAgent` - Code quality review, security checks, best practices

### Execution Flow

```
User Query
    ↓
[QueryRouter] Semantic matching → route + confidence
    ↓
[Orchestrator] Build initial state
    ↓
[LangGraph] START → HeadAgent
    ↓
[HeadAgent] Validate route, check confidence
    ├─ confidence < 0.45 → decompose query
    ├─ route unknown → FALLBACK (answer directly)
    └─ route valid → dispatch to specialist
    ↓
[Specialist] Execute task with auto-escalation
    ├─ select_model() → local/mid/cloud
    ├─ run_task() → LLM call via LiteLLM
    └─ score_output() → quality check
    ↓
[TailAgent] Aggregate + format final output
    ↓
[END] Return final state with metrics
```

## Adding Custom Agents

1. **Define agent class:**

```python
# In ganglion/agents_hub/agents.py
from ganglion.agents_hub import AgentCore, Task

class CustomAgent(AgentCore):
    def __init__(self, session_id: str):
        super().__init__(
            agent_id="custom_agent",  # Used for routing
            session_id=session_id,
            interleaved_thinking=False,
        )

    @property
    def system_prompt(self) -> str:
        return "You are a specialist in X. Your task is to..."

    @property
    def tools(self) -> list:
        return [
            # Tool definitions (optional)
        ]

    def score_output(self, task: Task, output: str) -> float:
        # Custom quality scoring (0.0-1.0)
        return 0.8

    def select_model(self, task: Task) -> str:
        # Custom model selection logic
        if task._escalate:
            return "cloud"
        return "mid"
```

2. **Add route definition:**

```yaml
# In ganglion/router/routes.yaml
routes:
  - name: custom_agent
    description: Handles X tasks
    utterances:
      - "example query 1"
      - "example query 2"
      # Add 10-20 examples
```

3. **Instantiate during initialization:**

```python
from ganglion.agents_hub.agents import CustomAgent

custom = CustomAgent(session_id=session_id)  # Auto-registers
```

The orchestrator will automatically discover and integrate your agent via the registry pattern.

## Metrics & Monitoring

### Per-Task Metrics

Every task execution records:
```python
result = agent.run_task(task)
print(f"Score: {result.score:.2f}")
print(f"Cost: ${result.cost_usd:.6f}")
print(f"Latency: {result.latency_ms:.0f}ms")
print(f"Model: {result.model_used}")
print(f"Retries: {result.retries}")
print(f"Success: {result.success}")
```

### Per-Agent Metrics

```python
metrics = agent.get_metrics()
print(f"Tasks total: {metrics.tasks_total}")
print(f"Pass rate: {metrics.tasks_passed / metrics.tasks_total:.1%}")
print(f"Avg score: {metrics.avg_score:.2f}")
print(f"Avg latency: {metrics.avg_latency_ms:.0f}ms")
print(f"Total cost: ${metrics.total_cost_usd:.4f}")
print(f"Model usage: {metrics.model_usage}")
```

### Session Metrics

```python
result = orchestrator.run(query="...")
print(f"Total cost: ${result['total_cost_usd']:.4f}")
print(f"Handled by: {result['handled_by']}")
print(f"Route confidence: {result['confidence']:.2f}")
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest ganglion/

# Run specific test suite
python -m pytest ganglion/router/test_routes.py
python -m pytest ganglion/agents_hub/test_core.py
python -m pytest ganglion/orchestration/test_graph.py

# Run with verbose output
python -m pytest -v ganglion/
```

### Test Routing

```bash
# Test semantic router
python -m ganglion.router.test_routes

# Test semantic cache
python -m ganglion.router.test_cache
```

### Project Structure

```
ganglion/
├── router/
│   ├── routes.py          # QueryRouter implementation
│   ├── routes.yaml        # Route definitions (EDIT THIS)
│   ├── cache.py           # Semantic caching
│   ├── config.py          # LiteLLM config loader
│   └── litellm_config.yaml # Model tiers (EDIT THIS)
├── agents_hub/
│   ├── core.py            # AgentCore base class (DO NOT EDIT LOOP)
│   ├── agents.py          # Specialist agent implementations
│   ├── prompts.py         # System prompts
│   ├── models.py          # Task, TaskResult, AgentMetrics
│   └── memory.py          # Memory interface (stub)
└── orchestration/
    ├── orchestrator.py    # LangGraph orchestrator
    └── state.py           # OrchestratorState schema
```

## Design Patterns

### Agent Registry Pattern

All agents self-register on instantiation:
```python
class AgentCore:
    live_agents: dict[str, "AgentCore"] = {}

    def __init__(self, agent_id: str, ...):
        AgentCore.live_agents[agent_id] = self  # Auto-register
```

The orchestrator dynamically discovers agents when building the graph:
```python
for agent_id in AgentCore.live_agents.keys():
    graph.add_node(agent_id, AgentCore.get_agent(agent_id))
```

**Benefit:** Adding new agents automatically extends the graph without code changes.

### Quality-Based Escalation

```
Attempt 1: select_model() picks tier from complexity signals
  ↓
LLM call + score_output()
  ↓
score < 0.6? → Set _escalate=True, retry
  ↓
Attempt 2: select_model() sees _escalate, bumps tier (local→mid→cloud)
  ↓
Attempt 3: Cloud model, final attempt
  ↓
All fail: TaskResult(success=False)
```

### Memory Stub Pattern

Memory uses interface abstraction:
```python
class AbstractMemory:
    def write(self, key: str, value: Any) -> None: ...
    def read(self, key: str) -> Optional[Any]: ...
    def search(self, query: str, top_k: int = 5) -> list: ...
```

Current implementation is `MemoryStub` (no-op). Swap for Chroma/Qdrant/pgvector without touching agent code.

## Cost Optimization Strategies

1. **Start local:** Set default model tier to `local` in `select_model()`
2. **Tune thresholds:** Adjust `success_threshold` per agent based on task requirements
3. **Cache aggressively:** Enable semantic caching for repeated queries
4. **Profile agents:** Use metrics to identify which agents have low scores (need better prompts or higher tier models)
5. **Optimize utterances:** Better semantic routing reduces head agent LLM calls

## Roadmap

- [ ] **Parallel execution** - LangGraph Send API for concurrent subtasks
- [ ] **Memory implementation** - Chroma/Qdrant integration for RAG
- [ ] **RouteLLM integration** - ML-based model selection from production logs
- [ ] **Streaming responses** - Real-time token streaming from specialists
- [ ] **Tool execution** - Actual tool implementations for code/search/data agents
- [ ] **Web UI** - Dashboard for metrics and session monitoring

## Contributing

When contributing:
1. **Never modify the agent loop** in `AgentCore.run_task()`
2. Use the registry pattern for new agents
3. Add semantic routing utterances for discoverability
4. Include quality scoring logic in `score_output()`
5. Add tests for new agents/features
6. Follow existing logging patterns

## License

[Your License Here]

## Resources

- **Design Document:** See `ganglion_build_plan.md` for detailed architecture decisions
- **AI Development Guide:** See `CLAUDE.md` for guidance on working with this codebase using AI assistants

---

**Built with:** Python • LiteLLM • semantic-router • LangGraph • Anthropic Claude