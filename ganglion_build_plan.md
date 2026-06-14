# Ganglion — agentic AI build plan

_Captured from design session — June 2026_

---

## What we decided

A multi-agent system with three clear layers:

1. **Router** — semantic dispatch + model gateway. OSS, configured not built.
2. **Agent-core** — the base loop every agent inherits. Built once, reused everywhere.
3. **Specialist agents** — extend agent-core, add domain tools. Head agent is also a specialist.

Memory is a **stub** for now — interface defined, implementation deferred.

No LangChain / LangGraph needed. Plain Python + Anthropic SDK + two pip installs.

---

## Package structure

```
ganglion/
├── packages/
│   ├── router-pkg/          # configure OSS libs, no custom logic
│   ├── agent-core-pkg/      # build once — loop, metrics, LLM call, memory stub
│   ├── head-agent-pkg/      # extends agent-core, adds decompose + dispatch
│   ├── code-agent-pkg/      # extends agent-core, adds code tools
│   ├── search-agent-pkg/    # extends agent-core, adds retrieval tools
│   └── data-agent-pkg/      # extends agent-core, adds DB/transform tools
└── ganglion_build_plan.md   # this file
```

---

## Package 1 — `router-pkg`

### What it does
- Receives raw user query
- Semantic cache check first (skip everything if hit)
- Embeds query → cosine similarity → picks specialist agent
- Forwards to LiteLLM which maps route label → actual model endpoint

### Libraries (install, don't build)
```bash
pip install semantic-router   # aurelio-labs/semantic-router
pip install litellm           # BerriAI/litellm
```

### What you write
**Route definitions** — one per specialist agent, plain English:
```python
from semantic_router import Route, RouteLayer
from semantic_router.encoders import FastEmbedEncoder

routes = [
    Route(
        name="code_agent",
        utterances=[
            "write a function that",
            "debug this code",
            "refactor this class",
            "add unit tests for",
            "fix the bug in",
            "implement an API endpoint",
        ],
    ),
    Route(
        name="search_agent",
        utterances=[
            "find information about",
            "search for recent",
            "what does the documentation say",
            "look up",
            "retrieve context on",
        ],
    ),
    Route(
        name="data_agent",
        utterances=[
            "query the database",
            "transform this CSV",
            "aggregate results from",
            "write a SQL query",
            "parse this JSON",
        ],
    ),
]

encoder = FastEmbedEncoder()   # runs locally, ~90MB, no GPU needed
router = RouteLayer(encoder=encoder, routes=routes)
```

**LiteLLM config** — maps route label → model:
```yaml
# litellm_config.yaml
model_list:
  - model_name: local          # simple tasks
    litellm_params:
      model: ollama/codestral
      api_base: http://localhost:11434

  - model_name: mid            # moderate complexity
    litellm_params:
      model: deepseek/deepseek-coder
      api_key: os.environ/DEEPSEEK_KEY

  - model_name: cloud          # complex / ambiguous
    litellm_params:
      model: claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  enable_pre_call_checks: true
  redis_url: os.environ/REDIS_URL   # for semantic cache
```

### What the router returns
```python
# task packet passed to head agent
{
    "query": str,
    "route": "code_agent" | "search_agent" | "data_agent",
    "confidence": float,        # cosine sim score
    "cache_hit": bool,
    "timestamp": str
}
```

### Decision logic
```
confidence >= 0.75  →  route directly to specialist
confidence 0.45-0.75  →  route to head agent to decide
confidence < 0.45  →  head agent handles with LLM planning
```

---

## Package 2 — `agent-core-pkg`

### What it does
Base class every agent inherits. Defines the loop — every measurement, every
retry, every LLM call goes through here. Specialist agents never touch the loop.

### Key design decisions
- **`interleaved_thinking`** — keyword param, default `False`. When `True`, uses
  Anthropic SDK directly with `thinking` enabled so Claude reasons between tool
  calls. When `False`, uses LiteLLM standard completion (works with any model).
- **`thinking_budget`** — keyword param, default `5000` tokens. Only used when
  `interleaved_thinking=True`. Tune per agent based on task complexity.
- Both are set at agent instantiation, not per-task — each specialist decides
  whether it needs interleaved thinking based on its own task profile.

### When to use interleaved thinking

| Agent | Use it? | Why |
|---|---|---|
| Head agent | Yes | Needs to reason between decomposition steps |
| Code agent | Optional | Worth it for multi-file refactors, not boilerplate |
| Search agent | No | Linear: retrieve → summarise, no mid-task reasoning |
| Data agent | Optional | Worth it for complex query planning |

### The agent loop

```python
import time
import anthropic
import litellm
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskResult:
    task_id: str
    output: Any
    success: bool
    score: float               # 0.0–1.0 quality score
    cost_usd: float
    latency_ms: float
    model_used: str
    thinking_used: bool = False
    error: Optional[str] = None
    retries: int = 0


@dataclass
class AgentMetrics:
    agent_id: str
    session_id: str
    tasks_total: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    avg_score: float = 0.0
    model_usage: dict = field(default_factory=dict)
    thinking_calls: int = 0    # how often thinking fired


class AgentCore:
    """
    Base class for all agents. Extend this — do not modify the loop.
    Specialist agents override: tools, system_prompt, score_output(),
    select_model().

    Args:
        agent_id:             unique name for this agent instance
        session_id:           session identifier for grouping metrics
        interleaved_thinking: if True, use Anthropic SDK with thinking enabled.
                              Requires model to be claude-sonnet-4-6 or higher.
                              Default: False (uses LiteLLM, works with any model)
        thinking_budget:      max tokens for thinking blocks per call.
                              Only used when interleaved_thinking=True.
                              Default: 5000. Increase for complex reasoning tasks.
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        interleaved_thinking: bool = False,
        thinking_budget: int = 5000,
    ):
        self.agent_id = agent_id
        self.session_id = session_id
        self.interleaved_thinking = interleaved_thinking
        self.thinking_budget = thinking_budget
        self.metrics = AgentMetrics(agent_id=agent_id, session_id=session_id)
        self.memory = MemoryStub()
        self.context: list[dict] = []
        self.max_retries = 3
        self._anthropic = anthropic.Anthropic() if interleaved_thinking else None

    # ── override in subclass ──────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    @property
    def tools(self) -> list[dict]:
        return []

    def score_output(self, task: dict, output: str) -> float:
        """
        Return 0.0–1.0. Override per specialist.
        Default: crude length heuristic — replace this.
        """
        return min(1.0, len(output) / 500)

    def select_model(self, task: dict) -> str:
        """
        Return 'local', 'mid', or 'cloud'.
        Override per specialist for domain-specific complexity signals.
        When interleaved_thinking=True, 'cloud' must be claude-sonnet-4-6+.
        """
        if task.get("_escalate"):
            current = task.get("_last_model", "local")
            return {"local": "mid", "mid": "cloud"}.get(current, "cloud")
        tokens = len(task.get("content", "").split())
        if tokens < 100:
            return "local"
        elif tokens < 400:
            return "mid"
        return "cloud"

    # ── core loop — do not override ──────────────────────────────────────

    def run_task(self, task: dict) -> TaskResult:
        """Single task with measurement, escalation, and retry."""
        task_id = task.get("id", "unknown")
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            attempt += 1
            t0 = time.monotonic()

            try:
                model = self.select_model(task)
                task["_last_model"] = model
                messages = self._build_messages(task)

                if self.interleaved_thinking and model == "cloud":
                    output, cost = self._call_with_thinking(messages)
                    thinking_used = True
                else:
                    output, cost = self._call_litellm(model, messages)
                    thinking_used = False

                latency = (time.monotonic() - t0) * 1000
                score = self.score_output(task, output)

                result = TaskResult(
                    task_id=task_id,
                    output=output,
                    success=score >= 0.6,
                    score=score,
                    cost_usd=cost,
                    latency_ms=latency,
                    model_used=model,
                    thinking_used=thinking_used,
                    retries=attempt - 1,
                )

                self._update_metrics(result)
                self._update_context(task, output)
                self.memory.write(task_id, result)

                if result.success:
                    return result

                task["_escalate"] = True

            except Exception as e:
                last_error = str(e)
                latency = (time.monotonic() - t0) * 1000
                self._update_metrics(TaskResult(
                    task_id=task_id, output="", success=False,
                    score=0.0, cost_usd=0.0, latency_ms=latency,
                    model_used="unknown", error=last_error, retries=attempt,
                ))

        return TaskResult(
            task_id=task_id, output="", success=False, score=0.0,
            cost_usd=0.0, latency_ms=0.0, model_used="unknown",
            error=last_error, retries=self.max_retries,
        )

    def run_tasks(self, tasks: list[dict]) -> list[TaskResult]:
        """Sequential. For parallel: wrap in asyncio.gather()."""
        return [self.run_task(t) for t in tasks]

    # ── LLM call backends ────────────────────────────────────────────────

    def _call_litellm(self, model: str, messages: list[dict]) -> tuple[str, float]:
        """Standard call via LiteLLM — works with any model tier."""
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=self.tools or None,
        )
        output = response.choices[0].message.content or ""
        cost = litellm.completion_cost(response)
        return output, cost

    def _call_with_thinking(self, messages: list[dict]) -> tuple[str, float]:
        """
        Interleaved thinking via Anthropic SDK.
        Claude reasons between tool calls — loop continues until
        stop_reason is 'end_turn'. Only fires when interleaved_thinking=True
        and model tier is 'cloud'.
        """
        output_text = ""
        total_input = 0
        total_output = 0
        msgs = list(messages)

        while True:
            response = self._anthropic.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                },
                tools=self.tools or [],
                system=self.system_prompt,
                messages=msgs,
            )

            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            for block in response.content:
                if block.type == "text":
                    output_text += block.text
                # thinking blocks: log but don't surface as output
                # TODO: send to debug store / metrics

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_results = self._execute_tools(response.content)
                msgs.append({"role": "assistant", "content": response.content})
                msgs.append({"role": "user", "content": tool_results})

        # claude-sonnet-4-6: $3/M input, $15/M output
        cost = (total_input * 3 + total_output * 15) / 1_000_000
        return output_text, cost

    def _execute_tools(self, content_blocks: list) -> list[dict]:
        """
        Execute tool_use blocks → tool_result list.
        Override in subclass to wire actual tool implementations.
        Default: stub returning empty result.
        """
        results = []
        for block in content_blocks:
            if block.type == "tool_use":
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "",   # stub — override per specialist
                })
        return results

    # ── private helpers ───────────────────────────────────────────────────

    def _build_messages(self, task: dict) -> list[dict]:
        messages = []
        if not self.interleaved_thinking:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.context[-6:])
        messages.append({"role": "user", "content": task["content"]})
        return messages

    def _update_context(self, task: dict, output: str):
        self.context.append({"role": "user", "content": task["content"]})
        self.context.append({"role": "assistant", "content": output})
        if len(self.context) > 20:
            self.context = self.context[-20:]

    def _update_metrics(self, result: TaskResult):
        m = self.metrics
        m.tasks_total += 1
        if result.success:
            m.tasks_passed += 1
        else:
            m.tasks_failed += 1
        m.total_cost_usd += result.cost_usd
        m.avg_latency_ms = (
            (m.avg_latency_ms * (m.tasks_total - 1) + result.latency_ms)
            / m.tasks_total
        )
        m.avg_score = (
            (m.avg_score * (m.tasks_total - 1) + result.score)
            / m.tasks_total
        )
        m.model_usage[result.model_used] = (
            m.model_usage.get(result.model_used, 0) + 1
        )
        if result.thinking_used:
            m.thinking_calls += 1


# ── Memory stub ───────────────────────────────────────────────────────────

class MemoryStub:
    """
    No-op placeholder. Interface is fixed — swap implementation
    (Chroma / Qdrant / pgvector) without touching any agent code.
    """
    def write(self, key: str, value: Any) -> None:
        pass

    def read(self, key: str) -> Optional[Any]:
        return None

    def search(self, query: str, top_k: int = 5) -> list:
        return []
```

### Metric probes — what to measure per task

| Probe | What | Why |
|---|---|---|
| `score` | 0–1 from `score_output()` | catches quality degradation per agent |
| `cost_usd` | actual API cost | shows where spend concentrates |
| `latency_ms` | wall-clock per task | catches slow model tier choices |
| `model_used` | local / mid / cloud | shows routing distribution |
| `thinking_used` | bool | cost of thinking vs quality gain |
| `retries` | attempt count | signals threshold needs tuning |
| `success` | score >= threshold | aggregate pass rate |

### Escalation logic

```
attempt 1  →  select_model() picks tier from complexity signals
score < 0.6  →  _escalate=True, retry
attempt 2  →  select_model() sees _escalate, bumps one tier up
attempt 3  →  cloud model, last attempt
all fail  →  TaskResult(success=False), head agent re-routes
```

---

## Package 3 — `head-agent-pkg`

Head agent uses `interleaved_thinking=True` — it needs to reason between
decomposition and dispatch steps.

```python
class HeadAgent(AgentCore):

    def __init__(self, session_id: str):
        super().__init__(
            agent_id="head_agent",
            session_id=session_id,
            interleaved_thinking=True,   # reasons between planning steps
            thinking_budget=8000,        # higher budget — complex planning
        )

    @property
    def system_prompt(self) -> str:
        return """
        You are an orchestrator. Given a user query, decompose it into
        a list of atomic subtasks. Return JSON only:
        {"tasks": [{"id": str, "content": str, "agent": str, "depends_on": [str]}]}
        Agent must be one of: code_agent, search_agent, data_agent.
        """

    def run(self, task_packet: dict) -> dict:
        decomp_result = self.run_task({
            "content": task_packet["query"],
            "id": "decomp",
        })
        subtasks = self._parse_subtasks(decomp_result.output)
        results = {}
        for task in self._resolve_order(subtasks):
            agent = self._get_agent(task["agent"])
            results[task["id"]] = agent.run_task(task)
        return self._aggregate(results)

    def score_output(self, task, output) -> float:
        try:
            import json
            parsed = json.loads(output)
            return 1.0 if "tasks" in parsed and len(parsed["tasks"]) > 0 else 0.3
        except Exception:
            return 0.0
```

---

## Package 4–6 — specialist agents

Each extends `AgentCore`. Override three things. Set `interleaved_thinking`
per agent — not all agents need it.

```python
# Code agent — thinking is optional, passed in at instantiation
class CodeAgent(AgentCore):

    def __init__(self, session_id: str, interleaved_thinking: bool = False):
        super().__init__(
            agent_id="code_agent",
            session_id=session_id,
            interleaved_thinking=interleaved_thinking,
            thinking_budget=5000,
        )

    @property
    def system_prompt(self) -> str:
        return "You are an expert software engineer. ..."

    @property
    def tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": "run_code", "...": "..."}},
            {"type": "function", "function": {"name": "run_tests", "...": "..."}},
            {"type": "function", "function": {"name": "lint_file", "...": "..."}},
        ]

    def score_output(self, task, output) -> float:
        # stub — replace with actual test runner result
        return 0.8

    def select_model(self, task) -> str:
        if task.get("_escalate"):
            return {"local": "mid", "mid": "cloud"}.get(
                task.get("_last_model", "local"), "cloud"
            )
        tokens = len(task.get("content", "").split())
        if tokens < 60 and "boilerplate" in task.get("content", "").lower():
            return "local"
        elif tokens < 300:
            return "mid"
        return "cloud"


# Search agent — no thinking needed
class SearchAgent(AgentCore):

    def __init__(self, session_id: str):
        super().__init__(
            agent_id="search_agent",
            session_id=session_id,
            interleaved_thinking=False,   # linear: retrieve → summarise
        )


# Data agent — thinking optional
class DataAgent(AgentCore):

    def __init__(self, session_id: str, interleaved_thinking: bool = False):
        super().__init__(
            agent_id="data_agent",
            session_id=session_id,
            interleaved_thinking=interleaved_thinking,
            thinking_budget=4000,
        )
```

### Instantiation examples

```python
# default — no thinking, cheap, works with any model
agent = CodeAgent(session_id="sess_001")

# complex refactor session — enable thinking
agent = CodeAgent(session_id="sess_002", interleaved_thinking=True)

# head agent always uses thinking (hardcoded in __init__)
head = HeadAgent(session_id="sess_001")
```

---

## TODO — ordered build sequence

### Phase 1 — router-pkg (week 1)
- [ ] `pip install semantic-router litellm`
- [ ] Write route descriptions for 3 agents
- [ ] Set up `litellm_config.yaml` with local (Ollama) + cloud (Claude) endpoints
- [ ] Test: does cosine sim route correctly on 20 sample queries?
- [ ] Add semantic cache (Redis or in-memory dict for now)
- [ ] Measure: cache hit rate, routing confidence distribution

### Phase 2 — agent-core-pkg (week 1–2)
- [ ] Implement `AgentCore` with `interleaved_thinking` and `thinking_budget` params
- [ ] Implement `_call_litellm()` — standard path, any model
- [ ] Implement `_call_with_thinking()` — Anthropic SDK path, cloud only
- [ ] Implement `MemoryStub`, `TaskResult`, `AgentMetrics`
- [ ] Test: `interleaved_thinking=False` → LiteLLM call, metrics populated
- [ ] Test: `interleaved_thinking=True` → thinking loop runs, `thinking_calls` increments
- [ ] Test escalation: force low score, verify model bumps one tier
- [ ] Compare same task with/without thinking: is score delta worth cost delta?

### Phase 3 — head-agent-pkg (week 2)
- [ ] Extend `AgentCore` → `HeadAgent`
- [ ] Implement decomposition prompt (JSON output)
- [ ] Implement `score_output()` — valid JSON with tasks array
- [ ] Implement dependency resolver (topological sort on `depends_on`)
- [ ] Test: decompose 5 real queries, check subtask quality

### Phase 4 — specialist agents (week 2–3)
- [ ] `CodeAgent` — extend core, stub `score_output()` (test runner later)
- [ ] `SearchAgent` — extend core, wire web_fetch / RAG tool
- [ ] `DataAgent` — extend core, wire DB query tool
- [ ] For each: test 10 tasks, check `avg_score` and `model_usage` distribution

### Phase 5 — measurement & autoresearch loop (week 3–4)
- [ ] Log `AgentMetrics` to file / SQLite after each session
- [ ] Build simple dashboard: per-agent score, cost, latency, model split
- [ ] Identify the leakage point: which agent has lowest avg_score?
- [ ] Run autoresearch loop: change one thing (prompt / threshold / route description), measure delta
- [ ] Commit improvement or revert (karpathy ratchet pattern)

### Phase 6 — RouteLLM (week 4+, needs data)
- [ ] Collect (query, model_used, score) triples from production logs
- [ ] Train RouteLLM classifier once ~1,500 labelled examples exist
- [ ] Swap into LiteLLM as router policy — no agent code changes needed

### Memory (deferred — stub until phase 5+)
- [ ] Define `MemoryInterface` (read / write / search)
- [ ] `MemoryStub` is the placeholder — agents already call it, no-op for now
- [ ] Wire real implementation (Chroma / Qdrant / pgvector) when ready
- [ ] Conflict resolver: recency · confidence · provenance · consensus scoring

---

## IDE usage

### Starting a session
```
Read ganglion_build_plan.md.
We are on Phase [X].
Help me implement [specific TODO item].
Keep all code consistent with the AgentCore base class in this file.
```

### Enabling thinking for a session
```
Read ganglion_build_plan.md.
Implement _call_with_thinking() in agent-core.
Use interleaved_thinking=True for HeadAgent and test on a decomposition task.
```

### What to update in this file as you build
- Route descriptions once tested and tuned
- Real `score_output()` per agent once test runners are wired
- Metric baselines from first real session (avg_score, cost/task, latency)
- Thinking cost/quality tradeoff findings per agent

---

## Key principles

1. **Loop in agent-core, not in specialists** — measurement is consistent everywhere
2. **`interleaved_thinking` is a param, not a hardcode** — toggle per agent, per session
3. **Router routes to agents · agents route to models** — separate concerns
4. **Memory is a stub** — interface fixed now, implementation later, zero agent changes
5. **One change at a time** — autoresearch ratchet, keep or revert
6. **Measure leakage per agent** — score every task, not just final output
7. **Local first, cloud on complexity** — cost discipline from day one
8. **No LangChain / LangGraph** — plain Python + Anthropic SDK + LiteLLM
