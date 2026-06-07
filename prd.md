# Product Requirements Document (PRD)

# ModelDirector

## Vision

ModelDirector is an open-source, stateless AI model selection engine that determines the cheapest capable model for a given task before execution.

Instead of hardcoding model choices, ModelDirector evaluates a prompt against a user-defined set of candidate models and returns confidence scores for each model, reasoning behind the scores, and the optimal model selection according to configurable policies.

ModelDirector does not execute prompts and does not act as an AI gateway.

Its sole responsibility is deciding:

> "Which model should handle this request?"

ModelDirector should be usable by:

- OpenClaw
- Hermes
- Claude Code wrappers
- Codex wrappers
- Gemini CLI wrappers
- Continue
- Roo Code
- Cline
- OpenCode
- Custom agents
- MCP clients
- Python applications
- FastAPI services

The project must be provider-agnostic and model-agnostic.

---

## Core Principles

### Stateless

- No database.
- No user storage.
- No learning.
- No analytics.
- No telemetry.
- No persistence.

Every request is evaluated independently.

---

### Model Agnostic

ModelDirector must never contain hardcoded model assumptions.

Users define their own models. The engine does not know what "cheap" or
"premium" mean. The model id is just a name; the policy decides.

Examples:

```yaml
models:
  - id: gpt5mini
    name: openai/gpt-4o-mini
    cost: { input: 0.15, output: 0.60 }
  - id: sonnet
    name: anthropic/claude-3.5-sonnet
    cost: { input: 3.00, output: 15.00 }
  - id: opus
    name: anthropic/claude-opus-4
    cost: { input: 15.00, output: 75.00 }
```

Or:

```yaml
models:
  - id: local-qwen
    name: ollama/qwen3-32b
    cost: { input: 0.0, output: 0.0 }
  - id: deepseek
    name: openrouter/deepseek-chat
    cost: { input: 0.27, output: 1.10 }
  - id: gemini
    name: google/gemini-2.5-pro
    cost: { input: 1.25, output: 5.00 }
```

Or:

```yaml
models:
  - id: my-finetune
    name: openai/ft:gpt-4o-mini:my-org:custom:abc123
    cost: { input: 0.30, output: 1.20 }
```

ModelDirector should not care. The engine treats every profile uniformly.

---

### Configuration First

Everything is configurable.

- No fixed tiers.
- No fixed model names.
- No fixed providers.

Users define:

- candidate models
- ranking
- selection policies
- confidence thresholds

---

## Problem Statement

Current AI systems force users to manually choose models.

Users repeatedly ask:

- Should I use GPT-5 Mini?
- Should I use Sonnet?
- Should I use Opus?
- Should I use Gemini?
- Should I use DeepSeek?

Most users select expensive models unnecessarily.

ModelDirector automates this decision.

---

## High Level Flow

```
Prompt
   |
   v
ModelDirector
   |
   +--> Score Model A
   +--> Score Model B
   +--> Score Model C
   |
   v
Decision Policy
   |
   v
Selected Model
```

---

## Architecture

### Components

#### Selector Engine

Responsible for scoring candidate models.

Input:

```json
{
  "prompt": "...",
  "models": [...]
}
```

Output:

```json
{
  "scores": {}
}
```

---

#### Policy Engine

Responsible for selecting the final model.

Input:

```json
{
  "scores": {}
}
```

Output:

```json
{
  "selected_model": "..."
}
```

---

#### Adapters

Interfaces for:

- Python SDK
- MCP Server
- REST API
- CLI

All adapters use the same internal engine.

---

## Scoring System

For every candidate model, ModelDirector must estimate:

### Success Confidence

Probability that the model can successfully complete the task.

Range: `0-100`

### Reasoning Confidence

Probability that reasoning depth is sufficient.

Range: `0-100`

### Context Confidence

Probability that context requirements fit the model.

Range: `0-100`

### Coding Confidence

Probability that coding quality is sufficient.

Range: `0-100`

### Overall Confidence

Weighted score.

Range: `0-100`

---

## Model Profile System

Users define model capabilities, strengths, and per-1M-token cost.

Example:

```yaml
models:
  - id: gpt5mini
    name: gpt-4o-mini
    display_name: GPT-4o mini
    description: |
      OpenAI's small, fast, low-cost general-purpose model. Good for
      simple classification, routing, and short-form tasks. Weaker at
      long-horizon reasoning and large code refactors.
    strengths:
      - classification
      - routing
      - short_summarisation
      - simple_qa
    capabilities:
      reasoning: 80
      coding: 85
      context: 80
      creativity: 70
    priority: 1
    cost:
      input: 0.15
      output: 0.60

  - id: sonnet
    name: claude-3.5-sonnet
    display_name: Claude 3.5 Sonnet
    description: |
      Anthropic's mid-tier model. Strong at coding, instruction
      following, and long-context reasoning. 200k context window. Default
      workhorse for most agentic tasks.
    strengths:
      - coding
      - architecture
      - refactoring
      - long_context
    capabilities:
      reasoning: 90
      coding: 95
      context: 95
      creativity: 85
    priority: 2
    cost:
      input: 3.00
      output: 15.00

  - id: opus
    name: claude-opus-4
    display_name: Claude Opus 4
    description: |
      Anthropic's flagship model. Best-in-class reasoning, complex
      multi-step planning, and nuanced code generation. Use only when
      cheaper models are unlikely to succeed.
    strengths:
      - hard_reasoning
      - complex_coding
      - architecture_design
    capabilities:
      reasoning: 99
      coding: 98
      context: 99
      creativity: 95
    priority: 3
    cost:
      input: 15.00
      output: 75.00
```

The `description` field is strongly recommended. Smaller selector models
may not have intrinsic knowledge of what "Sonnet", "Opus", "deepseek-r1",
or a user's custom model name means. The description gives the selector
enough context to score accurately and is always passed into the
generated prompt alongside the model name and capability scores.

`description` is optional in the schema, but profiles without one rely on
the selector model recognizing the model name on its own.

### Strengths (recommended)

```yaml
strengths:
  - coding
  - architecture
  - long_context
```

`strengths` is a structured list of task-type tags the model is good at.
Small selector models frequently don't know the latest capabilities of
every model. Tagging strengths explicitly is more reliable than relying
on the model name being recognised.

### Cost (required)

```yaml
cost:
  input: 0.15      # USD per 1M input tokens
  output: 0.60     # USD per 1M output tokens
```

A bare number (`cost: 1`) is also accepted for back-compat and is
treated as `cost: {input: 1, output: 1}`. New configs should always use
the `{input, output}` form.

Users can create unlimited profiles.

---

## Evaluation Strategy

ModelDirector uses a selector model.

The selector model is configurable.

Examples:

```yaml
selector_model:
  provider: openai
  model: gpt-5-mini
```

Or:

```yaml
selector_model:
  provider: anthropic
  model: claude-sonnet
```

Or:

```yaml
selector_model:
  provider: ollama
  model: qwen3
```

The selector model evaluates the prompt against candidate models.

---

## Selector Prompt

ModelDirector dynamically generates a prompt containing:

- user task
- model profiles (id, name, display name, description, **strengths**,
  capabilities, **cost per 1M tokens** in USD, priority)
- scoring instructions

The selector must return JSON only.

Example:

```json
{
  "models": [
    {
      "id": "gpt5mini",
      "overall": 87,
      "reasoning": 82,
      "coding": 91,
      "context": 85,
      "explanation": "..."
    }
  ]
}
```

---

## Decision Policies

### Cheapest Capable

Default.

Select the first model (by priority asc, then by input cost asc) whose
confidence exceeds the configured threshold.

Example:

```yaml
threshold: 85
```

Scores:

- gpt5mini = 89
- sonnet = 94
- opus = 99

Result: **gpt5mini**

### Highest Confidence

Select highest score.

Scores:

- gpt5mini = 89
- sonnet = 94
- opus = 99

Result: **opus**

### Best Value

Score: `confidence / cost.input`

Highest ratio wins. Uses `cost.input` (deterministic — the user always
pays for input tokens and we know the input length) rather than a
blended input+output figure (which would require guessing the output
length). Renamed from the legacy `balanced` policy (v0.0.x).

### User Defined

Users may implement custom policies by subclassing `Policy`.

---

## Output Format

### Standard Output

```json
{
  "selected_model": "sonnet",
  "policy": "cheapest_capable",
  "scores": {
    "gpt5mini": {
      "overall": 74,
      "reasoning": 70,
      "coding": 82,
      "context": 71,
      "explanation": "..."
    },
    "sonnet": {
      "overall": 91,
      "reasoning": 90,
      "coding": 95,
      "context": 93,
      "explanation": "..."
    },
    "opus": {
      "overall": 97,
      "reasoning": 98,
      "coding": 96,
      "context": 99,
      "explanation": "..."
    }
  },
  "estimated_cost_usd": {
    "gpt5mini": 0.0001,
    "sonnet":   0.0020,
    "opus":     0.0100
  },
  "input_tokens": 7,
  "reason": "'sonnet' is the first model exceeding the configured threshold of 85 (overall=91, priority=2, cost=$3.00/1M in, $15.00/1M out)."
}
```

`estimated_cost_usd` is a first-class field: a per-model USD cost
estimate for the actual prompt, computed from each model's
per-1M-token cost and a token-count estimate of the input. The
caller can render this directly to show users the cost trade-off.

---

## Interfaces

### Python SDK

```python
from modeldirector import ModelDirector

director = ModelDirector(config)

result = director.select(
    prompt=user_prompt
)
```

### MCP Server

Tool: `select_model`

Input:

```json
{
  "prompt": "...",
  "models": [...]
}
```

Output:

```json
{
  "selected_model": "..."
}
```

### REST API

Endpoint: `POST /select`

### CLI

```bash
modeldirector select prompt.txt
```

Output:

```json
{
  "..."
}
```

---

## Configuration

Single YAML file.

Example:

```yaml
selector:
  provider: openai
  model: gpt-4o-mini

policy:
  type: cheapest_capable
  threshold: 85

models:
  - id: gpt5mini
    name: openai/gpt-4o-mini
    description: OpenAI's small, fast, low-cost model for simple tasks.
    strengths: [classification, routing, simple_qa]
    cost: { input: 0.15, output: 0.60 }

  - id: sonnet
    name: anthropic/claude-3.5-sonnet
    description: Anthropic's mid-tier model. Strong at coding and long-context reasoning.
    strengths: [coding, architecture, long_context]
    cost: { input: 3.00, output: 15.00 }

  - id: opus
    name: anthropic/claude-opus-4
    description: Anthropic's flagship. Best reasoning, used when cheaper models are unlikely to succeed.
    strengths: [hard_reasoning, complex_coding, architecture_design]
    cost: { input: 15.00, output: 75.00 }
```

---

## Technology Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI
- **MCP:** FastMCP
- **Configuration:** Pydantic, PyYAML
- **Provider Layer:** LiteLLM
- **Packaging:** uv
- **Testing:** pytest

---

## Repository Structure

```
modeldirector/
├── modeldirector/           # The engine
│   ├── config.py            # Pydantic config models (Cost, ModelProfile, ...)
│   ├── loader.py            # YAML / dict loader, ${ENV} expansion
│   ├── models.py            # Public types: ModelScore, SelectionResult
│   ├── policy.py            # Policy engine (3 built-ins + custom)
│   ├── selector.py          # Selector engine + USD cost estimator
│   └── adapters/            # The interfaces (derived from the engine)
│       ├── cli.py
│       ├── rest.py
│       └── mcp.py
├── benchmarks/              # 30-task real-LLM benchmark
├── examples/                # ready-to-use config
├── tests/                   # 43 unit + 4 integration tests
├── docs/                    # hero image
├── prd.md
├── pyproject.toml
└── README.md
```

The engine is the product. The adapters are derived. Adding a new
interface is a matter of constructing a `ModelDirector` and translating
the input/output to the adapter's protocol.

---

## Success Criteria

ModelDirector is successful when any AI agent can call a single function:

```python
selected_model = director.select(
    prompt=user_prompt
)
```

and receive a deterministic, explainable recommendation based on:

- configurable models
- configurable policies
- confidence scoring

without any dependency on a specific provider, model family, database, or AI framework.
