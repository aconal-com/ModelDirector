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

Users define their own models.

Examples:

```yaml
models:
  - id: cheap
    name: gpt-5-mini
  - id: medium
    name: claude-sonnet
  - id: premium
    name: claude-opus
```

Or:

```yaml
models:
  - id: cheap
    name: gpt-4o
  - id: medium
    name: minimax-m1
  - id: premium
    name: minimax-m2
```

Or:

```yaml
models:
  - id: cheap
    name: qwen3-32b
  - id: medium
    name: deepseek-r1
  - id: premium
    name: gemini-2.5-pro
```

ModelDirector should not care.

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

Users define model capabilities.

Example:

```yaml
models:
  - id: gpt5mini
    name: gpt-5-mini
    display_name: GPT-5 Mini
    description: OpenAI's small, fast, low-cost general-purpose model. Good for simple classification, routing, and short-form tasks. Weaker at long-horizon reasoning and large code refactors.
    capabilities:
      reasoning: 80
      coding: 85
      context: 80
      creativity: 70
    priority: 1

  - id: sonnet
    name: claude-sonnet
    display_name: Claude Sonnet
    description: Anthropic's mid-tier model. Strong at coding, instruction following, and long-context reasoning. 200k context window. Default workhorse for most agentic tasks.
    capabilities:
      reasoning: 90
      coding: 95
      context: 95
      creativity: 85
    priority: 2

  - id: opus
    name: claude-opus
    display_name: Claude Opus
    description: Anthropic's flagship model. Best-in-class reasoning, complex multi-step planning, and nuanced code generation. Use only when cheaper models are unlikely to succeed.
    capabilities:
      reasoning: 99
      coding: 98
      context: 99
      creativity: 95
    priority: 3
```

The `description` field is strongly recommended. Smaller selector models may not have intrinsic knowledge of what "Sonnet", "Opus", "deepseek-r1", or a user's custom model name means. The description gives the selector enough context to score accurately and is always passed into the generated prompt alongside the model name and capability scores.

`description` is optional in the schema, but profiles without one rely on the selector model recognizing the model name on its own.

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
- model profiles (id, name, display name, description, capabilities, priority)
- scoring instructions

The selector must return JSON only.

Example:

```json
{
  "models": [
    {
      "id": "gpt5mini",
      "overall_confidence": 87,
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

Select the first model whose confidence exceeds threshold.

Example:

```yaml
threshold: 85
```

Scores:

- GPT-5 Mini = 89
- Sonnet = 94
- Opus = 99

Result: **GPT-5 Mini**

### Highest Confidence

Select highest score.

Scores:

- GPT-5 Mini = 89
- Sonnet = 94
- Opus = 99

Result: **Opus**

### Balanced

Score: `confidence / cost`

Highest ratio wins.

### User Defined

Users may implement custom policies.

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
      "context": 71
    },
    "sonnet": {
      "overall": 91,
      "reasoning": 90,
      "coding": 95,
      "context": 93
    },
    "opus": {
      "overall": 97,
      "reasoning": 98,
      "coding": 96,
      "context": 99
    }
  },
  "reason": "Sonnet is the first model exceeding the configured threshold of 85."
}
```

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
  model: gpt-5-mini

policy:
  type: cheapest_capable
  threshold: 85

models:
  - id: cheap
    name: gpt-5-mini
    description: OpenAI's small, fast, low-cost model for simple tasks.
    cost: 1
  - id: medium
    name: sonnet
    description: Anthropic's mid-tier model. Strong at coding and long-context reasoning.
    cost: 5
  - id: premium
    name: opus
    description: Anthropic's flagship. Best reasoning, used when cheaper models are unlikely to succeed.
    cost: 20
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
├── modeldirector/
├── selector/
├── policies/
├── adapters/
│   ├── sdk/
│   ├── mcp/
│   ├── rest/
│   └── cli/
├── providers/
├── examples/
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

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
