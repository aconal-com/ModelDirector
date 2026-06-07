"""Selector engine - asks an LLM to score each candidate model for a given prompt.

The selector:
  1. Builds a JSON-only prompt that lists the user task, each model's profile
     (id, name, display name, description, capabilities, strengths, priority,
     cost), and the scoring rubric.
  2. Calls the configured LLM via LiteLLM.
  3. Validates that the response is JSON matching the ``ModelScore`` shape.
  4. Computes per-model estimated cost in USD and returns a ``SelectionResult``.
  5. Returns the parsed scores; the Policy engine then makes the final pick.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import litellm
from pydantic import ValidationError

from modeldirector.config import Config
from modeldirector.models import ModelScore, SelectionResult
from modeldirector.policy import build_policy, select_model

log = logging.getLogger(__name__)

# Tunable cap on the prompt we send to the selector.  Keeps selector cost
# predictable even with very large user prompts.
_MAX_PROMPT_CHARS = 8_000


# --- Token estimation ---------------------------------------------------------
#
# Best-effort token count. Used only to compute `estimated_cost_usd` in the
# output - this is an estimate, not a bill. We try tiktoken (OpenAI's
# tokenizer, ~2MB dep) for a real count, and fall back to a 4-chars-per-token
# heuristic if it's not installed or doesn't know the model.

try:
    import tiktoken  # type: ignore[import-not-found]

    def _count_tokens(text: str, model_hint: str | None = None) -> int:
        # Pick the right encoding. cl100k_base is a safe default for most
        # modern OpenAI / Anthropic / OpenRouter models.
        try:
            if model_hint:
                enc = tiktoken.encoding_for_model(model_hint)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

except ImportError:  # pragma: no cover

    def _count_tokens(text: str, model_hint: str | None = None) -> int:
        # ~4 chars per token is a common rule of thumb.
        return max(1, len(text) // 4)


# --- Errors -------------------------------------------------------------------


class SelectorError(RuntimeError):
    """Raised when the selector model fails to produce a valid score response."""


# --- Public entry point -------------------------------------------------------


class ModelDirector:
    """High-level entry point.  Load a config, call ``.select(prompt)``."""

    def __init__(self, config: Config) -> None:
        self.config = config

    # --- public API ---

    def select(self, prompt: str, *, assumed_output_tokens: int | None = None) -> SelectionResult:
        """Pick the best model for the given prompt.  Returns a SelectionResult.

        `assumed_output_tokens` lets the caller override the output-token
        estimate used in `estimated_cost_usd` (defaults to mirroring the
        input-token count, which is a reasonable assumption for chat-style
        tasks). Set to 0 if you only care about input cost.
        """
        scores, input_tokens = self._score(prompt)
        estimated_cost = _estimate_cost_usd(
            models=self.config.models,
            input_tokens=input_tokens,
            assumed_output_tokens=(
                assumed_output_tokens
                if assumed_output_tokens is not None
                else input_tokens
            ),
        )
        result = select_model(
            scores=scores,
            models=self.config.models,
            policy=build_policy(self.config.policy),
        )
        # The policy-built result doesn't have the cost fields; attach them.
        return result.model_copy(
            update={
                "estimated_cost_usd": estimated_cost,
                "input_tokens": input_tokens,
            }
        )

    def score(self, prompt: str) -> dict[str, ModelScore]:
        """Public: return raw per-model scores without applying a policy."""
        scores, _ = self._score(prompt)
        return scores

    # --- internals ---

    def _score(self, prompt: str) -> tuple[dict[str, ModelScore], int]:
        selector_prompt = _build_selector_prompt(prompt, self.config.models)
        input_tokens = _count_tokens(prompt, model_hint=self.config.selector.model)
        raw = self._call_selector(selector_prompt)
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict) or "models" not in parsed:
            raise SelectorError(
                f"Selector response missing 'models' key. Got: {parsed!r}"
            )
        models_field = parsed["models"]
        if not isinstance(models_field, list):
            raise SelectorError(f"'models' must be a list, got {type(models_field).__name__}")

        scores: dict[str, ModelScore] = {}
        for entry in models_field:
            try:
                s = ModelScore.model_validate(entry)
            except ValidationError as e:
                raise SelectorError(
                    f"Selector returned an invalid score entry: {entry!r} ({e})"
                ) from e
            scores[s.id] = s

        # Ensure every configured model got a score.
        missing = [m.id for m in self.config.models if m.id not in scores]
        if missing:
            raise SelectorError(
                f"Selector did not return scores for: {missing}. "
                f"Got: {sorted(scores)}"
            )

        return scores, input_tokens

    def _call_selector(self, prompt: str) -> str:
        sel = self.config.selector
        # Build the LiteLLM model name.  Always route through the configured
        # provider unless the model string already starts with that provider
        # (e.g. `openrouter/anthropic/claude-3.5-haiku`).
        first_segment = sel.model.split("/", 1)[0]
        if first_segment == sel.provider:
            model_name = sel.model
        else:
            model_name = f"{sel.provider}/{sel.model}"

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are ModelDirector, a routing model. You score how well "
                        "each candidate model can handle a user task. Always reply "
                        "with strict JSON. No prose, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": sel.temperature,
        }
        if sel.api_key:
            kwargs["api_key"] = sel.resolved_api_key()
        if sel.api_base:
            kwargs["api_base"] = sel.api_base
        if sel.max_tokens:
            kwargs["max_tokens"] = sel.max_tokens

        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:
            raise SelectorError(f"Selector LLM call failed: {e}") from e

        try:
            return resp.choices[0].message.content  # type: ignore[union-attr]
        except (AttributeError, IndexError, KeyError) as e:
            raise SelectorError(f"Malformed LLM response: {resp!r}") from e


# --- Cost estimation ---------------------------------------------------------


def _estimate_cost_usd(
    *,
    models: list,
    input_tokens: int,
    assumed_output_tokens: int,
) -> dict[str, float]:
    """Compute a per-model USD cost estimate for the given input size.

    Returns a dict keyed by model id. Models with missing or zero cost are
    included with 0.0 so the output is always complete.
    """
    out: dict[str, float] = {}
    for m in models:
        in_cost = float(getattr(m.cost, "input", 0.0))
        out_cost = float(getattr(m.cost, "output", 0.0))
        usd = (input_tokens / 1_000_000.0) * in_cost + (
            assumed_output_tokens / 1_000_000.0
        ) * out_cost
        out[m.id] = round(usd, 6)
    return out


# --- Prompt construction -----------------------------------------------------


def _build_selector_prompt(user_prompt: str, models: list) -> str:
    """Build the selector prompt.

    The schema we ask for is intentionally strict JSON, validated server-side
    with Pydantic. The selector is told that `description` and `strengths`
    are its primary source of context for unfamiliar model names.
    """
    # Truncate the user prompt if it would blow up the selector's context.
    truncated = user_prompt
    if len(truncated) > _MAX_PROMPT_CHARS:
        truncated = truncated[:_MAX_PROMPT_CHARS] + "\n\n[... truncated for length ...]"

    model_blocks = []
    for m in models:
        cap = m.capabilities
        block = {
            "id": m.id,
            "name": m.name,
            "display_name": m.display_name or m.name,
            "description": m.description or "(no description provided)",
            "strengths": list(m.strengths) if m.strengths else [],
            "capabilities": {
                "reasoning": cap.reasoning,
                "coding": cap.coding,
                "context": cap.context,
                **({"creativity": cap.creativity} if cap.creativity is not None else {}),
            },
            "priority": m.priority,
            "cost_per_1m_tokens_usd": {
                "input": m.cost.input,
                "output": m.cost.output,
            },
        }
        model_blocks.append(block)

    schema_hint = json.dumps(
        {
            "models": [
                {
                    "id": "<model id>",
                    "overall": "<0-100 weighted score>",
                    "reasoning": "<0-100>",
                    "coding": "<0-100>",
                    "context": "<0-100>",
                    "creativity": "<0-100 or omit>",
                    "explanation": "<one short sentence>",
                }
            ]
        },
        indent=2,
    )

    return (
        "Score each candidate model for the user task below. "
        "Return JSON only, matching the schema exactly.\n\n"
        "## User task\n"
        f"{truncated}\n\n"
        "## Candidate models\n"
        f"{json.dumps(model_blocks, indent=2)}\n\n"
        "## Scoring rubric\n"
        "- `overall`: weighted score (0-100).  Higher is more suitable.\n"
        "- `reasoning`, `coding`, `context`, `creativity`: per-axis fit 0-100.\n"
        "- For each model, weigh its `description` and `strengths` heavily - "
        "you may not recognise the model name.\n"
        "- Be honest: a cheap model is usually fine for a cheap task.\n\n"
        "## Required output schema\n"
        f"{schema_hint}\n"
    )


# --- JSON extraction ---------------------------------------------------------


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response.

    Handles:
      * raw JSON
      * JSON wrapped in ```json ... ``` fences
      * JSON with stray prose before/after
    """
    text = text.strip()

    # Strip code fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Raw parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the outermost { ... } block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise SelectorError(f"Could not parse selector JSON: {e}\nText was:\n{text}") from e

    raise SelectorError(f"Could not find JSON in selector response:\n{text}")
