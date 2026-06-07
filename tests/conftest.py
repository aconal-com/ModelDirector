"""Shared test fixtures and helpers."""
from __future__ import annotations

import pytest

from modeldirector.config import (
    Capabilities,
    Config,
    Cost,
    ModelProfile,
    PolicyConfig,
    SelectorConfig,
)


@pytest.fixture
def three_models() -> list[ModelProfile]:
    """A standard 3-model candidate set used in many tests.

    Ids are model names, not tier labels. The engine treats every profile
    the same; the configured policy decides.
    """
    return [
        ModelProfile(
            id="gpt5mini",
            name="gpt-4o-mini",
            description="OpenAI's small, fast, low-cost model for simple tasks.",
            strengths=["classification", "short_summarisation", "simple_qa"],
            capabilities=Capabilities(reasoning=70, coding=75, context=70, creativity=65),
            priority=1,
            cost=Cost(input=0.15, output=0.60),
        ),
        ModelProfile(
            id="sonnet",
            name="claude-3.5-sonnet",
            description="Anthropic's mid-tier model. Strong at coding and reasoning.",
            strengths=["coding", "architecture", "refactoring"],
            capabilities=Capabilities(reasoning=90, coding=92, context=92, creativity=85),
            priority=2,
            cost=Cost(input=3.00, output=15.00),
        ),
        ModelProfile(
            id="opus",
            name="claude-opus-4",
            description="Anthropic's flagship. Best reasoning, used for hard tasks.",
            strengths=["hard_reasoning", "complex_coding", "architecture_design"],
            capabilities=Capabilities(reasoning=99, coding=97, context=99, creativity=95),
            priority=3,
            cost=Cost(input=15.00, output=75.00),
        ),
    ]


@pytest.fixture
def three_model_config(three_models) -> Config:
    return Config(
        selector=SelectorConfig(provider="openrouter", model="openai/gpt-4o-mini"),
        policy=PolicyConfig(type="cheapest_capable", threshold=80),
        models=three_models,
    )


def make_score(
    model_id: str,
    overall: int,
    *,
    reasoning: int | None = None,
    coding: int | None = None,
    context: int | None = None,
    creativity: int | None = None,
    explanation: str = "test",
) -> dict:
    """Build a ModelScore-shaped dict for the selector to return."""
    return {
        "id": model_id,
        "overall": overall,
        "reasoning": reasoning if reasoning is not None else overall,
        "coding": coding if coding is not None else overall,
        "context": context if context is not None else overall,
        **({"creativity": creativity} if creativity is not None else {}),
        "explanation": explanation,
    }
