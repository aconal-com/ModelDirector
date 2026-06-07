"""Shared test fixtures and helpers."""
from __future__ import annotations

import pytest

from modeldirector.config import (
    Capabilities,
    Config,
    ModelProfile,
    PolicyConfig,
    SelectorConfig,
)


@pytest.fixture
def cheap_mid_premium() -> list[ModelProfile]:
    """A standard 3-tier candidate set used in many tests."""
    return [
        ModelProfile(
            id="cheap",
            name="gpt-5-mini",
            description="OpenAI's small, fast, low-cost model for simple tasks.",
            capabilities=Capabilities(reasoning=70, coding=75, context=70, creativity=65),
            priority=1,
            cost=1,
        ),
        ModelProfile(
            id="mid",
            name="claude-sonnet",
            description="Anthropic's mid-tier model. Strong at coding and reasoning.",
            capabilities=Capabilities(reasoning=90, coding=92, context=92, creativity=85),
            priority=2,
            cost=5,
        ),
        ModelProfile(
            id="premium",
            name="claude-opus",
            description="Anthropic's flagship. Best reasoning, used for hard tasks.",
            capabilities=Capabilities(reasoning=99, coding=97, context=99, creativity=95),
            priority=3,
            cost=20,
        ),
    ]


@pytest.fixture
def cheap_mid_premium_config(cheap_mid_premium) -> Config:
    return Config(
        selector=SelectorConfig(provider="openrouter", model="openai/gpt-5-mini"),
        policy=PolicyConfig(type="cheapest_capable", threshold=80),
        models=cheap_mid_premium,
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
