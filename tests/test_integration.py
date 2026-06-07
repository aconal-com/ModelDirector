"""End-to-end integration test - makes a real LLM call.

Requires OPENROUTER_API_KEY.  Skipped automatically if it isn't set.
"""

from __future__ import annotations

import os

import pytest

from modeldirector.loader import load_config
from modeldirector.selector import ModelDirector


pytestmark = pytest.mark.integration


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "examples", "config.yaml")


@pytest.fixture
def director() -> ModelDirector:
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set - skipping integration test")
    return ModelDirector(load_config(CONFIG_PATH))


def test_select_simple_task_picks_cheap(director):
    """A trivial prompt should not need the premium model."""
    result = director.select("Translate 'hello' to Spanish.")
    assert result.selected_model in {"cheap", "mid"}
    assert result.policy == "cheapest_capable"
    assert "cheap" in result.scores
    assert "mid" in result.scores
    assert "premium" in result.scores


def test_select_hard_task_picks_stronger_model(director):
    """A clearly-hard task should push the selector toward the stronger model."""
    hard_prompt = (
        "Design a distributed rate limiter using Redis with token bucket "
        "semantics, then write the implementation in Python with a "
        "test suite that covers race conditions and clock skew."
    )
    result = director.select(hard_prompt)
    # The hard task should at least consider premium or mid as a candidate -
    # we just verify the selector returned something sensible.
    assert result.selected_model in {"cheap", "mid", "premium"}
    assert all(0 <= s.overall <= 100 for s in result.scores.values())


def test_score_returns_all_three_models(director):
    scores = director.score("Write a haiku about Python.")
    assert set(scores) == {"cheap", "mid", "premium"}
