"""Tests for the selector engine.  Mocks the LLM call so no API is hit."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from modeldirector.config import Config, ModelProfile, PolicyConfig, SelectorConfig
from modeldirector.models import ModelScore
from modeldirector.selector import ModelDirector, SelectorError, _extract_json
from tests.conftest import make_score


def _make_director(config: Config) -> ModelDirector:
    return ModelDirector(config)


def test_score_parses_clean_json_response(cheap_mid_premium_config):
    director = _make_director(cheap_mid_premium_config)
    fake_response = json.dumps(
        {
            "models": [
                make_score("cheap", 90, explanation="Easy task, cheap model is fine"),
                make_score("mid", 85, explanation="More than enough"),
                make_score("premium", 60, explanation="Overkill"),
            ]
        }
    )

    with patch.object(director, "_call_selector", return_value=fake_response):
        scores = director.score("Summarise this article.")

    assert set(scores) == {"cheap", "mid", "premium"}
    assert scores["cheap"].overall == 90
    assert scores["premium"].overall == 60


def test_score_strips_markdown_fences(cheap_mid_premium_config):
    director = _make_director(cheap_mid_premium_config)
    fenced = "```json\n" + json.dumps(
        {"models": [make_score("cheap", 80), make_score("mid", 80), make_score("premium", 80)]}
    ) + "\n```"

    with patch.object(director, "_call_selector", return_value=fenced):
        scores = director.score("test prompt")
    assert set(scores) == {"cheap", "mid", "premium"}


def test_score_rejects_missing_models_key(cheap_mid_premium_config):
    director = _make_director(cheap_mid_premium_config)
    with patch.object(director, "_call_selector", return_value='{"foo": "bar"}'):
        with pytest.raises(SelectorError, match="missing 'models' key"):
            director.score("test")


def test_score_rejects_missing_candidate(cheap_mid_premium_config):
    director = _make_director(cheap_mid_premium_config)
    response = json.dumps({"models": [make_score("cheap", 90), make_score("mid", 85)]})  # premium missing

    with patch.object(director, "_call_selector", return_value=response):
        with pytest.raises(SelectorError, match="did not return scores for"):
            director.score("test")


def test_score_rejects_invalid_entry(cheap_mid_premium_config):
    director = _make_director(cheap_mid_premium_config)
    response = json.dumps(
        {
            "models": [
                {"id": "cheap", "overall": 200, "reasoning": 80, "coding": 80, "context": 80, "explanation": "x"},
                make_score("mid", 80),
                make_score("premium", 80),
            ]
        }
    )

    with patch.object(director, "_call_selector", return_value=response):
        with pytest.raises(SelectorError, match="invalid score entry"):
            director.score("test")


def test_select_applies_default_cheapest_capable_policy(cheap_mid_premium_config):
    director = _make_director(cheap_mid_premium_config)
    response = json.dumps(
        {
            "models": [
                make_score("cheap", 90, explanation="Easy"),
                make_score("mid", 95, explanation="Ok"),
                make_score("premium", 99, explanation="Overkill"),
            ]
        }
    )
    with patch.object(director, "_call_selector", return_value=response):
        result = director.select("Summarise this short article.")

    assert result.selected_model == "cheap"
    assert result.policy == "cheapest_capable"


def test_extracted_json_handles_prose_around_block():
    raw = 'Here you go:\n{"models": []}\nDone!'
    parsed = _extract_json(raw)
    assert parsed == {"models": []}


def test_extracted_json_raises_on_garbage():
    with pytest.raises(SelectorError):
        _extract_json("not json at all")


def test_extracted_json_raises_when_no_braces():
    with pytest.raises(SelectorError, match="Could not find JSON"):
        _extract_json("just plain text")


def test_prompt_includes_description(cheap_mid_premium_config):
    """The selector prompt must surface each model's description to the LLM."""
    director = _make_director(cheap_mid_premium_config)
    captured: dict[str, Any] = {}

    def fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps(
            {"models": [make_score("cheap", 80), make_score("mid", 80), make_score("premium", 80)]}
        )

    with patch.object(director, "_call_selector", side_effect=fake_call):
        director.score("test")

    assert "OpenAI's small, fast, low-cost model" in captured["prompt"]
    assert "Anthropic's flagship" in captured["prompt"]
    assert "id" in captured["prompt"]
    assert "capabilities" in captured["prompt"]


def test_prompt_truncates_huge_user_prompt(cheap_mid_premium_config):
    director = _make_director(cheap_mid_premium_config)
    captured: dict[str, Any] = {}

    def fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps(
            {"models": [make_score("cheap", 80), make_score("mid", 80), make_score("premium", 80)]}
        )

    huge = "x" * 20_000
    with patch.object(director, "_call_selector", side_effect=fake_call):
        director.score(huge)

    assert "[... truncated for length ...]" in captured["prompt"]
    # The full 20k should not make it into the prompt
    assert "x" * 20_000 not in captured["prompt"]
