"""Tests for config loading and validation."""
from __future__ import annotations

from textwrap import dedent

import pytest

from modeldirector.config import (
    Capabilities,
    Cost,
    ModelProfile,
    SelectorConfig,
)
from modeldirector.loader import load_config


def test_loads_minimal_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            """\
            selector:
              provider: openrouter
              model: openai/gpt-4o-mini
            models:
              - id: gpt5mini
                name: gpt-4o-mini
                description: small model
                capabilities:
                  reasoning: 80
                  coding: 80
                  context: 80
                priority: 1
                cost: { input: 0.15, output: 0.60 }
            """
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.selector.provider == "openrouter"
    assert cfg.selector.model == "openai/gpt-4o-mini"
    assert cfg.policy.type == "cheapest_capable"
    assert cfg.policy.threshold == 85
    assert len(cfg.models) == 1
    assert cfg.models[0].id == "gpt5mini"
    assert cfg.models[0].cost.input == 0.15
    assert cfg.models[0].cost.output == 0.60


def test_loads_dict_directly():
    cfg = load_config(
        {
            "selector": {"provider": "anthropic", "model": "claude-sonnet"},
            "policy": {"type": "best_value"},
            "models": [
                {
                    "id": "sonnet",
                    "name": "claude-sonnet",
                    "description": "test",
                    "capabilities": {"reasoning": 90, "coding": 90, "context": 90},
                    "priority": 2,
                    "cost": {"input": 3.0, "output": 15.0},
                }
            ],
        }
    )
    assert cfg.policy.type == "best_value"


def test_legacy_balanced_aliases_to_best_value():
    """`balanced` (v0.0.x) is accepted and coerced to `best_value`."""
    cfg = load_config(
        {
            "selector": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
            "policy": {"type": "balanced"},
            "models": [
                {
                    "id": "sonnet",
                    "name": "claude-sonnet",
                    "description": "x",
                    "capabilities": {"reasoning": 80, "coding": 80, "context": 80},
                    "priority": 1,
                    "cost": {"input": 3.0, "output": 15.0},
                }
            ],
        }
    )
    assert cfg.policy.type == "best_value"


def test_cost_accepts_legacy_single_number():
    """A bare number for `cost` is coerced to Cost(input=x, output=x) for back-compat."""
    p = ModelProfile(
        id="gpt5mini",
        name="gpt-4o-mini",
        description="x",
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=1,
        cost=1.0,
    )
    assert isinstance(p.cost, Cost)
    assert p.cost.input == 1.0
    assert p.cost.output == 1.0


def test_cost_accepts_dict_form():
    p = ModelProfile(
        id="gpt5mini",
        name="gpt-4o-mini",
        description="x",
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=1,
        cost={"input": 0.15, "output": 0.60},
    )
    assert p.cost.input == 0.15
    assert p.cost.output == 0.60


def test_strengths_default_to_empty_list():
    p = ModelProfile(
        id="x",
        name="x",
        description="x",
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=1,
        cost={"input": 1.0, "output": 1.0},
    )
    assert p.strengths == []


def test_strengths_persist_from_config():
    p = ModelProfile(
        id="sonnet",
        name="claude-sonnet",
        description="x",
        strengths=["coding", "architecture", "refactoring"],
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=1,
        cost={"input": 3.0, "output": 15.0},
    )
    assert p.strengths == ["coding", "architecture", "refactoring"]


def test_expands_env_var_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "sk-test-123")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            """\
            selector:
              provider: openrouter
              model: openai/gpt-4o-mini
              api_key: ${MY_TEST_KEY}
            models:
              - id: gpt5mini
                name: gpt-4o-mini
                description: test
                capabilities: {reasoning: 80, coding: 80, context: 80}
                priority: 1
                cost: { input: 0.15, output: 0.60 }
            """
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.selector.resolved_api_key() == "sk-test-123"


def test_resolved_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    sel = SelectorConfig(provider="openrouter", model="openai/gpt-4o-mini")
    assert sel.resolved_api_key() == "sk-from-env"


def test_resolved_api_key_errors_when_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sel = SelectorConfig(provider="openrouter", model="openai/gpt-4o-mini")
    with pytest.raises(ValueError, match="No API key"):
        sel.resolved_api_key()


def test_rejects_duplicate_model_ids():
    with pytest.raises(ValueError, match="Duplicate model ids"):
        load_config(
            {
                "selector": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                "models": [
                    {
                        "id": "dup",
                        "name": "x",
                        "description": "x",
                        "capabilities": {"reasoning": 80, "coding": 80, "context": 80},
                        "priority": 1,
                        "cost": {"input": 1.0, "output": 1.0},
                    },
                    {
                        "id": "dup",
                        "name": "y",
                        "description": "y",
                        "capabilities": {"reasoning": 80, "coding": 80, "context": 80},
                        "priority": 2,
                        "cost": {"input": 1.0, "output": 1.0},
                    },
                ],
            }
        )


def test_rejects_empty_models_list():
    with pytest.raises(ValueError, match="at least 1"):
        load_config(
            {
                "selector": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                "models": [],
            }
        )


def test_rejects_invalid_capability_score():
    with pytest.raises(ValueError):
        ModelProfile(
            id="m",
            name="m",
            description="x",
            capabilities=Capabilities(reasoning=200, coding=80, context=80),
            priority=1,
            cost={"input": 1.0, "output": 1.0},
        )


def test_rejects_empty_id():
    with pytest.raises(ValueError):
        ModelProfile(
            id="   ",
            name="m",
            description="x",
            capabilities=Capabilities(reasoning=80, coding=80, context=80),
            priority=1,
            cost={"input": 1.0, "output": 1.0},
        )


def test_rejects_negative_cost():
    with pytest.raises(ValueError):
        Cost(input=-1.0, output=0.0)
