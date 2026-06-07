"""Tests for config loading and validation."""
from __future__ import annotations

import os
from textwrap import dedent

import pytest
import yaml

from modeldirector.config import (
    Capabilities,
    Config,
    ModelProfile,
    PolicyConfig,
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
              model: openai/gpt-5-mini
            models:
              - id: cheap
                name: gpt-5-mini
                description: small model
                capabilities:
                  reasoning: 80
                  coding: 80
                  context: 80
                priority: 1
            """
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.selector.provider == "openrouter"
    assert cfg.selector.model == "openai/gpt-5-mini"
    assert cfg.policy.type == "cheapest_capable"
    assert cfg.policy.threshold == 85
    assert len(cfg.models) == 1
    assert cfg.models[0].id == "cheap"
    assert cfg.models[0].cost == 1.0  # default from priority 1


def test_loads_dict_directly():
    cfg = load_config(
        {
            "selector": {"provider": "anthropic", "model": "claude-sonnet"},
            "policy": {"type": "balanced"},
            "models": [
                {
                    "id": "m",
                    "name": "claude-sonnet",
                    "description": "test",
                    "capabilities": {"reasoning": 90, "coding": 90, "context": 90},
                    "priority": 2,
                }
            ],
        }
    )
    assert cfg.policy.type == "balanced"


def test_expands_env_var_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "sk-test-123")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            """\
            selector:
              provider: openrouter
              model: openai/gpt-5-mini
              api_key: ${MY_TEST_KEY}
            models:
              - id: cheap
                name: gpt-5-mini
                description: test
                capabilities: {reasoning: 80, coding: 80, context: 80}
                priority: 1
            """
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.selector.resolved_api_key() == "sk-test-123"


def test_resolved_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    sel = SelectorConfig(provider="openrouter", model="openai/gpt-5-mini")
    assert sel.resolved_api_key() == "sk-from-env"


def test_resolved_api_key_errors_when_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sel = SelectorConfig(provider="openrouter", model="openai/gpt-5-mini")
    with pytest.raises(ValueError, match="No API key"):
        sel.resolved_api_key()


def test_rejects_duplicate_model_ids():
    with pytest.raises(ValueError, match="Duplicate model ids"):
        load_config(
            {
                "selector": {"provider": "openrouter", "model": "openai/gpt-5-mini"},
                "models": [
                    {
                        "id": "dup",
                        "name": "x",
                        "description": "x",
                        "capabilities": {"reasoning": 80, "coding": 80, "context": 80},
                        "priority": 1,
                    },
                    {
                        "id": "dup",
                        "name": "y",
                        "description": "y",
                        "capabilities": {"reasoning": 80, "coding": 80, "context": 80},
                        "priority": 2,
                    },
                ],
            }
        )


def test_rejects_empty_models_list():
    with pytest.raises(ValueError, match="at least 1"):
        load_config(
            {
                "selector": {"provider": "openrouter", "model": "openai/gpt-5-mini"},
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
        )


def test_rejects_empty_id():
    with pytest.raises(ValueError):
        ModelProfile(
            id="   ",
            name="m",
            description="x",
            capabilities=Capabilities(reasoning=80, coding=80, context=80),
            priority=1,
        )


def test_default_cost_uses_priority_bucket():
    p1 = ModelProfile(
        id="a", name="a", description="x",
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=1,
    )
    p2 = ModelProfile(
        id="b", name="b", description="x",
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=3,
    )
    p99 = ModelProfile(
        id="c", name="c", description="x",
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=99,
    )
    assert p1.cost == 1.0
    assert p2.cost == 20.0
    assert p99.cost == 990.0  # priority * 10 fallback


def test_explicit_cost_overrides_default():
    p = ModelProfile(
        id="a", name="a", description="x",
        capabilities=Capabilities(reasoning=80, coding=80, context=80),
        priority=1, cost=42.0,
    )
    assert p.cost == 42.0
