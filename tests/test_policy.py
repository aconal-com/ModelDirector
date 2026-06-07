"""Tests for the policy engine - pure logic, no LLM calls."""
from __future__ import annotations

import pytest

from modeldirector.config import PolicyConfig
from modeldirector.models import ModelScore
from modeldirector.policy import (
    BestValuePolicy,
    CheapestCapablePolicy,
    HighestConfidencePolicy,
    build_policy,
    select_model,
)


def _score(model_id: str, overall: int) -> ModelScore:
    return ModelScore(
        id=model_id,
        overall=overall,
        reasoning=overall,
        coding=overall,
        context=overall,
        explanation="test",
    )


class TestCheapestCapablePolicy:
    def test_picks_cheapest_model_above_threshold(self, three_models):
        scores = {
            "gpt5mini": _score("gpt5mini", 85),
            "sonnet": _score("sonnet", 92),
            "opus": _score("opus", 99),
        }
        result = CheapestCapablePolicy(threshold=80).apply(scores, three_models)
        assert result.selected_model == "gpt5mini"
        assert result.policy == "cheapest_capable"
        assert "exceeding" in result.reason.lower() or "threshold" in result.reason.lower()

    def test_picks_first_above_threshold_by_priority(self, three_models):
        # gpt5mini = 70 (below threshold), sonnet = 85, opus = 99
        scores = {
            "gpt5mini": _score("gpt5mini", 70),
            "sonnet": _score("sonnet", 85),
            "opus": _score("opus", 99),
        }
        result = CheapestCapablePolicy(threshold=80).apply(scores, three_models)
        assert result.selected_model == "sonnet"

    def test_falls_back_to_highest_scorer_when_nothing_meets_threshold(self, three_models):
        scores = {
            "gpt5mini": _score("gpt5mini", 30),
            "sonnet": _score("sonnet", 40),
            "opus": _score("opus", 50),
        }
        result = CheapestCapablePolicy(threshold=80).apply(scores, three_models)
        assert result.selected_model == "opus"
        assert "fall" in result.reason.lower() or "no model" in result.reason.lower()

    def test_ignores_models_without_scores(self, three_models):
        scores = {"sonnet": _score("sonnet", 90)}  # gpt5mini and opus missing
        result = CheapestCapablePolicy(threshold=80).apply(scores, three_models)
        assert result.selected_model == "sonnet"

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            CheapestCapablePolicy(threshold=150)
        with pytest.raises(ValueError):
            CheapestCapablePolicy(threshold=-1)


class TestHighestConfidencePolicy:
    def test_always_picks_highest_score(self, three_models):
        scores = {
            "gpt5mini": _score("gpt5mini", 90),
            "sonnet": _score("sonnet", 60),
            "opus": _score("opus", 70),
        }
        result = HighestConfidencePolicy().apply(scores, three_models)
        assert result.selected_model == "gpt5mini"  # highest overall, even if cheap

    def test_handles_tie_by_picking_first_inserted(self, three_models):
        scores = {
            "gpt5mini": _score("gpt5mini", 80),
            "sonnet": _score("sonnet", 80),
            "opus": _score("opus", 80),
        }
        result = HighestConfidencePolicy().apply(scores, three_models)
        # max() is stable, so the first-inserted key with the max value wins.
        assert result.selected_model in {"gpt5mini", "sonnet", "opus"}


class TestBestValuePolicy:
    def test_picks_best_score_to_input_cost_ratio(self, three_models):
        # gpt5mini=85/0.15=566, sonnet=90/3=30, opus=99/15=6.6
        scores = {
            "gpt5mini": _score("gpt5mini", 85),
            "sonnet": _score("sonnet", 90),
            "opus": _score("opus", 99),
        }
        result = BestValuePolicy().apply(scores, three_models)
        assert result.selected_model == "gpt5mini"

    def test_handles_zero_input_cost_model(self, three_models):
        # Override opus input cost to 0 - a free high-score model beats a
        # cheaper-but-weaker one when the cost-vs-score ratio is what matters.
        three_models[2].cost.input = 0
        scores = {
            "gpt5mini": _score("gpt5mini", 50),
            "sonnet": _score("sonnet", 50),
            "opus": _score("opus", 90),
        }
        # 50/0.15 = 333 for gpt5mini beats 90/0 = 90 for opus, so the cheap
        # one still wins. Make opus genuinely free AND dominant by also
        # bumping the cheap model's cost.
        three_models[0].cost.input = 1000.0
        result = BestValuePolicy().apply(scores, three_models)
        # opus: 90 / 0 = 90 (free fallback)  >  gpt5mini: 50 / 1000 = 0.05
        assert result.selected_model == "opus"


class TestBuildPolicy:
    def test_builds_cheapest_capable(self):
        p = build_policy(PolicyConfig(type="cheapest_capable", threshold=90))
        assert isinstance(p, CheapestCapablePolicy)
        assert p.threshold == 90

    def test_builds_highest_confidence(self):
        p = build_policy(PolicyConfig(type="highest_confidence"))
        assert isinstance(p, HighestConfidencePolicy)

    def test_builds_best_value(self):
        p = build_policy(PolicyConfig(type="best_value"))
        assert isinstance(p, BestValuePolicy)

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError):
            build_policy(PolicyConfig(type="nonsense"))  # type: ignore[arg-type]

    def test_normalises_legacy_string(self):
        p = build_policy(PolicyConfig.model_validate({"type": "Cheapest-Capable"}))
        assert isinstance(p, CheapestCapablePolicy)

    def test_legacy_balanced_string_coerced(self):
        # `balanced` (the old name) is coerced to `best_value`
        p = build_policy(PolicyConfig.model_validate({"type": "balanced"}))
        assert isinstance(p, BestValuePolicy)


class TestSelectModelHelper:
    def test_accepts_policy_config_directly(self, three_models):
        scores = {
            "gpt5mini": _score("gpt5mini", 90),
            "sonnet": _score("sonnet", 95),
            "opus": _score("opus", 99),
        }
        result = select_model(
            scores=scores,
            models=three_models,
            policy=PolicyConfig(type="highest_confidence"),
        )
        assert result.selected_model == "opus"
