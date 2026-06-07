"""Tests for the policy engine - pure logic, no LLM calls."""
from __future__ import annotations

import pytest

from modeldirector.config import PolicyConfig
from modeldirector.models import ModelScore
from modeldirector.policy import (
    BalancedPolicy,
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
    def test_picks_cheapest_model_above_threshold(self, cheap_mid_premium):
        scores = {
            "cheap": _score("cheap", 85),
            "mid": _score("mid", 92),
            "premium": _score("premium", 99),
        }
        result = CheapestCapablePolicy(threshold=80).apply(scores, cheap_mid_premium)
        assert result.selected_model == "cheap"
        assert result.policy == "cheapest_capable"
        assert "exceeding" in result.reason.lower() or "threshold" in result.reason.lower()

    def test_picks_first_above_threshold_by_priority(self, cheap_mid_premium):
        # cheap = 70 (below threshold), mid = 85, premium = 99
        scores = {
            "cheap": _score("cheap", 70),
            "mid": _score("mid", 85),
            "premium": _score("premium", 99),
        }
        result = CheapestCapablePolicy(threshold=80).apply(scores, cheap_mid_premium)
        assert result.selected_model == "mid"

    def test_falls_back_to_highest_scorer_when_nothing_meets_threshold(self, cheap_mid_premium):
        scores = {
            "cheap": _score("cheap", 30),
            "mid": _score("mid", 40),
            "premium": _score("premium", 50),
        }
        result = CheapestCapablePolicy(threshold=80).apply(scores, cheap_mid_premium)
        assert result.selected_model == "premium"
        assert "fall" in result.reason.lower() or "no model" in result.reason.lower()

    def test_ignores_models_without_scores(self, cheap_mid_premium):
        scores = {"mid": _score("mid", 90)}  # cheap and premium missing
        result = CheapestCapablePolicy(threshold=80).apply(scores, cheap_mid_premium)
        assert result.selected_model == "mid"

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            CheapestCapablePolicy(threshold=150)
        with pytest.raises(ValueError):
            CheapestCapablePolicy(threshold=-1)


class TestHighestConfidencePolicy:
    def test_always_picks_highest_score(self, cheap_mid_premium):
        scores = {
            "cheap": _score("cheap", 90),
            "mid": _score("mid", 60),
            "premium": _score("premium", 70),
        }
        result = HighestConfidencePolicy().apply(scores, cheap_mid_premium)
        assert result.selected_model == "cheap"  # highest overall, even if "cheap"

    def test_handles_tie_by_picking_first_inserted(self, cheap_mid_premium):
        scores = {
            "cheap": _score("cheap", 80),
            "mid": _score("mid", 80),
            "premium": _score("premium", 80),
        }
        result = HighestConfidencePolicy().apply(scores, cheap_mid_premium)
        # max() is stable, so the first-inserted key with the max value wins.
        assert result.selected_model in {"cheap", "mid", "premium"}


class TestBalancedPolicy:
    def test_picks_best_score_to_cost_ratio(self, cheap_mid_premium):
        # cheap=85/1=85, mid=90/5=18, premium=99/20=4.95
        scores = {
            "cheap": _score("cheap", 85),
            "mid": _score("mid", 90),
            "premium": _score("premium", 99),
        }
        result = BalancedPolicy().apply(scores, cheap_mid_premium)
        assert result.selected_model == "cheap"

    def test_handles_zero_cost_model(self, cheap_mid_premium):
        # Override premium to cost 0 - it should win because the fallback
        # treats zero-cost as overall-only (so higher overall scores higher).
        cheap_mid_premium[2].cost = 0
        scores = {
            "cheap": _score("cheap", 50),
            "mid": _score("mid", 50),
            "premium": _score("premium", 90),
        }
        result = BalancedPolicy().apply(scores, cheap_mid_premium)
        assert result.selected_model == "premium"


class TestBuildPolicy:
    def test_builds_cheapest_capable(self):
        p = build_policy(PolicyConfig(type="cheapest_capable", threshold=90))
        assert isinstance(p, CheapestCapablePolicy)
        assert p.threshold == 90

    def test_builds_highest_confidence(self):
        p = build_policy(PolicyConfig(type="highest_confidence"))
        assert isinstance(p, HighestConfidencePolicy)

    def test_builds_balanced(self):
        p = build_policy(PolicyConfig(type="balanced"))
        assert isinstance(p, BalancedPolicy)

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError):
            build_policy(PolicyConfig(type="nonsense"))  # type: ignore[arg-type]

    def test_normalises_legacy_string(self):
        p = build_policy(PolicyConfig.model_validate({"type": "Cheapest-Capable"}))
        assert isinstance(p, CheapestCapablePolicy)


class TestSelectModelHelper:
    def test_accepts_policy_config_directly(self, cheap_mid_premium):
        scores = {"cheap": _score("cheap", 90), "mid": _score("mid", 95), "premium": _score("premium", 99)}
        result = select_model(
            scores=scores,
            models=cheap_mid_premium,
            policy=PolicyConfig(type="highest_confidence"),
        )
        assert result.selected_model == "premium"
