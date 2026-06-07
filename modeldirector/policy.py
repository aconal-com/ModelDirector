"""Policy engine - takes a set of scores + a policy + the model list, returns a selection.

Three built-in policies:
  * ``cheapest_capable``  - first model (by priority) whose overall score >= threshold
  * ``highest_confidence``- the model with the highest overall score
  * ``best_value``        - highest score / cost ratio (renamed from ``balanced``)

Users can subclass ``Policy`` to define custom policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from modeldirector.config import PolicyConfig
from modeldirector.models import ModelScore, SelectionResult

if TYPE_CHECKING:
    from modeldirector.config import ModelProfile


class Policy(ABC):
    """Base class for all policies."""

    name: str

    @abstractmethod
    def apply(
        self,
        scores: dict[str, ModelScore],
        models: list["ModelProfile"],
    ) -> SelectionResult:
        """Return a SelectionResult given the per-model scores and full profile list."""


class CheapestCapablePolicy(Policy):
    """First model (by priority asc, then by cost asc) whose overall >= threshold."""

    name = "cheapest_capable"

    def __init__(self, threshold: int) -> None:
        if not 0 <= threshold <= 100:
            raise ValueError("threshold must be 0-100")
        self.threshold = threshold

    def apply(
        self,
        scores: dict[str, ModelScore],
        models: list["ModelProfile"],
    ) -> SelectionResult:
        # Sort by priority asc, then by input cost asc.
        ordered = sorted(models, key=lambda m: (m.priority, m.cost.input))
        chosen: "ModelProfile | None" = None
        chosen_score: "ModelScore | None" = None
        for m in ordered:
            s = scores.get(m.id)
            if s is None:
                continue
            if s.overall >= self.threshold:
                chosen = m
                chosen_score = s
                break

        if chosen is None or chosen_score is None:
            # Fall back to the highest-scoring model so we never return empty.
            best = max(scores.values(), key=lambda s: s.overall)
            chosen = _profile_by_id(models, best.id)
            assert chosen is not None  # always present
            chosen_score = best
            reason = (
                f"No model reached the threshold of {self.threshold}. "
                f"Falling back to the highest-scoring model: '{chosen.id}' (overall={best.overall})."
            )
        else:
            reason = (
                f"'{chosen.id}' is the first model exceeding the threshold of {self.threshold} "
                f"(overall={chosen_score.overall}, priority={chosen.priority}, "
                f"cost=${chosen.cost.input:.2f}/1M in, ${chosen.cost.output:.2f}/1M out)."
            )

        return SelectionResult(
            selected_model=chosen.id,
            policy=self.name,  # type: ignore[arg-type]
            scores=scores,
            reason=reason,
        )


class HighestConfidencePolicy(Policy):
    """Always pick the highest-scoring model."""

    name = "highest_confidence"

    def apply(
        self,
        scores: dict[str, ModelScore],
        models: list["ModelProfile"],
    ) -> SelectionResult:
        best_id, best_score = max(scores.items(), key=lambda kv: kv[1].overall)
        return SelectionResult(
            selected_model=best_id,
            policy=self.name,  # type: ignore[arg-type]
            scores=scores,
            reason=(
                f"'{best_id}' has the highest overall score ({best_score.overall}). "
                f"Policy is 'highest_confidence' so we always pick the best scorer."
            ),
        )


class BestValuePolicy(Policy):
    """Pick the model with the highest (overall / cost.input) ratio.

    Uses `cost.input` (deterministic - the user always pays for input tokens
    and we know the input length) rather than a blended input+output figure
    (which would require guessing the output length).
    """

    name = "best_value"

    def apply(
        self,
        scores: dict[str, ModelScore],
        models: list["ModelProfile"],
    ) -> SelectionResult:
        def ratio(model_id: str) -> float:
            s = scores[model_id]
            cost_in = _input_cost_for(models, model_id)
            if cost_in <= 0:
                return float(s.overall)  # free model - just use score
            return s.overall / cost_in

        best_id = max(scores, key=ratio)
        s = scores[best_id]
        cost_in = _input_cost_for(models, best_id)
        return SelectionResult(
            selected_model=best_id,
            policy=self.name,  # type: ignore[arg-type]
            scores=scores,
            reason=(
                f"'{best_id}' has the best confidence-per-USD ratio "
                f"({s.overall} / ${cost_in:.2f}/1M in = {ratio(best_id):.2f})."
            ),
        )


# --- helpers ------------------------------------------------------------------


def _profile_by_id(models: list["ModelProfile"], model_id: str) -> "ModelProfile | None":
    for m in models:
        if m.id == model_id:
            return m
    return None


def _input_cost_for(models: list["ModelProfile"], model_id: str) -> float:
    p = _profile_by_id(models, model_id)
    if p is None:
        return 0.0
    return float(p.cost.input)


# --- public factory -----------------------------------------------------------


def build_policy(cfg: PolicyConfig) -> Policy:
    """Instantiate the policy described by `cfg`."""
    if cfg.type == "cheapest_capable":
        return CheapestCapablePolicy(threshold=cfg.threshold)
    if cfg.type == "highest_confidence":
        return HighestConfidencePolicy()
    if cfg.type == "best_value":
        return BestValuePolicy()
    raise ValueError(f"Unknown policy type: {cfg.type!r}")


def select_model(
    scores: dict[str, ModelScore],
    models: list["ModelProfile"],
    policy: Policy | PolicyConfig,
) -> SelectionResult:
    """Top-level helper: apply a Policy (or PolicyConfig) to a score map."""
    if isinstance(policy, PolicyConfig):
        policy = build_policy(policy)
    return policy.apply(scores, models)
