"""Public data types: model scores, selection result, and policy enum."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PolicyType = Literal["cheapest_capable", "highest_confidence", "balanced"]


class ModelScore(BaseModel):
    """Score assigned to a single candidate model by the selector."""

    id: str = Field(..., description="The model profile id being scored")
    overall: int = Field(..., ge=0, le=100, description="Weighted overall confidence 0-100")
    reasoning: int = Field(..., ge=0, le=100, description="Reasoning capability fit 0-100")
    coding: int = Field(..., ge=0, le=100, description="Coding capability fit 0-100")
    context: int = Field(..., ge=0, le=100, description="Context length fit 0-100")
    creativity: int | None = Field(None, ge=0, le=100, description="Optional creativity fit 0-100")
    explanation: str = Field(..., min_length=1, description="Why this model got this score")

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be empty")
        return v


class SelectionResult(BaseModel):
    """The full output of a model selection call."""

    selected_model: str = Field(..., description="The id of the chosen model")
    policy: PolicyType = Field(..., description="The policy that made the decision")
    scores: dict[str, ModelScore] = Field(..., description="All candidate scores keyed by id")
    reason: str = Field(..., min_length=1, description="Human-readable explanation of the decision")

    def cost_saved_vs(
        self,
        baseline_model_id: str,
        models: list[dict],
    ) -> float:
        """Return the cost saved if we used `self.selected_model` instead of `baseline_model_id`.

        Both models are looked up in `models` (a list of dicts with `id` and `cost`).
        Returns a fraction between -1.0 and 1.0 (positive = saved, negative = cost more).
        """
        baseline_cost = None
        chosen_cost = None
        for m in models:
            if m["id"] == baseline_model_id:
                baseline_cost = float(m.get("cost", 0))
            if m["id"] == self.selected_model:
                chosen_cost = float(m.get("cost", 0))

        if baseline_cost is None:
            raise ValueError(f"baseline_model_id '{baseline_model_id}' not in models")
        if chosen_cost is None:
            raise ValueError(f"selected_model '{self.selected_model}' not in models")
        if baseline_cost == 0:
            return 0.0

        return (baseline_cost - chosen_cost) / baseline_cost
