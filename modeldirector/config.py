"""Pydantic configuration models for ModelDirector.

A `Config` is a single YAML file (or programmatic dict) that fully describes:
  * which LLM to use as the selector
  * which decision policy to apply
  * which candidate models to score against
"""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Fields that get a default cost if the user did not specify one.
# Costs are arbitrary units - the user defines the scale.  The benchmark
# only uses the *relative* values to compute savings.
_DEFAULT_COST_BY_PRIORITY = {1: 1, 2: 5, 3: 20}


class Capabilities(BaseModel):
    """Static capability scores for a model (user-defined, 0-100)."""

    reasoning: int = Field(..., ge=0, le=100)
    coding: int = Field(..., ge=0, le=100)
    context: int = Field(..., ge=0, le=100)
    creativity: int | None = Field(None, ge=0, le=100)


class ModelProfile(BaseModel):
    """A single candidate model the selector can choose from."""

    id: str = Field(..., min_length=1, description="Stable identifier used in output")
    name: str = Field(..., min_length=1, description="Provider-specific model name passed to LiteLLM")
    display_name: str | None = Field(None, description="Human-friendly label (optional)")
    description: str = Field(
        "",
        description=(
            "Free-form description of the model. Strongly recommended - smaller "
            "selector models may not recognise bare model names."
        ),
    )
    capabilities: Capabilities
    priority: int = Field(1, ge=1, description="Lower = cheaper / preferred. Used to break ties and as a default cost.")
    cost: float | None = Field(
        None,
        ge=0,
        description=(
            "Relative cost unit. Optional; if omitted, ModelDirector derives a "
            "sensible default from the model's priority."
        ),
    )

    @field_validator("id", "name")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace")
        return v

    @model_validator(mode="after")
    def _apply_default_cost(self) -> "ModelProfile":
        if self.cost is None:
            # Use the priority bucket, then escalate if priority is out of the table.
            self.cost = float(_DEFAULT_COST_BY_PRIORITY.get(self.priority, self.priority * 10))
        return self


class SelectorConfig(BaseModel):
    """Configuration of the LLM that scores the candidates."""

    provider: str = Field(..., min_length=1, description="LiteLLM provider, e.g. 'openrouter', 'openai'")
    model: str = Field(..., min_length=1, description="Model name in LiteLLM format")
    api_key: str | None = Field(
        None,
        description=(
            "API key. If unset, ModelDirector falls back to the env var "
            "<PROVIDER>_API_KEY (uppercased)."
        ),
    )
    api_base: str | None = Field(None, description="Optional base URL override")
    temperature: float = Field(0.0, ge=0, le=2, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Optional cap on selector output")

    def resolved_api_key(self) -> str:
        """Return the API key, falling back to the provider's env var."""
        if self.api_key and not self.api_key.startswith("${"):
            return self.api_key
        if self.api_key and self.api_key.startswith("${"):
            return _resolve_env(self.api_key)
        env_name = f"{self.provider.upper()}_API_KEY"
        val = os.environ.get(env_name)
        if not val:
            raise ValueError(
                f"No API key for provider '{self.provider}'. "
                f"Set 'selector.api_key' or env var ${env_name}."
            )
        return val


class PolicyConfig(BaseModel):
    """Decision policy applied after scoring."""

    type: Literal["cheapest_capable", "highest_confidence", "balanced"] = "cheapest_capable"
    threshold: int = Field(85, ge=0, le=100, description="Confidence threshold for 'cheapest_capable'")
    cost_per_million: bool = Field(
        True,
        description=(
            "If true, costs are interpreted as USD per 1M tokens for the 'balanced' "
            "policy report. Does not affect the decision - just the metric."
        ),
    )

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_legacy(cls, v: object) -> object:
        # accept `cheapest_capable`, `cheapest-capable`, `Cheapest Capable` etc.
        if isinstance(v, str):
            return v.strip().lower().replace("-", "_").replace(" ", "_")
        return v

    @classmethod
    def default(cls) -> "PolicyConfig":
        """Return a default policy config (helper for Pydantic field defaults)."""
        # pyright thinks cls(...) requires threshold + cost_per_million, but they
        # have defaults in the class body. Suppress the false positive.
        return cls(type="cheapest_capable")  # type: ignore[call-arg]


class Config(BaseModel):
    """Top-level configuration."""

    selector: SelectorConfig
    policy: PolicyConfig = Field(default_factory=PolicyConfig.default)
    models: list[ModelProfile] = Field(..., min_length=1)

    @field_validator("models")
    @classmethod
    def _unique_ids(cls, v: list[ModelProfile]) -> list[ModelProfile]:
        ids = [m.id for m in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate model ids: {dupes}")
        return v


def _resolve_env(placeholder: str) -> str:
    """Resolve ${VAR_NAME} style placeholders against the process environment."""
    m = re.match(r"^\$\{([A-Z0-9_]+)\}$", placeholder)
    if not m:
        raise ValueError(f"Invalid env placeholder: {placeholder!r}")
    val = os.environ.get(m.group(1))
    if not val:
        raise ValueError(f"Environment variable ${m.group(1)} is not set")
    return val
