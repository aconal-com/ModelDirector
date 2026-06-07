"""ModelDirector - Stateless AI model selection engine."""

from modeldirector.config import Config, ModelProfile, PolicyConfig, SelectorConfig
from modeldirector.models import ModelScore, SelectionResult
from modeldirector.policy import Policy, select_model
from modeldirector.selector import ModelDirector
from modeldirector.loader import load_config

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ModelProfile",
    "PolicyConfig",
    "SelectorConfig",
    "ModelScore",
    "SelectionResult",
    "Policy",
    "select_model",
    "ModelDirector",
    "load_config",
]
