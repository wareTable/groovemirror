"""One-to-one onset metrics (no instrument classification or rhythm scoring)."""
from .onsets import EvaluationConfig, EvaluationResult, OnsetMatch, evaluate_onsets

__all__ = ["EvaluationConfig", "EvaluationResult", "OnsetMatch", "evaluate_onsets"]
