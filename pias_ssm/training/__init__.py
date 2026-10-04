"""Training, evaluation, and statistical reporting."""

from .engine import build_model, run_training
from .evaluator import EvaluationResult, evaluate_model

__all__ = ["EvaluationResult", "build_model", "evaluate_model", "run_training"]
