"""Evaluation metrics and benchmark runner for ReBranch and baselines."""

from rebranch.eval.metrics import (
    EvalMetrics,
    compute_branch_discrimination_accuracy,
    compute_metrics,
    policy_has_sensory_branch,
    summarize_trace_for_llm,
    trace_used_probe,
)

__all__ = [
    "EvalMetrics",
    "compute_branch_discrimination_accuracy",
    "compute_metrics",
    "policy_has_sensory_branch",
    "summarize_trace_for_llm",
    "trace_used_probe",
]
