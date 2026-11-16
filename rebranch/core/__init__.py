"""ReBranch core algorithm modules."""

from rebranch.core.branch import (
    BranchVerifyResult,
    build_branch_prompt,
    format_conflict_summary,
    format_observables_summary,
    format_trace_feedback,
    synthesize_branch,
    verify_branch,
)
from rebranch.core.conflict import (
    CaseSolution,
    DistinguishingObservable,
    distinguishing_observable,
    incompatible_pairs,
    per_case_solutions,
    unsat_core,
)
from rebranch.core.counterexamples import (
    Counterexample,
    find_ablation_counterexamples,
    find_counterexamples,
)
from rebranch.core.lifting import (
    ConstSpec,
    hoist_inline_literals,
    instantiate,
    lift_constants,
)
from rebranch.core.loop import ReBranchConfig, ReBranchResult, RoundLog, run_rebranch
from rebranch.core.retune import RetuneResult, Retuner, score_traces
from rebranch.core.screen import sensitivity_screen
from rebranch.core.signature import (
    FailureSignature,
    failure_signature,
    first_divergence,
    top_deviation_signs,
)

__all__ = [
    "BranchVerifyResult",
    "CaseSolution",
    "ConstSpec",
    "Counterexample",
    "DistinguishingObservable",
    "FailureSignature",
    "ReBranchConfig",
    "ReBranchResult",
    "RetuneResult",
    "Retuner",
    "RoundLog",
    "build_branch_prompt",
    "distinguishing_observable",
    "failure_signature",
    "find_ablation_counterexamples",
    "find_counterexamples",
    "first_divergence",
    "format_conflict_summary",
    "format_observables_summary",
    "format_trace_feedback",
    "hoist_inline_literals",
    "incompatible_pairs",
    "instantiate",
    "lift_constants",
    "per_case_solutions",
    "run_rebranch",
    "score_traces",
    "sensitivity_screen",
    "synthesize_branch",
    "top_deviation_signs",
    "unsat_core",
    "verify_branch",
]
