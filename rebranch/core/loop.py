"""Main ReBranch iterative loop: screen -> CE -> retune/conflict/branch -> admit."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Sequence

import numpy as np

from rebranch.api.primitives import API_DOC, PHYSCAP_EXTRA_PRIMITIVES
from rebranch.core.branch import (
    format_observables_summary,
    format_trace_feedback,
    synthesize_branch,
)
from rebranch.core.conflict import (
    CaseSolution,
    _norm_solution_dist,
    cross_evaluate_solutions,
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
from rebranch.core.lifting import ConstSpec, hoist_inline_literals, lift_constants
from rebranch.core.retune import RetuneResult, Retuner, score_traces
from rebranch.core.screen import sensitivity_screen
from rebranch.exec.evaluator import BudgetExceededError, Evaluator
from rebranch.exec.sandbox import StaticCheckError, check_policy_source
from rebranch.exec.trace import Trace
from rebranch.llm.client import LLMClient, extract_python_code, LimitedLLMClient, LLMBudgetExceededError
from rebranch.core.development import AdmissionGate, polish_candidate
from rebranch.llm.prompts import (
    PHYSCAP_EXTRA_DOC,
    SYSTEM_PROMPT,
    build_repair_prompt,
)
from rebranch.sim.backend import TaskBackend


@dataclass
class ReBranchConfig:
    """Configuration and §7 ablation toggles for `run_rebranch`."""

    max_rounds: int = 4
    n_screen_base: int = 8
    n_ce_samples: int = 200
    k_max: int = 6
    n_bisect: int = 6
    divergence_delta: float = 3.0
    retune_iters: int = 20
    retune_margin_target: float = 0.25
    max_branch_retries: int = 2
    admission_n_eval: int = 24
    seed: int = 0
    max_llm_calls: int = 8
    # §7 ablation toggles
    random_ce: bool = False
    worst_ce: bool = False
    no_retune_or_branch: bool = False
    no_lifting: bool = False
    no_branch_verify: bool = False
    physcap_primitives: bool = False


@dataclass
class RoundLog:
    """Per-round telemetry and policy snapshot."""

    round_idx: int
    action: str
    n_counterexamples: int
    n_regression: int
    active_params: list[str]
    source_before: str
    source_after: str
    admitted: bool
    eval_rate_before: float
    eval_rate_after: float
    n_rollouts_total: int
    llm_calls_total: int
    details: dict[str, Any] = field(default_factory=dict)
    admission_feedback: str = ""


@dataclass
class ReBranchResult:
    """Complete result returned by `run_rebranch`."""

    final_source: str
    initial_source: str
    active_params: list[str]
    rounds: list[RoundLog]
    regression_set: list[dict[str, Any]]
    converged: bool
    n_rollouts: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    stop_reason: str = "round_limit"


def _format_repair_summary(
    traces: Sequence[Trace],
    *,
    nominal_trace: Trace | None = None,
    backend: TaskBackend | None = None,
) -> str:
    """Summarize failure traces strictly via SDK observables (zero theta leakage)."""
    failing = [tr for tr in traces if not tr.success]
    if not failing:
        return "No failures recorded."
    parts: list[str] = [format_trace_feedback(failing)]
    if nominal_trace is not None and nominal_trace.success and backend is not None:
        obs_ranked = distinguishing_observable(
            [nominal_trace, failing[0]],
            [(0, 1)],
            backend.obs_scale,
            obs_api_map=backend.obs_api_map,
            nominal_trace=nominal_trace,
        )
        if obs_ranked:
            parts.append(
                "Observables differing from passing nominal execution:\n"
                + format_observables_summary(obs_ranked, obs_api_map=backend.obs_api_map)
            )
    return "\n\n".join(parts)


def _repair_with_retries(
    backend: TaskBackend,
    evaluator: Evaluator,
    llm: LLMClient,
    work_src: str,
    repair_traces: Sequence[Trace],
    eval_cases: Sequence[dict[str, Any]],
    *,
    nominal_trace: Trace,
    cfg: ReBranchConfig,
    retuner: Retuner,
    extra_prims: tuple[str, ...],
    api_doc: str,
    admission_feedback: str = "",
) -> str:
    """Synthesize a structural repair with SDK call-trace feedback retries."""
    base_summary = _format_repair_summary(
        repair_traces, nominal_trace=nominal_trace, backend=backend
    )
    feedback = admission_feedback
    best_cand = work_src
    best_rank = (-1, -1e9)
    n_attempts = 1 + max(0, int(cfg.max_branch_retries))

    for _ in range(n_attempts):
        prompt = build_repair_prompt(
            backend.task_description,
            work_src,
            base_summary,
            extra_notes=feedback,
            api_doc=api_doc,
        )
        try:
            resp = llm.complete(prompt, system=SYSTEM_PROMPT)
        except LLMBudgetExceededError:
            break  # Admit the best already-evaluated candidate before stopping.
        cand = extract_python_code(resp.text)
        try:
            check_policy_source(cand, extra_primitives=extra_prims)
        except StaticCheckError as exc:
            feedback = (
                f"Previous Candidate:\n```python\n{cand.strip()}\n```\n\n"
                f"Static sandbox check failed: {exc}"
            )
            continue

        cand_eval = _evaluate_fixed(evaluator, cand, eval_cases)
        if not cfg.no_lifting:
            opt = polish_candidate(evaluator, cand, eval_cases, seed=retuner.seed,
                                   reserve=retuner.budget_reserve)
            if opt is not None and (opt.n_success, opt.mean_score) > (cand_eval.n_success, cand_eval.mean_score):
                cand, cand_eval = opt.best_source, opt

        rank = (cand_eval.n_success, cand_eval.mean_score)
        if rank > best_rank:
            best_rank = rank
            best_cand = cand
        if cand_eval.success_all:
            return cand

        feedback = (
            f"Previous Candidate:\n```python\n{cand.strip()}\n```\n\n"
            f"Verification Trace Feedback:\n{format_trace_feedback(cand_eval.traces)}"
        )

    return best_cand


def _lift_with_optional_hoist(source: str) -> tuple[str, list[ConstSpec]]:
    consts = lift_constants(source)
    if consts:
        return source, consts
    hoisted = hoist_inline_literals(source)
    return hoisted, lift_constants(hoisted)


def _dedup_cases(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for c in cases:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _eval_admission(
    evaluator: Evaluator,
    candidate: str,
    val_cases: Sequence[dict[str, Any]],
    cur_rate: float,
    extra_primitives: tuple[str, ...],
    incumbent_source: str,
    admission_gate: AdmissionGate,
) -> tuple[bool, float]:
    try:
        check_policy_source(candidate, extra_primitives=extra_primitives)
    except StaticCheckError:
        return False, cur_rate
    return admission_gate.consider(candidate)


def _evaluate_fixed(
    evaluator: Evaluator, source: str, cases: Sequence[dict[str, Any]]
) -> RetuneResult:
    traces = evaluator.run_many(source, cases)
    ok_all, n_succ, worst, mean, obj = score_traces(traces)
    return RetuneResult(
        success_all=ok_all,
        best_values={},
        best_source=source,
        best_objective=obj,
        n_success=n_succ,
        n_cases=len(traces),
        worst_score=worst,
        mean_score=mean,
        traces=traces,
        n_evals=1,
    )


def run_rebranch(
    backend: TaskBackend,
    evaluator: Evaluator,
    llm: LLMClient,
    config: ReBranchConfig | None = None,
    initial_source: str | None = None,
    oracle_filter: bool = False,
) -> ReBranchResult:
    """Execute the full §3.12 ReBranch diagnosis, retune, and branch synthesis loop."""
    if oracle_filter:
        raise ValueError("Privileged oracle filtering is not part of protocol v2")
    t_start = time.perf_counter()
    cfg = config or ReBranchConfig()
    llm = LimitedLLMClient(llm, cfg.max_llm_calls)
    rng = np.random.default_rng(cfg.seed)

    src = initial_source if initial_source is not None else backend.draft_policy
    init_src = src
    extra_prims = tuple(
        dict.fromkeys(
            (
                *evaluator.extra_primitives,
                *(PHYSCAP_EXTRA_PRIMITIVES if cfg.physcap_primitives else ()),
            )
        )
    )
    api_doc = (
        f"{API_DOC}\n\n{PHYSCAP_EXTRA_DOC}" if cfg.physcap_primitives else API_DOC
    )

    val_cases = [
        backend.param_space.sample(rng) for _ in range(max(0, int(cfg.admission_n_eval)))
    ]

    rounds: list[RoundLog] = []
    regression_set: list[dict[str, Any]] = []
    active_names: list[str] = list(backend.param_space.names)
    converged = False
    cur_val_rate = 0.0
    rollouts_before = evaluator.n_rollouts
    llm_calls_before = int(llm.stats.get("calls", 0))
    prompt_tok_before = int(llm.stats.get("prompt_tokens", 0))
    comp_tok_before = int(llm.stats.get("completion_tokens", 0))

    admission_gate = None
    try:
        admission_gate = AdmissionGate(evaluator, val_cases, src)
        cur_val_rate = admission_gate.rate
        active_names = sensitivity_screen(
            evaluator, src, backend.param_space, rng, n_base=cfg.n_screen_base
        )
        active_space = backend.param_space.with_active(active_names)
        nominal_theta = active_space.nominal()
        regression_set = [nominal_theta]

        for round_idx in range(cfg.max_rounds):
            if round_idx > 0 and llm.attempts >= llm.limit:
                raise LLMBudgetExceededError("Proposal limit reached after prior candidate admission")
            src_before = src
            rate_before = cur_val_rate
            nominal_trace = evaluator.run(src, nominal_theta, seed=0)

            if cfg.random_ce:
                ces, ce_stats = find_ablation_counterexamples(
                    evaluator,
                    src,
                    active_space,
                    nominal_trace,
                    backend.obs_scale,
                    rng,
                    mode="random_ce",
                    n_samples=cfg.n_ce_samples,
                    k_max=cfg.k_max,
                    delta=cfg.divergence_delta,
                )
            elif cfg.worst_ce:
                ces, ce_stats = find_ablation_counterexamples(
                    evaluator,
                    src,
                    active_space,
                    nominal_trace,
                    backend.obs_scale,
                    rng,
                    mode="worst_ce",
                    n_samples=cfg.n_ce_samples,
                    k_max=cfg.k_max,
                    delta=cfg.divergence_delta,
                )
            else:
                ces, ce_stats = find_counterexamples(
                    evaluator,
                    src,
                    active_space,
                    nominal_trace,
                    backend.obs_scale,
                    rng,
                    n_samples=cfg.n_ce_samples,
                    k_max=cfg.k_max,
                    n_bisect=cfg.n_bisect,
                    delta=cfg.divergence_delta,
                )

            if not ces and nominal_trace.success:
                converged = True
                rounds.append(
                    RoundLog(
                        admission_feedback=admission_gate.feedback if admission_gate is not None else "",
                        round_idx=round_idx,
                        action="CONVERGED",
                        n_counterexamples=0,
                        n_regression=len(regression_set),
                        active_params=list(active_names),
                        source_before=src_before,
                        source_after=src,
                        admitted=True,
                        eval_rate_before=rate_before,
                        eval_rate_after=cur_val_rate,
                        n_rollouts_total=evaluator.n_rollouts - rollouts_before,
                        llm_calls_total=int(llm.stats.get("calls", 0)) - llm_calls_before,
                        details={"ce_stats": ce_stats},
                    )
                )
                break

            new_ce_thetas = [ce.theta for ce in ces]
            target_cases = _dedup_cases(list(regression_set) + new_ce_thetas)

            if cfg.no_retune_or_branch:
                traces_fail = [ce.trace for ce in ces] or [nominal_trace]
                summary = _format_repair_summary(traces_fail)
                prompt = build_repair_prompt(
                    backend.task_description, src, summary, extra_notes=admission_gate.feedback, api_doc=api_doc
                )
                resp = llm.complete(prompt, system=SYSTEM_PROMPT)
                cand = extract_python_code(resp.text)
                admitted, cand_rate = _eval_admission(
                    evaluator, cand, val_cases, cur_val_rate, extra_prims, src, admission_gate
                )
                if admitted:
                    src = cand
                    cur_val_rate = cand_rate
                    regression_set = _dedup_cases(regression_set + new_ce_thetas)
                rounds.append(
                    RoundLog(
                        admission_feedback=admission_gate.feedback if admission_gate is not None else "",
                        round_idx=round_idx,
                        action="ABLATION_REPAIR" if admitted else "REJECTED",
                        n_counterexamples=len(ces),
                        n_regression=len(regression_set),
                        active_params=list(active_names),
                        source_before=src_before,
                        source_after=src,
                        admitted=admitted,
                        eval_rate_before=rate_before,
                        eval_rate_after=cand_rate if admitted else rate_before,
                        n_rollouts_total=evaluator.n_rollouts - rollouts_before,
                        llm_calls_total=int(llm.stats.get("calls", 0)) - llm_calls_before,
                        details={"ce_stats": ce_stats},
                    )
                )
                continue

            if cfg.no_lifting:
                work_src, consts = src, []
            else:
                work_src, consts = _lift_with_optional_hoist(src)

            admission_reserve = (
                min(len(val_cases) + 20, evaluator.budget_limit // 2)
                if evaluator.budget_limit is not None
                else 0
            )
            retuner = Retuner(
                evaluator,
                max_iters=cfg.retune_iters,
                margin_target=cfg.retune_margin_target,
                seed=cfg.seed + round_idx * 17,
                budget_reserve=admission_reserve,
            )

            if consts and not cfg.no_lifting:
                case_sols = per_case_solutions(
                    retuner,
                    work_src,
                    consts,
                    target_cases,
                    max_rollouts_per_case=60,
                )
            else:
                cur_traces = evaluator.run_many(work_src, target_cases)
                case_sols = [
                    CaseSolution(
                        index=idx,
                        case=c,
                        solvable=tr.success,
                        best_values={},
                        best_source=work_src,
                        score=float(tr.score),
                        trace=tr,
                    )
                    for idx, (c, tr) in enumerate(zip(target_cases, cur_traces))
                ]

            # Keep every training instance; failed single-case search calls for repair.
            surviving_cases = [sol.case for sol in case_sols]
            surviving_sols = list(case_sols)
            repair_traces = [sol.trace for sol in case_sols if not sol.solvable]
            n_dead_twins = 0  # Legacy telemetry field; v2 never filters instances.

            if repair_traces:
                cand = _repair_with_retries(
                    backend,
                    evaluator,
                    llm,
                    work_src,
                    repair_traces,
                    surviving_cases,
                    nominal_trace=nominal_trace,
                    cfg=cfg,
                    retuner=retuner,
                    extra_prims=extra_prims,
                    api_doc=api_doc,
                    admission_feedback=admission_gate.feedback,
                )
                admitted, cand_rate = _eval_admission(
                    evaluator, cand, val_cases, cur_val_rate, extra_prims, src, admission_gate
                )
                if admitted:
                    src = cand
                    cur_val_rate = cand_rate
                    regression_set = _dedup_cases(surviving_cases)
                rounds.append(
                    RoundLog(
                        admission_feedback=admission_gate.feedback if admission_gate is not None else "",
                        round_idx=round_idx,
                        action="REPAIR" if admitted else "REJECTED",
                        n_counterexamples=len(ces),
                        n_regression=len(regression_set),
                        active_params=list(active_names),
                        source_before=src_before,
                        source_after=src,
                        admitted=admitted,
                        eval_rate_before=rate_before,
                        eval_rate_after=cand_rate if admitted else rate_before,
                        n_rollouts_total=evaluator.n_rollouts - rollouts_before,
                        llm_calls_total=int(llm.stats.get("calls", 0)) - llm_calls_before,
                        details={"n_dead_twins": n_dead_twins, "ce_stats": ce_stats},
                    )
                )
                continue

            # Short-circuit joint retuning if cross-eval confirms an incompatible pair.
            cross_ok: dict[tuple[int, int], bool] | None = None
            confirmed_unsat_pair: tuple[int, int] | None = None
            if consts and not cfg.no_lifting and len(surviving_cases) >= 2:
                cross_ok = cross_evaluate_solutions(
                    evaluator,
                    surviving_cases,
                    surviving_sols,
                    budget_reserve=admission_reserve,
                )
                mut_pairs = [
                    (i, j)
                    for i in range(len(surviving_cases))
                    for j in range(i + 1, len(surviving_cases))
                    if cross_ok.get((i, j)) is False
                    and cross_ok.get((j, i)) is False
                ]
                mut_pairs.sort(
                    key=lambda ij: (
                        -_norm_solution_dist(
                            consts,
                            surviving_sols[ij[0]].best_values,
                            surviving_sols[ij[1]].best_values,
                        ),
                        ij[0],
                        ij[1],
                    )
                )
                if mut_pairs:
                    i_star, j_star = mut_pairs[0]
                    pair_retuner = Retuner(
                        evaluator,
                        max_iters=min(cfg.retune_iters, 5),
                        popsize=6,
                        margin_target=1e-6,
                        seed=retuner.seed,
                        max_rollouts=70,
                        budget_reserve=admission_reserve,
                    )
                    pair_res = pair_retuner.optimize(
                        work_src,
                        consts,
                        [surviving_cases[i_star], surviving_cases[j_star]],
                    )
                    if pair_res.n_evals > 0 and not pair_res.success_all:
                        confirmed_unsat_pair = (i_star, j_star)

            # All surviving cases are individually solvable -> attempt joint retune.
            if confirmed_unsat_pair is not None:
                joint_res = RetuneResult(
                    success_all=False,
                    best_values={},
                    best_source=work_src,
                    best_objective=-float("inf"),
                    n_success=0,
                    n_cases=len(surviving_cases),
                    worst_score=-1.0,
                    mean_score=-1.0,
                    traces=[],
                    n_evals=0,
                )
            elif consts and not cfg.no_lifting:
                joint_cap = (
                    max(80, (evaluator.budget_limit - admission_reserve - evaluator.n_rollouts) // 2)
                    if evaluator.budget_limit is not None
                    else None
                )
                joint_retuner = Retuner(
                    evaluator,
                    max_iters=cfg.retune_iters,
                    margin_target=cfg.retune_margin_target,
                    seed=retuner.seed,
                    max_rollouts=joint_cap,
                    budget_reserve=admission_reserve,
                )
                joint_res = joint_retuner.optimize(work_src, consts, surviving_cases)
            else:
                joint_res = _evaluate_fixed(evaluator, work_src, surviving_cases)

            if joint_res.success_all:
                cand = joint_res.best_source
                admitted, cand_rate = _eval_admission(
                    evaluator, cand, val_cases, cur_val_rate, extra_prims, src, admission_gate
                )
                if admitted:
                    src = cand
                    cur_val_rate = cand_rate
                    regression_set = _dedup_cases(surviving_cases)
                rounds.append(
                    RoundLog(
                        admission_feedback=admission_gate.feedback if admission_gate is not None else "",
                        round_idx=round_idx,
                        action="RETUNE" if admitted else "REJECTED",
                        n_counterexamples=len(ces),
                        n_regression=len(regression_set),
                        active_params=list(active_names),
                        source_before=src_before,
                        source_after=src,
                        admitted=admitted,
                        eval_rate_before=rate_before,
                        eval_rate_after=cand_rate if admitted else rate_before,
                        n_rollouts_total=evaluator.n_rollouts - rollouts_before,
                        llm_calls_total=int(llm.stats.get("calls", 0)) - llm_calls_before,
                        details={
                            "n_dead_twins": n_dead_twins,
                            "best_values": joint_res.best_values,
                            "ce_stats": ce_stats,
                        },
                    )
                )
                continue

            # Joint retuning failed -> unsat_core + BRANCH.
            repair_feedback_traces = [ce.trace for ce in ces] or [nominal_trace]
            if len(surviving_cases) < 2:
                cand = _repair_with_retries(
                    backend,
                    evaluator,
                    llm,
                    work_src,
                    repair_feedback_traces,
                    surviving_cases,
                    nominal_trace=nominal_trace,
                    cfg=cfg,
                    retuner=retuner,
                    extra_prims=extra_prims,
                    api_doc=api_doc,
                    admission_feedback=admission_gate.feedback,
                )
                action_label = "REPAIR"
                details: dict[str, Any] = {"n_dead_twins": n_dead_twins}
            else:
                core_idx = unsat_core(
                    retuner,
                    work_src,
                    consts,
                    surviving_cases,
                    case_solutions=surviving_sols,
                    cross_ok=cross_ok,
                    confirmed_unsat_pair=confirmed_unsat_pair,
                )
                include_zero = (
                    0 not in core_idx
                    and 0 < len(surviving_cases)
                    and surviving_sols[0].solvable
                    and cross_ok is not None
                    and any(cross_ok.get((c, 0)) is False for c in core_idx)
                )
                branch_idx = sorted(set(core_idx) | ({0} if include_zero else set()))
                idx_map = {orig: new for new, orig in enumerate(branch_idx)}
                core_cases = [surviving_cases[i] for i in branch_idx]
                core_sols = [surviving_sols[i] for i in branch_idx]
                branch_cross = (
                    {
                        (idx_map[a], idx_map[b]): v
                        for (a, b), v in cross_ok.items()
                        if a in idx_map and b in idx_map
                    }
                    if cross_ok is not None
                    else None
                )
                branch_confirmed = (
                    [(idx_map[confirmed_unsat_pair[0]], idx_map[confirmed_unsat_pair[1]])]
                    if confirmed_unsat_pair is not None
                    and confirmed_unsat_pair[0] in idx_map
                    and confirmed_unsat_pair[1] in idx_map
                    else None
                )
                incompat = incompatible_pairs(
                    evaluator,
                    work_src,
                    consts,
                    core_cases,
                    core_sols,
                    retuner=retuner if consts else None,
                    cross_ok=branch_cross,
                    confirmed_pairs=branch_confirmed,
                )
                if not incompat:
                    cand = _repair_with_retries(
                        backend, evaluator, llm, work_src,
                        repair_feedback_traces, surviving_cases,
                        nominal_trace=nominal_trace, cfg=cfg, retuner=retuner,
                        extra_prims=extra_prims, api_doc=api_doc,
                        admission_feedback=admission_gate.feedback,
                    )
                    ver = None
                    obs_ranked = []
                else:
                    core_traces = evaluator.run_many(work_src, core_cases)
                    obs_ranked = distinguishing_observable(
                        core_traces,
                        incompat,
                        backend.obs_scale,
                        obs_api_map=backend.obs_api_map,
                        nominal_trace=nominal_trace,
                    )
                    cand, ver = synthesize_branch(
                        evaluator,
                        llm,
                        backend.task_description,
                        work_src,
                        core_cases,
                        core_traces,
                        incompat,
                        obs_ranked,
                        case_solutions=core_sols,
                        obs_api_map=backend.obs_api_map,
                        retuner=retuner,
                        max_retries=cfg.max_branch_retries,
                        no_branch_verify=cfg.no_branch_verify,
                        no_lifting=cfg.no_lifting,
                        physcap_primitives=cfg.physcap_primitives,
                        initial_feedback=admission_gate.feedback,
                    )
                if not cfg.no_lifting and ver is not None and ver.passed:
                    try:
                        check_policy_source(cand, extra_primitives=extra_prims)
                        cand_eval = _evaluate_fixed(evaluator, cand, surviving_cases)
                        if not (
                            cand_eval.success_all
                            and cand_eval.worst_score >= cfg.retune_margin_target
                        ):
                            opt = polish_candidate(evaluator, cand, surviving_cases,
                                seed=retuner.seed, reserve=admission_reserve)
                            if opt is not None and (opt.n_success, opt.mean_score) > (cand_eval.n_success, cand_eval.mean_score):
                                cand = opt.best_source
                    except StaticCheckError:
                        pass
                action_label = "BRANCH" if incompat else "REPAIR"
                details = {
                    "n_dead_twins": n_dead_twins,
                    "core_size": len(core_cases),
                    "incompatible_pairs": incompat,
                    "top_observables": [o.key for o in obs_ranked[:3]],
                    "branch_verified": ver.passed if ver is not None else False,
                    "ce_stats": ce_stats,
                }

            admitted, cand_rate = _eval_admission(
                evaluator, cand, val_cases, cur_val_rate, extra_prims, src, admission_gate
            )
            if admitted:
                src = cand
                cur_val_rate = cand_rate
                regression_set = _dedup_cases(surviving_cases)
            rounds.append(
                RoundLog(
                        admission_feedback=admission_gate.feedback if admission_gate is not None else "",
                    round_idx=round_idx,
                    action=action_label if admitted else "REJECTED",
                    n_counterexamples=len(ces),
                    n_regression=len(regression_set),
                    active_params=list(active_names),
                    source_before=src_before,
                    source_after=src,
                    admitted=admitted,
                    eval_rate_before=rate_before,
                    eval_rate_after=cand_rate if admitted else rate_before,
                    n_rollouts_total=evaluator.n_rollouts - rollouts_before,
                    llm_calls_total=int(llm.stats.get("calls", 0)) - llm_calls_before,
                    details=details,
                )
            )
    except (BudgetExceededError, LLMBudgetExceededError) as exc:
        stop_kind = "LLM_BUDGET_STOP" if isinstance(exc, LLMBudgetExceededError) else "SIM_BUDGET_STOP"
        rounds.append(
            RoundLog(
                        admission_feedback=admission_gate.feedback if admission_gate is not None else "",
                round_idx=len(rounds),
                action=stop_kind,
                n_counterexamples=0,
                n_regression=len(regression_set),
                active_params=list(active_names),
                source_before=src,
                source_after=src,
                admitted=False,
                eval_rate_before=cur_val_rate,
                eval_rate_after=cur_val_rate,
                n_rollouts_total=evaluator.n_rollouts - rollouts_before,
                llm_calls_total=int(llm.stats.get("calls", 0)) - llm_calls_before,
                details={"sim_budget_limit": evaluator.budget_limit, "llm_budget_limit": cfg.max_llm_calls, "reason": str(exc)},
            )
        )

    wall_s = time.perf_counter() - t_start
    return ReBranchResult(
        final_source=src,
        initial_source=init_src,
        active_params=list(active_names),
        rounds=rounds,
        regression_set=regression_set,
        converged=converged,
        n_rollouts=evaluator.n_rollouts - rollouts_before,
        llm_calls=int(llm.stats.get("calls", 0)) - llm_calls_before,
        prompt_tokens=int(llm.stats.get("prompt_tokens", 0)) - prompt_tok_before,
        completion_tokens=int(llm.stats.get("completion_tokens", 0)) - comp_tok_before,
        wall_s=wall_s,
        stop_reason=("converged" if converged else
                     (rounds[-1].action if rounds and rounds[-1].action in ("SIM_BUDGET_STOP", "LLM_BUDGET_STOP") else
                      ("LLM_BUDGET_STOP" if llm.attempts >= llm.limit else "round_limit"))),
    )
