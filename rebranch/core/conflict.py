"""Conflict analysis: per-case solvability, greedy unsat core, incompatible pairs, distinguishing observables."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from rebranch.core.lifting import ConstSpec
from rebranch.core.retune import Retuner
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.trace import Trace
from rebranch.sim.backend import SNAPSHOT_KEYS


@dataclass
class CaseSolution:
    """Single-case constant optimization outcome."""

    index: int
    case: Any
    solvable: bool
    best_values: dict[str, Any]
    best_source: str
    score: float
    trace: Trace


@dataclass(frozen=True)
class DistinguishingObservable:
    """SDK observable ranked by separation across conflicting cases."""

    key: str
    score: float
    call_index: int
    call_name: str
    lineno: int
    val_a: float
    val_b: float
    opposite_sign: bool
    api_expr: str


def per_case_solutions(
    retuner: Retuner,
    source: str,
    consts: Sequence[ConstSpec],
    cases: Sequence[Any],
    max_rollouts_per_case: int | None = None,
) -> list[CaseSolution]:
    """Optimize constants independently for each case in `cases`."""
    # Per-case solvability only needs any passing solution (stop immediately on success).
    fast_retuner = Retuner(
        retuner.evaluator,
        max_iters=retuner.max_iters,
        popsize=retuner.popsize,
        margin_target=1e-6,
        sigma0=retuner.sigma0,
        seed=retuner.seed,
        max_rollouts=(
            max_rollouts_per_case
            if max_rollouts_per_case is not None
            else retuner.max_rollouts
        ),
        budget_reserve=retuner.budget_reserve,
    )
    sols: list[CaseSolution] = []
    for idx, case in enumerate(cases):
        res = fast_retuner.optimize(source, consts, [case])
        if res.traces:
            tr = res.traces[0]
        else:
            tr = Trace(
                theta=dict(case) if isinstance(case, dict) else {},
                seed=0,
                calls=[],
                lines=[],
                success=False,
                score=-1.0,
                failure="budget_exhausted",
            )
        sols.append(
            CaseSolution(
                index=idx,
                case=case,
                solvable=res.success_all,
                best_values=dict(res.best_values),
                best_source=res.best_source,
                score=float(res.worst_score),
                trace=tr,
            )
        )
    return sols


def _norm_solution_dist(
    consts: Sequence[ConstSpec],
    vals_a: Mapping[str, Any],
    vals_b: Mapping[str, Any],
) -> float:
    sq_sum = 0.0
    for c in consts:
        if (
            c.kind == "continuous"
            and c.lo is not None
            and c.hi is not None
            and c.hi > c.lo
        ):
            va = float(vals_a.get(c.name, c.initial))
            vb = float(vals_b.get(c.name, c.initial))
            diff = (va - vb) / (float(c.hi) - float(c.lo))
            sq_sum += diff * diff
        elif c.kind == "discrete":
            va = vals_a.get(c.name, c.initial)
            vb = vals_b.get(c.name, c.initial)
            sq_sum += 0.0 if va == vb else 1.0
    return math.sqrt(sq_sum)


def cross_evaluate_solutions(
    evaluator: Evaluator,
    cases: Sequence[Any],
    case_solutions: Sequence[CaseSolution],
    budget_reserve: int = 0,
) -> dict[tuple[int, int], bool]:
    """Cross-evaluate each per-case solution on other cases without exceeding budget."""
    n = len(cases)
    cross_ok: dict[tuple[int, int], bool] = {}
    for i, sol in enumerate(case_solutions):
        if not sol.solvable:
            continue  # No successful source exists to cross-evaluate.
        other_indices = [j for j in range(n) if j != i]
        if not other_indices:
            continue
        if evaluator.budget_limit is not None:
            rem = evaluator.budget_limit - max(0, int(budget_reserve)) - evaluator.n_rollouts
            if rem < len(other_indices):
                continue  # Missing entries mean unmeasured, never failed.
        other_cases = [cases[j] for j in other_indices]
        traces = evaluator.run_many(sol.best_source, other_cases)
        for j, tr in zip(other_indices, traces):
            cross_ok[(i, j)] = bool(tr.success)
    return cross_ok


def unsat_core(
    retuner: Retuner,
    source: str,
    consts: Sequence[ConstSpec],
    cases: Sequence[Any],
    case_solutions: Sequence[CaseSolution] | None = None,
    cross_ok: Mapping[tuple[int, int], bool] | None = None,
    confirmed_unsat_pair: tuple[int, int] | None = None,
) -> list[int]:
    """Find a subset unresolved by budgeted retuning; this is not an unsat proof."""
    n = len(cases)
    if n <= 2:
        return list(range(n))
    if confirmed_unsat_pair is not None:
        return sorted(list(confirmed_unsat_pair))

    if case_solutions is not None and consts:
        c_ok = (
            cross_ok
            if cross_ok is not None
            else cross_evaluate_solutions(
                retuner.evaluator,
                cases,
                case_solutions,
                budget_reserve=retuner.budget_reserve,
            )
        )
        mut_pairs = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if c_ok.get((i, j)) is False and c_ok.get((j, i)) is False
        ]
        mut_pairs.sort(
            key=lambda ij: (
                -_norm_solution_dist(
                    consts,
                    case_solutions[ij[0]].best_values,
                    case_solutions[ij[1]].best_values,
                ),
                ij[0],
                ij[1],
            )
        )
        if mut_pairs:
            pair_retuner = Retuner(
                retuner.evaluator,
                max_iters=min(retuner.max_iters, 5),
                popsize=6,
                margin_target=1e-6,
                sigma0=retuner.sigma0,
                seed=retuner.seed,
                max_rollouts=70,
                budget_reserve=retuner.budget_reserve,
            )
            for i, j in mut_pairs[:2]:
                joint = pair_retuner.optimize(source, consts, [cases[i], cases[j]])
                if joint.n_evals > 0 and not joint.success_all:
                    return [i, j]

    del_retuner = Retuner(
        retuner.evaluator,
        max_iters=min(retuner.max_iters, 5),
        popsize=6,
        margin_target=retuner.margin_target,
        sigma0=retuner.sigma0,
        seed=retuner.seed,
        max_rollouts=90,
        budget_reserve=retuner.budget_reserve,
    )
    core = list(range(n))
    for idx in list(reversed(core)):
        if len(core) <= 2:
            break
        trial = [j for j in core if j != idx]
        res = del_retuner.optimize(source, consts, [cases[j] for j in trial])
        if res.n_evals == 0:
            break
        if not res.success_all:
            core = trial
    return core


def incompatible_pairs(
    evaluator: Evaluator,
    source: str,
    consts: Sequence[ConstSpec],
    cases: Sequence[Any],
    case_solutions: Sequence[CaseSolution],
    retuner: Retuner | None = None,
    cross_ok: Mapping[tuple[int, int], bool] | None = None,
    confirmed_pairs: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Select candidate pairs using cross-evaluations and budgeted retuning."""
    n = len(cases)
    if n < 2:
        return []

    reserve = retuner.budget_reserve if retuner is not None else 0
    c_ok = (
        dict(cross_ok)
        if cross_ok is not None
        else cross_evaluate_solutions(
            evaluator, cases, case_solutions, budget_reserve=reserve
        )
    )


    pair_retuner: Retuner | None = None
    if retuner is not None and consts:
        pair_retuner = Retuner(
            evaluator,
            max_iters=min(retuner.max_iters, 5),
            popsize=6,
            margin_target=1e-6,
            sigma0=retuner.sigma0,
            seed=retuner.seed,
            max_rollouts=50,
            budget_reserve=retuner.budget_reserve,
        )

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            # A search failure or a caller label is not cross-failure evidence.
            mutually_incompatible = (
                case_solutions[i].solvable and case_solutions[j].solvable
                and c_ok.get((i, j)) is False and c_ok.get((j, i)) is False
            )
            if mutually_incompatible:
                if pair_retuner is not None:
                    joint = pair_retuner.optimize(source, consts, [cases[i], cases[j]])
                    if joint.n_evals > 0 and joint.success_all:
                        continue
                pairs.append((i, j))

    return pairs


def distinguishing_observable(
    traces: Sequence[Trace] | Mapping[int, Trace],
    pairs: Sequence[tuple[int, int]],
    obs_scale: Mapping[str, float],
    obs_api_map: Mapping[str, str] | None = None,
    nominal_trace: Trace | None = None,
    top_k: int = 8,
) -> list[DistinguishingObservable]:
    """Rank SDK observables by normalized separation across `pairs` (2x weight for opposite signs)."""
    if not pairs:
        return []

    def _get_trace(idx: int) -> Trace:
        return traces[idx]

    best_by_key: dict[str, DistinguishingObservable] = {}

    # Inspect all keys in SNAPSHOT_KEYS (plus any extra present in calls).
    candidate_keys: list[str] = list(SNAPSHOT_KEYS)
    first_tr = _get_trace(pairs[0][0])
    if first_tr.calls:
        for k in first_tr.calls[0].obs:
            if k not in candidate_keys:
                candidate_keys.append(k)

    # Determine common aligned call indices across all pairs once before scanning keys.
    max_common = min(
        min(len(_get_trace(i).calls), len(_get_trace(j).calls)) for i, j in pairs
    )
    if max_common <= 0:
        return []

    valid_common = 0
    for c_idx in range(max_common):
        ref_call = _get_trace(pairs[0][0]).calls[c_idx]
        if any(
            _get_trace(i).calls[c_idx].name != ref_call.name
            or _get_trace(j).calls[c_idx].name != ref_call.name
            for i, j in pairs
        ):
            break
        valid_common = c_idx + 1
    if valid_common <= 0:
        return []

    eval_end = valid_common
    if valid_common >= 2:
        first_div_idx: int | None = None
        for c_idx in range(valid_common):
            for key in candidate_keys:
                scale = float(obs_scale.get(key, 1.0))
                if scale <= 0.0:
                    scale = 1.0
                if any(
                    abs(
                        float(_get_trace(i).calls[c_idx].obs.get(key, 0.0))
                        - float(_get_trace(j).calls[c_idx].obs.get(key, 0.0))
                    )
                    / scale
                    >= 3.0
                    for i, j in pairs
                ):
                    first_div_idx = c_idx
                    break
            if first_div_idx is not None:
                break
        if first_div_idx is not None:
            eval_end = first_div_idx + 1

    for key in candidate_keys:
        scale = float(obs_scale.get(key, 1.0))
        if scale <= 0.0:
            scale = 1.0

        best_score = -1.0
        best_obs: DistinguishingObservable | None = None

        for c_idx in range(eval_end):
            ref_call = _get_trace(pairs[0][0]).calls[c_idx]

            pair_scores: list[float] = []
            any_opp = False
            rep_va = 0.0
            rep_vb = 0.0
            for p_idx, (i, j) in enumerate(pairs):
                va = float(_get_trace(i).calls[c_idx].obs.get(key, 0.0))
                vb = float(_get_trace(j).calls[c_idx].obs.get(key, 0.0))
                norm_diff = abs(va - vb) / scale

                opp = False
                if norm_diff >= 1.0:
                    if va * vb < 0.0:
                        opp = True
                    elif nominal_trace is not None and c_idx < len(nominal_trace.calls):
                        v_nom = float(nominal_trace.calls[c_idx].obs.get(key, 0.0))
                        if (va - v_nom) * (vb - v_nom) < 0.0:
                            opp = True

                weight = norm_diff * (2.0 if opp else 1.0)
                pair_scores.append(weight)
                any_opp = any_opp or opp
                if p_idx == 0:
                    rep_va, rep_vb = va, vb

            avg_score = sum(pair_scores) / len(pair_scores)
            if avg_score > best_score:
                best_score = avg_score
                api_expr = (
                    obs_api_map.get(key, key) if obs_api_map is not None else key
                )
                best_obs = DistinguishingObservable(
                    key=key,
                    score=float(avg_score),
                    call_index=c_idx,
                    call_name=ref_call.name,
                    lineno=int(ref_call.lineno),
                    val_a=rep_va,
                    val_b=rep_vb,
                    opposite_sign=any_opp,
                    api_expr=api_expr,
                )

        if best_obs is not None and best_obs.score > 1e-6:
            best_by_key[key] = best_obs

    ranked = sorted(best_by_key.values(), key=lambda d: (-d.score, d.key))
    return ranked[: int(top_k)]
