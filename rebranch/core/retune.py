"""Joint constant retuning using CMA-ES (ND), 1D grid+CEM, and discrete enumeration."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from rebranch.core.lifting import ConstSpec, instantiate
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.trace import Trace


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cma  # type: ignore[import-untyped]


@dataclass
class RetuneResult:
    """Outcome of constant retuning across a set of target cases."""

    success_all: bool
    best_values: dict[str, Any]
    best_source: str
    best_objective: float
    n_success: int
    n_cases: int
    worst_score: float
    mean_score: float
    traces: list[Trace]
    n_evals: int


def score_traces(traces: Sequence[Trace]) -> tuple[bool, int, float, float, float]:
    """Compute `(success_all, n_success, worst_score, mean_score, objective)`."""
    if not traces:
        return True, 0, 1.0, 1.0, 1.0
    n_success = sum(1 for t in traces if t.success)
    scores = [float(t.score) for t in traces]
    worst_score = min(scores)
    mean_score = sum(scores) / len(scores)
    obj = float(n_success) + 0.5 * worst_score + 0.1 * mean_score
    return (n_success == len(traces)), n_success, worst_score, mean_score, obj


class Retuner:
    """Constant optimizer over lifted continuous and discrete policy constants."""

    def __init__(
        self,
        evaluator: Evaluator,
        max_iters: int = 25,
        popsize: int | None = None,
        margin_target: float = 0.25,
        sigma0: float = 0.25,
        seed: int = 0,
        max_rollouts: int | None = None,
        budget_reserve: int = 0,
    ) -> None:
        self.evaluator = evaluator
        self.max_iters = int(max_iters)
        self.popsize = popsize
        self.margin_target = float(margin_target)
        self.sigma0 = float(sigma0)
        self.seed = int(seed)
        self.max_rollouts = None if max_rollouts is None else int(max_rollouts)
        self.budget_reserve = max(0, int(budget_reserve))
        self._opt_start_rollouts: int = 0
        self.generations_completed: int = 0

    def _can_afford(self, n_cases: int, n_candidates: int = 1) -> int:
        if n_cases <= 0:
            return n_candidates
        cap = n_candidates
        if self.evaluator.budget_limit is not None:
            rem_global = (
                self.evaluator.budget_limit
                - self.budget_reserve
                - self.evaluator.n_rollouts
            )
            cap = min(cap, max(0, rem_global // n_cases))
        if self.max_rollouts is not None:
            used = self.evaluator.n_rollouts - self._opt_start_rollouts
            rem_local = self.max_rollouts - used
            cap = min(cap, max(0, rem_local // n_cases))
        return cap

    def _eval_candidate(
        self,
        source: str,
        values: dict[str, Any],
        thetas: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
    ) -> tuple[str, list[Trace], bool, int, float, float, float]:
        cand_src = instantiate(source, values)
        traces = self.evaluator.run_many(cand_src, thetas)
        ok_all, n_succ, worst, mean, obj = score_traces(traces)
        return cand_src, traces, ok_all, n_succ, worst, mean, obj

    @staticmethod
    def _to_values(
        cont_consts: Sequence[ConstSpec],
        u_vec: Sequence[float],
        disc_dict: Mapping[str, Any],
        fixed_cont: Mapping[str, Any],
    ) -> dict[str, Any]:
        vals: dict[str, Any] = dict(fixed_cont)
        vals.update(disc_dict)
        for spec, u in zip(cont_consts, u_vec):
            lo = float(spec.lo)  # type: ignore[arg-type]
            hi = float(spec.hi)  # type: ignore[arg-type]
            uc = min(max(float(u), 0.0), 1.0)
            value = lo + uc * (hi - lo)
            if spec.is_int:
                vals[spec.name] = min(max(int(round(value)), math.ceil(lo)), math.floor(hi))
            else:
                vals[spec.name] = value
        return vals

    def _optimize_1d(
        self,
        source: str,
        cont: ConstSpec,
        disc_dict: Mapping[str, Any],
        fixed_cont: Mapping[str, Any],
        thetas: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
        rng: np.random.Generator,
        best: RetuneResult,
        init_u: float | None,
        init_obj: float | None,
    ) -> RetuneResult:
        lo = float(cont.lo)  # type: ignore[arg-type]
        hi = float(cont.hi)  # type: ignore[arg-type]
        span = max(hi - lo, 1e-12)
        pop = self.popsize if self.popsize is not None else 8
        history: list[tuple[float, float]] = []
        if init_u is not None and init_obj is not None:
            history.append((init_u, init_obj))

        # Initial 1D grid across [0, 1]
        grid = [k / 8.0 for k in range(9)]
        if init_u is not None:
            grid = [u for u in grid if abs(u - init_u) > 1e-6]

        n_afford = self._can_afford(len(thetas), len(grid))
        for u in grid[:n_afford]:
            vals = self._to_values([cont], [u], disc_dict, fixed_cont)
            cand_src, traces, ok_all, n_succ, worst, mean, obj = self._eval_candidate(
                source, vals, thetas
            )
            best.n_evals += 1
            history.append((u, obj))
            if obj > best.best_objective:
                best = RetuneResult(
                    success_all=ok_all,
                    best_values=vals,
                    best_source=cand_src,
                    best_objective=obj,
                    n_success=n_succ,
                    n_cases=len(traces),
                    worst_score=worst,
                    mean_score=mean,
                    traces=traces,
                    n_evals=best.n_evals,
                )
            if best.success_all and best.worst_score >= self.margin_target:
                return best

        if not history:
            return best

        # 1D CEM refinement
        n_iters = max(1, min(self.max_iters, 8))
        for _ in range(n_iters):
            n_afford = self._can_afford(len(thetas), pop)
            if n_afford <= 0:
                break
            elites = sorted(history, key=lambda kv: -kv[1])[:3]
            elite_u = np.array([e[0] for e in elites], dtype=float)
            mu = float(elite_u[0])
            std = float(max(np.std(elite_u), 0.03))
            samples = np.clip(rng.normal(mu, std, size=n_afford), 0.0, 1.0)
            # Also probe midpoint toward second-best if distinct
            if len(elites) >= 2 and n_afford >= 2 and abs(elites[0][0] - elites[1][0]) > 1e-3:
                samples[0] = 0.5 * (elites[0][0] + elites[1][0])

            for u_raw in samples:
                u = float(u_raw)
                if any(abs(u - h_u) * span < 1e-6 * span for h_u, _ in history):
                    continue
                vals = self._to_values([cont], [u], disc_dict, fixed_cont)
                cand_src, traces, ok_all, n_succ, worst, mean, obj = self._eval_candidate(
                    source, vals, thetas
                )
                best.n_evals += 1
                history.append((u, obj))
                if obj > best.best_objective:
                    best = RetuneResult(
                        success_all=ok_all,
                        best_values=vals,
                        best_source=cand_src,
                        best_objective=obj,
                        n_success=n_succ,
                        n_cases=len(traces),
                        worst_score=worst,
                        mean_score=mean,
                        traces=traces,
                        n_evals=best.n_evals,
                    )
                if best.success_all and best.worst_score >= self.margin_target:
                    return best
            self.generations_completed += 1
        return best

    def _optimize_cem_nd(
        self,
        source: str,
        cont_consts: Sequence[ConstSpec],
        disc_dict: Mapping[str, Any],
        fixed_cont: Mapping[str, Any],
        thetas: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
        rng: np.random.Generator,
        best: RetuneResult,
        u0: list[float],
    ) -> RetuneResult:
        dim = len(cont_consts)
        pop = self.popsize if self.popsize is not None else max(8, 4 + 2 * dim)
        mu = np.array(u0, dtype=float)
        std = np.full(dim, self.sigma0, dtype=float)
        n_elite = max(2, pop // 3)

        for _ in range(self.max_iters):
            n_afford = self._can_afford(len(thetas), pop)
            if n_afford <= 0:
                break
            pop_u = np.clip(
                rng.normal(mu, std, size=(n_afford, dim)), 0.0, 1.0
            )
            batch: list[tuple[np.ndarray, float]] = []
            for u_vec in pop_u:
                vals = self._to_values(cont_consts, u_vec.tolist(), disc_dict, fixed_cont)
                cand_src, traces, ok_all, n_succ, worst, mean, obj = self._eval_candidate(
                    source, vals, thetas
                )
                best.n_evals += 1
                batch.append((u_vec, obj))
                if obj > best.best_objective:
                    best = RetuneResult(
                        success_all=ok_all,
                        best_values=vals,
                        best_source=cand_src,
                        best_objective=obj,
                        n_success=n_succ,
                        n_cases=len(traces),
                        worst_score=worst,
                        mean_score=mean,
                        traces=traces,
                        n_evals=best.n_evals,
                    )
                if best.success_all and best.worst_score >= self.margin_target:
                    return best
            if len(batch) >= n_elite:
                batch.sort(key=lambda kv: -kv[1])
                elite_arr = np.stack([b[0] for b in batch[:n_elite]], axis=0)
                mu = np.mean(elite_arr, axis=0)
                std = np.maximum(np.std(elite_arr, axis=0), 0.02)
            self.generations_completed += 1
        return best

    def _optimize_cma_nd(
        self,
        source: str,
        cont_consts: Sequence[ConstSpec],
        disc_dict: Mapping[str, Any],
        fixed_cont: Mapping[str, Any],
        thetas: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
        rng: np.random.Generator,
        best: RetuneResult,
        u0: list[float],
        cma_seed: int,
    ) -> RetuneResult:
        dim = len(cont_consts)
        assert dim >= 2
        pop = (
            self.popsize
            if self.popsize is not None
            else min(10, max(6, 4 + int(3 * math.log(dim))))
        )
        max_iters = min(self.max_iters, 8) if dim > 8 else self.max_iters
        u0_safe = [min(max(float(u), 1e-4), 1.0 - 1e-4) for u in u0]
        try:
            es = cma.CMAEvolutionStrategy(
                u0_safe,
                self.sigma0,
                {
                    "bounds": [0.0, 1.0],
                    "seed": max(1, int(cma_seed)),
                    "verbose": -9,
                    "popsize": pop,
                },
            )
        except Exception:
            return self._optimize_cem_nd(
                source, cont_consts, disc_dict, fixed_cont, thetas, rng, best, u0
            )

        for _ in range(max_iters):
            if es.stop():
                break
            sols = es.ask()
            n_afford = self._can_afford(len(thetas), len(sols))
            if n_afford < len(sols):
                # Cannot afford a full CMA-ES generation; evaluate up to n_afford and stop.
                for u_vec in sols[:n_afford]:
                    vals = self._to_values(cont_consts, u_vec, disc_dict, fixed_cont)
                    cand_src, traces, ok_all, n_succ, worst, mean, obj = (
                        self._eval_candidate(source, vals, thetas)
                    )
                    best.n_evals += 1
                    if obj > best.best_objective:
                        best = RetuneResult(
                            success_all=ok_all,
                            best_values=vals,
                            best_source=cand_src,
                            best_objective=obj,
                            n_success=n_succ,
                            n_cases=len(traces),
                            worst_score=worst,
                            mean_score=mean,
                            traces=traces,
                            n_evals=best.n_evals,
                        )
                    if best.success_all and best.worst_score >= self.margin_target:
                        return best
                break

            losses: list[float] = []
            for u_vec in sols:
                vals = self._to_values(cont_consts, u_vec, disc_dict, fixed_cont)
                cand_src, traces, ok_all, n_succ, worst, mean, obj = self._eval_candidate(
                    source, vals, thetas
                )
                best.n_evals += 1
                losses.append(-obj)
                if obj > best.best_objective:
                    best = RetuneResult(
                        success_all=ok_all,
                        best_values=vals,
                        best_source=cand_src,
                        best_objective=obj,
                        n_success=n_succ,
                        n_cases=len(traces),
                        worst_score=worst,
                        mean_score=mean,
                        traces=traces,
                        n_evals=best.n_evals,
                    )
                if best.success_all and best.worst_score >= self.margin_target:
                    return best
            es.tell(sols, losses)
            self.generations_completed += 1

        return best

    def optimize(
        self,
        source: str,
        consts: Sequence[ConstSpec],
        thetas: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
    ) -> RetuneResult:
        """Optimize `consts` in `source` over `thetas` to maximize success + margin."""
        self._opt_start_rollouts = self.evaluator.n_rollouts
        self.generations_completed = 0
        init_values = {c.name: c.initial for c in consts}
        if not thetas:
            return RetuneResult(
                success_all=True,
                best_values=init_values,
                best_source=source,
                best_objective=1.0,
                n_success=0,
                n_cases=0,
                worst_score=1.0,
                mean_score=1.0,
                traces=[],
                n_evals=0,
            )

        if self._can_afford(len(thetas), 1) <= 0:
            return RetuneResult(
                success_all=False,
                best_values=init_values,
                best_source=source,
                best_objective=-float("inf"),
                n_success=0,
                n_cases=len(thetas),
                worst_score=-1.0,
                mean_score=-1.0,
                traces=[],
                n_evals=0,
            )

        # Always evaluate initial constants first.
        cand_src, traces, ok_all, n_succ, worst, mean, obj = self._eval_candidate(
            source, init_values, thetas
        )
        best = RetuneResult(
            success_all=ok_all,
            best_values=init_values,
            best_source=cand_src,
            best_objective=obj,
            n_success=n_succ,
            n_cases=len(traces),
            worst_score=worst,
            mean_score=mean,
            traces=traces,
            n_evals=1,
        )
        if not consts or (best.success_all and best.worst_score >= self.margin_target):
            return best

        cont_consts = [
            c
            for c in consts
            if c.kind == "continuous"
            and c.lo is not None
            and c.hi is not None
            and c.hi > c.lo
        ]
        fixed_cont = {
            c.name: c.initial
            for c in consts
            if c.kind == "continuous" and c not in cont_consts
        }
        disc_consts = [c for c in consts if c.kind == "discrete" and c.choices]

        if disc_consts:
            raw_combos = list(itertools.product(*(c.choices for c in disc_consts)))
            init_combo = tuple(c.initial for c in disc_consts)
            if init_combo in raw_combos:
                raw_combos.remove(init_combo)
                disc_combos = [init_combo] + raw_combos
            else:
                disc_combos = raw_combos
        else:
            disc_combos = [()]

        rng = np.random.default_rng(self.seed)
        u0 = [
            min(max((float(c.initial) - float(c.lo)) / (float(c.hi) - float(c.lo)), 0.0), 1.0)  # type: ignore[arg-type]
            for c in cont_consts
        ]

        for combo_idx, combo in enumerate(disc_combos):
            disc_dict = {c.name: val for c, val in zip(disc_consts, combo)}
            is_init_combo = combo_idx == 0 and (
                not disc_consts or combo == tuple(c.initial for c in disc_consts)
            )
            init_obj_for_combo: float | None = best.best_objective if is_init_combo else None

            if not cont_consts:
                if not is_init_combo:
                    if self._can_afford(len(thetas), 1) <= 0:
                        break
                    vals = dict(fixed_cont)
                    vals.update(disc_dict)
                    cand_src, traces, ok_all, n_succ, worst, mean, obj = (
                        self._eval_candidate(source, vals, thetas)
                    )
                    best.n_evals += 1
                    if obj > best.best_objective:
                        best = RetuneResult(
                            success_all=ok_all,
                            best_values=vals,
                            best_source=cand_src,
                            best_objective=obj,
                            n_success=n_succ,
                            n_cases=len(traces),
                            worst_score=worst,
                            mean_score=mean,
                            traces=traces,
                            n_evals=best.n_evals,
                        )
            elif len(cont_consts) == 1:
                best = self._optimize_1d(
                    source,
                    cont_consts[0],
                    disc_dict,
                    fixed_cont,
                    thetas,
                    rng,
                    best,
                    init_u=u0[0] if is_init_combo else None,
                    init_obj=init_obj_for_combo,
                )
            else:
                best = self._optimize_cma_nd(
                    source,
                    cont_consts,
                    disc_dict,
                    fixed_cont,
                    thetas,
                    rng,
                    best,
                    u0,
                    cma_seed=self.seed + 100 * combo_idx + 1,
                )

            if best.success_all and best.worst_score >= self.margin_target:
                return best

        return best
