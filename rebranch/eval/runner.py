"""End-to-end experiment runner for ReBranch, baselines, and §7 ablations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import threading
import time
from typing import Any, Sequence

from rebranch.api.primitives import PHYSCAP_EXTRA_PRIMITIVES
from rebranch.baselines import BaselineResult
from rebranch.baselines.agentic_genplan import run_agentic_genplan
from rebranch.baselines.oneshot_calib import run_oneshot_calib
from rebranch.baselines.oracle_twin import run_oracle_twin
from rebranch.baselines.physcap_agent import PhysCaPEpisodeWrapper, run_physcap
from rebranch.baselines.physcap_appendix import PhysCaPTaskBackend, run_physcap_appendix
from rebranch.baselines.single_twin import run_single_twin
from rebranch.baselines.strong import run_fixed_compliant, run_parameter_only
from rebranch.baselines.t10_controls import run_t10_control_development
from rebranch.baselines.twin_mpc import run_twin_mpc
from rebranch.baselines.twin_mpc_receding import run_twin_mpc_receding
from rebranch.baselines.uniform_dr import run_uniform_dr
from rebranch.core.loop import ReBranchConfig, ReBranchResult, run_rebranch
from rebranch.core.development import admission_cases
from rebranch.eval.metrics import EvalMetrics, compute_metrics
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.trace import Trace
from rebranch.llm.client import (
    CodexCLIClient,
    DiskCacheLLM,
    LLMClient,
    MockLLMClient,
    LimitedLLMClient,
    OpenAICompatibleClient,
)
from rebranch.sim.backend import Episode, TaskBackend
from rebranch.sim.toy_backend import ToyBranchTask, ToyRetuneTask


ABLATION_VARIANTS: tuple[str, ...] = (
    "full",
    "random_ce",
    "worst_ce",
    "no_retune_or_branch",
    "no_lifting",
    "no_branch_verify",
    "physcap_primitives",
)


class PhysCaPTaskBackendWrapper(TaskBackend):
    """Wraps any `TaskBackend` so `make_episode()` exposes `get_mass()` and `get_stiffness()`."""

    def __init__(self, inner: TaskBackend) -> None:
        self._inner = inner
        self.name = inner.name
        self.task_description = inner.task_description
        self.requires_branching = inner.requires_branching
        self.param_space = inner.param_space
        self.obs_scale = inner.obs_scale
        self.obs_api_map = inner.obs_api_map
        self.draft_policy = inner.draft_policy
        self.probe_config = getattr(inner, "probe_config", None)

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        return PhysCaPEpisodeWrapper(self._inner.make_episode(theta, seed=seed))

    def hidden_test_set(self, seed: int = 2026, n: int = 50) -> list[dict[str, Any]]:
        return self._inner.hidden_test_set(seed=seed, n=n)

    def ood_test_set(self, seed: int = 3026, n: int = 10) -> list[dict[str, Any]]:
        return self._inner.ood_test_set(seed=seed, n=n)

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        return self._inner.oracle_policy(theta)

    def conservative_policy(self) -> str:
        return self._inner.conservative_policy()

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return self._inner.latent_mode(theta)


def load_task_backend(task_name: str) -> TaskBackend:
    """Instantiate a `TaskBackend` by name (`toy_retune`, `toy_branch`, or `t1`..`t8`)."""
    if task_name == "toy_retune":
        return ToyRetuneTask()
    if task_name == "toy_branch":
        return ToyBranchTask()
    from rebranch.tasks import make_task

    return make_task(task_name)


def build_llm_client(
    provider: str = "codex",
    *,
    model: str = "gpt-6.1-sol",
    effort: str = "low",
    cache_dir: str | Path = ".cache/rebranch_llm",
    timeout_s: int = 180,
) -> LLMClient:
    """Construct a cached LLM client from config parameters."""
    if provider == "codex":
        inner: LLMClient = CodexCLIClient(
            model=model, effort=effort, timeout_s=timeout_s
        )
        return DiskCacheLLM(inner, cache_dir=cache_dir)
    if provider == "openai":
        inner = OpenAICompatibleClient(model=model, timeout_s=timeout_s)
        return DiskCacheLLM(inner, cache_dir=cache_dir)
    if provider == "mock":
        return MockLLMClient(
            ["```python\ndef run(api):\n    api.zero_ee_wrench()\n```"]
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def _make_rebranch_config(
    variant: str,
    *,
    seed: int,
    max_rounds: int = 3,
    n_ce_samples: int = 80,
    n_screen_base: int = 6,
    k_max: int = 4,
    n_bisect: int = 5,
    retune_iters: int = 12,
    admission_n_eval: int = 24,
) -> ReBranchConfig:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ReBranch variant: {variant!r}")
    cfg = ReBranchConfig(
        max_rounds=max_rounds,
        n_screen_base=n_screen_base,
        n_ce_samples=n_ce_samples,
        k_max=k_max,
        n_bisect=n_bisect,
        retune_iters=retune_iters,
        admission_n_eval=admission_n_eval,
        seed=seed,
    )
    if variant == "random_ce":
        cfg.random_ce = True
    elif variant == "worst_ce":
        cfg.worst_ce = True
    elif variant == "no_retune_or_branch":
        cfg.no_retune_or_branch = True
    elif variant == "no_lifting":
        cfg.no_lifting = True
    elif variant == "no_branch_verify":
        cfg.no_branch_verify = True
    elif variant == "physcap_primitives":
        cfg.physcap_primitives = True
    return cfg


def _rebranch_to_baseline_result(
    res: ReBranchResult, variant: str
) -> BaselineResult:
    round_logs: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = [
        {"round": -1, "accepted": True, "source": res.initial_source}
    ]
    for r in res.rounds:
        round_logs.append(
            {
                "round": r.round_idx,
                "action": r.action,
                "n_counterexamples": r.n_counterexamples,
                "n_regression": r.n_regression,
                "active_params": r.active_params,
                "admitted": r.admitted,
                "eval_rate_before": r.eval_rate_before,
                "eval_rate_after": r.eval_rate_after,
                "n_rollouts": r.n_rollouts_total,
                "dev_llm_calls": r.llm_calls_total,
                "details": r.details,
                "admission_feedback": r.admission_feedback,
            }
        )
        if r.admitted or r.action == "CONVERGED":
            snapshots.append(
                {"round": r.round_idx, "accepted": r.admitted, "source": r.source_after}
            )
    return BaselineResult(
        method="rebranch",
        policy_source=res.final_source,
        n_rollouts=res.n_rollouts,
        dev_llm_calls=res.llm_calls,
        dev_prompt_tokens=res.prompt_tokens,
        dev_completion_tokens=res.completion_tokens,
        round_logs=round_logs,
        program_snapshots=snapshots,
        wall_time_s=res.wall_s,
        extra={
            "variant": variant,
            "converged": res.converged,
            "stop_reason": res.stop_reason,
            "active_params": res.active_params,
        },
    )


def _eval_runtime_agent_episode(
    agent: Any,
    backend: TaskBackend,
    theta: dict[str, Any],
    seed: int,
) -> tuple[Trace, dict[str, Any]]:
    return agent.evaluate_episode(backend, theta, seed=seed)


def _evaluate_split(
    backend: TaskBackend,
    res: BaselineResult,
    test_cases: list[dict[str, Any]],
    *,
    seed: int,
    workers: int,
    extra_primitives: Sequence[str] = (),
) -> EvalMetrics:
    if res.agent is not None:
        n_cases = len(test_cases)
        traces: list[Trace | None] = [None] * n_cases
        policy_sources: list[str] = [res.policy_source] * n_cases
        sim_preds: list[bool | None] = [None] * n_cases
        rt_calls: list[int] = [0] * n_cases
        rt_tokens: list[int | None] = [0] * n_cases
        exec_times: list[float] = [0.0] * n_cases
        rt_sim = [0] * n_cases
        rt_sim_time = [0.0] * n_cases
        runtime_episode_records: list[dict[str, Any] | None] = [None] * n_cases

        def _eval_one(idx: int, th: dict[str, Any]) -> None:
            tr, meta = res.agent.evaluate_episode(  # type: ignore[union-attr]
                backend, th, seed=seed * 1000 + idx
            )
            traces[idx] = tr
            policy_sources[idx] = str(meta.get("policy_source", res.policy_source))
            sim_preds[idx] = meta.get("sim_predicted_success")
            rt_calls[idx] = int(meta.get("runtime_llm_calls", 0))
            rt_tokens[idx] = meta.get("runtime_tokens", 0)
            rt_sim[idx] = int(meta.get("runtime_sim_rollouts", 0))
            rt_sim_time[idx] = float(meta.get("runtime_sim_time_s", 0.0))
            exec_times[idx] = float(
                meta.get("wall_time_s", tr.sim_time + tr.wall_time)
            )
            runtime_episode_records[idx] = {"trace": asdict(tr), "metadata": meta}

        thr_cap = 6 if res.method in {"physcap", "physcap_appendix"} else 16
        max_thr = min(max(1, int(workers)), max(1, n_cases), thr_cap)
        if max_thr > 1 and n_cases > 1:
            if res.method in {"physcap", "physcap_appendix"}:
                with ThreadPoolExecutor(max_workers=max_thr) as ex:
                    futs = [ex.submit(_eval_one, idx, th) for idx, th in enumerate(test_cases)]
                    for f in futs:
                        f.result()
            else:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=max_thr, mp_context=ctx) as ex:
                    futs = [
                        (idx, ex.submit(_eval_runtime_agent_episode, res.agent, backend, th, seed * 1000 + idx))
                        for idx, th in enumerate(test_cases)
                    ]
                    for idx, f in futs:
                        tr, meta = f.result()
                        traces[idx] = tr
                        policy_sources[idx] = str(meta.get("policy_source", res.policy_source))
                        sim_preds[idx] = meta.get("sim_predicted_success")
                        rt_calls[idx] = int(meta.get("runtime_llm_calls", 0))
                        rt_tokens[idx] = meta.get("runtime_tokens", 0)
                        rt_sim[idx] = int(meta.get("runtime_sim_rollouts", 0))
                        rt_sim_time[idx] = float(meta.get("runtime_sim_time_s", 0.0))
                        exec_times[idx] = float(
                            meta.get("wall_time_s", tr.sim_time + tr.wall_time)
                        )
                        runtime_episode_records[idx] = {"trace": asdict(tr), "metadata": meta}
        else:
            for idx, th in enumerate(test_cases):
                _eval_one(idx, th)

        final_traces = [t for t in traces if t is not None]
        res.extra.setdefault("runtime_episode_records_by_seed", {})[str(seed)] = runtime_episode_records
        return compute_metrics(
            backend,
            final_traces,
            policy_sources=policy_sources,
            sim_predicted_success=sim_preds,
            dev_llm_calls=res.dev_llm_calls,
            dev_tokens=res.dev_prompt_tokens + res.dev_completion_tokens,
            runtime_llm_calls_per_ep=rt_calls,
            runtime_tokens_per_ep=rt_tokens,
            exec_times_s=exec_times,
            n_rollouts_dev=res.n_rollouts,
            runtime_sim_rollouts_per_ep=rt_sim,
            runtime_sim_times_per_ep=rt_sim_time,
        )

    # Static policy evaluation via multiprocessed Evaluator
    nom_theta = backend.param_space.nominal()
    with Evaluator(
        backend, workers=1, extra_primitives=extra_primitives
    ) as nom_ev:
        nom_ok = bool(nom_ev.run(res.policy_source, nom_theta, seed=seed).success)

    cases_with_seeds = [(th, seed * 1000 + idx) for idx, th in enumerate(test_cases)]
    with Evaluator(
        backend, workers=workers, extra_primitives=extra_primitives
    ) as ev:
        traces = ev.run_many(res.policy_source, cases_with_seeds)

    return compute_metrics(
        backend,
        traces,
        policy_sources=[res.policy_source] * len(traces),
        sim_predicted_success=[nom_ok] * len(traces),
        dev_llm_calls=res.dev_llm_calls,
        dev_tokens=res.dev_prompt_tokens + res.dev_completion_tokens,
        runtime_llm_calls_per_ep=[0] * len(traces),
        runtime_tokens_per_ep=[0] * len(traces),
        n_rollouts_dev=res.n_rollouts,
    )


def _save_run_artifacts(
    output_dir: Path,
    task_name: str,
    method: str,
    variant: str,
    seed: int,
    res: BaselineResult,
    split_metrics: dict[str, EvalMetrics],
    budget_info: dict[str, Any],
) -> list[dict[str, Any]]:
    rounds_dir = output_dir / "rounds"
    snaps_dir = output_dir / "snapshots"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{task_name}__{method}__{variant}__seed{seed}"
    round_path = rounds_dir / f"{stem}.jsonl"
    round_path.write_text(
        "\n".join(json.dumps(r) for r in res.round_logs) + ("\n" if res.round_logs else ""),
        encoding="utf-8",
    )

    for snap in res.program_snapshots:
        r_idx = snap.get("round", 0)
        p = snaps_dir / f"{stem}_round{r_idx}.py"
        p.write_text(str(snap.get("source", "")), encoding="utf-8")

    final_snap = snaps_dir / f"{stem}_final.py"
    final_snap.write_text(res.policy_source, encoding="utf-8")

    records: list[dict[str, Any]] = []
    runs_jsonl = output_dir / "runs.jsonl"
    runs_csv = output_dir / "runs.csv"

    for split_name, m in split_metrics.items():
        rec = {
            "task": task_name,
            "method": method,
            "variant": variant,
            "split": split_name,
            "seed": seed,
            "n_rollouts_dev": res.n_rollouts,
            "budget": budget_info,
            "metrics": m.to_dict(),
            "method_details": res.extra,
            "final_policy_path": str(final_snap),
            "rounds_path": str(round_path),
            "wall_time_s": round(res.wall_time_s, 3),
        }
        records.append(rec)
        with runs_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        flat_row: dict[str, Any] = {
            "task": task_name,
            "method": method,
            "variant": variant,
            "split": split_name,
            "seed": seed,
            "wall_time_s": round(res.wall_time_s, 3),
            **m.to_dict(),
        }
        write_header = not runs_csv.exists()
        with runs_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat_row)

    return records


def _completed_keys(output_dir: Path) -> set[tuple[str, str, str, int]]:
    runs_jsonl = output_dir / "runs.jsonl"
    if not runs_jsonl.exists():
        return set()
    done: set[tuple[str, str, str, int]] = set()
    for line in runs_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        done.add((str(r["task"]), str(r["method"]), str(r.get("variant", "full")), int(r["seed"])))
    return done


def run_single_method(
    task_name: str,
    method: str,
    llm: LLMClient,
    *,
    variant: str = "full",
    seed: int = 0,
    workers: int = 1,
    n_hidden: int = 50,
    n_ood: int = 10,
    max_rollouts: int = 2000,
    max_llm_calls: int = 8,
    max_rounds: int = 3,
    n_ce_samples: int = 80,
    output_dir: str | Path = "outputs/single",
    oracle_filter: bool = False,
    runtime_llm_budget: int = 100,
    runtime_sim_budget_per_episode: int = 24,
    twin_mpc_prediction_horizon: int = 3,
    physcap_max_turns: int = 10,
    validation_cases: int = 24,
    training_cases_per_restart: int = 24,
    optimizer_restart_rollouts: int = 2000,
) -> dict[str, Any]:
    """Run development + hidden/OOD evaluation for one `(task, method, variant, seed)`."""
    backend = load_task_backend(task_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_prims: tuple[str, ...] = ()
    eval_backend: TaskBackend = backend
    measurement_track = getattr(backend, "measurement_track", None)

    if measurement_track == "A":
        extra_prims = ("get_mass",)

    if measurement_track is not None:
        if method == "physcap":
            raise ValueError("legacy physcap proxy is not an evaluation method for T10")
        if measurement_track == "B" and method == "physcap_appendix":
            raise ValueError("PhysCaP appendix measurement is unavailable on T10 Track B")
        if method == "rebranch" and variant == "physcap_primitives":
            raise ValueError("physcap_primitives ablation is not defined for T10")
    if method.startswith("t10_") and measurement_track is None:
        raise ValueError(f"{method} is defined only for T10")
    if method == "t10_mass_balance" and measurement_track != "A":
        raise ValueError("t10_mass_balance requires T10 Track A")

    if method == "rebranch":
        cfg = _make_rebranch_config(
            variant, seed=seed, max_rounds=max_rounds, n_ce_samples=n_ce_samples
        )
        cfg.max_llm_calls = max_llm_calls
        if cfg.physcap_primitives:
            eval_backend = PhysCaPTaskBackendWrapper(backend)
            extra_prims = PHYSCAP_EXTRA_PRIMITIVES
        with Evaluator(
            eval_backend,
            workers=workers,
            budget_limit=max_rollouts,
            extra_primitives=extra_prims,
        ) as ev:
            rb_res = run_rebranch(
                eval_backend,
                ev,
                llm,
                config=cfg,
                oracle_filter=oracle_filter,
            )
        res = _rebranch_to_baseline_result(rb_res, variant)
    elif method == "agentic_genplan":
        res = run_agentic_genplan(
            backend, llm, seed=seed, max_rollouts=max_rollouts,
            max_llm_calls=max_llm_calls,
        )
    elif method == "single_twin":
        res = run_single_twin(
            backend, llm, seed=seed, max_rollouts=min(20, max_rollouts), max_llm_calls=min(5, max_llm_calls)
        )
    elif method == "uniform_dr":
        res = run_uniform_dr(
            backend,
            llm,
            seed=seed,
            workers=workers,
            max_rollouts=max_rollouts,
            max_llm_calls=max_llm_calls,
        )
    elif method == "physcap":
        extra_prims = PHYSCAP_EXTRA_PRIMITIVES
        res = run_physcap(backend, llm, max_dev_llm_calls=min(1, max_llm_calls), runtime_llm_budget=runtime_llm_budget)
    elif method == "physcap_appendix":
        eval_backend = PhysCaPTaskBackend(backend)
        extra_prims = ("get_mass",)
        res = run_physcap_appendix(
            eval_backend, LimitedLLMClient(llm, runtime_llm_budget), max_turns=physcap_max_turns
        )
    elif method == "twin_mpc":
        res = run_twin_mpc(
            backend,
            llm,
            seed=seed,
            workers=workers,
            max_rollouts=max_rollouts,
            max_llm_calls=max_llm_calls,
            runtime_sim_budget_per_episode=runtime_sim_budget_per_episode,
        )
    elif method == "twin_mpc_receding":
        res = run_twin_mpc_receding(
            backend, llm, seed=seed, workers=workers,
            max_rollouts=max_rollouts, max_llm_calls=max_llm_calls,
            runtime_sim_budget_per_episode=runtime_sim_budget_per_episode,
            prediction_horizon=twin_mpc_prediction_horizon,
        )
    elif method == "oneshot_calib":
        res = run_oneshot_calib(
            backend,
            llm,
            seed=seed,
            workers=workers,
            max_rollouts=max_rollouts,
            max_llm_calls=max_llm_calls,
        )
    elif method == "oracle_twin":
        res = run_oracle_twin(backend)
    elif method == "fixed_compliant":
        res = run_fixed_compliant(backend)
    elif method == "parameter_only":
        res = run_parameter_only(
            backend,
            seed=seed,
            workers=workers,
            max_rollouts=max_rollouts,
            validation_n=validation_cases,
            training_cases_per_restart=training_cases_per_restart,
            optimizer_restart_rollouts=optimizer_restart_rollouts,
        )
    elif method == "compliant_parameter_only":
        res = run_parameter_only(
            backend,
            seed=seed,
            workers=workers,
            max_rollouts=max_rollouts,
            validation_n=validation_cases,
            training_cases_per_restart=training_cases_per_restart,
            optimizer_restart_rollouts=optimizer_restart_rollouts,
            compliant=True,
        )
    elif method == "t10_parameter_only":
        res = run_parameter_only(
            backend,
            seed=seed,
            workers=workers,
            max_rollouts=max_rollouts,
            validation_n=validation_cases,
            training_cases_per_restart=training_cases_per_restart,
            optimizer_restart_rollouts=optimizer_restart_rollouts,
        )
    elif method in {"t10_fixed_blind", "t10_mass_balance", "t10_tilt_feedback"}:
        dev_cases = admission_cases(backend, seed=seed, n=validation_cases)
        res = run_t10_control_development(
            backend,
            method,
            dev_cases,
            seed=seed,
            workers=workers,
            max_rollouts=max_rollouts,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    hidden_cases = eval_backend.hidden_test_set(seed=2026, n=n_hidden)
    ood_cases = eval_backend.ood_test_set(seed=3026, n=n_ood)

    split_metrics: dict[str, EvalMetrics] = {}
    if hidden_cases:
        split_metrics["hidden"] = _evaluate_split(
            eval_backend,
            res,
            hidden_cases,
            seed=seed,
            workers=workers,
            extra_primitives=extra_prims,
        )
    if ood_cases:
        split_metrics["ood"] = _evaluate_split(
            eval_backend,
            res,
            ood_cases,
            seed=seed + 500,
            workers=workers,
            extra_primitives=extra_prims,
        )

    development_protocol = (
        "strong-baselines-v1"
        if method in {"fixed_compliant", "parameter_only", "compliant_parameter_only"}
        else "t10-strong-controls-v1" if method.startswith("t10_")
        else "agentic-genplan-platform-v1" if method == "agentic_genplan"
        else "twin-mpc-receding-v1" if method == "twin_mpc_receding"
        else "repair-development-v3"
    )
    res.extra["development_protocol"] = development_protocol
    test_rollouts = {
        "hidden": len(hidden_cases),
        "ood": len(ood_cases),
        "total": len(hidden_cases) + len(ood_cases),
    }
    res.extra["actual_test_rollouts_by_split"] = test_rollouts
    budget_info = {"max_rollouts": max_rollouts, "max_llm_calls": max_llm_calls,
                   "dev_rollout_budget": max_rollouts, "dev_llm_budget": max_llm_calls,
                   "actual_dev_rollouts": res.n_rollouts,
                   "actual_test_rollouts_by_split": test_rollouts,
                   "baseline_dev_rollout_budget": res.extra.get("dev_rollout_budget", max_rollouts),
                   "runtime_llm_budget": runtime_llm_budget if method == "physcap" else 0,
                   "runtime_llm_call_budget": runtime_llm_budget if method in {"physcap", "physcap_appendix"} else 0,
                   "runtime_sim_budget_per_episode": runtime_sim_budget_per_episode if method in {"twin_mpc", "twin_mpc_receding"} else 0,
                   "twin_mpc_prediction_horizon": twin_mpc_prediction_horizon if method == "twin_mpc_receding" else None,
                   "development_protocol": development_protocol,
                   "simulation_protocol": "rebranch-mujoco-v2" if type(backend).__module__.startswith("rebranch.tasks.") else "unversioned"}
    records = _save_run_artifacts(
        out_dir, task_name, method, variant, seed, res, split_metrics, budget_info
    )
    return {
        "task": task_name,
        "method": method,
        "variant": variant,
        "seed": seed,
        "result": res,
        "metrics": {k: v.to_dict() for k, v in split_metrics.items()},
        "records": records,
    }


def run_benchmark_suite(
    tasks: Sequence[str],
    methods: Sequence[str],
    llm: LLMClient,
    *,
    ablations: Sequence[str] = (),
    ablation_tasks: Sequence[str] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    workers: int = 16,
    n_hidden: int = 50,
    n_ood: int = 10,
    max_rollouts: int = 2000,
    max_llm_calls: int = 8,
    max_rounds: int = 3,
    n_ce_samples: int = 80,
    output_dir: str | Path = "outputs/suite",
    match_rebranch_budget: bool = True,  # Compatibility flag; caps are always predeclared.
    runtime_llm_budget: int = 100,
    runtime_sim_budget_per_episode: int = 24,
    twin_mpc_prediction_horizon: int = 3,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run all `(task, seed, method/ablation)` jobs with matched rollout/LLM budgets."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(out_dir) if resume else set()
    outputs: list[dict[str, Any]] = []

    for task_name in tasks:
        for seed in seeds:
            # All methods retain the same predeclared caps regardless of early stopping.
            if "rebranch" in methods:
                if (task_name, "rebranch", "full", seed) not in done:
                    out = run_single_method(
                        task_name,
                        "rebranch",
                        llm,
                        variant="full",
                        seed=seed,
                        workers=workers,
                        n_hidden=n_hidden,
                        n_ood=n_ood,
                        max_rollouts=max_rollouts,
                        max_llm_calls=max_llm_calls,
                        max_rounds=max_rounds,
                        n_ce_samples=n_ce_samples,
                        output_dir=out_dir,
                        runtime_llm_budget=runtime_llm_budget,
                        runtime_sim_budget_per_episode=runtime_sim_budget_per_episode,
                    )
                    outputs.append(out)

            for method in methods:
                if method == "rebranch":
                    continue
                if (task_name, method, "full", seed) in done:
                    continue
                m_rollouts = max_rollouts
                m_llm = max_llm_calls
                out = run_single_method(
                    task_name,
                    method,
                    llm,
                    variant="full",
                    seed=seed,
                    workers=workers,
                    n_hidden=n_hidden,
                    n_ood=n_ood,
                    max_rollouts=m_rollouts,
                    max_llm_calls=m_llm,
                    max_rounds=max_rounds,
                    n_ce_samples=n_ce_samples,
                    output_dir=out_dir,
                    runtime_llm_budget=runtime_llm_budget,
                    runtime_sim_budget_per_episode=runtime_sim_budget_per_episode,
                    twin_mpc_prediction_horizon=twin_mpc_prediction_horizon,
                )
                outputs.append(out)

            abl_task_set = set(ablation_tasks) if ablation_tasks is not None else set(tasks)
            for abl in ablations if task_name in abl_task_set else ():
                if (task_name, "rebranch", abl, seed) in done:
                    continue
                out = run_single_method(
                    task_name,
                    "rebranch",
                    llm,
                    variant=abl,
                    seed=seed,
                    workers=workers,
                    n_hidden=n_hidden,
                    n_ood=n_ood,
                    max_rollouts=max_rollouts,
                    max_llm_calls=max_llm_calls,
                    max_rounds=max_rounds,
                    n_ce_samples=n_ce_samples,
                    output_dir=out_dir,
                    runtime_llm_budget=runtime_llm_budget,
                    runtime_sim_budget_per_episode=runtime_sim_budget_per_episode,
                )
                outputs.append(out)

    return outputs
