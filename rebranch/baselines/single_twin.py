"""Single-Twin + CaP baseline (nominal twin debugging)."""

from __future__ import annotations

import time
from typing import Any

from rebranch.baselines import BaselineResult
from rebranch.eval.metrics import summarize_trace_for_llm
from rebranch.exec.evaluator import BudgetExceededError, Evaluator
from rebranch.exec.sandbox import StaticCheckError, check_policy_source
from rebranch.llm.client import LLMClient, extract_python_code
from rebranch.llm.prompts import SYSTEM_PROMPT, build_single_twin_prompt
from rebranch.sim.backend import TaskBackend


def run_single_twin(
    backend: TaskBackend,
    llm: LLMClient,
    *,
    seed: int = 0,
    max_rollouts: int = 20,
    max_llm_calls: int = 5,
    n_nominal_seeds: int = 1,
) -> BaselineResult:
    """Debug `backend.draft_policy` strictly on the nominal twin `param_space.nominal()`.

    Produces a single static policy validated on the nominal twin without seeing any
    parameter variations (`theta`), matching the Single-Twin + CaP baseline in §7.
    """
    t0 = time.perf_counter()
    theta_nom = backend.param_space.nominal()
    current_src = backend.draft_policy
    best_src = current_src

    dev_llm_calls = 0
    dev_prompt_tokens = 0
    dev_completion_tokens = 0
    round_logs: list[dict[str, Any]] = []
    program_snapshots: list[dict[str, Any]] = [
        {"round": 0, "accepted": True, "source": current_src}
    ]

    with Evaluator(backend, workers=1, budget_limit=max(1, int(max_rollouts))) as ev:
        nom_cases = [(theta_nom, seed + i) for i in range(max(1, int(n_nominal_seeds)))]
        traces = ev.run_many(current_src, nom_cases)
        best_sr = sum(1 for t in traces if t.success) / len(traces)
        best_score = sum(t.score for t in traces) / len(traces)
        round_logs.append(
            {
                "round": 0,
                "nominal_sr": best_sr,
                "nominal_score": best_score,
                "n_rollouts": ev.n_rollouts,
                "dev_llm_calls": dev_llm_calls,
            }
        )

        rnd = 0
        while best_sr < 1.0 and dev_llm_calls < max_llm_calls:
            rnd += 1
            failing = [t for t in traces if not t.success]
            summary = summarize_trace_for_llm(failing[0])
            prompt = build_single_twin_prompt(
                task_description=backend.task_description,
                current_code=best_src,
                failure_summary=summary,
            )
            resp = llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.2)
            dev_llm_calls += 1
            dev_prompt_tokens += int(resp.prompt_tokens)
            dev_completion_tokens += int(resp.completion_tokens)

            cand_src = extract_python_code(resp.text)
            try:
                check_policy_source(cand_src)
            except StaticCheckError as exc:
                round_logs.append(
                    {
                        "round": rnd,
                        "accepted": False,
                        "error": str(exc),
                        "n_rollouts": ev.n_rollouts,
                        "dev_llm_calls": dev_llm_calls,
                    }
                )
                continue

            if ev.n_rollouts + len(nom_cases) > ev.budget_limit:
                break
            try:
                cand_traces = ev.run_many(cand_src, nom_cases)
            except BudgetExceededError:
                break

            cand_sr = sum(1 for t in cand_traces if t.success) / len(cand_traces)
            cand_score = sum(t.score for t in cand_traces) / len(cand_traces)
            accepted = (cand_sr > best_sr) or (
                cand_sr == best_sr and cand_score > best_score
            )
            if accepted:
                best_src = cand_src
                best_sr = cand_sr
                best_score = cand_score
                traces = cand_traces
                program_snapshots.append(
                    {"round": rnd, "accepted": True, "source": best_src}
                )
            round_logs.append(
                {
                    "round": rnd,
                    "accepted": accepted,
                    "nominal_sr": cand_sr,
                    "nominal_score": cand_score,
                    "n_rollouts": ev.n_rollouts,
                    "dev_llm_calls": dev_llm_calls,
                }
            )

        n_rollouts = ev.n_rollouts

    return BaselineResult(
        method="single_twin",
        policy_source=best_src,
        n_rollouts=n_rollouts,
        dev_llm_calls=dev_llm_calls,
        dev_prompt_tokens=dev_prompt_tokens,
        dev_completion_tokens=dev_completion_tokens,
        round_logs=round_logs,
        program_snapshots=program_snapshots,
        wall_time_s=time.perf_counter() - t0,
        extra={"nominal_sr": best_sr, "nominal_score": best_score},
    )
