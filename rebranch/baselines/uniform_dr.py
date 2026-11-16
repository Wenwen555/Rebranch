"""Uniform-DR + CaP baseline under strictly matched rollout and LLM budgets."""

from __future__ import annotations

from collections import Counter
import time
from typing import Any

import numpy as np

from rebranch.baselines import BaselineResult
from rebranch.eval.metrics import summarize_trace_for_llm
from rebranch.exec.evaluator import BudgetExceededError, Evaluator
from rebranch.exec.sandbox import StaticCheckError, check_policy_source
from rebranch.exec.trace import Trace
from rebranch.llm.client import LLMClient, extract_python_code
from rebranch.llm.prompts import SYSTEM_PROMPT, build_uniform_dr_prompt
from rebranch.sim.backend import TaskBackend


def _summarize_dr_failures(traces: list[Trace], max_examples: int = 3) -> str:
    """Summarize multi-twin failures strictly via SDK observables (zero theta leak)."""
    n_total = len(traces)
    failed = [t for t in traces if not t.success]
    succeeded = [t for t in traces if t.success]
    fail_counts = Counter((t.failure or "none", t.error or "none", t.safety_violation or "none") for t in traces)
    lines = [
        f"Batch summary: {len(succeeded)}/{n_total} succeeded, {len(failed)}/{n_total} failed.",
        "All-trace breakdown (task_failure, execution_error, safety_violation -> count):",
    ]
    for (ftype, err, sv), cnt in fail_counts.most_common():
        lines.append(f"  - task_failure={ftype}; execution_error={err} (safety={sv}): {cnt}")

    if succeeded and failed:
        # Report SDK-observable statistics separating failed vs succeeded trials
        succ_obs = [t.calls[-1].obs for t in succeeded if t.calls]
        fail_obs = [t.calls[-1].obs for t in failed if t.calls]
        if succ_obs and fail_obs:
            diffs: list[tuple[float, str, float, float]] = []
            for k in ("fx", "fy", "fz", "tx", "ty", "tz", "grip_width", "grip_effort", "ee_x", "ee_y", "ee_z"):
                m_s = sum(o.get(k, 0.0) for o in succ_obs) / len(succ_obs)
                m_f = sum(o.get(k, 0.0) for o in fail_obs) / len(fail_obs)
                diffs.append((abs(m_s - m_f), k, m_s, m_f))
            diffs.sort(reverse=True)
            lines.append("Top final-SDK mean shifts (succeeded vs failed):")
            for _, k, m_s, m_f in diffs[:4]:
                lines.append(f"  - {k}: succeeded_mean={m_s:.4f}, failed_mean={m_f:.4f}")

    lines.append(f"Representative failure traces (up to {max_examples}):")
    for idx, tr in enumerate(failed[:max_examples]):
        lines.append(f"[Failure #{idx + 1}]")
        lines.append(summarize_trace_for_llm(tr, max_calls=8))
    return "\n".join(lines)


def run_uniform_dr(
    backend: TaskBackend, llm: LLMClient, *, seed: int = 0, workers: int = 1,
    max_rollouts: int = 2000, max_llm_calls: int = 8,
    twins_per_round: int | None = None, admission_n_eval: int = 24,
) -> BaselineResult:
    """Ordinary multi-trajectory repair with fixed admission and shared Retuner.

    All measured candidates, rejected feedback, optimizer and admission rollouts
    count against one budget. No shortened admission batch at budget exhaustion.
    """
    from rebranch.core.development import admission_cases, AdmissionGate, polish_candidate, DEVELOPMENT_PROTOCOL_VERSION
    t0=time.perf_counter(); rng=np.random.default_rng(seed + 1)
    budget=max(0,int(max_rollouts)); llm_budget=max(0,int(max_llm_calls))
    n_batch=max(2,int(twins_per_round or 24))
    val_cases=admission_cases(backend,seed,admission_n_eval)
    train_cases=[backend.param_space.sample(rng) for _ in range(n_batch)]
    current_src=backend.draft_policy; rejected_attempt=""
    snapshots=[{"round":0,"accepted":True,"source":current_src}]; logs=[]
    calls=pt=ct=0
    with Evaluator(backend,workers=workers,budget_limit=budget) as ev:
        if budget < len(train_cases)+len(val_cases):
            return BaselineResult(method="uniform_dr",policy_source=current_src,n_rollouts=0,
                dev_llm_calls=0,dev_prompt_tokens=0,dev_completion_tokens=0,
                program_snapshots=snapshots,extra={"protocol":DEVELOPMENT_PROTOCOL_VERSION,"stop":"insufficient_initial_budget"})
        train=ev.run_many(current_src,train_cases,seed=seed)
        gate=AdmissionGate(ev,val_cases,current_src)
        for rnd in range(1,llm_budget+1):
            reserve=len(val_cases)+20
            # Keep full verification and admission batches; never truncate difficult cases.
            if budget-ev.n_rollouts < len(train_cases)+reserve:
                logs.append({"round":rnd,"stop":"insufficient_full_batch_budget"});break
            if all(t.success for t in train):
                # Discover more failures without changing the fixed admission set.
                fresh=[backend.param_space.sample(rng) for _ in range(n_batch)]
                if budget-ev.n_rollouts < 2*len(train_cases)+reserve: break
                train_cases=fresh;train=ev.run_many(current_src,train_cases,seed=seed)
                if all(t.success for t in train): continue
            prompt=build_uniform_dr_prompt(task_description=backend.task_description,
                current_code=current_src,failure_summary=_summarize_dr_failures(train),
                rejected_attempt=rejected_attempt)
            resp=llm.complete(prompt,system=SYSTEM_PROMPT,temperature=.2)
            calls+=1;pt+=int(resp.prompt_tokens);ct+=int(resp.completion_tokens)
            candidate=extract_python_code(resp.text)
            try: check_policy_source(candidate)
            except StaticCheckError as exc:
                rejected_attempt=f"```python\n{candidate}\n```\nStatic check: {exc}"
                snapshots.append({"round":rnd,"accepted":False,"source":candidate})
                logs.append({"round":rnd,"accepted":False,"static_error":str(exc),"feedback":rejected_attempt,"n_rollouts":ev.n_rollouts,"dev_llm_calls":calls})
                continue
            candidate_train=ev.run_many(candidate,train_cases,seed=seed)
            opt=polish_candidate(ev,candidate,train_cases,seed=seed+(rnd-1)*17,reserve=reserve)
            if opt is not None and opt.traces and (opt.n_success,opt.mean_score) > (sum(t.success for t in candidate_train),sum(t.score for t in candidate_train)/len(candidate_train)):
                candidate=opt.best_source;candidate_train=opt.traces
            accepted, candidate_rate=gate.consider(candidate)
            new_val=gate.last_candidate
            feedback=_summarize_dr_failures(candidate_train)
            if accepted:
                current_src=candidate;train=candidate_train;rejected_attempt=""
            else: rejected_attempt=feedback+"\n"+gate.feedback
            snapshots.append({"round":rnd,"accepted":accepted,"source":candidate})
            logs.append({"round":rnd,"accepted":accepted,"train_summary":feedback,
                "admission_successes":sum(t.success for t in new_val),"admission_n":len(val_cases),"admission_feedback":gate.feedback,"admission_mean_score":sum(t.score for t in new_val)/len(new_val),
                "n_rollouts":ev.n_rollouts,"dev_llm_calls":calls})
        used=ev.n_rollouts
    return BaselineResult(method="uniform_dr",policy_source=current_src,n_rollouts=used,
        dev_llm_calls=calls,dev_prompt_tokens=pt,dev_completion_tokens=ct,
        round_logs=logs,program_snapshots=snapshots,wall_time_s=time.perf_counter()-t0,
        extra={"protocol":DEVELOPMENT_PROTOCOL_VERSION,"admission_cases":val_cases,
               "admission_noise_seed":10000,"optimizer":"Retuner/cap120/iters6/pop8",
               "stop_reason":"LLM_BUDGET_STOP" if calls>=llm_budget else (logs[-1].get("stop","training_discovery_round_limit") if logs else "training_discovery_round_limit")})
