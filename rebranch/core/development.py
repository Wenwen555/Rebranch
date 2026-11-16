"""Shared v2 development protocol for ordinary and structured code repair."""
from rebranch.core.lifting import lift_constants, hoist_inline_literals
from rebranch.core.retune import Retuner

DEVELOPMENT_PROTOCOL_VERSION = 'repair-development-v4-bounded-feedback'


def admission_cases(backend, seed, n=24):
    import numpy as np
    rng=np.random.default_rng(seed)
    return [backend.param_space.sample(rng) for _ in range(n)]


def polish_candidate(evaluator, source, cases, *, seed, reserve=0, max_iters=6, margin=.25):
    """Identical constant extraction, optimizer, cap and seed route for all repairs."""
    consts=lift_constants(source)
    if not consts:
        source=hoist_inline_literals(source)
        consts=lift_constants(source)
    if not consts:
        return None
    return Retuner(evaluator,max_iters=max_iters,popsize=8,margin_target=margin,
                   seed=seed,max_rollouts=120,budget_reserve=reserve).optimize(source,consts,cases)


def admission_passes(incumbent, candidate):
    """Preserve solved instances and strictly improve (success count, mean score)."""
    if not incumbent or len(incumbent) != len(candidate):
        return False
    preserved = all(not old.success or new.success for old,new in zip(incumbent,candidate))
    def rank(traces):
        return sum(t.success for t in traces), sum(t.score for t in traces)/len(traces)
    return preserved and rank(candidate) > rank(incumbent)


class AdmissionGate:
    """One fixed-noise incumbent evaluation; update its cache only on admission."""
    def __init__(self, evaluator, cases, source):
        self.evaluator, self.cases, self.source = evaluator, list(cases), source
        self.traces = evaluator.run_many(source, self.cases, seed=10000)
        self.last_candidate = self.traces
        self.feedback = ""

    @property
    def rate(self):
        return sum(t.success for t in self.traces)/len(self.traces) if self.traces else 0.

    def consider(self, candidate):
        new = (self.traces if candidate == self.source else
               self.evaluator.run_many(candidate, self.cases, seed=10000))
        self.last_candidate = new
        accepted = admission_passes(self.traces, new)
        rate = sum(t.success for t in new)/len(new) if new else 0.
        if accepted:
            self.source, self.traces = candidate, new
            self.feedback = ""
        else:
            from rebranch.eval.metrics import summarize_trace_for_llm
            old_rank = (sum(t.success for t in self.traces), sum(t.score for t in self.traces)/len(self.traces))
            new_rank = (sum(t.success for t in new), sum(t.score for t in new)/len(new))
            regressions = [i for i,(old,cur) in enumerate(zip(self.traces,new)) if old.success and not cur.success]
            score_drops = sorted((i for i,(old,cur) in enumerate(zip(self.traces,new)) if cur.score < old.score),
                                 key=lambda i: self.traces[i].score-new[i].score, reverse=True)
            evidence = list(dict.fromkeys(regressions + score_drops))[:3]
            if not evidence and new:
                evidence = [min(range(len(new)), key=lambda i: new[i].score)]
            from collections import Counter
            failure_counts = Counter((t.failure or 'none', t.error or 'none') for t in new)
            counts = "\n".join(f"  - task_failure={failure}; execution_error={error}: {count}"
                               for (failure,error),count in failure_counts.most_common())
            self.feedback = (f"Fixed admission rejected. Strict comparison old={old_rank!r}, new={new_rank!r}; "
                             f"success regressions at admission indices {regressions}.\n"
                             f"All {len(new)} admission cases retained; task failure/execution error counts:\n{counts}\n"
                             f"Rejected candidate:\n```python\n{candidate}\n```\n" +
                             "\n".join(f"Admission case {i}:\n" + summarize_trace_for_llm(new[i],max_calls=8) for i in evidence))
        return accepted, rate
