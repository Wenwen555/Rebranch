from types import SimpleNamespace
import sys

import pytest

from rebranch.baselines.strong import compare_validation, fixed_compliant_source
from rebranch.core.lifting import lift_constants
from rebranch.core.retune import Retuner
import rebranch.core.retune as retune_module
import scripts.run_method as run_method_cli
import rebranch.baselines.strong as strong_module
from rebranch.eval.runner import run_single_method
from rebranch.tasks.t4_cabinet import T4CabinetTask
from rebranch.tasks.t5_drawer import T5DrawerTask


def _trace(success, score):
    return SimpleNamespace(success=success, score=score, failure=None, error=None)


def test_parameter_validation_rank_allows_case_regression_for_higher_success_count():
    old = [_trace(True, 0.8), _trace(False, 0.2), _trace(False, 0.1)]
    new = [_trace(False, 0.9), _trace(True, 0.8), _trace(True, 0.7)]
    accepted, old_rank, new_rank, regressions = compare_validation(old, new)
    assert accepted
    assert old_rank[0] == 1 and new_rank[0] == 2
    assert regressions == [0]


@pytest.mark.parametrize("backend", [T4CabinetTask(), T5DrawerTask()])
def test_fixed_compliant_template_uses_declared_pull_parameters(backend):
    source = fixed_compliant_source(backend)
    assert "PULL_DIST = 0.125" in source
    assert "PULL_SPEED = 0.04" in source
    assert "GRIP_FORCE = 18.0" in source
    assert "move_compliant([h[0] - PULL_DIST, h[1], h[2]]" in source
    assert "speed=PULL_SPEED" in source
    assert 'free_axes="yzXYZ"' in source
    force = next(c for c in lift_constants(source) if c.name == "GRIP_FORCE")
    assert force.initial == 18.0 and not force.is_int


def test_formal_runner_accepts_compliant_parameter_only_without_model_calls(tmp_path):
    result = run_single_method(
        task_name="t4_cabinet",
        method="compliant_parameter_only",
        llm=object(),
        max_rollouts=0,
        n_hidden=0,
        n_ood=0,
        output_dir=tmp_path,
    )
    assert result["method"] == "compliant_parameter_only"
    assert result["result"].dev_llm_calls == 0
    assert result["result"].n_rollouts == 0
    assert result["result"].extra["stop_reason"] == "cap_below_fixed_validation_cost"
    assert "move_compliant" in result["result"].policy_source


def test_retuner_counts_completed_cma_tell_generations(monkeypatch):
    tell_calls = []

    class MockCMA:
        def __init__(self, *args, **kwargs):
            self.generation = 0

        def stop(self):
            return self.generation >= 5

        def ask(self):
            return [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]

        def tell(self, solutions, losses):
            assert len(solutions) == len(losses) == 4
            self.generation += 1
            tell_calls.append(self.generation)

    monkeypatch.setattr(retune_module.cma, "CMAEvolutionStrategy", MockCMA)

    class MockEvaluator:
        budget_limit = 21

        def __init__(self):
            self.n_rollouts = 0

        def run_many(self, source, cases):
            self.n_rollouts += len(cases)
            return [_trace(False, 0.1) for _ in cases]

    source = "A = 0.5  # [0.0, 1.0]\nB = 0.5  # [0.0, 1.0]\ndef run(api):\n    pass\n"
    evaluator = MockEvaluator()
    optimizer = Retuner(evaluator, max_iters=5, popsize=4, seed=11, max_rollouts=21)
    optimizer.optimize(source, lift_constants(source), [{"case": 0}])
    assert optimizer.generations_completed == len(tell_calls) == 5


def test_cli_forwards_formal_parameter_optimizer_defaults(monkeypatch, capsys):
    captured = {}
    monkeypatch.setattr(run_method_cli, "build_llm_client", lambda **kwargs: object())
    monkeypatch.setattr(
        run_method_cli,
        "run_single_method",
        lambda **kwargs: captured.update(kwargs) or {"metrics": {}},
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_method", "--task", "t4_cabinet", "--method", "fixed_compliant", "--n-hidden", "0", "--n-ood", "0"],
    )
    run_method_cli.main()
    capsys.readouterr()
    assert captured["validation_cases"] == 24
    assert captured["training_cases_per_restart"] == 24
    assert captured["optimizer_restart_rollouts"] == 2000
    assert captured["max_rollouts"] == 2000


def test_parameter_snapshots_keep_initial_source_at_round_zero(monkeypatch):
    initial = "A = 0.5  # [0.0, 1.0]\ndef run(api):\n    pass\n"
    candidate = initial.replace("A = 0.5", "A = 0.7")

    class FakeBackend:
        name = "t2_cup"
        draft_policy = initial

        class Space:
            @staticmethod
            def sample(rng):
                return {"sampled": 1}

        param_space = Space()

    class FakeEvaluator:
        def __init__(self, backend, workers, budget_limit):
            self.budget_limit = budget_limit
            self.n_rollouts = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run_many(self, source, cases, seed=0):
            self.n_rollouts += len(cases)
            success = source == candidate
            return [_trace(success, 0.4 if success else 0.2) for _ in cases]

    class FakeRetuner:
        def __init__(self, evaluator, **kwargs):
            self.evaluator = evaluator
            self.generations_completed = 0

        def optimize(self, source, consts, cases):
            traces = self.evaluator.run_many(candidate, cases)
            return SimpleNamespace(
                best_source=candidate, n_evals=1, success_all=False,
                worst_score=0.1, n_success=len(traces), mean_score=0.4,
                traces=traces,
            )

    monkeypatch.setattr(strong_module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(strong_module, "Retuner", FakeRetuner)
    result = strong_module.run_parameter_only(
        FakeBackend(), max_rollouts=50, training_cases_per_restart=1,
        optimizer_restart_rollouts=26,
    )
    assert result.program_snapshots[0] == {"round": 0, "source": initial}
    assert result.program_snapshots[-1] == {"round": 1, "source": candidate}
    assert result.policy_source == candidate
