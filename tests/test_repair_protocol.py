"""Protocol v2 regression checks: measured evidence, timing and active measurement safety."""
import inspect
import numpy as np
import pytest
from rebranch.baselines.physcap_agent import PhysCaPAPIWrapper
from rebranch.core.conflict import CaseSolution, cross_evaluate_solutions, incompatible_pairs
from rebranch.core.loop import run_rebranch
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.sandbox import run_policy
from rebranch.exec.trace import Trace
from rebranch.sim.toy_backend import ToyBranchTask


def test_unknown_cross_results_do_not_create_pairs():
    task=ToyBranchTask(); cases=[task.param_space.nominal()]*2
    solutions=[CaseSolution(i,c,True,{},task.draft_policy,1.,Trace(c,0)) for i,c in enumerate(cases)]
    with Evaluator(task,budget_limit=0) as ev:
        cross=cross_evaluate_solutions(ev,cases,solutions)
        assert cross=={}
        assert incompatible_pairs(ev,task.draft_policy,[],cases,solutions,cross_ok=cross,confirmed_pairs=[(0,1)])==[]
        assert ev.n_rollouts==0
        assert incompatible_pairs(ev,task.draft_policy,[],cases,solutions,cross_ok={(0,1):False})==[]
        assert incompatible_pairs(ev,task.draft_policy,[],cases,solutions,cross_ok={(0,1):False,(1,0):False})==[(0,1)]


def test_micro_contact_search_and_explicit_hold():
    from rebranch.sim.mujoco_episode import MujocoEpisode
    ep=MujocoEpisode(noise_enabled=False)
    p=ep.get_ee_pose()['pos']; target=[p[0],p[1],p[2]+.001]
    r=ep.move_until_contact(target,max_force=6.,max_torque=.8)
    assert not r['contact']
    assert r['pos'][2]-p[2]==pytest.approx(.001,abs=.0001)
    before=ep.sim_time;setpoint=ep._x_des.copy();p=ep.get_ee_pose()
    ep.move_compliant(p['pos'],quat=p['quat'],duration=.02)
    assert ep.sim_time-before==pytest.approx(.02)
    assert np.array_equal(setpoint,ep._x_des)
    first=ep.get_gripper_state();time=ep.sim_time
    assert all(ep.get_gripper_state()==first for _ in range(7))
    assert ep.sim_time==time


def test_tare_invalidates_paired_mass_and_suspended_squeeze_does_not_act():
    from rebranch.tasks import make_task
    t=make_task('t2');th=t.param_space.nominal();th['cup_type']='sand'
    ep=t.make_episode(th);api=PhysCaPAPIWrapper(ep);p=api.get_object_pose('cup')['pos']
    api.get_object_pose('table')  # A static queried object must not authorize suspended squeeze.
    api.open_gripper(.068);api.move_to([p[0],p[1],p[2]+.08]);api.move_to(p)
    assert api._unloaded
    api.zero_ee_wrench()
    assert not api._unloaded
    api.close_gripper(12.);api.move_to([p[0],p[1],p[2]+.08],speed=.03)
    assert api.get_mass() is None
    start=ep.sim_time;force=ep._target_force
    assert api.get_stiffness() is None
    assert ep.sim_time==start and ep._target_force==force


def test_runner_default_has_no_privileged_filter():
    from rebranch.eval.runner import run_single_method
    assert inspect.signature(run_rebranch).parameters['oracle_filter'].default is False
    assert inspect.signature(run_single_method).parameters['oracle_filter'].default is False


def test_llm_cap_prevents_extra_underlying_requests():
    from rebranch.llm.client import LimitedLLMClient, MockLLMClient
    from rebranch.llm.client import LLMBudgetExceededError
    inner=MockLLMClient(lambda prompt: "def run(api): pass")
    client=LimitedLLMClient(inner,1)
    client.complete('first')
    with pytest.raises(LLMBudgetExceededError): client.complete('second')
    assert inner.stats['calls']==1


def test_admission_preserves_individual_successes():
    from rebranch.core.development import admission_passes
    old=[Trace({},0,success=True,score=1.),Trace({},1,success=False,score=-1.)]
    swapped=list(reversed(old))
    assert not admission_passes(old,swapped)
    assert not admission_passes(old,old)


def test_actual_probe_matching_survives_reset(tmp_path):
    from scripts.repair_stage1_probe_matching import main
    summary=main(tmp_path)
    assert all(v['selected_indices']==[0,1] for v in summary.values())
