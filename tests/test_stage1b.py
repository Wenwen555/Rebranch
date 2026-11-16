"""Stage1b protocol regressions using mock requests; no external model calls."""
from dataclasses import asdict
import subprocess
import pytest
from rebranch.core.development import AdmissionGate, admission_passes
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.trace import Trace,CallRecord
from rebranch.llm.client import MockLLMClient,LimitedLLMClient,LLMBudgetExceededError
from rebranch.sim.toy_backend import ToyRetuneTask,ToyBranchTask

A='def run(api):\n    api.open_gripper(0.061)\n'
B='def run(api):\n    api.open_gripper(0.062)\n'

class ScoreEvaluator(Evaluator):
    def __init__(self, backend=None, **kw):
        super().__init__(backend or ToyRetuneTask(), **kw);self.events=[]
    def run_many(self,source,cases,seed=0):
        self._consume_budget(len(cases));self.events.append((source,len(cases),seed))
        score = -.3 if source==B else (-.6 if source==A else -.9)
        return [Trace(theta=dict(c[0] if isinstance(c,tuple) else c),seed=seed+i,score=score,failure='diagnostic_failure',calls=[CallRecord(0,'get_gripper_state',2,1,(),{}, {'width':.05},{'grip_width':.05})]) for i,c in enumerate(cases)]
    def run(self,source,theta,seed=0):return self.run_many(source,[theta],seed)[0]

def test_strict_admission_and_cached_incumbent():
    def tr(score,ok=False):return Trace({},0,success=ok,score=score)
    assert not admission_passes([tr(-.4)],[tr(-.4)])
    assert not admission_passes([tr(-.4)],[tr(-.5)])
    assert admission_passes([tr(-.4)],[tr(-.3)])  # 0-success score progress allowed.
    assert not admission_passes([tr(.2,True),tr(-.9)],[tr(-.1),tr(.9,True)])
    ev=ScoreEvaluator();cases=[{}]*24;g=AdmissionGate(ev,cases,'initial')
    assert not g.consider('initial')[0] and ev.n_rollouts==24
    assert g.consider(A)[0] and ev.n_rollouts==48
    assert not g.consider(A)[0] and ev.n_rollouts==48
    assert g.consider(B)[0] and ev.n_rollouts==72
    assert sum(n for src,n,seed in ev.events if src=='initial' and seed==10000)==24

def test_two_requests_keep_best_repair_and_admit(monkeypatch):
    import rebranch.core.loop as loop
    task=ToyRetuneTask();ev=ScoreEvaluator(task,budget_limit=2000)
    monkeypatch.setattr(loop,'sensitivity_screen',lambda *a,**kw:list(task.param_space.names))
    monkeypatch.setattr(loop,'find_counterexamples',lambda *a,**kw:([],{}))
    llm=MockLLMClient([A,B]);cfg=loop.ReBranchConfig(max_rounds=1,max_llm_calls=2,max_branch_retries=2,no_lifting=True,admission_n_eval=24)
    result=loop.run_rebranch(task,ev,llm,cfg)
    assert llm.stats['calls']==2
    assert result.final_source==B
    assert result.rounds[0].admitted
    assert result.stop_reason=="LLM_BUDGET_STOP"
    assert any(src==B and n==24 and seed==10000 for src,n,seed in ev.events)

def test_branch_budget_exhaustion_returns_measured_best():
    from rebranch.core.branch import synthesize_branch
    ev=ScoreEvaluator(budget_limit=2000);llm=MockLLMClient([A,B])
    src,ver=synthesize_branch(ev,LimitedLLMClient(llm,2),'task','initial',[{},{}],ev.run_many('initial',[{},{}]),[(0,1)],[],max_retries=2,no_lifting=True)
    assert src==B and len(ver.traces)==2 and llm.stats['calls']==2
    gate=AdmissionGate(ev,[{}]*24,'initial')
    assert gate.consider(src)[0]

def test_admission_feedback_contains_actual_regression_sdk():
    ev=ScoreEvaluator();gate=AdmissionGate(ev,[{}]*24,B)
    assert not gate.consider(A)[0]
    assert 'Strict comparison' in gate.feedback and 'Admission case' in gate.feedback
    assert 'diagnostic_failure' in gate.feedback and 'get_gripper_state' in gate.feedback

def test_runner_default_caps_and_parity(monkeypatch,tmp_path):
    import inspect
    import rebranch.eval.runner as runner
    assert inspect.signature(runner._make_rebranch_config).parameters['admission_n_eval'].default==24
    assert inspect.signature(runner.run_single_method).parameters['max_rollouts'].default==2000
    seen=[]
    def fake(task,method,llm,**kw):
        seen.append((method,kw));return {'result':type('R',(),{'n_rollouts':1,'dev_llm_calls':0})()}
    monkeypatch.setattr(runner,'run_single_method',fake)
    runner.run_benchmark_suite(['toy_branch'],['rebranch','uniform_dr'],MockLLMClient([A]),seeds=[0],output_dir=tmp_path,max_rollouts=2000,max_llm_calls=2)
    assert [(k['max_rollouts'],k['max_llm_calls']) for m,k in seen]==[(2000,2),(2000,2)]

def test_proxy_failed_requests_budget_and_none_prediction():
    from rebranch.baselines.physcap_agent import run_physcap
    task=ToyBranchTask()
    def fail(prompt): raise RuntimeError('transient request failure')
    result=run_physcap(task,MockLLMClient(fail),max_dev_llm_calls=0,runtime_llm_budget=1)
    th=task.param_space.nominal();tr,meta=result.agent.evaluate_episode(task,th)
    assert meta['runtime_llm_calls']==1 and meta['runtime_llm_failed_attempts']==1
    assert meta['runtime_tokens'] is None and meta['sim_predicted_success'] is None
    assert len(meta['runtime_request_events'])==1 and meta['fallback_reason']=='runtime_llm_budget_exhausted'
    tr,meta2=result.agent.evaluate_episode(task,th)
    assert meta2['runtime_llm_calls']==0 and meta2['runtime_llm_budget_used']==1

def test_none_prediction_survives_runner_and_metrics():
    from rebranch.baselines.physcap_agent import run_physcap
    from rebranch.eval.runner import _evaluate_split
    task=ToyBranchTask();llm=MockLLMClient([task.draft_policy]);res=run_physcap(task,llm,max_dev_llm_calls=0,runtime_llm_budget=0)
    metrics=_evaluate_split(task,res,task.hidden_test_set(n=2),seed=0,workers=1)
    assert metrics.prediction_coverage==0 and metrics.sim_false_positive_rate is None
    assert metrics.runtime_llm_calls==0

def test_twin_runtime_simulations_are_bounded_and_counted():
    from rebranch.baselines import build_safe_probe_source,probe_features
    from rebranch.baselines.twin_mpc import TwinMPCAgent
    from rebranch.exec.sandbox import run_policy
    task=ToyBranchTask();bank=task.hidden_test_set(n=2);source=build_safe_probe_source(task)
    obs=[probe_features(run_policy(source,task.make_episode(th),th),source) for th in bank]
    agent=TwinMPCAgent(source,bank,obs,[task.draft_policy,A],runtime_sim_budget_per_episode=3)
    _,meta=agent.evaluate_episode(task,bank[0])
    assert meta['runtime_sim_rollouts']==2 and meta['runtime_sim_time_s']>0
    agent.runtime_sim_budget_per_episode=0
    _,meta=agent.evaluate_episode(task,bank[0])
    assert meta['runtime_sim_rollouts']==0 and meta['sim_predicted_success'] is None

def test_udr_rejection_reports_admission_even_if_train_passes(monkeypatch):
    import rebranch.baselines.uniform_dr as udr
    import rebranch.core.development as dev
    class RegressionEvaluator(ScoreEvaluator):
        def run_many(self,source,cases,seed=0):
            traces=super().run_many(source,cases,seed)
            for t in traces:
                t.success = (source not in [A,B]) if seed==10000 else (source in [A,B])
                t.score = .5 if t.success else -.5
            return traces
    monkeypatch.setattr(udr,'Evaluator',RegressionEvaluator)
    monkeypatch.setattr(dev,'polish_candidate',lambda *a,**kw:None)
    llm=MockLLMClient([A,B]);result=udr.run_uniform_dr(ToyBranchTask(),llm,max_llm_calls=2)
    assert llm.stats['calls']==2 and all(not r['accepted'] for r in result.round_logs)
    prompt=llm.history[1]['prompt']
    assert 'Fixed admission rejected' in prompt and 'Strict comparison' in prompt
    assert 'Admission case' in prompt and 'get_gripper_state' in prompt

def test_rebranch_outer_rejection_feedback_reaches_next_proposal(monkeypatch):
    import rebranch.core.loop as loop
    task=ToyRetuneTask()
    class OuterRegressionEvaluator(ScoreEvaluator):
        def run_many(self,source,cases,seed=0):
            traces=super().run_many(source,cases,seed)
            if seed==10000:
                for t in traces:t.score= -.6 if source==A else (-.1 if source==B else -.3)
            return traces
    ev=OuterRegressionEvaluator(task,budget_limit=2000)
    monkeypatch.setattr(loop,'sensitivity_screen',lambda *a,**kw:list(task.param_space.names))
    monkeypatch.setattr(loop,'find_counterexamples',lambda *a,**kw:([],{}))
    llm=MockLLMClient([A,B]);cfg=loop.ReBranchConfig(max_rounds=2,max_llm_calls=2,max_branch_retries=0,no_lifting=True)
    result=loop.run_rebranch(task,ev,llm,cfg)
    assert result.rounds[0].action=='REJECTED' and result.final_source==B
    assert 'Fixed admission rejected' in llm.history[1]['prompt']
    assert 'get_gripper_state' in llm.history[1]['prompt']

def test_default_single_entry_two_complete_proposals(tmp_path):
    from rebranch.eval.runner import run_single_method
    for method in ['uniform_dr','rebranch']:
        llm=MockLLMClient([A,B])
        r=run_single_method('toy_branch',method,llm,max_llm_calls=2,n_hidden=0,n_ood=0,workers=1,output_dir=tmp_path/method)
        assert llm.stats['calls']==2
        assert r['result'].n_rollouts<=2000
        assert not any(x.get('action')=='SIM_BUDGET_STOP' for x in r['result'].round_logs)

def test_unloaded_reference_hold_is_public_sim_time():
    from rebranch.sim.mujoco_episode import MujocoEpisode
    from rebranch.baselines.physcap_agent import PhysCaPAPIWrapper
    plain=MujocoEpisode(noise_enabled=False);wrapped=MujocoEpisode(noise_enabled=False)
    pose=plain.get_ee_pose()['pos'];plain.move_to(pose)
    api=PhysCaPAPIWrapper(wrapped);api.move_to(pose)
    assert wrapped.sim_time-plain.sim_time==pytest.approx(.24)
    assert api.measurements[-1]['hold_s']==.24

def test_twin_runtime_rollouts_reach_split_metrics():
    from rebranch.baselines import BaselineResult,build_safe_probe_source,probe_features
    from rebranch.baselines.twin_mpc import TwinMPCAgent
    from rebranch.exec.sandbox import run_policy
    from rebranch.eval.runner import _evaluate_split
    t=ToyBranchTask();bank=t.hidden_test_set(n=2);probe=build_safe_probe_source(t)
    obs=[probe_features(run_policy(probe,t.make_episode(th),th),probe) for th in bank]
    agent=TwinMPCAgent(probe,bank,obs,[t.draft_policy,A],runtime_sim_budget_per_episode=3)
    res=BaselineResult('twin_mpc',t.draft_policy,0,0,0,0,agent=agent)
    m=_evaluate_split(t,res,bank,seed=0,workers=1)
    assert m.n_rollouts_runtime_sim==4 and m.runtime_sim_time_s>0
    agent.runtime_sim_budget_per_episode=0
    m=_evaluate_split(t,res,bank,seed=0,workers=1)
    assert m.n_rollouts_runtime_sim==0 and m.prediction_coverage==0 and m.sim_false_positive_rate is None

def test_no_conflict_repair_prompt_uses_current_program_failure_sdk(monkeypatch):
    import rebranch.core.loop as loop
    from rebranch.core.conflict import CaseSolution
    from rebranch.core.counterexamples import Counterexample
    from rebranch.core.retune import RetuneResult

    task=ToyRetuneTask();ev=ScoreEvaluator(task,budget_limit=2000)
    failure=Trace(theta={'hidden_theta_marker':987654},seed=7,success=False,
        score=-.7,failure='physical_contact_failure',
        calls=[CallRecord(0,'move_until_contact',2,1,(0,0,.02),{},
            {'contact':False},{'ee_z':.12,'grip_width':.04})])
    ce=Counterexample({'hidden_theta_marker':987654},failure,None,1,.7,.2,7)

    monkeypatch.setattr(loop,'sensitivity_screen',lambda *a,**kw:list(task.param_space.names))
    monkeypatch.setattr(loop,'find_counterexamples',lambda *a,**kw:([ce],{}))
    monkeypatch.setattr(loop,'_lift_with_optional_hoist',lambda src:(src,[object()]))
    monkeypatch.setattr(loop,'per_case_solutions',lambda retuner,source,consts,cases,**kw:[
        CaseSolution(i,c,True,{},source,.2,Trace({},i,success=True,score=.2))
        for i,c in enumerate(cases)])
    monkeypatch.setattr(loop,'cross_evaluate_solutions',lambda *a,**kw:{})
    monkeypatch.setattr(loop,'unsat_core',lambda *a,**kw:[0,1])
    monkeypatch.setattr(loop,'incompatible_pairs',lambda *a,**kw:[])
    monkeypatch.setattr(loop,'polish_candidate',lambda *a,**kw:None)

    class FailedJointRetuner:
        def __init__(self,*a,**kw): self.seed=0;self.budget_reserve=0
        def optimize(self,source,consts,cases):
            return RetuneResult(False,{},source,-1,0,len(cases),-1,-1,[],1)
    monkeypatch.setattr(loop,'Retuner',FailedJointRetuner)

    llm=MockLLMClient([B])
    cfg=loop.ReBranchConfig(max_rounds=1,max_llm_calls=1,max_branch_retries=0,
        admission_n_eval=2,retune_margin_target=.1)
    loop.run_rebranch(task,ev,llm,cfg)
    prompt=llm.history[0]['prompt']
    assert 'physical_contact_failure' in prompt
    assert 'move_until_contact' in prompt and 'ee_z' in prompt
    assert 'No failures recorded.' not in prompt
    assert 'hidden_theta_marker' not in prompt and '987654' not in prompt
