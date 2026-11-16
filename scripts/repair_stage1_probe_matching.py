"""Actual MuJoCo probe-stage discrimination for both runtime selectors; no LLM."""
from dataclasses import asdict
import json
from pathlib import Path
from rebranch.tasks import make_task
from rebranch.baselines import build_safe_probe_source,probe_features
from rebranch.baselines.twin_mpc import TwinMPCAgent
from rebranch.baselines.oneshot_calib import OneShotCalibAgent
from rebranch.exec.sandbox import run_policy

def main(output=None):
 t=make_task('t2');probe=build_safe_probe_source(t)
 bank=[];obs=[];bank_records=[]
 for typ in ['empty','sand']:
  th=t.param_space.nominal();th['cup_type']=typ;tr=run_policy(probe,t.make_episode(th,seed=100),th,seed=100)
  bank.append(th);obs.append(probe_features(tr,probe));bank_records.append(asdict(tr))
 policies=[t.draft_policy,t.draft_policy.replace('GRIP_FORCE = 3.0','GRIP_FORCE = 15.0')]
 agents={'oneshot':OneShotCalibAgent(probe,bank,obs,policies,[True,True]),'twin_proxy':TwinMPCAgent(probe,bank,obs,policies)}
 rows=[]
 for name,agent in agents.items():
  for typ in ['empty','sand']:
   for seed in range(5):
    th=t.param_space.nominal();th['cup_type']=typ
    tr,meta=agent.evaluate_episode(t,th,seed=120+seed)
    chosen=policies.index(meta['policy_source'])
    rows.append(dict(agent=name,type=typ,noise_enabled=True,noise_seed=120+seed,chosen=chosen,probe_features=meta['probe_features'],trace=asdict(tr)))
 root=Path(output or 'outputs/repair_validation/stage1/probe_matching');root.mkdir(exist_ok=True)
 (root/'probe.py').write_text(probe);(root/'bank.json').write_text(json.dumps(dict(features=obs,traces=bank_records),indent=2));(root/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 summary={}
 for name in agents:
  rs=[r for r in rows if r['agent']==name]; summary[name]=dict(n=len(rs),success=sum(r['trace']['success'] for r in rs),correct_matches=sum(r['chosen']==int(r['type']=='sand') for r in rs),selected_indices=sorted(set(r['chosen'] for r in rs)))
 (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 assert all(len(v['selected_indices'])==2 and v['correct_matches']>=8 for v in summary.values())
 return summary
if __name__=='__main__':main()
