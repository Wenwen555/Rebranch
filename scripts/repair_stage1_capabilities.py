"""No-LLM capability diagnostics. Never filter samples or feed theta to ordinary controls."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from rebranch.tasks import TASKS, make_task
from rebranch.exec.sandbox import run_policy
from rebranch.sim.mujoco_episode import SIM_PROTOCOL_VERSION

def run_one(job):
    name, split, i, th, policy = job
    task = make_task(name)
    src = task.oracle_policy(th) if policy == 'oracle_reference' else task.draft_policy
    if policy == 'compliant_control':
        src = task.draft_policy.replace('GRIP_FORCE = 14.0', 'GRIP_FORCE = 18.0').replace('GRIP_FORCE = 15.0', 'GRIP_FORCE = 18.0')
        src = src.replace('PULL_DIST = 0.09 ', 'PULL_DIST = 0.125 ').replace('PULL_DIST = 0.095 ', 'PULL_DIST = 0.125 ')
        src = src.replace('api.move_to([h[0] - PULL_DIST, h[1], h[2]], speed=PULL_SPEED)', 'api.move_compliant([h[0] - PULL_DIST, h[1], h[2]], free_axes="yzXYZ", speed=0.04)')
    ep=task.make_episode(th, seed=700000+i)
    tr=run_policy(src,ep,th,seed=700000+i)
    return dict(task=name,split=split,index=i,policy=policy,theta=th,source=src,trace=tr.to_dict() if hasattr(tr,'to_dict') else vars(tr))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--n',type=int,default=20); p.add_argument('--out',default='outputs/repair_validation/stage1/capabilities'); a=p.parse_args()
    jobs=[]
    for name in TASKS:
        t=make_task(name)
        cases=[('nominal',0,t.param_space.nominal())]+[('development',i,th) for i,th in enumerate(t.hidden_test_set(seed=12026,n=a.n))]+[('boundary',i,th) for i,th in enumerate(t.ood_test_set(seed=13026,n=10))]
        for split,i,th in cases:
            for policy in ['draft','oracle_reference']+(['compliant_control'] if name in ['t4_cabinet','t5_drawer'] else []): jobs.append((name,split,i,th,policy))
    root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    summary={}
    with ProcessPoolExecutor(max_workers=8) as pool, (root/'traces.jsonl').open('w') as f:
        for row in pool.map(run_one,jobs):
            f.write(json.dumps(row,default=lambda x:vars(x))+'\n'); f.flush()
            key=row['task']+'/'+row['policy']+'/'+row['split']; z=summary.setdefault(key,dict(n=0,success=0,failures=Counter()))
            tr=row['trace']; z['n']+=1; z['success']+=tr['success']; z['failures'][str(tr['failure'] or tr['error'])]+=not tr['success']
    (root/'summary.json').write_text(json.dumps(dict(protocol=SIM_PROTOCOL_VERSION,summary=summary),indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
