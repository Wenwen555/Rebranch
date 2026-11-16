"""Fixed SDK feedback control under close timing perturbations; no theta input."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from rebranch.tasks import make_task
from rebranch.exec.sandbox import run_policy

SOURCE='''def sample_width(api):
    total = 0.0
    for i in range(24):
        hold = api.get_ee_pose()
        api.move_compliant(hold['pos'], quat=hold['quat'], duration=0.02)
        total += api.get_gripper_state()['width']
    return total / 24

def run(api):
    api.zero_ee_wrench()
    p = api.get_object_pose('cup')['pos']
    api.open_gripper(0.068)
    api.move_to([p[0], p[1], p[2]+0.08], speed=0.08)
    api.move_to(p, speed=0.04)
    api.close_gripper(force=1.0)
    low = sample_width(api)
    api.close_gripper(force=3.0)
    high = sample_width(api)
    force = 3.0 if high < 0.049 else 15.0
    api.close_gripper(force=force)
    api.move_to([p[0], p[1], p[2]+0.11], speed=0.04)
'''

def work(job):
 split,i,th,extra=job;t=make_task('t2');ep=t.make_episode(th,seed=83000+i);close=ep.close_gripper
 def delayed(force=8.): close(force);ep.step_idle(round(extra/ep.dt))
 ep.close_gripper=delayed
 tr=run_policy(SOURCE,ep,th,seed=83000+i)
 return dict(split=split,index=i,extra_wait=extra,theta=th,trace=asdict(tr))

def main():
 t=make_task('t2');cases=[('development',i,th) for i,th in enumerate(t.hidden_test_set(seed=12026,n=20))]+[('boundary',i,th) for i,th in enumerate(t.ood_test_set(seed=13026,n=10))]
 jobs=[(split,i,th,extra) for split,i,th in cases for extra in [0.,.2,.5]]
 root=Path('outputs/repair_validation/stage1/timing_width_control');root.mkdir(exist_ok=True);(root/'control.py').write_text(SOURCE)
 with ProcessPoolExecutor(max_workers=8) as pool: rows=list(pool.map(work,jobs))
 (root/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));s={}
 for r in rows:
  k=f"{r['split']}/{r['extra_wait']}";v=s.setdefault(k,dict(n=0,success=0,failures=Counter()));v['n']+=1;v['success']+=r['trace']['success'];v['failures'][str(r['trace']['failure'] or r['trace']['error'])]+=not r['trace']['success']
 (root/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
if __name__=='__main__':main()
