"""Replay all reference failures; retain boundary instances and test control alternatives."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from rebranch.tasks import make_task
from rebranch.exec.sandbox import run_policy

def work(job):
 row,label,src=job;task=make_task(row['task']);th=row['theta'];ep=task.make_episode(th,seed=700000+row['index']);tr=run_policy(src,ep,th,seed=700000+row['index'])
 return dict(task=row['task'],split=row['split'],index=row['index'],theta=th,label=label,source=src,trace=asdict(tr))

def main():
 rows=[json.loads(x) for x in Path('outputs/repair_validation/stage1/capabilities/traces.jsonl').read_text().splitlines()]
 failures=[r for r in rows if r['policy']=='oracle_reference' and not r['trace']['success']]
 jobs=[]
 for r in failures:
  src=r['source'];jobs.append((r,'reference_replay',src))
  if r['task']=='t3_push_can':
   for h in [.028,.030,.032,.036]:
    for speed in [.01,.02]:jobs.append((r,f'low_push_h{h}_v{speed}',src.replace('PUSH_HEIGHT_OFFSET = 0.032',f'PUSH_HEIGHT_OFFSET = {h}').replace('PUSH_SPEED = 0.028',f'PUSH_SPEED = {speed}')))
  if r['task']=='t5_drawer':
   # Ordinary, no theta: one compliant x pull shared across all skew conditions.
   for speed in [.01,.02,.04,.06]:
    for f in [14.,18.]:
     x=make_task('t5').draft_policy.replace('GRIP_FORCE = 15.0',f'GRIP_FORCE = {f}').replace('PULL_DIST = 0.095','PULL_DIST = 0.125').replace('api.move_to([h[0] - PULL_DIST, h[1], h[2]], speed=PULL_SPEED)',f'api.move_compliant([h[0] - PULL_DIST, h[1], h[2]], free_axes="yzXYZ", speed={speed})')
     jobs.append((r,f'fixed_compliant_f{f}_v{speed}',x))
 out=Path('outputs/repair_validation/stage1/residuals');out.mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=8) as pool: result=list(pool.map(work,jobs))
 (out/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
 for r in result: print(r['task'],r['split'],r['index'],r['label'],r['trace']['success'],r['trace']['failure'])
if __name__=='__main__':main()
