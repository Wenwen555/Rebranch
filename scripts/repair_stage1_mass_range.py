"""Independent noisy paired-payload checks, tare drift, and actual pose errors."""
import json
import sys
from pathlib import Path
from rebranch.tasks import make_task
from rebranch.baselines.physcap_agent import PhysCaPAPIWrapper
from rebranch.sim.mujoco_episode import MujocoEpisode
rows=[]
for mass in [.015,.30,.35,.40,.45,.50]:
 for seed in range(5):
  t=make_task('t2');th=t.param_space.nominal();th['cup_type']='empty' if mass==.015 else 'sand';th['sand_mass_kg']=mass
  ep=t.make_episode(th,seed=81000+seed);api=PhysCaPAPIWrapper(ep);p=api.get_object_pose('cup')['pos'];api.open_gripper(.068)
  api.move_to([p[0],p[1],p[2]+.08],speed=.03);api.zero_ee_wrench()
  # Explicit paired unloaded reference is acquired AFTER the new tare.
  api.move_to([p[0],p[1],p[2]+.08],speed=.03)
  api.move_to(p,speed=.03);api.close_gripper(3. if mass==.015 else 14.)
  api.move_to([p[0],p[1],p[2]+.08],speed=.02);estimated=api.get_mass()
  rows.append(dict(true_mass=mass,noise_seed=81000+seed,noise_enabled=ep._noise.enabled,estimate=estimated,relative_error=None if estimated is None else abs(estimated-mass)/mass,object_lift=ep.get_object_pose('cup')['pos'][2]-p[2],measurement=api.measurements[-1]))
drift=[]
for seed in range(5):
 ep=MujocoEpisode(seed=91000+seed);samples=[]
 for sec in [0.,.1,.5,1.]:
  if sec:ep.step_idle(round(sec/ep.dt))
  samples.append(dict(time=ep.sim_time,wrench=ep.get_ee_wrench()))
 drift.append(dict(seed=seed,samples=samples))
summary={}
for mass in [.015,.30,.35,.40,.45,.50]:
 rs=[r for r in rows if r['true_mass']==mass];valid=[r for r in rs if r['estimate'] is not None]
 summary[str(mass)]=dict(n=len(rs),valid=len(valid),unmeasurable=len(rs)-len(valid),max_relative_error=max([r['relative_error'] for r in valid],default=None))
out=Path('outputs/repair_validation/stage1');(out/(sys.argv[1]+'.json' if len(sys.argv)>1 else 'mass_range.json')).write_text(json.dumps(dict(rows=rows,summary=summary,unloaded_drift=drift),indent=2));print(json.dumps(summary,indent=2))
