"""Controlled SDK measurement and timing diagnostics; no LLM calls."""
import json
from pathlib import Path
from rebranch.tasks import make_task
from rebranch.baselines.physcap_agent import PhysCaPAPIWrapper
from rebranch.sim.mujoco_episode import MujocoEpisode, SIM_PROTOCOL_VERSION

rows=[]
for typ in ['empty','sand']:
 for seed in range(3):
  for wait in [0.,.2,.5]:
   t=make_task('t2'); th=t.param_space.nominal(); th['cup_type']=typ
   ep=t.make_episode(th,seed=seed); api=PhysCaPAPIWrapper(ep)
   p=api.get_object_pose('cup')['pos']; api.open_gripper(.068)
   api.move_to([p[0],p[1],p[2]+.08],speed=.04)
   api.move_to(p,speed=.04); api.close_gripper(1.); ep.step_idle(round(wait/ep.dt))
   a=api.get_gripper_state(); true_a=float(ep.data.qpos[6]-ep.data.qpos[7]); stamp=ep.sim_time
   repeated=[api.get_gripper_state() for _ in range(5)]
   api.close_gripper(3.); ep.step_idle(round(wait/ep.dt)); b=api.get_gripper_state(); true_b=float(ep.data.qpos[6]-ep.data.qpos[7])
   k=api.get_stiffness()
   # This is an explicit type-specialized capability measurement, NOT baseline input.
   api.close_gripper(3. if typ=='empty' else 12.)
   api.move_to([p[0],p[1],p[2]+.08],speed=.02)
   mass=api.get_mass(); z=api.get_object_pose('cup')['pos'][2]
   rows.append(dict(type=typ,seed=seed,extra_wait=wait,a=a,b=b,true_widths=[true_a,true_b],identical_repeated_reads=all(x==repeated[0] for x in repeated),stiffness=k,mass=mass,object_lift=z-p[2],measurements=api.measurements))
# Tiny movements without contact must execute; contact search must not report free-space inertia as contact.
for dz in [.001,.003,.01]:
 ep=MujocoEpisode(noise_enabled=False)
 p=ep.get_ee_pose()['pos']; start=ep.sim_time
 r=ep.move_until_contact([p[0],p[1],p[2]+dz],speed=.04,max_force=6.,max_torque=.8)
 rows.append(dict(kind='micro_contact_search',requested_dz=dz,actual_dz=r['pos'][2]-p[2],contact=r['contact'],elapsed=ep.sim_time-start,wrench=r['force']))
root=Path('outputs/repair_validation/stage1'); (root/'physics_checked.json').write_text(json.dumps(dict(protocol=SIM_PROTOCOL_VERSION,rows=rows),indent=2))
for r in rows:
 if 'type' in r: print(r['type'],r['seed'],r['extra_wait'],'widths',r['true_widths'],'k',r['stiffness'],'m',r['mass'],'dz',r['object_lift'])
 else: print(r)
