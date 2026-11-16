"""Check explicit probes alone and subsequent capability on all tasks, no LLM."""
import concurrent.futures
from dataclasses import asdict
import importlib.util,json,sys
from pathlib import Path
from rebranch.tasks import TASKS,make_task
from rebranch.baselines import build_safe_probe_source
from rebranch.baselines.physcap_agent import _augment_probe_with_physcap,PhysCaPEpisodeWrapper
from rebranch.api.primitives import PHYSCAP_EXTRA_PRIMITIVES
from rebranch.exec.sandbox import run_policy

def work(job):
 name,i,th,proxy=job;t=make_task(name);ep=t.make_episode(th,seed=48000+i);src=build_safe_probe_source(t)
 if proxy: ep=PhysCaPEpisodeWrapper(ep);src=_augment_probe_with_physcap(src)
 before={n:ep.api.get_object_pose(n) for n in ([t.probe_config[0]] if t.probe_config else [])}
 tr=run_policy(src,ep,th,seed=48000+i,extra_primitives=PHYSCAP_EXTRA_PRIMITIVES if proxy else ())
 after={n:ep.api.get_object_pose(n) for n in before}
 capability=run_policy(t.oracle_policy(th),ep,th,seed=48000+i,extra_primitives=PHYSCAP_EXTRA_PRIMITIVES if proxy else ())
 reference=run_policy(t.oracle_policy(th),t.make_episode(th,seed=48000+i),th,seed=48000+i)
 return dict(reference_without_probe=asdict(reference),task=name,index=i,proxy=proxy,theta=th,source=src,before=before,after=after,probe=asdict(tr),capability_after=asdict(capability))

def main():
 out=Path('outputs/repair_validation/stage1b/probes_validated');out.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('stage1_legacy_probe',out.parent/'originals/rebranch/baselines/__init__.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
 old=[];t=make_task('t3')
 for i,th in enumerate(t.hidden_test_set(seed=33001,n=5)):
  src=legacy.build_safe_probe_source(t.draft_policy);ep=t.make_episode(th,seed=i);before=ep.get_object_pose('can');tr=run_policy(src,ep,th,seed=i)
  old.append(dict(theta=th,source=src,before=before,after=ep.get_object_pose('can'),trace=asdict(tr)))
 (out/'legacy_t3.json').write_text(json.dumps(old,indent=2))
 jobs=[]
 for name in TASKS:
  t=make_task(name);cases=t.hidden_test_set(seed=33002,n=3)+t.ood_test_set(seed=33003,n=2)
  jobs += [(name,i,th,proxy) for i,th in enumerate(cases) for proxy in [False,True]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool: rows=list(pool.map(work,jobs))
 (out/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));summary={}
 for r in rows:
  k=r['task']+'/'+('proxy' if r['proxy'] else 'base');s=summary.setdefault(k,dict(n=0,probe_damage=0,probe_errors=0,probe_locked=0,subsequent_success=0,reference_without_probe_success=0,probe_induced_reference_failure=0));s['n']+=1;s['probe_damage']+=bool(r['probe']['safety_violation']);s['probe_errors']+=bool(r['probe']['error']);s['probe_locked']+=r['probe']['failure'] in ['rail_jammed','jammed_excessive_force','crushed','tipped'];s['reference_without_probe_success']+=r['reference_without_probe']['success'];s['probe_induced_reference_failure']+=r['reference_without_probe']['success'] and not r['capability_after']['success'];s['subsequent_success']+=r['capability_after']['success']
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 assert all(s['probe_damage']==s['probe_errors']==0 and s['probe_locked']==0 and s['probe_induced_reference_failure']==0 for s in summary.values())
if __name__=='__main__':main()
