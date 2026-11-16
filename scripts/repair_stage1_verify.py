"""Record exact commands and exit statuses for the frozen stage-1 verification."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path('outputs/repair_validation/stage1');env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1')
commands=[
 ['-m','pytest','tests/test_foundation.py','tests/test_m0_core.py','tests/test_baselines_eval.py','tests/test_repair_protocol.py','-k','not codex_cli_smoke','-q'],
 ['-m','pytest','tests/test_piperx_sim.py::test_rollout_speed_determinism_snapshot_and_callbacks','tests/test_tasks_m3.py::test_t2_cup_crush_hook_and_proprioceptive_separability','-q'],
 ['scripts/repair_stage1_mass_range.py','mass_range_final'],
]
records=[]
for i,args in enumerate(commands):
 log=root/f'final_check_{i}.log';start=time.time();command=[sys.executable,*args]
 with log.open('w') as f: r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append(dict(command=command,exit_code=r.returncode,wall_seconds=time.time()-start,log=str(log)))
 (root/'verification_commands_final.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
if any(r['exit_code'] for r in records):sys.exit(1)
