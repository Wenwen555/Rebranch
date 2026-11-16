"""Bounded offline stage1b verification, preserve exact commands and exits."""
import json,os,subprocess,sys,time
from pathlib import Path
out=Path('outputs/repair_validation/stage1b');env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1')
commands=[['-m','pytest','tests/test_foundation.py','tests/test_m0_core.py','tests/test_baselines_eval.py','tests/test_repair_protocol.py','tests/test_stage1b.py','-k','not codex_cli_smoke','-q'],['scripts/repair_stage1b_probes.py']]
records=[]
for i,args in enumerate(commands):
 command=[sys.executable,*args];log=out/f'final_check_{i}.log';start=time.time()
 with log.open('w') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append(dict(command=command,exit_code=r.returncode,wall_seconds=time.time()-start,log=str(log)))
 (out/'verification_commands.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
if any(r['exit_code'] for r in records):sys.exit(1)
