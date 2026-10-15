"""D236 verification only: no optimizer, collection, or training driver."""
import argparse,subprocess,sys,json,datetime,traceback,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--deadline',required=True);a=p.parse_args()
r=a.root;r.mkdir(parents=True,exist_ok=True)
if (r/'started.json').exists():raise RuntimeError('Already started; no retry')
start=time.monotonic();deadline=datetime.datetime.fromisoformat(a.deadline.replace('Z','+00:00'))
def save(name,value):(r/name).write_text(json.dumps(value,indent=2))
save('started.json',dict(deadline=a.deadline,scope='full BF16 reload verification only',training_updates=0))
try:
 budget=min(1200,int((deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds())-120)
 if budget<180:raise RuntimeError('Insufficient verification time remaining')
 args=[sys.executable,'-u',str(Path(__file__).with_name('audit_d232_fixed_inputs.py')),'--precision','bf16','--baseline','C:/adapters/d207_stage2_adapter_s8000','--checkpoint','C:/runs/d234_early_review/diagnostic_checkpoint','--evidence','C:/runs/d234_early_review/diagnostic_evidence','--output',str(r/'fixed_inputs')]
 with (r/'audit.log').open('w') as log:proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=budget)
 if proc.returncode:raise RuntimeError('Full-model diagnostic exited '+str(proc.returncode))
 report=json.loads((r/'fixed_inputs/same_input.json').read_text())
 if report.get('passed') is not True or len(report.get('cases',[]))!=8:raise RuntimeError('Missing eight-case passing report')
 save('completion.json',dict(passed=True,scope='verification only',training_updates=0,cases=8,elapsed_seconds=time.monotonic()-start))
except Exception as e:
 save('failure.json',dict(passed=False,error=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start,training_updates=0));raise
