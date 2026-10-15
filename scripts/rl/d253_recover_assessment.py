"""User-authorized bounded recovery; frozen D253 model/protocol. No AWS."""
import os,sys,time,json,hashlib,shutil,subprocess,socket,urllib.request,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/d253_state_critic'; OLD=OUT/'test_flights'; REC=OUT/'recovery'
PY=Path('D:/drone_vla_pilot/venv_qwen/Scripts/python.exe'); SIMPY=Path('D:/drone_vla_pilot/venv_uaveval/Scripts/python.exe')
EVAL=Path('D:/drone_vla_pilot/win_bundle/UAV-Flow-Eval'); ADAPTER=Path('D:/drone_vla_pilot/adapters/d207_s8000')
BASE='D:/drone_vla_pilot/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016190651cd'
BASE='D:/drone_vla_pilot/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'
DEADLINE=datetime.datetime.fromisoformat('2026-10-15T16:37:16+00:00')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def remaining():return (DEADLINE-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
def kill(p):
 if p is not None and p.poll() is None:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
def main():
 REC.mkdir(exist_ok=False);server=worker=None;start=time.monotonic()
 fit=json.loads((OUT/'fit.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
 assert sha(OUT/'critic.pt')==fit['critic_sha256'] and sha(OUT/'protocol.json')==fit['protocol_sha256']
 for n,h in fit['source_hashes'].items():assert sha(ROOT/n)==h,n
 original=OLD/'episodes/sim_2025-03-27_15-41-24_r0.json';original_hash=sha(original)
 save(REC/'start.json',dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_utc=DEADLINE.isoformat(),original_flight_sha256=original_hash,critic_sha256=fit['critic_sha256'],criteria_unchanged=True,actor_updates=0,aws_used=False,interruption_evidence='Windows System events 41 and 6008 record unclean reboot; precise run interruption time unknown',recovery_note='Preserve first complete flight; remaining fixed pairs in three simulator sessions with original scene seed201; no outcome-based replacement.'))
 env=os.environ.copy();env.update(UE_MAX_FPS='10',UAV_EVAL_OFFSCREEN='1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',QWEN_MODEL=BASE,PYTHONPATH=str(ROOT/'scripts')+';'+str(ROOT/'scripts/rl'))
 try:
  with socket.socket() as sock:sock.bind(('127.0.0.1',5011))
  with (REC/'server.log').open('w') as log:
   server=subprocess.Popen([str(PY),'-u',str(ROOT/'scripts/uav_flow_eval_server.py'),'--model','qwen','--path',str(ADAPTER),'--prefix-adapter',str(ADAPTER),'--precision','nf4','--chunk','8','--port','5011','--log',str(REC/'server_calls.jsonl'),'--rollout-temperature','1.0','--progress','--goal-memory','--numeric-goals','--box'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   for _ in range(150):
    if server.poll() is not None:raise RuntimeError('server exited')
    try:urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:5011/reset',data=b'',method='POST'),timeout=2).close();break
    except Exception:time.sleep(2)
   else:raise TimeoutError('server readiness')
   collection_deadline=min(time.monotonic()+2400,time.monotonic()+remaining()-1200)
   groups=[(protocol['test']['tasks'][0],1,1),(protocol['test']['tasks'][1],0,2),(protocol['test']['tasks'][2],0,2)]
   rows=json.loads((OLD/'episodes/summary.json').read_text());assert len(rows)==1
   for i,(task,first,repeats) in enumerate(groups):
    tasks=REC/f'tasks_{i}';tasks.mkdir();shutil.copyfile(OLD/'tasks'/task,tasks/task)
    dest=REC/f'episodes_{i}'
    with (REC/f'practice_{i}.log').open('w') as log2:
     worker=subprocess.Popen([str(SIMPY),'-u',str(ROOT/'scripts/rl/nonlanding_preflight.py'),'--tasks',str(tasks),'--out',str(dest),'--eval-root',str(EVAL),'--port','5011','--all-selected','--repeats',str(repeats),'--repeat-start',str(first),'--seed','201','--max-steps','128'],cwd=EVAL,env=env,stdout=log2,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
     save(REC/'status.json',dict(stage='collecting',group=i,completed=len(rows),server_pid=server.pid,worker_pid=worker.pid))
     rc=worker.wait(timeout=max(1,collection_deadline-time.monotonic()))
     if rc:raise RuntimeError(f'collection group{i} exit{rc}')
    done=json.loads((dest/'completion.json').read_text());assert done['episodes']==repeats
    image_dest=OLD/'rollout_images/recovery';image_dest.mkdir(exist_ok=True)
    for image in (REC/'rollout_images').glob('*.png'):shutil.copyfile(image,image_dest/image.name)
    for file in dest.glob('*_r?.json'):
     e=json.loads(file.read_text())
     def relocate(x):
      if isinstance(x,dict):
       if 'image' in x and isinstance(x['image'],str) and 'image_sha256' in x:
        assert sha(image_dest/x['image'])==x['image_sha256'];x['image']='recovery/'+x['image']
       for v in x.values():relocate(v)
      elif isinstance(x,list):
       for v in x:relocate(v)
     relocate(e);target=OLD/'episodes'/file.name;assert not target.exists();save(target,e)
    rows.extend(json.loads((dest/'summary.json').read_text()));save(OLD/'episodes/summary.json',rows)
   assert len(rows)==6 and sha(original)==original_hash
   save(OLD/'episodes/completion.json',dict(status='complete',episodes=6,recovered=True,actor_updates=0,elapsed_seconds=time.monotonic()-start))
  kill(server);server=None
  for script,logname in [('d253_evaluate_critic.py','evaluation.log'),('d253_verify_report.py','verification.log')]:
   save(REC/'status.json',dict(stage=script,completed=6))
   with (OUT/logname).open('w') as log:
    worker=subprocess.Popen([str(PY),'-u',str(ROOT/'scripts/rl'/script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    rc=worker.wait(timeout=max(1,min(900,remaining()-60)))
    if rc:raise RuntimeError(f'{script} exit{rc}')
  save(REC/'completion.json',dict(status='assessment_complete',elapsed_seconds=time.monotonic()-start,aws_used=False,actor_updates=0))
 except Exception as e:
  save(REC/'failure.json',dict(error=repr(e),elapsed_seconds=time.monotonic()-start,aws_used=False,actor_updates=0));raise
 finally:
  kill(worker);kill(server);save(REC/'cleanup.json',dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if __name__=='__main__':main()
