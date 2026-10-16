import json,os,socket,subprocess,sys,time,traceback
from pathlib import Path
import requests
root=Path('C:/Users/Asael/PycharmProjects/drone_small_vla_training');out=root/'reports/d257_hf_release';package=Path('D:/drone_vla_pilot/release/hf_fresh_d257/model')
def save(n,x):(out/n).write_text(json.dumps(x,indent=2))
assert json.loads((out/'hf_download_verification.json').read_text())['status']=='fresh_download_hash_verified'
env=os.environ.copy();env.update(UE_MAX_FPS='10',UAV_EVAL_OFFSCREEN='1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH='')
server=worker=None;start=time.monotonic()
try:
 with socket.socket() as sock:sock.bind(('127.0.0.1',5017))
 cmd=[sys.executable,'-u',str(package/'serve.py'),'--precision','nf4','--port','5017','--log',str(out/'server_calls.jsonl')]
 with (out/'server.log').open('w') as log:
  server=subprocess.Popen(cmd,cwd=package,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);save('status.json',dict(stage='loading_fresh_package',server_pid=server.pid,command=cmd))
  while time.monotonic()-start<300:
   if server.poll() is not None:raise RuntimeError('Package server exited')
   try:
    r=requests.post('http://127.0.0.1:5017/reset',json={'seed':7},timeout=2);r.raise_for_status();break
   except requests.RequestException:time.sleep(2)
  else:raise TimeoutError('Package readiness cap')
  with (out/'simulator.log').open('w') as log2:
   worker=subprocess.Popen(['D:/drone_vla_pilot/venv_uaveval/Scripts/python.exe','-u',str(root/'scripts/d257_sim_smoke.py'),'--package',str(package),'--out',str(out/'flights'),'--tasks',str(root/'reports/d238_two_task/tasks/development')],cwd='D:/drone_vla_pilot/win_bundle/UAV-Flow-Eval',env=env,stdout=log2,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   save('status.json',dict(stage='simulator_smoke',server_pid=server.pid,worker_pid=worker.pid));code=worker.wait(timeout=max(1,1800-(time.monotonic()-start)));assert code==0,f'simulator exit {code}'
 save('completion.json',dict(elapsed_seconds=time.monotonic()-start,training=False,aws=False))
except Exception:save('failure.json',dict(traceback=traceback.format_exc()));raise
finally:
 for proc in (worker,server):
  if proc is not None and proc.poll() is None:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW);proc.wait(timeout=20)
 save('cleanup.json',dict(server_exit=None if server is None else server.poll(),worker_exit=None if worker is None else worker.poll()))
