"""D228 bounded laptop-only no-learning numeric practice collection."""
import json,os,subprocess,sys,time,urllib.request,socket,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/d228_local_numeric_flights/finish'
EVAL=Path('D:/drone_vla_pilot/win_bundle/UAV-Flow-Eval')
ADAPTER=Path('D:/drone_vla_pilot/adapters/d207_s8000')
BASE='D:/drone_vla_pilot/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'
def main():
    OUT.mkdir(exist_ok=False)
    def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
    env=os.environ.copy();env.update(UE_MAX_FPS='10',UAV_EVAL_OFFSCREEN='1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',QWEN_MODEL=BASE,PYTHONPATH=str(ROOT/'scripts')+';'+str(ROOT/'scripts/rl'))
    server=worker=None;started=time.monotonic();deadline=started+1200
    try:
        with socket.socket() as sock: sock.bind(('127.0.0.1',5009))
        sys.path.insert(0,str(ROOT/'scripts/rl'))
        from composite_policy import adapter_digest
        digest=adapter_digest(ADAPTER)
        assert digest=='75d55cf748d084c463ccb26842c4266abecbc7e038b29960ee94fea887d1a711'
        tasks=OUT/'tasks';tasks.mkdir()
        src=ROOT/'reports/d224_reward_gated_ppo/tasks/sim_2025-03-27_19-11-33.json'
        shutil.copyfile(src,tasks/src.name)
        files=list(tasks.glob('*.json'));assert len(files)==1
        assert all(json.loads(f.read_text(encoding='utf-8-sig'))['motion_kind']!='Land' for f in files)
        save('manifest.json',dict(adapter_sha256=digest,precision='nf4',seeds=[7,8],scene_seed=7,max_steps=128,wall_limit_seconds=1200,training=False,tasks={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['scripts/uav_flow_eval_server.py','scripts/numeric_goals.py','scripts/rl/nonlanding_preflight.py','scripts/rl/movement_reward.py','scripts/rl/path_reward.py','scripts/rl/flight_collector.py']}))
        with (OUT/'server.log').open('w') as log:
            server=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/uav_flow_eval_server.py'),'--model','qwen','--path',str(ADAPTER),'--prefix-adapter',str(ADAPTER),'--precision','nf4','--chunk','8','--port','5009','--log',str(OUT/'server_calls.jsonl'),'--rollout-temperature','1.0','--progress','--goal-memory','--numeric-goals','--box'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save('status.json',dict(stage='loading_policy',server_pid=server.pid))
            for _ in range(150):
                if server.poll() is not None:raise RuntimeError('policy server exited during loading')
                try:
                    request=urllib.request.Request('http://127.0.0.1:5009/reset',data=b'',method='POST')
                    urllib.request.urlopen(request,timeout=2).close();break
                except Exception:time.sleep(2)
            else:raise TimeoutError('policy readiness limit')
            with (OUT/'practice.log').open('w') as output:
                worker=subprocess.Popen(['D:/drone_vla_pilot/venv_uaveval/Scripts/python.exe','-u',str(ROOT/'scripts/rl/nonlanding_preflight.py'),'--tasks',str(tasks),'--out',str(OUT/'episodes'),'--eval-root',str(EVAL),'--port','5009','--all-selected','--repeats','2','--seed','7','--max-steps','128'],cwd=EVAL,env=env,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                save('status.json',dict(stage='collecting',server_pid=server.pid,worker_pid=worker.pid))
                status=worker.wait(timeout=max(1,deadline-time.monotonic()))
                if status:raise RuntimeError('collector exited '+str(status))
        save('completion.json',dict(status='collected',elapsed_seconds=time.monotonic()-started,training_started=False))
    except Exception as e:
        save('failure.json',dict(error=repr(e),elapsed_seconds=time.monotonic()-started,training_started=False));raise
    finally:
        for proc in (worker,server):
            if proc is not None and proc.poll() is None:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
        save('cleanup.json',dict(server_exit=None if server is None else server.poll(),worker_exit=None if worker is None else worker.poll(),training_started=False))
if __name__=='__main__':main()
