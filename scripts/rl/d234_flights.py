"""D234 mixed nonlanding instruction-routed flights; no learning."""
import argparse,json,os,subprocess,sys,time,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kinds',nargs='+',default=['Approach/Move','Turn','Pass','Surround']);p.add_argument('--max-steps',type=int,choices=range(1,321));p.add_argument('--repeats',type=int,choices=(1,2),default=2);p.add_argument('--repeat-start',type=int,choices=(0,1),default=0);p.add_argument('--seed',type=int);p.add_argument('--selection-index',type=int,choices=(0,1),default=0);p.add_argument('--action-adapter',type=Path,default=Path('C:/adapters/d207_stage2_adapter_s8000'));p.add_argument('--probe-only',action='store_true');p.add_argument('--tasks',type=Path,default=Path(__file__).resolve().parents[2]/'reports/d234_full_ppo_preparation/tasks');p.add_argument('--validation-only',action='store_true');p.add_argument('--wall-seconds',type=int,default=4800);a=p.parse_args()
    root=a.root;root.mkdir(parents=True,exist_ok=False);code=Path(__file__).resolve().parents[2]
    env=os.environ.copy();env.update(UE_MAX_FPS='10',UAV_EVAL_OFFSCREEN='1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',QWEN_MODEL='C:/models/qwen3vl4b',PYTHONPATH=str(code/'scripts')+';'+str(code/'scripts/rl'))
    server=None;worker=None;start=time.monotonic()
    try:
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen([sys.executable,'-u',str(code/'scripts/uav_flow_eval_server.py'),'--model','qwen','--path',str(a.action_adapter),'--prefix-adapter','C:/adapters/d207_stage2_adapter_s8000','--precision','bf16','--chunk','8','--port','5008','--log',str(root/'server_calls.jsonl'),'--rollout-temperature','1.0','--progress','--goal-memory','--auto-numeric-goals','--box'],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            for _ in range(180):
                if server.poll() is not None:raise RuntimeError('policy server exited during loading')
                try:
                    request=urllib.request.Request('http://127.0.0.1:5008/reset',data=b'',method='POST')
                    urllib.request.urlopen(request,timeout=2).close();break
                except Exception:time.sleep(2)
            else:raise TimeoutError('server readiness limit')
            with (root/'practice.log').open('w') as output:
                worker=subprocess.Popen(['C:/venvs/eval/Scripts/python.exe','-u',str(code/'scripts/rl/nonlanding_preflight.py'),'--tasks',str(a.tasks),'--out',str(root/'episodes'),'--eval-root','C:/win_bundle/UAV-Flow-Eval',*(['--kinds','Approach/Move','--max-steps','2'] if a.probe_only else ['--all-selected']),*(['--validation-only'] if a.validation_only else []),'--wall-seconds',str(a.wall_seconds),'--repeats',str(a.repeats),'--repeat-start',str(a.repeat_start),'--selection-index',str(a.selection_index),*(['--seed',str(a.seed)] if a.seed is not None else []),*(['--max-steps',str(a.max_steps)] if a.max_steps else [])],cwd='C:/win_bundle/UAV-Flow-Eval',env=env,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                status=worker.wait(timeout=a.wall_seconds+60)
            if status:raise RuntimeError('practice collector exited '+str(status))
        (root/'completion.json').write_text(json.dumps(dict(status='complete',elapsed_seconds=time.monotonic()-start,training_started=False)))
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=str(e),elapsed_seconds=time.monotonic()-start)));raise
    finally:
        for proc in (worker,server):
            if proc is not None and proc.poll() is None:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
if __name__=='__main__':main()
