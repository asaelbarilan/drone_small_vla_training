"""D215 single-owner, deadline-bounded subprocess stages. No retries/cloud API.

The cloud wrapper must independently set shutdown before this runner starts and
verify stop afterwards. This runner never treats process startup as completion.
"""
import json,os,subprocess,time,uuid
from pathlib import Path


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def run_stages(stages,root,*,wall_seconds,reserve_seconds=30,env=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if wall_seconds<=reserve_seconds or reserve_seconds<0:raise ValueError('invalid wall budget/reserve')
    if not stages or any(not s.get('argv') or s.get('timeout_seconds',0)<=0 for s in stages):raise ValueError('bounded stages required')
    if (root/'completion.json').exists() or (root/'failure.json').exists():
        raise ValueError('run directory already has final outcome')
    lock=root/'RUNNING.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.monotonic();done=[]
    try:
        for index,stage in enumerate(stages):
            remaining=wall_seconds-reserve_seconds-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('save/shutdown reserve reached')
            timeout=min(remaining,stage['timeout_seconds'])
            atomic_json(root/'status.json',dict(stage=index,name=stage['name'],state='running',completed=done))
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            with (root/f'{index:02d}.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(stage['argv'],cwd=stage.get('cwd'),env=env,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=flags,start_new_session=os.name!='nt')
                try:code=process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        import signal
                        os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=10)
                    raise TimeoutError('stage exceeded deadline: '+stage['name'])
            if code!=0:raise RuntimeError(f"stage {stage['name']} exited {code}")
            required=stage.get('required_outputs',[])
            if any(not Path(p).is_file() or Path(p).stat().st_size==0 for p in required):
                raise ValueError('stage missing required output: '+stage['name'])
            done.append(stage['name'])
        result=dict(state='complete',completed=done,elapsed_seconds=time.monotonic()-started)
        atomic_json(root/'completion.json',result);return result
    except Exception as e:
        atomic_json(root/'failure.json',dict(state='failed',error=str(e),completed=done,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        lock.unlink(missing_ok=True)
