"""No-update audit of actual practice image/context/action replay."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from train_path_ppo import make_forward
from ppo_math import replay_parity
from composite_policy import CompositePolicy,runtime_identity,adapter_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--probe-run',type=Path);p.add_argument('--expected-episodes',type=int,default=8);p.add_argument('--precision',choices=['bf16','nf4'],default='bf16');p.add_argument('--adapter',type=Path,default=Path('C:/adapters/d207_stage2_adapter_s8000'));p.add_argument('--prefix-adapter',type=Path);a=p.parse_args()
    from train_uav_flow_vla import setup,official_stats
    from uav_flow_action_tokenizer import ActionTokenizer
    start=time.monotonic();report=dict(status='running',training_started=False,records=[])
    try:
        api,processor,base,_=setup('qwen',a.precision)
        adapter=a.adapter
        prefix=a.prefix_adapter or adapter
        policy=CompositePolicy.load(base,adapter,prefix,train_action=False)
        tokenizer=ActionTokenizer(processor.tokenizer,official_stats())
        runtime=runtime_identity(policy.model,processor,official_stats())
        identity=dict(action_adapter_sha256=adapter_digest(adapter),prefix_adapter_sha256=adapter_digest(prefix),precision=a.precision,**runtime)
        paths=[(run,path) for run in a.run for path in sorted((run/'episodes').glob('sim_*_r*.json'))]
        if len(paths)!=a.expected_episodes:raise ValueError('unexpected completed practice episode count')
        if a.probe_run:
            probes=list((a.probe_run/'episodes').glob('sim_*_r*.json'))
            if len(probes)!=1 or not (a.probe_run/'completion.json').exists():raise ValueError('horizon probe incomplete')
            probe=json.loads(probes[0].read_text())
            if probe['end_reason']!='horizon' or sum(probe['executed_counts'])!=2:raise ValueError('horizon probe must execute exactly two moves')
            paths.extend((a.probe_run,p) for p in probes)
        keys=set();report['bootstrap_records']=[]
        for run,path in paths:
            forward,bootstrap=make_forward(api,processor,tokenizer,policy,run/'rollout_images',8)
            ep=json.loads(path.read_text());samples=ep['samples']
            key=('probe' if run==a.probe_run else 'practice',ep['task'],ep['repeat'])
            if key in keys:raise ValueError('duplicate task/repeat evidence')
            keys.add(key)
            selected=range(len(samples))
            for i in selected:
                sample=samples[i]
                if any(sample['policy_provenance'].get(k)!=v for k,v in identity.items()):raise ValueError('live policy identity mismatch')
                call_start=time.monotonic()
                with torch.no_grad():lp,_=forward([sample])
                result=replay_parity([sample['logps']],lp.cpu().tolist())
                report['records'].append(dict(task=ep['task'],repeat=ep['repeat'],call=i,elapsed_seconds=time.monotonic()-call_start,**result))
                print(json.dumps(report['records'][-1]),flush=True)
            if samples[-1]['truncated']:
                with torch.no_grad():features=bootstrap(samples[-1])
                if not torch.isfinite(features).all():raise ValueError('nonfinite final-observation features')
                report['bootstrap_records'].append(dict(task=ep['task'],repeat=ep['repeat'],shape=list(features.shape)))
        if len(report['records'])<8:raise ValueError('incomplete practice evidence')
        if not report['bootstrap_records']:raise ValueError('real horizon bootstrap was not exercised')
        report['status']='passed'
    except Exception as e:
        import traceback
        report.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        a.out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
