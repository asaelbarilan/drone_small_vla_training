"""D217 bounded-pilot gates and matched practice comparison. No cloud API."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))

def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assets','seed','admit','after','compare']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    if a.stage=='assets':
        manifest=ROOT/'reports/uav_flow_official_10shard/manifest.json'
        stats=json.loads(manifest.read_text(encoding='utf-8'))['action_stats_train']
        digest=hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()
        data=json.loads((r/'batch/rollouts.json').read_text())
        for sample in data['samples']:
            if sample['policy_provenance']['action_stats_sha256']!=digest:raise ValueError('normalization manifest differs from collection')
        required=[ROOT/'reports/d210_dagger_audit/practice/rollout_images/ep00064_c000.png']
        for path in required:
            if not path.is_file():raise FileNotFoundError(path)
        tasks=list((ROOT/'reports/d210_dagger_audit/practice/tasks').glob('*.json'))
        if len(tasks)!=100:raise ValueError('expected complete100-task practice inventory')
        write(r/'assets_gate.json',dict(status='passed',action_stats_sha256=digest,practice_tasks=len(tasks)))
    elif a.stage=='seed':
        import torch
        from PIL import Image
        from uav_flow_eval_server import qwen_loader,STATE,app
        STATE.update(episode=0,call=0,goal=None,yaw_total=0.,last_yaw=None)
        adapter='C:/adapters/d207_stage2_adapter_s8000'
        predictor=qwen_loader(adapter,8,'bf16',temperature=1.,progress=True,goal_memory=True,box=True,prefix_adapter=adapter)
        practice=ROOT/'reports/d210_dagger_audit/practice'
        task=json.loads((practice/'tasks/sim_2025-04-30_15-37-22.json').read_text())
        image=Image.open(practice/'rollout_images/ep00064_c000.png').convert('RGB');records=[]
        client=app.test_client()
        for _ in range(2):
            response=client.post('/reset',json={'seed':7});assert response.status_code==200 and response.json['seed']==7
            predictor(image,[0,0,0,0],task['instruction']);records.append(dict(STATE.pop('sample')))
        for key in ['ids','logps','context_input_ids','box_text','progress_text']:
            if records[0][key]!=records[1][key]:raise ValueError('fixed-seed sampling differs: '+key)
        write(r/'seed_gate.json',dict(status='passed',seed=7,same_saved_input=True,checks=['ids','logps','context_input_ids','box_text','progress_text'],simulator_bitwise_determinism_claimed=False))
    elif a.stage=='admit':
        from ppo_math import validate_ppo_samples
        source=r/'batch/rollouts.json';data=json.loads(source.read_text());before=json.loads((r/'before/episodes/summary.json').read_text())
        if len(before)!=4 or len({x['task'] for x in before})!=4:raise ValueError('four development flights required')
        training_tasks={s['task'] for s in data['samples']}
        if training_tasks & {x['task'] for x in before}:raise ValueError('development/train overlap')
        if json.loads((r/'seed_gate.json').read_text())['status']!='passed':raise ValueError('seed gate missing')
        for f in (r/'before/episodes').glob('sim_*_r*.json'):
            ep=json.loads(f.read_text())
            if ep['seed']!=7 or ep['reward']['invalid_transitions']:raise ValueError('invalid development flight')
        if len(data['samples'])!=72:raise ValueError('unexpected training batch size')
        for s in data['samples']:
            path=(r/'batch/images'/s['image']).resolve()
            if not path.is_relative_to((r/'batch/images').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=s['image_sha256']:raise ValueError('image integrity')
            s['reward_spec_approved']=True
        if validate_ppo_samples(data['samples'],8,require_final_value=False)!=8:raise ValueError('expected eight training flights')
        data.update(preflight_passed=True,status='admitted_D217',pending=[],admission=dict(user_authorized=True,seed_gate='seed_gate.json',baseline='before/episodes/summary.json',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='one round at most four updates'))
        write(r/'admitted_rollouts.json',data);write(r/'admission.json',data['admission'])
    elif a.stage=='after':
        from checkpoint_state import verified_checkpoint
        report=json.loads((r/'training/report.json').read_text())
        if not 1<=report['step']<=4 or not report['frozen_adapters_unchanged'] or not report['reload_logps_exact']:raise ValueError('training verification failed')
        action=verified_checkpoint(r/'training/checkpoints')/'adapter/action'
        subprocess.run([sys.executable,str(ROOT/'scripts/rl/practice_preflight_driver.py'),'--root',str(r/'after'),'--repeats','1','--seed','7','--selection-index','1','--action-adapter',str(action)],check=True)
    else:
        from path_reward import path_reward
        before={};after={}
        for name,dest in [('before',before),('after',after)]:
            for f in (r/name/'episodes').glob('sim_*_r*.json'):
                ep=json.loads(f.read_text());task=json.loads((ROOT/'reports/d210_dagger_audit/practice/tasks'/ep['task']).read_text(encoding='utf-8-sig'))
                if ep['seed']!=7:raise ValueError('unmatched seed')
                if ep['reward']['invalid_transitions']:raise ValueError('invalid measured transition')
                dest[ep['task']]={k:ep[k] for k in ['kind','seed','end_reason']}
                dest[ep['task']].update(steps=sum(ep['executed_counts']),reward=ep['reward']['reward'],complete=ep['reward']['complete'],progress=ep['reward']['progress_fraction'],endpoint_error=ep['reward']['endpoint_error'],endpoint_yaw_error_deg=ep['reward']['endpoint_yaw_error_deg'])
        if before.keys()!=after.keys() or len(before)!=4:raise ValueError('unmatched development tasks')
        rows=[dict(task=k,before=before[k],after=after[k],reward_delta=after[k]['reward']-before[k]['reward']) for k in before]
        write(r/'comparison.json',dict(scope='four matched practice development flights, not benchmark',seed=7,rows=rows,before_complete=sum(x['complete'] for x in before.values()),after_complete=sum(x['complete'] for x in after.values()),training=json.loads((r/'training/report.json').read_text()),automatic_promotion=False))
if __name__=='__main__':main()
