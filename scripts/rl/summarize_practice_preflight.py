"""Independently verify and summarize completed D216 practice evidence."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from executed_rollouts import attach_path_rewards
from path_reward import path_reward,PathRewardConfig

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);episodes=a.out/'episodes';episodes.mkdir(exist_ok=True)
    rows=[];keys=set();manifest=[];batch=[];image_root=a.out/'ppo_batch/images';image_root.mkdir(parents=True,exist_ok=True)
    for root in a.runs:
        for f in sorted((root/'episodes').glob('sim_*_r*.json')):
            ep=json.loads(f.read_text(encoding='utf-8'));key=(ep['task'],ep['repeat'])
            if key in keys:raise ValueError('duplicate episode')
            keys.add(key);task_path=a.tasks/ep['task'];task=json.loads(task_path.read_text(encoding='utf-8-sig'))
            legacy_config=dict(ep['reward']['config']);legacy_config.setdefault('endpoint_yaw_tolerance_deg',180.)
            _,reward=attach_path_rewards(ep['samples'],ep['poses'],ep['executed_counts'],task['reference_path_preprocessed'],end_reason=ep['end_reason'],final_observation=ep['final_observation'],config=PathRewardConfig(**legacy_config))
            np.testing.assert_allclose(reward['transition_rewards'],ep['reward']['transition_rewards'],atol=1e-12,rtol=0)
            if reward['complete']!=ep['reward']['complete']:raise ValueError('completion mismatch')
            for sample in ep['samples']:
                image=root/'rollout_images'/sample['image']
                if hashlib.sha256(image.read_bytes()).hexdigest()!=sample['image_sha256']:raise ValueError('image hash mismatch')
            legacy_reward=reward
            new_samples,reward=attach_path_rewards(ep['samples'],ep['poses'],ep['executed_counts'],task['reference_path_preprocessed'],end_reason=ep['end_reason'],final_observation=ep['final_observation'])
            for sample in new_samples:
                relative=Path(root.name)/sample['image'];dest=image_root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/'rollout_images'/sample['image'],dest);sample['image']=relative.as_posix();batch.append(sample)
            ep['rescored_reward_v2']=reward
            yaw=np.rad2deg(np.unwrap(np.deg2rad(np.asarray(ep['poses'])[:,4])))
            row={k:ep[k] for k in ('task','kind','repeat','elapsed_seconds','end_reason')}
            row.update(steps=sum(ep['executed_counts']),calls=len(ep['samples']),reward=reward['reward'],progress=reward['progress_fraction'],complete=reward['complete'],endpoint_error=reward['endpoint_error'],max_deviation=reward['max_deviation'],invalid_transitions=reward['invalid_transitions'],net_yaw_degrees=float(yaw[-1]-yaw[0]),spatial_endpoint_error_m=float(np.linalg.norm(np.asarray(ep['poses'])[-1,:3]-np.asarray(task['reference_path_preprocessed'])[-1,:3])/100),heading_feedback_corrections=len(ep['rotation_corrections']) if 'rotation_corrections' in ep else None)
            row['legacy_reward']=legacy_reward['reward'];row['legacy_complete']=legacy_reward['complete'];row['endpoint_yaw_error_deg']=reward['endpoint_yaw_error_deg']
            rows.append(row);(episodes/f.name).write_text(json.dumps(ep,indent=2))
            manifest.append(dict(episode=str(f),episode_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),task=task_path.name,task_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest()))
    if len(rows)!=8 or any(sum(x['kind']==kind for x in rows)!=2 for kind in ('Approach/Move','Turn','Pass','Surround')):raise ValueError('expected two episodes per selected kind')
    summary=dict(status='verified',episodes=rows,training_started=False,benchmark_tasks_used=False,comparison_to_legacy_validation_valid=False)
    artifact=dict(schema='executed-path-ppo-v1',preflight_passed=False,status='prepared_not_launch_ready',pending=['fixed-seed development runner verification','bounded pilot launch decision'],chunk=8,reward_version='path-reward-candidate-v2',samples=batch)
    (a.out/'ppo_batch/rollouts.json').write_text(json.dumps(artifact,indent=2))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
