"""D249: saved model-action reward/PPO-direction check; no VLA/cloud execution."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from checkpoint_state import verified_checkpoint,sha256
from executed_rollouts import attach_path_rewards
from rollout_samples import ppo_advantages
from ppo_engine import prepare_advantages
from ppo_math import joint_ppo_loss

def terminal_direction(sample,value):
    if not sample['terminated'] or sample['truncated']:raise ValueError('exact terminal return required')
    # True sampled token logps, ratio=1. Independent logp perturbation checks the
    # surrogate derivative only, NOT a VLA weight update or policy probability.
    old=torch.tensor([sample['logps']],dtype=torch.float64)
    new=old.clone().requires_grad_(True);a=float(sample['reward_v2']-value)
    loss,_=joint_ppo_loss(new,old,torch.tensor([a],dtype=torch.float64));loss.backward()
    gradient=-new.grad.detach().numpy()
    assert np.isfinite(gradient).all()
    return dict(value_baseline=value,raw_terminal_advantage=a,selected_logp_descent_direction=float(gradient.mean()))

def run():
    checkpoint=verified_checkpoint(ROOT/'reports/d239_kl_diagnosis/checkpoints')
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
    stored=state['metadata']['samples'];batch=[];rows=[];sources={};terminal_samples=[];sample_count=0
    groups=[('D238_BF16',ROOT/'reports/d238_two_task/evidence/round_1/attempts/episodes'),('D228_NF4_finish',ROOT/'reports/d228_local_numeric_flights/finish/episodes'),('D228_NF4_retry1',ROOT/'reports/d228_local_numeric_flights/retry1/episodes')]
    for group,folder in groups:
        for path in sorted(folder.glob('*_r?.json')):
            episode=json.loads(path.read_text(encoding='utf-8'));sources[str(path.relative_to(ROOT))]=sha256(path)
            if group.startswith('D238'):
                taskpath=ROOT/'reports/d238_two_task/evidence/round_1/tasks'/episode['task']
                calls=[s for s in stored if s['task']==episode['task'] and s['passage']==episode['repeat']]
                for a,b in zip(calls,episode['samples'],strict=True):
                    assert all(a[k]==b[k] for k in ['ids','logps','context_input_ids','image_sha256','executed_count'])
            else:
                taskpath=ROOT/'reports/d224_reward_gated_ppo/tasks'/episode['task']
                manifest=json.loads((folder.parent/'manifest.json').read_text(encoding='utf-8'))
                assert sha256(taskpath)==manifest['tasks'][episode['task']]
                calls=episode['samples']
            task=json.loads(taskpath.read_text(encoding='utf-8'));sources[str(taskpath.relative_to(ROOT))]=sha256(taskpath)
            assert [s['call_index'] for s in calls]==list(range(len(calls)))
            for s in calls:
                assert len(s['ids'])==len(s['logps'])==32 and np.isfinite(s['logps']).all()
            updated,score=attach_path_rewards(calls,episode['poses'],episode['executed_counts'],task['reference_path_preprocessed'],kind=episode['kind'],end_reason=episode['end_reason'],final_observation=episode.get('final_observation'),stopping_mode='accepted_goal')
            assert not any(s['reward_spec_approved'] for s in updated)
            assert abs(sum(s['reward_v2'] for s in updated)-score['reward'])<1e-9
            sample_count+=len(updated)
            row=dict(group=group,task=episode['task'],repeat=episode['repeat'],calls=len(updated),end=episode['end_reason'],complete=score['complete'],endpoint_m=score['endpoint_position_error_m'],heading_deg=score['endpoint_yaw_error_deg'],last_call_reward=updated[-1]['reward_v2'],new_total_reward=score['reward'],first_accepted_pose=score['first_accepted_pose'])
            if updated[-1]['terminated']:
                label='accepted_stop' if score['complete'] else ('premature_position_stop' if score['endpoint_position_error_m']>.5 else 'other_failed_stop')
                row['label']=label;row['baseline_controls']=[terminal_direction(updated[-1],v) for v in [-.5,0,.5,.9]]
                for c in row['baseline_controls']:assert (c['selected_logp_descent_direction']>0)==score['complete']
                terminal_samples.append((updated[-1],score['complete']))
            else:row['label']='timeout_no_terminal_direction_claim'
            if group.startswith('D238'):batch.extend(updated)
            rows.append(row)
    # Diagnostic sensitivity only: old values belong to a DIFFERENT reward.
    ppo_advantages(batch,1,.95)
    centered=prepare_advantages([s['gae'] for s in batch]);scale=prepare_advantages([s['gae'] for s in batch],'scale_only')
    sensitivity=[dict(task=s['task'],passage=s['passage'],last_reward=s['reward_v2'],stale_value=s['value_old'],raw_gae=s['gae'],centered=float(a),scale_only=float(b)) for s,a,b in zip(batch,centered,scale) if s['terminated']]
    good=[s['reward_v2'] for s,success in terminal_samples if success];bad=[s['reward_v2'] for s,success in terminal_samples if not success]
    lower=max(bad);upper=min(good)
    assert lower<upper and all(r['raw_gae']>0 and r['centered']>0 and r['scale_only']>0 for r in sensitivity)
    return dict(source_sha256=sha256(Path(__file__)),sources=sources,checkpoint_state_sha256=sha256(checkpoint/'state.pt'),episodes=rows,
        counts=dict(episodes=len(rows),sampled_chunks=sample_count,accepted_stops=len(good),failed_stops=len(bad),timeouts=sum(r['end']=='horizon' for r in rows)),
        terminal_baseline_interval=dict(lower_exclusive=lower,upper_exclusive=upper,meaning='If each terminal state value lies within these bounds, raw stopping advantages have the desired sign for every recorded terminal chunk. Not a measured/calibrated new critic.'),
        d238_stale_critic_sensitivity=sensitivity,
        verdict=dict(recorded_reward_and_surrogate_direction_passed=True,new_critic_calibrated=False,actual_weight_update_tested=False,paid_training_ready=False),
        limitations=['Two distinct historical policies/precisions remain separate; no combined PPO normalization or batch admission.', 'Saved action log probabilities are used only for a ratio-one surrogate derivative. No model was loaded or its parameter gradient computed.', 'New reward requires fresh value fitting and advantage computation. Stored old values are sensitivity evidence, not valid new targets.', 'Timeouts retain explicit censoring; no fabricated terminal success or counterfactual tokens.'])
if __name__=='__main__':
    out=ROOT/'reports/d249_real_action_stopping';out.mkdir(exist_ok=True);result=run()
    (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['counts','terminal_baseline_interval','d238_stale_critic_sensitivity','verdict']},indent=2))