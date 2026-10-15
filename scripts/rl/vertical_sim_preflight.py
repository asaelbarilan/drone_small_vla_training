"""D222 scripted simulator plumbing checks, not model evaluation or training."""
import argparse,json,sys,time,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from simulator_transport import UnrealReadback
from flight_collector import collect_episode
from vertical_tasks import vertical_reward
def main():
    p=argparse.ArgumentParser();p.add_argument('--eval-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.eval_root));import batch_run_act_all as official
    def save(name,x):(a.out/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
    env=None;rows=[];start=time.monotonic()
    try:
        env=official.gym.make('UnrealTrack-DowntownWest-ContinuousColor-v0')
        env=official.time_dilation.TimeDilationWrapper(env,10)
        env.unwrapped.agents_category=['drone']
        env=official.configUE.ConfigUEWrapper(env,resolution=(256,256),offscreen=True)
        env=official.augmentation.RandomPopulationWrapper(env,2,2,random_target=False)
        env.seed(7);env.reset();api=env.unwrapped.unrealcv;actor=env.unwrapped.player_list[0]
        api.set_viewport(actor);api.client.request('vrun t.MaxFPS 10');api.set_phy(actor,0)
        for f in sorted((ROOT/'reports/d221_vertical_landing/tasks').glob('*.json')):
            if time.monotonic()-start>240:raise TimeoutError('local probe budget reached')
            task=json.loads(f.read_text());initial=task['initial_pos']
            api.set_obj_location(actor,initial[:3]);reset=initial[4]-180
            api.set_rotation(actor,reset);time.sleep(2)
            reset-=(api.get_obj_rotation(actor)[1]-initial[4]+180)%360-180
            api.set_rotation(actor,reset);time.sleep(1)
            if abs((api.get_obj_rotation(actor)[1]-initial[4]+180)%360-180)>.005:
                raise ValueError('reset heading calibration failed')
            measured=UnrealReadback(api,actor,initial,lambda:official.set_cam(env),initial_command_yaw=reset)
            ref=np.asarray(task['reference_path_preprocessed'])
            commands=np.vstack([ref[1:],np.repeat(ref[-1:],12,axis=0)])
            cursor=0
            def policy(obs):
                nonlocal cursor
                part=commands[cursor:cursor+8];cursor+=len(part)
                if not len(part):raise ValueError('scripted stop not detected')
                return dict(proprio=[*obs['pose'][:3],obs['pose'][4]],
                            poses=[[float(r[0]),float(r[1]),float(r[2]),float(np.deg2rad(r[4]))] for r in part],
                            task=f.name,scripted_control=True)
            result=collect_episode(measured,policy,lambda o:{'id':o['id']},ref.tolist(),
                                   max_steps=len(commands),reward_kind='Ascend/Descend')
            verified=vertical_reward(result['poses'],task,stopped=result['end_reason']=='model_stop')
            err=float(np.linalg.norm(np.asarray(result['poses'][-1])[:3]-ref[-1,:3]))
            row=dict(task=f.name,scripted=True,end_reason=result['end_reason'],endpoint_error_cm=err,
                     reward=verified['reward'],complete=verified['complete'],
                     transition_sum_error=abs(sum(s['reward_v2'] for s in result['samples'])-verified['reward']),
                     training_samples=False)
            save(f.stem+'.json',dict(summary=row,poses=result['poses'],executed_counts=result['executed_counts']))
            if result['end_reason']!='model_stop' or not verified['complete'] or err>.05 or row['transition_sum_error']>1e-8:
                raise ValueError('measured endpoint/stop/reward mismatch: '+f.name)
            rows.append(row);save('progress.json',rows);print(json.dumps(row),flush=True)
        save('completion.json',dict(status='passed',tasks=rows,elapsed_seconds=time.monotonic()-start,
             actual_model_flown=False,training_started=False,landing_contact_verified=False))
    except Exception as e:
        save('failure.json',dict(error=repr(e),completed=rows));raise
    finally:
        if env is not None:env.close()
if __name__=='__main__':main()
