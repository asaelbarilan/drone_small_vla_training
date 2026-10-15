"""D216 small practice-only measured collection check. No optimizer/training."""
import argparse,json,time,traceback,sys
from pathlib import Path
import numpy as np
from simulator_transport import UnrealReadback,PolicyHTTP
from flight_collector import collect_episode

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--eval-root',type=Path,required=True);p.add_argument('--port',type=int,default=5008);p.add_argument('--probe',action='store_true');p.add_argument('--movement-probe',action='store_true');p.add_argument('--kinds',nargs='+',default=['Approach/Move','Turn','Pass','Surround'])
    p.add_argument('--max-steps',type=int,choices=range(1,321));p.add_argument('--repeats',type=int,choices=(1,2),default=2);p.add_argument('--repeat-start',type=int,choices=(0,1),default=0)
    p.add_argument('--all-selected',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--selection-index',type=int,choices=(0,1),default=0)
    p.add_argument("--validation-only",action="store_true",help="Preserve legacy validation instructions; no altitude training certification")
    p.add_argument("--wall-seconds",type=float,default=4800)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.eval_root))
    import batch_run_act_all as official
    choices=[(f,json.loads(f.read_text(encoding='utf-8-sig'))) for f in sorted(args.tasks.glob('*.json'))]
    if any(t['motion_kind']=='Land' for _,t in choices):raise ValueError('landing forbidden in D223')
    chosen=choices if args.all_selected else [[(f,t) for f,t in choices if t['motion_kind']==kind][args.selection_index] for kind in args.kinds]
    for _,task in chosen:
        if task['motion_kind']=='Ascend/Descend' and not args.validation_only:
            from vertical_tasks import vertical_reward
            vertical_reward(task['reference_path_preprocessed'],task,stopped=True)
    save(args.out/'selection.json',[dict(task=f.name,kind=t['motion_kind'],max_steps=args.max_steps or (320 if t['motion_kind']=='Surround' else 128),repeats=args.repeats,seed=args.seed,selection_index=args.selection_index) for f,t in chosen])
    save(args.out/'usage.json',dict(validation_only=args.validation_only,training_eligible=not args.validation_only))
    started=time.monotonic();env=None;summary=[];raw={}
    try:
        env=official.gym.make('UnrealTrack-DowntownWest-ContinuousColor-v0')
        env=official.time_dilation.TimeDilationWrapper(env,10)
        env.unwrapped.agents_category=['drone']
        env=official.configUE.ConfigUEWrapper(env,resolution=(256,256),offscreen=True)
        env=official.augmentation.RandomPopulationWrapper(env,2,2,random_target=False)
        env.seed(args.seed if args.seed is not None else 0);env.reset();api=env.unwrapped.unrealcv;actor=env.unwrapped.player_list[0]
        api.set_viewport(actor);api.client.request('vrun t.MaxFPS 10');api.set_phy(actor,0)
        api.new_obj('bp_character_C','BP_Character_21',[0,0,0]);api.set_appearance('BP_Character_21',0);api.set_obj_rotation('BP_Character_21',[0,0,0])
        api.new_obj('BP_BaseCar_C','BP_Character_22',[1000,0,0]);api.set_appearance('BP_Character_22',2);api.set_obj_rotation('BP_Character_22',[0,0,0]);api.set_phy('BP_Character_22',0)
        if args.probe:
            initial=chosen[0][1]['initial_pos'];api.set_obj_location(actor,initial[:3]);rows=[]
            for yaw in (initial[4]-180,initial[4]-180-0.233,initial[4]-180-0.233+309.15,initial[4]-180-0.233-50.85,initial[4]-180-0.233-48.93,initial[4]-180-0.233-47.01):
                api.set_rotation(actor,yaw)
                for wait in (.1,.5,1.,2.):
                    time.sleep(wait);official.set_cam(env)
                    rows.append(dict(command=yaw,wait=wait,actor_rotation=api.get_obj_rotation(actor),camera_rotation=api.get_cam_rotation(0)))
                    save(args.out/'rotation_probe.json',rows);print(json.dumps(rows[-1]),flush=True)
            return
        for f,task in chosen:
            for repeat in range(args.repeat_start,args.repeat_start+args.repeats):
                if time.monotonic()-started>args.wall_seconds:raise TimeoutError('practice test wall limit')
                tag=f.stem+'_r'+str(repeat);initial=task['initial_pos'];t=time.monotonic()
                target=task.get('target_pos',task.get('obj_pos'))
                official.create_obj_if_needed(env,dict(use_obj=task.get('use_obj'),obj_id=task.get('obj_id'),obj_pos=target[:3],obj_rot=target[3:] if len(target)==6 else [0,0,0]))
                api.set_obj_location(actor,initial[:3]);api.set_rotation(actor,initial[4]-180);official.set_cam(env);time.sleep(2)
                reset_command=initial[4]-180
                # The blueprint adds a mesh offset and interpolates rotation.
                # Calibrate the command so camera/actor yaw reaches task heading.
                offset=(api.get_obj_rotation(actor)[1]-initial[4]+180)%360-180
                reset_command-=offset;api.set_rotation(actor,reset_command);time.sleep(1)
                residual=(api.get_obj_rotation(actor)[1]-initial[4]+180)%360-180
                if abs(residual)>.005:raise ValueError('reset heading calibration failed')
                measured=UnrealReadback(api,actor,initial,lambda:official.set_cam(env),initial_command_yaw=reset_command)
                if not summary:
                    checks=[]
                    for command in ([20,10,5,np.deg2rad(10)],[-20,-10,-5,np.deg2rad(-10)],[0,0,0,0]):
                        measured.execute(command);o=measured.observe();expected=[*command[:3],0,np.rad2deg(command[3]),0]
                        checks.append(dict(command=command,measured=o['pose'],expected=expected))
                        save(args.out/'calibration.json',checks)
                        np.testing.assert_allclose(o['pose'],expected,atol=.05,rtol=0)
                    measured.observe()['image'].save(args.out/'calibration_image.png')
                    save(args.out/'camera.json',api.get_camera_config())
                if args.movement_probe:
                    checks=[]
                    for degrees in (10,-10,0,90,170,179,-179,-90,-50.85,-48.93,-47.01,0):
                        measured.execute([0,0,0,np.deg2rad(degrees)]);o=measured.observe();error=(o['pose'][4]-degrees+180)%360-180
                        checks.append(dict(degrees=degrees,actual=o['pose'][4],error=error,corrections=measured.rotation_corrections.copy()))
                        save(args.out/'movement_probe.json',checks);print(json.dumps(checks[-1]),flush=True)
                        assert abs(error)<=.01
                    return
                measured.observe()['image'].save(args.out/(tag+'_start.png'))
                client=PolicyHTTP('http://127.0.0.1:'+str(args.port),task['instruction'],f.name,repeat);client.reset(seed=None if args.seed is None else args.seed+repeat)
                raw=dict(task=f.name,repeat=repeat,calls=[],poses=[])
                class LoggedEnv:
                    def execute(self,c):return measured.execute(c)
                    def observe(self):
                        o=measured.observe();raw['poses'].append(o['pose']);return o
                def policy(o):
                    r=client(o);raw['calls'].append(r);return r
                result=collect_episode(LoggedEnv(),policy,client.observe_context,task['reference_path_preprocessed'],max_steps=args.max_steps or (320 if task['motion_kind']=='Surround' else 128),reward_kind=task['motion_kind'])
                result['executed_yaw_tolerance_deg']=0.05
                result['rotation_corrections']=measured.rotation_corrections
                result.update(seed=None if args.seed is None else args.seed+repeat,task=f.name,kind=task['motion_kind'],repeat=repeat,elapsed_seconds=time.monotonic()-t)
                save(args.out/(tag+'.json'),result)
                row={k:result[k] for k in ('task','kind','repeat','elapsed_seconds','end_reason')}
                row.update(steps=sum(result['executed_counts']),calls=len(result['samples']),reward={k:v for k,v in result['reward'].items() if k not in ('transition_rewards','progress_trace','deviation_trace')})
                summary.append(row);save(args.out/'summary.json',summary);print(json.dumps(row),flush=True)
        save(args.out/'completion.json',dict(status='complete',episodes=len(summary),elapsed_seconds=time.monotonic()-started,training_started=False))
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc(),partial=raw,completed=len(summary)));raise
    finally:
        if env is not None:env.close()
if __name__=='__main__':main()
