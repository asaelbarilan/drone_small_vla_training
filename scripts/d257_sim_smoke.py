"""D257 local fresh-HF-package smoke; no learning or test benchmark."""
import argparse,base64,io,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import requests

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(exist_ok=False);sys.path.insert(0,str(a.package/'code/scripts/rl'));sys.path.insert(0,'D:/drone_vla_pilot/win_bundle/UAV-Flow-Eval')
 from simulator_transport import UnrealReadback
 import batch_run_act_all as official
 selected=['sim_2025-03-27_17-34-24.json','sim_2025-03-27_16-41-53.json'];env=None;rows=[];started=time.monotonic()
 save(a.out/'protocol.json',dict(tasks=selected,seed=7,max_moves=128,wall_seconds=1500,precision='nf4',training=False,quiet_moves=10,quiet_position_cm=3,quiet_heading_deg=1,role='package smoke, not efficacy benchmark'))
 try:
  env=official.gym.make('UnrealTrack-DowntownWest-ContinuousColor-v0');env=official.time_dilation.TimeDilationWrapper(env,10);env.unwrapped.agents_category=['drone'];env=official.configUE.ConfigUEWrapper(env,resolution=(256,256),offscreen=True);env=official.augmentation.RandomPopulationWrapper(env,2,2,random_target=False);env.seed(7);env.reset()
  api=env.unwrapped.unrealcv;actor=env.unwrapped.player_list[0];api.set_viewport(actor);api.client.request('vrun t.MaxFPS 10');api.set_phy(actor,0)
  api.new_obj('bp_character_C','BP_Character_21',[0,0,0]);api.set_appearance('BP_Character_21',0);api.set_obj_rotation('BP_Character_21',[0,0,0]);api.new_obj('BP_BaseCar_C','BP_Character_22',[1000,0,0]);api.set_appearance('BP_Character_22',2);api.set_obj_rotation('BP_Character_22',[0,0,0]);api.set_phy('BP_Character_22',0)
  client=requests.Session();url='http://127.0.0.1:5017'
  for name in selected:
   task=json.loads((a.tasks/name).read_text(encoding='utf-8-sig'));initial=task['initial_pos'];target=task.get('target_pos',task.get('obj_pos'));tag=Path(name).stem
   official.create_obj_if_needed(env,dict(use_obj=task.get('use_obj'),obj_id=task.get('obj_id'),obj_pos=target[:3],obj_rot=target[3:] if len(target)==6 else [0,0,0]))
   api.set_obj_location(actor,initial[:3]);reset=initial[4]-180;api.set_rotation(actor,reset);official.set_cam(env);time.sleep(2);reset-=(api.get_obj_rotation(actor)[1]-initial[4]+180)%360-180;api.set_rotation(actor,reset);time.sleep(1)
   assert abs((api.get_obj_rotation(actor)[1]-initial[4]+180)%360-180)<=.005
   sim=UnrealReadback(api,actor,initial,lambda:official.set_cam(env),initial_command_yaw=reset)
   if not rows:
    calibration=[]
    for cmd in ([20,10,5,math.radians(10)],[-20,-10,-5,math.radians(-10)],[0,0,0,0]):
     sim.execute(cmd);o=sim.observe();np.testing.assert_allclose(o['pose'],[*cmd[:3],0,math.degrees(cmd[3]),0],atol=.05,rtol=0);calibration.append(dict(command=cmd,pose=o['pose']))
    save(a.out/'calibration.json',calibration)
   response=client.post(url+'/reset',json={'seed':7},timeout=10);response.raise_for_status();assert response.json()['seed']==7
   obs=sim.observe();obs['image'].save(a.out/(tag+'_start.png'));poses=[obs['pose']];calls=[];quiet=0;moves=0;reason='horizon'
   while moves<128 and quiet<10:
    if time.monotonic()-started>1500:raise TimeoutError('Smoke wall cap')
    b=io.BytesIO();obs['image'].resize((224,224)).save(b,format='PNG');pos=obs['pose'];payload=dict(image=base64.b64encode(b.getvalue()).decode(),proprio=[pos[0],pos[1],pos[2],pos[4]],instr=task['instruction'])
    r=client.post(url+'/predict',json=payload,timeout=180);r.raise_for_status();result=r.json();actions=np.asarray(result['action'],float);assert actions.ndim==2 and actions.shape[1]==4 and 1<=len(actions)<=8 and np.isfinite(actions).all()
    calls.append(dict(input_pose=pos,response=result));obs['image'].save(a.out/(tag+f'_call{len(calls):03d}.png'))
    for cmd in actions:
     if moves>=128 or quiet>=10:break
     previous=np.asarray(obs['pose']);sim.execute(cmd);obs=sim.observe();actual=np.asarray(obs['pose']);expected=np.array([*cmd[:3],0,math.degrees(cmd[3]),0]);assert np.linalg.norm(actual[:3]-expected[:3])<=.05;assert abs((actual[4]-expected[4]+180)%360-180)<=.05
     quiet=quiet+1 if np.max(np.abs(actual[:3]-previous[:3]))<3 and abs((actual[4]-previous[4]+180)%360-180)<1 else 0;moves+=1;poses.append(obs['pose'])
    save(a.out/(tag+'.json'),dict(task=task,poses=poses,calls=calls,moves=moves,quiet=quiet,complete=False))
    print(json.dumps(dict(task=tag,calls=len(calls),moves=moves,quiet=quiet)),flush=True)
   if quiet>=10:reason='quiet_stop'
   obs['image'].save(a.out/(tag+'_end.png'));goal=np.asarray(task['reference_path_preprocessed'][-1]);endpoint=float(np.linalg.norm(np.asarray(poses[-1])[:3]-goal[:3])/100);heading=float(abs((poses[-1][4]-goal[4]+180)%360-180))
   row=dict(task=name,kind=task['motion_kind'],moves=moves,calls=len(calls),end_reason=reason,endpoint_error_m=endpoint,heading_error_deg=heading,within_1m_5deg=endpoint<=1 and heading<=5)
   save(a.out/(tag+'.json'),dict(task=task,poses=poses,calls=calls,summary=row,rotation_corrections=sim.rotation_corrections,complete=True));rows.append(row);save(a.out/'summary.json',rows)
  save(a.out/'completion.json',dict(episodes=len(rows),elapsed_seconds=time.monotonic()-started,training=False))
 except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc(),completed=rows));raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
