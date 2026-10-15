"""Recover only a fully stopped, retained episode rejected by the old yaw tolerance.
Never turns an interrupted transport run into a completed flight.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from executed_rollouts import attach_path_rewards

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);a=p.parse_args()
    source=a.run/'episodes/failure.json';failure=json.loads(source.read_text())
    if not failure['traceback'].rstrip().endswith('ValueError: actual yaw mismatch'):raise ValueError('not the supported post-flight validation failure')
    raw=failure['partial'];poses=np.asarray(raw['poses']);calls=raw['calls'];counts=[len(x['poses']) for x in calls];counts[-1]-=sum(counts)-(len(poses)-1)
    if not 0<counts[-1]<=len(calls[-1]['poses']):raise ValueError('incomplete call boundary evidence')
    quiet=0;first_stop=None
    for i,(old,new) in enumerate(zip(poses,poses[1:]),1):
        small=np.max(abs(new[:3]-old[:3]))<3 and abs((new[4]-old[4]+180)%360-180)<1
        quiet=quiet+1 if small else 0
        if quiet>=10:first_stop=i;break
    if first_stop!=len(poses)-1:raise ValueError('stored path does not end exactly at the first ten-quiet-move stop')
    task=json.loads((a.tasks/raw['task']).read_text(encoding='utf-8-sig'))
    samples,reward=attach_path_rewards(calls,poses,counts,task['reference_path_preprocessed'],end_reason='model_stop')
    proposed=np.asarray([c for x in calls for c in x['poses']])[:len(poses)-1];error=(np.rad2deg(proposed[:,3])-poses[1:,4]+180)%360-180
    evidence=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reason='post-flight heading tolerance corrected from 0.01 to 0.05 degrees',first_stop_move=first_stop,max_yaw_error_deg=float(abs(error).max()),max_position_error_cm=float(abs(proposed[:,:3]-poses[1:,:3]).max()),training_approved=False)
    result=dict(samples=samples,reward=reward,poses=poses.tolist(),executed_counts=counts,end_reason='model_stop',pose_source='simulator_readback',final_observation=None,task=raw['task'],kind=task['motion_kind'],repeat=raw['repeat'],elapsed_seconds=None,recovery=evidence,executed_yaw_tolerance_deg=.05)
    a.out.mkdir(exist_ok=False);(a.out/'episodes').mkdir();shutil.copytree(a.run/'rollout_images',a.out/'rollout_images')
    (a.out/'episodes'/(Path(raw['task']).stem+'_r'+str(raw['repeat'])+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
    (a.out/'recovery.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(dict(recovery=evidence,reward={k:v for k,v in reward.items() if not k.endswith('_trace') and k!='transition_rewards'}),indent=2))
if __name__=='__main__':main()
