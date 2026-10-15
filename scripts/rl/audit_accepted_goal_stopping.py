"""D248 deterministic local execution controls, NOT VLA/physics flights."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from flight_collector import collect_episode
from movement_reward import movement_reward
from audit_movement_reward import line

class KinematicHarness:
    def __init__(self,start):self.pose=np.array(start,float)
    def execute(self,command):
        self.pose[:3]=np.round(command[:3],8);self.pose[4]=round(float(np.rad2deg(command[3])),8)
    def observe(self):return dict(pose=self.pose.tolist(),pose_source='simulator_readback')

def execute(path,ref,kind,mode):
    # This fake transport satisfies the readback contract for a CPU integration
    # test only. It does not claim images, real simulator execution or model tokens.
    env=KinematicHarness(path[0]);commands=list(path[1:])+[path[-1]]*32;cursor=0
    def policy(obs):
        nonlocal cursor
        part=commands[cursor:cursor+8];cursor+=len(part)
        assert part
        return dict(proprio=obs['pose'][:3]+[obs['pose'][4]],poses=[list(p[:3])+[float(np.deg2rad(p[4]))] for p in part],source='scripted_cpu_control')
    result=collect_episode(env,policy,lambda o:dict(id='cpu_control_final'),ref,max_steps=len(commands),reward_kind=kind,reward_stopping_mode=mode)
    assert result['end_reason']=='model_stop'
    assert abs(sum(s['reward_v2'] for s in result['samples'])-result['reward']['reward'])<1e-9
    assert not any(s['reward_spec_approved'] for s in result['samples'])
    return result

def cases():
    out=[];r=line(yaw=26,n=14)
    for name,yaws,tail,prefer in [('turn_far',[0,4,8,12],[16,20,24],'continue'),('turn_outside',[0,4,8,12,16,20,22],[24],'continue'),('turn_accepted',[0,4,8,12,16,20,24],[26],'stop')]:
        p=np.zeros((len(yaws),6));p[:,4]=yaws;t=np.zeros((len(tail),6));t[:,4]=tail
        out.append((name,'Turn',r,p,np.vstack([p,t]),prefer))
    r=line(x=3,z=2,n=25)
    for name,f,yaw,goal_f,goal_yaw,prefer in [('altitude_far',.5,0,.92,0,'continue'),('altitude_position_outside',.84,0,.92,0,'continue'),('altitude_heading_outside',1,6,1,4,'continue'),('altitude_accepted',.92,4,1,0,'stop')]:
        p=line(x=3*f,z=2*f,yaw=yaw,n=max(3,int(24*f)))
        end=np.array([300*goal_f,0,200*goal_f,0,goal_yaw,0])
        steps=max(1,int(np.ceil(np.linalg.norm(end[:3]-p[-1,:3])/20)),int(np.ceil(abs(goal_yaw-yaw)/2)))
        tail=np.linspace(p[-1],end,steps+1)[1:]
        out.append((name,'Ascend/Descend',r,p,np.vstack([p,tail]),prefer))
    base=ROOT/'reports/d238_two_task/evidence/round_1'
    for name,filename,call,prefer in [('saved_heading_outside','sim_2025-03-27_16-41-45_r0.json',10,'continue'),('saved_turn_accepted','sim_2025-03-27_19-38-18_r0.json',6,'stop')]:
        e=json.loads((base/'attempts/episodes'/filename).read_text());task=json.loads((base/'tasks'/e['task']).read_text())
        n=sum(e['executed_counts'][:call]);prefix=np.array(e['poses'][:n+1],float)
        end=prefix[-1].copy();end[4]=4. if prefer=='continue' else task['reference_path_preprocessed'][-1][4]
        out.append((name,e['kind'],np.array(task['reference_path_preprocessed']),prefix,np.vstack([prefix,end]),prefer))
    return out

def run():
    rows=[]
    for name,kind,ref,stop_path,continue_path,prefer in cases():
        item=dict(name=name,kind=kind,expected_preference=prefer)
        for mode in ['continuous','accepted_goal']:
            branches={}
            prefix_reward=movement_reward(stop_path,ref,kind=kind,stopped=False,stopping_mode=mode)['reward']
            for label,path in [('stop',stop_path),('continue',continue_path)]:
                e=execute(path,ref,kind,mode);q=e['reward']
                branches[label]=dict(remaining_reward=q['reward']-prefix_reward,reward=q['reward'],terminal_credit=q['terminal_credit'],complete=q['complete'],endpoint_m=q['endpoint_position_error_m'],heading_error_deg=q['endpoint_yaw_error_deg'],moves=sum(e['executed_counts']),poses=e['poses'],executed_counts=e['executed_counts'])
            item[mode]=branches
        new=item['accepted_goal'];other='continue' if prefer=='stop' else 'stop'
        assert new[prefer]['reward']>new[other]['reward'],item
        assert new[prefer]['complete']
        rows.append(item)
    return rows

def main():
    out=ROOT/'reports/d248_accepted_goal_stopping';out.mkdir(exist_ok=True)
    rows=run();(out/'examples.json').write_text(json.dumps(dict(execution='Scripted deterministic CPU transport through production collector; no VLA, AirSim, physics, AWS or training.',cases=rows),indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True,sharex=True)
    labels=[r['name'].replace('_',' ') for r in rows];y=np.arange(len(rows))
    for ax,mode,title in zip(axes,['continuous','accepted_goal'],['Original objective','Local accepted-goal objective']):
        ax.barh(y-.17,[r[mode]['stop']['remaining_reward'] for r in rows],.32,label='Stop now')
        ax.barh(y+.17,[r[mode]['continue']['remaining_reward'] for r in rows],.32,label='Continue, then stop')
        ax.set_yticks(y,labels);ax.set_title(title);ax.set_xlabel('Remaining reward from the same decision state');ax.legend();ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.suptitle('Scripted local controls: reward comparison, not trained-model improvement')
    fig.tight_layout();fig.savefig(out/'examples.png',dpi=150);plt.close(fig)
    print(json.dumps([dict(name=r['name'],prefer=r['expected_preference'],old=[r['continuous'][b]['remaining_reward'] for b in ['stop','continue']],new=[r['accepted_goal'][b]['remaining_reward'] for b in ['stop','continue']]) for r in rows],indent=2))
if __name__=='__main__':main()
