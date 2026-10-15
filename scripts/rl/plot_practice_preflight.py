"""Plot measured practice routes against reference geometry, without image scoring."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(f.read_text()) for f in sorted(a.episodes.glob('sim_*_r*.json'))]
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    colors=['#147D92','#D07827']
    for ax,kind in zip(axes.flat,('Approach/Move','Turn','Pass','Surround')):
        records=sorted([r for r in rows if r['kind']==kind],key=lambda r:r['repeat'])
        task=json.loads((a.tasks/records[0]['task']).read_text());ref=np.asarray(task['reference_path_preprocessed'])
        if kind=='Turn':
            target=np.rad2deg(np.unwrap(np.deg2rad(ref[:,4])))[-1]
            ax.axhline(target,color='#57606A',lw=2,label='Reference final heading')
            for r,color in zip(records,colors):
                actual=np.asarray(r['poses']);yaw=np.rad2deg(np.unwrap(np.deg2rad(actual[:,4])))
                ax.plot(range(len(yaw)),yaw,color=color,lw=2,label='Attempt '+str(r['repeat']+1))
            ax.set_xlabel('Executed moves');ax.set_ylabel('Cumulative yaw (degrees)')
        else:
            dim=2 if kind=='Approach/Move' else 1
            ax.plot(ref[:,0]/100,ref[:,dim]/100,color='#57606A',lw=3,alpha=.7,label='Reference route')
            ax.scatter(ref[-1,0]/100,ref[-1,dim]/100,color='#57606A',marker='x',s=65)
            for r,color in zip(records,colors):
                actual=np.asarray(r['poses'])/100
                ax.plot(actual[:,0],actual[:,dim],color=color,lw=1.8,label='Attempt '+str(r['repeat']+1))
                ax.scatter(actual[-1,0],actual[-1,dim],color=color,s=35)
            ax.scatter(0,0,color='black',marker='o',s=25)
            ax.set_xlabel('Start-frame x (m)');ax.set_ylabel('Start-frame '+('z' if dim==2 else 'y')+' (m)');ax.set_aspect('equal',adjustable='datalim')
        lines=[kind]
        for r in records:
            score=r.get('rescored_reward_v2',r['reward']);lines.append('Attempt %d: %.0f%% progress | reward %.3f | %s'%(r['repeat']+1,100*score['progress_fraction'],score['reward'],'complete' if score['complete'] else 'partial'))
        ax.set_title('\n'.join(lines),fontsize=11);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('D216 unchanged D208 policy: measured practice trajectories\nGeometry checks; no PPO learning run and no benchmark evaluation',fontsize=14)
    a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=160);plt.close(fig)
if __name__=='__main__':main()
