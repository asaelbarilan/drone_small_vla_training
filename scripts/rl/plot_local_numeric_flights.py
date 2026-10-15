"""D228 measured trajectory evidence; no inferred image-based success."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/d228_local_numeric_flights/retry1'
fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
titles=['60 degree right turn','4 m vertical descent','9 m / 85 degree ascent','5 m / 75 degree descent']
for ax,taskfile,title in zip(axes.flat,sorted((ROOT/'reports/d224_reward_gated_ppo/tasks').glob('*.json')),titles):
 task=json.loads(taskfile.read_text());ref=np.asarray(task['reference_path_preprocessed'],float)
 turn=task['motion_kind']=='Turn/Rotate';vertical='vertically' in task['instruction']
 for repeat,color in [(0,'#2876b9'),(1,'#c57024')]:
  f=OUT/'episodes'/(taskfile.stem+f'_r{repeat}.json')
  if not f.exists():f=OUT.parent/'finish/episodes'/f.name
  if not f.exists():continue
  ep=json.loads(f.read_text());a=np.asarray(ep['poses']);r=ep['reward']
  x=np.arange(len(a)) if turn or vertical else a[:,0]/100
  y=np.rad2deg(np.unwrap(np.deg2rad(a[:,4]))) if turn else a[:,2]/100
  ax.plot(x,y,color=color,label=f"Seed {7+repeat}: reward {r['reward']:.3f}; {ep['end_reason']}")
  ax.plot(x[-1],y[-1],'o',color=color)
 if turn or vertical:
  ax.axhline(ref[-1,4] if turn else ref[-1,2]/100,color='#555555',linestyle='--',label='Target')
 else:
  ax.plot(ref[:,0]/100,ref[:,2]/100,'--',color='#555555',label='Reference geometry')
  ax.set_aspect('equal',adjustable='datalim')
 ax.set_title(title);ax.set_xlabel('Executed move' if turn or vertical else 'Forward displacement (m)')
 ax.set_ylabel('Accumulated yaw (degrees)' if turn else 'Height change (m)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('D228 local NF4 flights — frozen policy, different sampling seeds\nMeasured poses; landing excluded; no learning')
fig.savefig(OUT/'flight_paths.png',dpi=160);plt.close(fig)
