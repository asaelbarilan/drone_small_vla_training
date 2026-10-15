"""Render saved practice scenes/trajectories; no model/simulator/cloud calls."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rollout_samples import match
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'reports/d210_dagger_audit/practice'
OUT=ROOT/'reports/d213_offline_preflight'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    matched=match(P/'pass1',P/'server_calls.jsonl')
    selected={}
    for task in sorted((P/'tasks').glob('*.json')):
        data=read(task)
        if task.name in matched:selected.setdefault(data['motion_kind'],task)
    items=list(selected.items());manifest=[]
    for page in range(0,len(items),3):
        subset=items[page:page+3]
        fig,axes=plt.subplots(len(subset),3,figsize=(15,3.7*len(subset)),squeeze=False,layout='constrained')
        for row,(kind,path) in enumerate(subset):
            task=read(path);calls=matched[path.name];ref=np.asarray(task['reference_path_preprocessed'])
            flown=read(P/'pass1'/path.name);xyz=np.array([x['state'][0] for x in flown])/100
            for col,call in enumerate([calls[0],calls[-1]]):
                image=P/'rollout_images'/call['image']
                if image.exists():axes[row,col].imshow(plt.imread(image))
                else:axes[row,col].text(.1,.5,'Saved image unavailable')
                axes[row,col].axis('off');axes[row,col].set_title(kind+(' | first call' if col==0 else ' | last call (not final pose)'))
            ax=axes[row,2]
            ax.plot(ref[:,0]/100,ref[:,1]/100,label='reference',color='#64748b')
            ax.plot(xyz[:,0],xyz[:,1],label='saved flight',color='#dc2626')
            ax.scatter([0],[0],marker='o',color='black',s=20)
            points=np.vstack([ref[:,:2]/100,xyz[:,:2],[[0,0]]])
            lo,hi=points.min(axis=0),points.max(axis=0)
            mid=(lo+hi)/2;half=max(float((hi-lo).max())*.55,.25)
            ax.set_xlim(mid[0]-half,mid[0]+half);ax.set_ylim(mid[1]-half,mid[1]+half)
            ax.set_aspect('equal',adjustable='box');ax.set_xlabel('start-frame x (m)');ax.set_ylabel('y (m)');ax.legend(fontsize=8)
            ax.set_title(path.stem,fontsize=8)
            axes[row,0].text(0,-.09,task['instruction'],transform=axes[row,0].transAxes,fontsize=8,wrap=True)
            manifest.append(dict(kind=kind,task=path.name,instruction=task['instruction'],
                                 page=f'review_{page//3+1}.png',
                                 first_image=str((P/'rollout_images'/calls[0]['image']).relative_to(ROOT)),
                                 last_image=str((P/'rollout_images'/calls[-1]['image']).relative_to(ROOT))))
        fig.savefig(OUT/f'review_{page//3+1}.png',dpi=120,bbox_inches='tight');plt.close(fig)
    (OUT/'review_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'REVIEW.md').write_text('# Saved practice review pack\n\n'
        'One example per kind, selected by sorted task ID. These images are observations before the first/last model call, '
        'not a guarantee of the final camera state. XY plots omit altitude and yaw; use task_review.json and raw traces '
        'for those dimensions. This pack supports review, not semantic certification.\n\n'
        + '\n\n'.join(f'![Practice examples {i}](review_{i}.png)' for i in range(1,(len(items)+2)//3+1))+'\n',encoding='utf-8')
    print(f'Rendered {len(items)} task-kind examples')
if __name__=='__main__':main()
