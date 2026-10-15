"""D220 local plots and concise report."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from audit_movement_reward import controls,variants,OUT,read
from movement_reward import movement_reward
from path_reward import path_reward
audit=read(OUT/'audit.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,3,figsize=(16,13),layout='constrained')
for ax,(kind,ref),row in zip(axes.flat,controls().items(),audit['synthetic']):
    paths=variants(ref);names=['correct','early_stop','shortcut' if kind in ('Pass','Surround') else 'wrong_direction']
    turn=kind.startswith('Turn');vertical=kind in ('Ascend/Descend','Land')
    for name,color in zip(names,['#087f8c','#d19016','#c03946']):
        a=paths[name]
        if turn:x=np.linspace(0,1,len(a));y=a[:,4]
        elif vertical:x=a[:,0]/100;y=a[:,2]/100
        else:x=a[:,0]/100;y=a[:,1]/100
        score=row['scores'][name]['new']['reward']
        ax.plot(x,y,color=color,lw=2,label=name.replace('_',' ')+' | '+f'{score:.2f}')
        ax.scatter(x[-1],y[-1],c=color,s=42,marker='s',zorder=5)
    if turn:ax.axhline(ref[-1,4],color='#555',ls='--',lw=1)
    else:ax.set_aspect('equal',adjustable='datalim')
    ax.set_title(kind+(' (synthetic contact known)' if kind=='Land' else ''),fontsize=11)
    ax.set_xlabel('Relative trajectory sequence' if turn else 'Forward x (m)')
    ax.set_ylabel('Heading (degrees)' if turn else 'Height z (m)' if vertical else 'Lateral y (m)')
    ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('Local reward examples: all nine movement categories\nSynthetic paths with known intent | legend = corrected reward | squares = stopping positions',fontsize=15)
fig.savefig(OUT/'all_movements.png',dpi=160);fig.savefig(OUT/'all_movements.pdf');plt.close(fig)
ref=np.zeros((101,6));ref[:,4]=np.linspace(0,-9,101)
ends=np.linspace(-12,-6,241);old=[];new=[]
for end in ends:
    p=ref.copy();p[:,4]=np.linspace(0,end,len(p))
    old.append(path_reward(p,ref,stopped=True)['reward'])
    new.append(movement_reward(p,ref,kind='Turn',stopped=True)['reward'])
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
ax.plot(ends,old,c='#d19016',label='Old reward: 98% bonus threshold')
ax.plot(ends,new,c='#087f8c',label='Corrected reward: continuous heading accuracy')
ax.axvline(-9,c='#333',ls='--',label='Requested heading: -9 degrees')
ax.set(xlabel='Final heading (degrees): overshoot left, undershoot right',ylabel='Episode reward',
       title='Turn: equal errors on either side should receive equal credit')
ax.grid(alpha=.2);ax.legend();fig.savefig(OUT/'turn_reward_fix.png',dpi=160);plt.close(fig)
coverage=[];saved=[]
for kind in controls():
    group=[r for r in audit['inventory'] if r['kind']==kind]
    coverage.append(f"| {kind} | {len(group)} | {sum(r.get('self_control',{}).get('geometric_complete',False) for r in group)}/{len(group)} |")
for kind in ['Approach/Move','Turn','Pass','Surround']:
    pair={r['phase']:r for r in audit['saved'] if r['kind']==kind};b,a=pair['before'],pair['after']
    saved.append(f"| {kind} | {b['old']['reward']:.3f} → {a['old']['reward']:.3f} | {b['new']['reward']:.3f} → {a['new']['reward']:.3f} | {b['new']['complete']} → {a['new']['complete']} |")
text=f"""# D220 — Local movement reward audit

The geometry reward passes **{audit['passed']}/{audit['total']} checks** over nine synthetic movement types, all 100 practice references with counterexamples, and eight saved before/after flights. These are software and reference-agreement checks, not independent flights or verified semantic labels. No AWS calls, simulator flights or training.

## What changed

- Replaced the discontinuous one-point completion bonus with continuous stopping credit. Geometric completion remains a separate diagnostic.
- In-place turns use symmetric final heading error. Equal undershoot and overshoot receive equal credit. Tolerance is 10% of requested rotation, capped at 5 degrees; these are provisional settings, not fitted labels.
- Turning excursions and reversals lose reward even if the drone returns to the final target.
- Numerical integration subdivides measured segments, then sums back to original executed actions. Duplicate poses and tested resampling do not create reward. No command per reference photo.
- Endpoint position error is reported in metres separately from heading error in degrees. Route deviation still uses the existing combined metre/radian metric (1 metre per radian), not pure distance.
- Original transitions are checked before subdivision: translation limit 1 m and heading limit 45 degrees are explicit kinematic validity assumptions. Wrong start frame, invalid move or measured collision cannot earn positive return.
- Land receives no completion credit without explicit measured contact. A category name or low altitude is not touchdown evidence.

Implementation: scripts/rl/movement_reward.py. Explicit opt-in through attach_path_rewards(kind=..., events=...). Existing callers and historical scores remain unchanged. New reward remains training_ready=False and attached samples remain unapproved. No previously admitted PPO batch was rewritten.

## Examples

![All movement examples](all_movements.png)

Synthetic cases, not new flights. Green follows the reference; amber stops early; red goes the wrong way or shortcuts a curved path. The Land positive example assumes known contact. Stationary, overshoot, detour, reversal, collision and timeout controls are also saved.

![Turn correction](turn_reward_fix.png)

For a requested -9 degree turn, the old score jumped at 98% progress, rewarding overshoot more than a closer shortfall. The corrected curve peaks at the target and changes smoothly on both sides. This property was specified independently of saved PPO flights.

## Saved flights rescored without changing their trajectories

| Task | Old reward: before → after PPO | Corrected reward: before → after PPO | Corrected geometric completion |
|---|---|---|---|
"""+'\n'.join(saved)+"""

Compare before/after within each reward version, not absolute values across versions. Turn now ranks the original flight higher, consistent with smaller heading error (0.617 vs 1.416 degrees). Its completion flips from before-fail/after-pass to before-pass/after-fail: a scorer correction, not a change in model ability. Approach still ranks the original higher. Pass and Surround improve geometrically but remain incomplete. Four one-seed tasks do not establish general improvement.

## Coverage

| Practice category | References | Geometric self-controls |
|---|---|---|
"""+'\n'.join(coverage)+"""

All 100 references were tested against stationary, early-stop, opposite-direction and perpendicular off-route variants. Passing a reference self-control establishes consistency only, not that the reference satisfies the instruction.

The first audit caught unpenalized out-and-back altitude excursions during turns; corrected with a path-integrated drift penalty and continuous excursion quality. An expanded test initially labelled a change along a steep ascent as off-path. The fixture was corrected to use a perpendicular excursion: different sampling/speed along the same route is not geometric failure.

## Remaining training blockers

1. **Land intent/contact:** one Land task says “Descend to 4.0 meters altitude”; its reference descends by 4 m. Four others request landing, but saved poses have no contact state. They cannot yet receive landing-success labels.
2. **Altitude vs travel distance:** “Ascend 9.0 meters at 30 degrees” moves 7.794 m forward and 4.5 m up: 9 m along the slope. “Climb to an altitude of 4.0 meters with a 5-degree pitch angle” rises only 0.349 m along a roughly 4 m slope. Resolve instruction/reference meaning before semantic reward training.
3. **Object identity, side and clearance:** geometry does not independently verify these. Pass/Surround/Approach/Retreat need reviewed metadata or a separately validated semantic judge. Outputs explicitly leave semantic completion unknown.
4. **Takeoff:** no separate Takeoff category exists in these 100 practice tasks. No takeoff/airborne validation claim.
5. **Scope:** this fixes reward logic, not DAgger labels, physics, PPO KL guard or horizon policy. No time-efficiency reward was invented; stored poses do not establish duration.

The next gate is reviewed task intent plus validated telemetry where needed. A learned reward network is possible, but would need labelled comparisons and these counterexample tests too. Geometry alone cannot certify all movements ready for training.

## Reproduction

Run python scripts/rl/audit_movement_reward.py, then python scripts/rl/render_movement_audit.py.
Run the movement/path/executed-rollout/collector unit tests.
audit.json retains task hashes, reward source hash, every check and scores. tests.txt records regression results.
"""
(OUT/'README.md').write_text(text,encoding='utf-8')
print(OUT)
