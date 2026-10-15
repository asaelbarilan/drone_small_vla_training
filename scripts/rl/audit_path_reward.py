"""D214 offline candidate reward audit over saved practice data."""
import json
from pathlib import Path
import numpy as np
from path_reward import path_reward
from rollout_samples import match
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'reports/d210_dagger_audit/practice'
OUT=ROOT/'reports/d214_path_reward'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def compact(r):return {k:v for k,v in r.items() if k not in ('transition_rewards','progress_trace','deviation_trace')}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    episodes=match(P/'pass1',P/'server_calls.jsonl');rows=[]
    for path in sorted((P/'tasks').glob('*.json')):
        task=read(path);ref=task['reference_path_preprocessed']
        row=dict(task=path.name,kind=task['motion_kind'])
        for label,poses in [('reference',ref),('stationary',[ref[0]]*len(ref)),('prefix104',ref[:105])]:
            try:row[label]=compact(path_reward(poses,ref,stopped=True))
            except ValueError as e:row[label]=dict(error=str(e))
        if path.name in episodes:
            calls=episodes[path.name];p=calls[0]['proprio'];start=[*p[:3],0,p[3],0]
            poses=[start]+[r['state'][0]+r['state'][1] for r in read(P/'pass1'/path.name)]
            row['saved']=compact(path_reward(poses,ref,stopped=bool(calls[-1]['stop'])))
        rows.append(row)
    (OUT/'scores.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    cir=[r for r in rows if r['kind']=='Surround']
    passed=sum(r['reference'].get('complete',False) for r in rows)
    lines=['# D214 path reward candidate','',
           'This reward uses the whole route geometry and ordered progress. It does not score camera endpoints or require one action per recorded photo.',
           '',f'Reference self-controls complete: {passed}/{len(rows)}. '
           f"Circle full > partial104 > stationary ordering: {sum(r['reference']['reward']>r['prefix104']['reward']>r['stationary']['reward'] for r in cir)}/{len(cir)}.",
           '', '## The pictured Pass flight','']
    example=next(r for r in rows if r['task']=='sim_2025-04-30_15-37-22.json')
    for label in ['reference','saved','stationary']:
        r=example[label]
        lines.append(f"- {label}: reward {r['reward']:.3f}; ordered progress {r['progress_fraction']:.1%}; "
                     f"maximum deviation {r['max_deviation']:.2f}; endpoint error {r['endpoint_error']:.2f}; complete {r['complete']}.")
    lines+=['','Deviation/error units combine metres and yaw radians weighted by1 m/radian. '
            'These are candidate engineering settings, not calibrated success thresholds. '
            'Saved-flight figures measure reference agreement, not independently certified task success.',
            '', '## Reward definition','',
            'Per transition: signed route progress / route length, minus distance-weighted deviation penalty, '
            'minus an extra reverse-motion penalty. At the final stopped state, add one completion bonus '
            'only after ordered route coverage, endpoint and whole-flight corridor checks pass. '
            'Progress matching is history-constrained; it cannot jump from the beginning to the end of a loop. '
            'Large moves invalidate completion instead of earning teleport progress.',
            '', '## Limits and next check','',
            'No training launched. Not connected to PPO yet. Weights, movement bound, corridor and discount interaction '
            'still require calibration on practice-only controls. Whole-flight corridor strictness can penalize recoverable detours; '
            'do not silently relax it to fit validation results. The underlying reference must represent the intended task. '
            'Physical obstacle clearance is not established by geometric proximity. Original reward and historical scores remain unchanged.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:12]))
if __name__=='__main__':main()
