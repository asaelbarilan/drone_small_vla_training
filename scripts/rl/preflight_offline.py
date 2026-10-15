"""Reproducible D213 saved-artifact audit. Never starts a model or simulator."""
import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from reward_v2 import fit_reference_orbit, orbit_diagnostic
from reward_uav_flow import reward

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'reports/d213_offline_preflight')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    practice=ROOT/'reports/d210_dagger_audit/practice'
    val=ROOT/'reports/d208_completion_20261009/tasks'
    taskfiles=sorted((practice/'tasks').glob('*.json'))
    if len(taskfiles)!=100:raise ValueError('expected saved 100 practice tasks')
    validation={p.stem for p in val.glob('*.json')}
    validation_sources={read(p).get('source_flight',p.stem) for p in val.glob('*.json')}
    excluded_path=ROOT/'reports/d209_preparation/code/release/qwen3vl4b_uavflow_vla/uav_flow_sim_excluded.json'
    heldout_path=ROOT/'reports/d209_preparation/code/release/qwen3vl4b_uavflow_vla/sim_val_flights.json'
    normal=lambda x: str(x).removeprefix('sim_').removesuffix('.json')
    excluded={normal(x) for x in read(excluded_path)}
    heldout={normal(x) for x in read(heldout_path)}
    records=[];orbit=[];inputs={str(excluded_path.relative_to(ROOT)):digest(excluded_path),
                              str(heldout_path.relative_to(ROOT)):digest(heldout_path)}
    for path in taskfiles:
        task=read(path);inputs[str(path.relative_to(ROOT))]=digest(path)
        target=task.get('target_pos',[])
        sentinel=len(target)<3 or target[2]==-10000
        reasons=['no independently approved semantic specification']
        if sentinel and task['motion_kind'] in ('Approach/Move','Pass','Retreat'):
            reasons.append('object-relative task lacks explicit target geometry')
        if task.get('object_from'):reasons.append('derived object metadata requires verification')
        if task['motion_kind']=='Land':reasons.append('instruction altitude semantics/contact require review')
        records.append(dict(task=path.name,kind=task['motion_kind'],instruction=task['instruction'],
                            instruction_unified=task.get('instruction_unified'),target_missing=sentinel,
                            object_provenance=task.get('object_from'),training_reward_approved=False,reasons=reasons))
        if normal(task.get('source_flight',path.stem)) in excluded | heldout:
            raise ValueError('practice source overlaps saved exclusion/heldout manifest')
        if path.stem in validation or task.get('source_flight',path.stem) in validation_sources:
            raise ValueError('practice/validation identity or source overlap')
        if task['motion_kind']!='Surround':continue
        ref=task['reference_path_preprocessed'];xyz=np.asarray(ref)[:,:3]/100
        row=dict(task=path.name)
        try:
            spec=fit_reference_orbit(ref);row['spec']=asdict(spec)
            row['reference_complete']=orbit_diagnostic(xyz,spec,stopped=True)
            row['stationary']=orbit_diagnostic(np.repeat(xyz[:1],len(xyz),axis=0),spec,stopped=True)
            row['partial']=orbit_diagnostic(xyz[:min(105,len(xyz))],spec,stopped=True)
            fp=practice/'pass1'/path.name
            if fp.exists():
                inputs[str(fp.relative_to(ROOT))]=digest(fp)
                flown=read(fp)
                poses=np.vstack([xyz[0],np.array([r['state'][0] for r in flown])/100])
                # Historical records lack authoritative terminal status. Passing True
                # is an optimistic upper bound on geometric completion, not a reward.
                row['saved_flight_stop_assumed_true']=orbit_diagnostic(poses,spec,stopped=True)
        except ValueError as e:row['excluded_reason']=str(e)
        orbit.append(row)
    # Reproduce v1 historical controls without changing the old scorer.
    audit=ROOT/'reports/d210_dagger_audit/surround_path_audit.json'
    controls=[]
    for row in read(audit):
        if row['run']!='D209':continue
        task=read(val/row['task']);ref=task['reference_path_preprocessed']
        stationary=[dict(state=[[0,0,0],[0,0,0]])]*10
        prefix=[dict(state=[r[:3],r[3:]]) for r in ref[1:105]]
        a=reward(stationary,ref,'Surround')['reward'];b=reward(prefix,ref,'Surround')['reward']
        if not np.isclose(a,row['stationary_reward']['reward'],rtol=0,atol=1e-10):raise AssertionError('historical score changed')
        if not np.isclose(b,row['reference_prefix_reward']['reward'],rtol=0,atol=1e-10):raise AssertionError('historical prefix score changed')
        if not a>b:raise AssertionError('expected saved v1 counterexample')
        controls.append(dict(task=row['task'],v1_stationary=a,v1_partial=b))
    summary=dict(status='BLOCKED_FOR_TRAINING',practice_tasks=len(records),
        kinds=dict(Counter(r['kind'] for r in records)),
        missing_target_geometry=sum(r['target_missing'] for r in records),
        object_relative_missing_geometry=sum(r['target_missing'] and r['kind'] in ('Approach/Move','Pass','Retreat') for r in records),
        derived_object_metadata=sum(bool(r['object_provenance']) for r in records),
        approved_semantic_rewards=0,practice_validation_overlap=0,saved_exclusion_manifest_overlap=0,
        reference_circle_diagnostics=sum('spec' in r for r in orbit),
        circle_fit_exclusions=sum('excluded_reason' in r for r in orbit),
        v1_counterexamples_reproduced=len(controls),
        unverified=['current benchmark manifest identity versus saved exclusion list','semantic review','camera/object identity','stop metadata',
                    'real-model probability parity','frozen-prefix adapter switching','complete checkpoint recovery'])
    for name,data in [('summary',summary),('task_review',records),('orbit_controls',orbit),('historical_controls',controls),('input_hashes',inputs)]:
        (args.out/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# D213 offline preflight review','',
           '**Training remains blocked.** Geometry diagnostics do not establish task semantics.','',
           f"Audited {len(records)} practice tasks; {summary['missing_target_geometry']} have missing/sentinel target geometry. "
           f"{summary['derived_object_metadata']} have derived object metadata requiring review. "
           f"Only {summary['object_relative_missing_geometry']} missing-target cases are object-relative Approach/Pass/Retreat tasks; "
           'a sentinel alone is not a defect for a displacement/rotation task.',
           f"Reproduced {len(controls)} historical stationary-circle reward counterexamples. Historical scorer unchanged.",
           f"Reference-circle controls usable for {summary['reference_circle_diagnostics']}/{len(orbit)} practice circles. "
           'Circle centres/directions fitted from references are diagnostics, not independent expert truth.','',
           '## Concrete semantic issue','',
           'Land task sim_2025-03-27_17-33-04 says “Descend to 4.0 meters altitude”; its unified instruction says “Land 4.0 meters”. '
           'These do not establish whether 4 m is a destination altitude or a displacement, or whether surface contact is intended. '
           'Do not assign a landing-success reward from this label alone.','',
           '## Practice circle controls','',
           '| Task | Reference full circle | Stationary | First 104 moves | Saved flight (optimistic stop) |',
           '|---|---|---|---|---|']
    for r in orbit:
        if 'excluded_reason' in r:lines.append(f"| {r['task']} | Excluded: {r['excluded_reason']} | — | — | — |")
        else:
            value=lambda k:str(r.get(k,{}).get('geometric_complete','missing'))
            lines.append(f"| {r['task']} | {value('reference_complete')} | {value('stationary')} | {value('partial')} | {value('saved_flight_stop_assumed_true')} |")
    lines+=['','## Next work','',
            'Resolve a small practice subset’s task meaning and geometry; review camera/trajectory examples. '
            'Complete PPO rollout provenance, frozen-prefix integration and checkpoint/failure tests before GPU smoke testing. '
            'No cloud run is justified by these offline checks alone.','',
            'Evidence: task_review.json contains every instruction and exclusion reason; orbit_controls.json contains geometry and failures; '
            'historical_controls.json preserves v1 counterexamples; input_hashes.json records exact practice inputs.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
