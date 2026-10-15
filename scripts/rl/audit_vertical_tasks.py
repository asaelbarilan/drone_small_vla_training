"""Build isolated D221 practice repairs and local counterexample evidence."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from vertical_tasks import repair_vertical_task,vertical_reward
OUT=ROOT/'reports/d221_vertical_landing'
def main():
    (OUT/'tasks').mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted((ROOT/'reports/d210_dagger_audit/practice/tasks').glob('*.json')):
        t=json.loads(path.read_text(encoding='utf-8-sig'))
        if t['motion_kind'] not in ('Land','Ascend/Descend'):continue
        new,spec=repair_vertical_task(t)
        row=dict(task=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),spec=spec)
        if spec['status']=='local_geometry_verified':
            (OUT/'tasks'/path.name).write_text(json.dumps(new,indent=2),encoding='utf-8')
            r=np.asarray(new['reference_path_preprocessed']);cases={'correct':r,'stationary':r[:1].repeat(2,axis=0),
                'opposite':-r,'half':r[:max(2,len(r)//2)],'overshoot':r*1.5}
            row['checks']={}
            for label,poses in cases.items():
                result=vertical_reward(poses,new,stopped=True)
                row['checks'][label]={k:result[k] for k in ['reward','complete','endpoint_position_error_m']}
            assert row['checks']['correct']['complete']
            for label in ['stationary','opposite','half','overshoot']:
                assert not row['checks'][label]['complete']
                assert row['checks']['correct']['reward']>row['checks'][label]['reward']
        rows.append(row)
    (OUT/'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(dict(repaired=sum(r['spec']['status']=='local_geometry_verified' for r in rows),
                         blocked=sum(r['spec']['status']!='local_geometry_verified' for r in rows)),indent=2))
if __name__=='__main__':main()
