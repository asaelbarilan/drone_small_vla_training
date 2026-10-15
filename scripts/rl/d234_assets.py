"""D234 inexpensive deployment prerequisites before loading GPU weights."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
code=Path(__file__).resolve().parents[2];r=a.root
stats_path=code/'reports/uav_flow_official_10shard/manifest.json'
manifest=json.loads(stats_path.read_text());stats=manifest['action_stats_train']
digest=hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()
files=sorted((r.parent/'diagnostic_evidence/attempts/episodes').glob('sim_*r0.json'))
if len(files)!=4:raise ValueError('four baseline diagnostic episodes required')
for f in files:
 ep=json.loads(f.read_text())
 for s in ep['samples']:
  if s['policy_provenance']['action_stats_sha256']!=digest:raise ValueError('action normalization differs from saved rollout')
parts=json.loads((code/'reports/d234_full_ppo_preparation/partitions.json').read_text())
for group in ('training','development','validation'):
 for item in parts[group]:
  path=code/item['path']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('task asset hash mismatch')
for path in ('C:/models/qwen3vl4b/config.json','C:/adapters/d207_stage2_adapter_s8000/adapter_config.json','C:/venvs/eval/Scripts/python.exe'):
 if not Path(path).is_file():raise ValueError('missing runtime asset '+path)
r.mkdir(parents=True,exist_ok=True)
(r/'assets.json').write_text(json.dumps(dict(status='passed',action_stats_sha256=digest,task_files=sum(len(parts[k]) for k in ('training','development','validation'))),indent=2))
