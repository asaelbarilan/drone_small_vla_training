"""D240: resume saved D238 step one, three more updates, two candidate flights, review."""
import argparse,os,sys,time,shutil,hashlib,traceback
from pathlib import Path
import torch
from bounded_job import run_stages
from checkpoint_state import verified_checkpoint
from d234_support import read,save,episodes,comparison,behavior_gate
from composite_policy import adapter_digest
CODE=Path(__file__).resolve().parents[2]
BASE=Path('C:/adapters/d207_stage2_adapter_s8000')

def verify_reuse(source,source_code,manifest):
 for group,root in [('files',source),('inference_code',source_code)]:
  for relative,digest in manifest[group].items():
   path=(root/relative).resolve()
   if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
    raise ValueError('baseline reuse checksum mismatch: '+relative)
 before=episodes(source/'before_dev')
 if sorted(e['task'] for e in before)!=sorted(manifest['expected_tasks']):raise ValueError('baseline task mismatch')
 for e in before:
  if e['seed']!=manifest['seed'] or e['reward']['version']!='movement-reward-candidate-v5':raise ValueError('baseline settings mismatch')
  for s in e['samples']:
   p=s['policy_provenance']
   if p['action_adapter_sha256']!=manifest['baseline_adapter_sha256'] or p['prefix_adapter_sha256']!=manifest['baseline_adapter_sha256'] or p['precision']!='bf16':raise ValueError('baseline policy mismatch')
 return before

def verify_four(report):
 if report['step']!=4 or [x['step'] for x in report['updates']]!=[1,2,3,4]:raise ValueError('four committed updates required')
 if report['kl_policy']!='full_batch_mean_v2' or not report['frozen_adapters_unchanged'] or not report['reload_logps_exact']:raise ValueError('training invariant failed')

def action_comparison(before,after,out):
 from transformers import AutoTokenizer
 from train_uav_flow_vla import official_stats
 from uav_flow_action_tokenizer import ActionTokenizer
 decoder=ActionTokenizer(AutoTokenizer.from_pretrained('C:/models/qwen3vl4b',local_files_only=True),official_stats())
 b={e['task']:e for e in episodes(before)};a={e['task']:e for e in episodes(after)};rows=[]
 for task in sorted(b):
  old=[i for s in b[task]['samples'] for i in s['ids']];new=[i for s in a[task]['samples'] for i in s['ids']]
  rows.append(dict(task=task,action_tokens_changed=old!=new,common_token_mismatches=sum(x!=y for x,y in zip(old,new)),before_tokens=old,after_tokens=new,before_decoded=[decoder.decode(s['ids'],8).tolist() for s in b[task]['samples']],after_decoded=[decoder.decode(s['ids'],8).tolist() for s in a[task]['samples']],before_poses=b[task]['poses'],after_poses=a[task]['poses'],before_executed_counts=b[task]['executed_counts'],after_executed_counts=a[task]['executed_counts']))
 save(out,dict(rows=rows,scope='Matched task/seed; images may differ; changes are not proof of improvement.'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('C:/runs/d238_two_task/results'));p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args()
 r=a.root;r.mkdir(parents=True,exist_ok=True);start=time.monotonic();completed=[]
 if (r/'started.json').exists():raise ValueError('already dispatched')
 save(r/'started.json',dict(pid=os.getpid(),wall_seconds=a.wall_seconds))
 def remaining():return a.wall_seconds-(time.monotonic()-start)
 def stage(name,args,limit,required,reserve):
  if remaining()<reserve+60:raise TimeoutError('comparison/upload reserve reached')
  save(r/'status.json',dict(name=name,state='running',completed=completed,remaining_seconds=remaining()))
  run_stages([dict(name=name,argv=[sys.executable,'-u',*map(str,args)],cwd=str(CODE),timeout_seconds=limit,required_outputs=[str(required)])],r/'stages'/name,wall_seconds=remaining(),reserve_seconds=reserve,env=os.environ.copy())
  completed.append(name)
 try:
  manifest=read(CODE/'reports/d240_resume_review/reuse_manifest.json');source=a.source
  verify_reuse(source,source.parent/'code',manifest)
  if adapter_digest(BASE)!=manifest['baseline_adapter_sha256']:raise ValueError('baseline adapter changed')
  oldroot=source/'round_1/training/checkpoints';cp=verified_checkpoint(oldroot)
  if read(oldroot/'CURRENT.json')!=manifest['checkpoint_pointer']:raise ValueError('unexpected resume checkpoint')
  state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False);m=state['metadata'];cfg=m['config']
  if state['step']!=1 or m['cursor']!=1 or len(m['orders'])!=4 or cfg['max_updates']!=4:raise ValueError('resume must start at one of four scheduled updates')
  rollouts=source/'round_1/admitted_rollouts.json'
  if hashlib.sha256(rollouts.read_bytes()).hexdigest()!=m['identity']['rollouts_sha256'] or m['identity']['rollouts_sha256']!=manifest['training_rollouts_sha256']:raise ValueError('rollouts changed')
  for sample in m['samples']:
   image=source/'round_1/batch/images'/sample['image']
   if hashlib.sha256(image.read_bytes()).hexdigest()!=sample['image_sha256']:raise ValueError('training image changed')
  del state
  for name in ['before_dev','dev_tasks']:shutil.copytree(source/name,r/name)
  train=r/'training';ck=train/'checkpoints';ck.mkdir(parents=True)
  shutil.copytree(cp,ck/cp.name);shutil.copyfile(oldroot/'CURRENT.json',ck/'CURRENT.json')
  save(r/'reuse.json',dict(baseline_reflights=0,new_collection_flights=0,source_checkpoint=cp.name,starting_step=1,additional_update_cap=3,source_config=cfg,baseline_manifest=manifest,kl_rule='full_batch_mean_v2',comparison_seed=7,automatic_continuation=False))
  completed.append('reuse_verified')
  # Keep saved engine config identical for optimizer resume. The external stage
  # and independent shutdown impose the shorter new wall deadline.
  budget=min(4500,remaining()-1800)
  if budget<1200:raise TimeoutError('insufficient training time with comparison reserve')
  stage('resume_to_4',['scripts/rl/train_path_ppo.py','--resume','--rollouts',rollouts,'--images',source/'round_1/batch/images','--action-adapter',BASE,'--prefix-adapter',BASE,'--out',train,'--max-updates',4,'--max-seconds',cfg['max_seconds'],'--lr','2.5e-7','--value-lr','1e-4','--batch-size',1,'--accum',8,'--pause-after-updates',4],budget,train/'report.json',1800)
  report=read(train/'report.json');verify_four(report)
  candidate=verified_checkpoint(ck)/'adapter/action'
  stage('after_dev',['scripts/rl/d234_flights.py','--root',r/'after_dev','--tasks',r/'dev_tasks','--action-adapter',candidate,'--repeats',1,'--seed',7,'--wall-seconds',1080],1500,r/'after_dev/completion.json',180)
  check=comparison(r/'before_dev',r/'after_dev',r/'comparison.json',2)
  action_comparison(r/'before_dev',r/'after_dev',r/'action_comparison.json')
  diagnostic=behavior_gate(r/'before_dev',r/'after_dev',check)
  save(r/'review_required.json',dict(reason='User review required before any continuation',updates=4,new_updates=3,automatic_continuation=False,diagnostic=diagnostic))
  save(r/'completion.json',dict(state='awaiting_user_review',updates=4,new_updates=3,comparison_pairs=2,baseline_reflights=0,checkpoint=str(candidate),automatic_continuation=False,elapsed_seconds=time.monotonic()-start))
  save(r/'status.json',dict(state='awaiting_user_review',completed=completed))
 except Exception as e:
  save(r/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),completed=completed,elapsed_seconds=time.monotonic()-start));raise
if __name__=='__main__':main()
