"""D234 fixed-input D232 checkpoint audit. No learning or flights."""
import os,sys,json,time,hashlib,argparse,atexit,traceback
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--precision',choices=['nf4','bf16'],required=True)
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('output must be a new directory; preserve earlier evidence')
import torch,numpy as np
from PIL import Image
from transformers import GenerationConfig,LogitsProcessor,LogitsProcessorList
from train_uav_flow_vla import setup,official_stats,encode
from uav_flow_action_tokenizer import ActionTokenizer
from composite_policy import CompositePolicy,adapter_digest,runtime_identity
from reload_audit import snapshot,describe,compare,restore_snapshot,save_evidence
OUT=args.output;EV=args.evidence
OUT.mkdir(parents=True)
results={};raw={};weights={};summary=[]
report=dict(passed=False,state='running',precision=args.precision,cases=summary,sampled=results,weights=weights)
def persist():save_evidence(OUT,raw,report)
def exception_hook(kind,value,tb):
 report.update(state='failed',error=str(value),traceback=''.join(traceback.format_exception(kind,value,tb)))
 persist();sys.__excepthook__(kind,value,tb)
sys.excepthook=exception_hook
atexit.register(persist)
persist()
OLD=args.baseline;NEW=args.checkpoint/'adapter/action'
assert hashlib.sha256((args.checkpoint/'manifest.json').read_bytes()).hexdigest()=='fffa58b398c0028ab4186907650810a587d524715f50553ea3bc7fea7cb79cd4', 'unexpected committed manifest'
m=json.loads((args.checkpoint/'manifest.json').read_text())
for name in ('adapter/action/adapter_config.json','adapter/action/adapter_model.safetensors','adapter/action/exact_restore.json'):
 hashes={k.replace('\\','/'):v for k,v in m['files'].items()}
 p=args.checkpoint/name
 if name not in hashes or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[name]:
  raise ValueError('missing or mismatched checkpoint artifact: '+name)
assert adapter_digest(OLD)=='75d55cf748d084c463ccb26842c4266abecbc7e038b29960ee94fea887d1a711'
assert adapter_digest(NEW)=='c79577c6035955272262b7815a66e14ab614797d940785b0deb63942f3fbf7ac'
api,processor,base,_=setup('qwen',args.precision)
policy=CompositePolicy.load(base,OLD,OLD);tok=ActionTokenizer(processor.tokenizer,official_stats())
initial_action=snapshot(policy.model);initial_prefix=snapshot(policy.model,'prefix')
report['runtime']=runtime_identity(policy.model,processor,official_stats())
low=tok.vocab_size-tok.n_bins;high=tok.vocab_size
cases=[]
for f in sorted((EV/'attempts/episodes').glob('sim_*r0.json')):
 ep=json.loads(f.read_text())
 for j in sorted(set([0,max(0,len(ep['samples'])-2)])):
  s=ep['samples'][j];cases.append((ep['task']+':'+str(j),s))
assert len(cases)==8, 'expected eight fixed D232 contexts'
class OnlyAction(LogitsProcessor):
 def __call__(self,ids,scores):
  masked=torch.full_like(scores,float('-inf'));masked[:,low:high]=scores[:,low:high];return masked
start=time.monotonic()
for label,path in [('old',OLD),('old_no_reload',OLD),('new',NEW),('old_reload_unrestored',OLD),('old_repeat',OLD)]:
 if label not in ('old','old_no_reload','old_repeat'):
  policy.model.delete_adapter('action');policy.model.load_adapter(str(path),adapter_name='action',is_trainable=False)
  if (path/'exact_restore.json').exists():policy.restore_action_weights_exact(path)
 if label=='old_repeat':restore_snapshot(policy.model,initial_action)
 policy.activate('prefix');policy.activate('action')
 weights[label]=dict(tensors=describe(snapshot(policy.model)),vs_initial=compare(initial_action,snapshot(policy.model)),prefix_vs_initial=compare(initial_prefix,snapshot(policy.model,'prefix')))
 persist()
 assert policy.model.active_adapters==['action']
 results[label]=[]
 for key,s in cases:
  im=EV/'attempts/rollout_images'/s['image'];assert hashlib.sha256(im.read_bytes()).hexdigest()==s['image_sha256']
  batch=encode(processor,tok,{k:s[k] for k in ('prompt','box_text','progress_text')},8,with_answer=False,image=Image.open(im).convert('RGB'))
  assert batch['input_ids'][0].tolist()==s['context_input_ids']
  n=batch['input_ids'].shape[1];batch=api.cuda(batch)
  class Force(LogitsProcessor):
   def __call__(self,ids,scores):
    out=torch.full_like(scores,float('-inf'));out[:,s['ids'][ids.shape[1]-n]]=0;return out
  with torch.inference_mode():
   forced=policy.model.generate(**batch,generation_config=GenerationConfig(do_sample=False,use_cache=True,min_new_tokens=32,max_new_tokens=32,output_logits=True,return_dict_in_generate=True,pad_token_id=processor.tokenizer.pad_token_id),use_model_defaults=False,logits_processor=LogitsProcessorList([Force()]))
   lp=torch.stack(forced.logits,dim=1)[0,:,low:high].float().log_softmax(-1).cpu().numpy()
   torch.manual_seed(7);torch.cuda.manual_seed_all(7)
   sampled=policy.model.generate(**batch,generation_config=GenerationConfig(do_sample=True,temperature=1.,top_k=0,top_p=1.,use_cache=True,min_new_tokens=32,max_new_tokens=32,pad_token_id=processor.tokenizer.pad_token_id),use_model_defaults=False,logits_processor=LogitsProcessorList([OnlyAction()]))[0,n:].tolist()
  raw[label+':'+key]=lp
  results[label].append(dict(case=key,ids=sampled,active=policy.model.active_adapters))
  report['elapsed_seconds']=time.monotonic()-start;persist()
  print(label,key,'done',round(time.monotonic()-start,1),flush=True)
for i,(key,s) in enumerate(cases):
 b=raw['old:'+key];a=raw['new:'+key];again=raw['old_repeat:'+key]
 report.setdefault('reload_controls',[]).append(dict(case=key,no_reload_max_logp_error=float(abs(b-raw['old_no_reload:'+key]).max()),unrestored_reload_max_logp_error=float(abs(b-raw['old_reload_unrestored:'+key]).max()),unrestored_reload_token_changes=sum(x!=y for x,y in zip(results['old'][i]['ids'],results['old_reload_unrestored'][i]['ids']))))
 summary.append(dict(case=key,old_repeat_max_logp_error=float(abs(b-again).max()),old_repeat_tokens_equal=results['old'][i]['ids']==results['old_repeat'][i]['ids'],max_logp_change=float(abs(b-a).max()),mean_full_categorical_KL=float((np.exp(b)*(b-a)).sum(-1).mean()),argmax_changes=int(np.sum(b.argmax(-1)!=a.argmax(-1))),sampled_token_changes=sum(x!=y for x,y in zip(results['old'][i]['ids'],results['new'][i]['ids']))))
persist()
assert all(x['no_reload_max_logp_error']==0 for x in report['reload_controls']), 'same-model repeat drift'
assert all(results['old'][i]['ids']==results['old_no_reload'][i]['ids'] for i in range(len(cases))), 'same-model sampled token drift'
assert all(x['prefix_vs_initial']['equal'] for x in weights.values()), 'prefix weights changed'
assert weights['old_repeat']['vs_initial']['equal'], 'restored baseline weights differ'
assert all(x['old_repeat_max_logp_error']==0 and x['old_repeat_tokens_equal'] for x in summary), 'exact baseline repeat output mismatch'
assert all(np.isfinite(x).all() for x in raw.values()), 'nonfinite distribution'
assert any(x['max_logp_change']>0 for x in summary), 'candidate has no observed effect'
report.update(passed=True,state='complete',scope='fixed-input adapter sensitivity; baseline repeat restored to initial effective weights; not flight improvement',elapsed_seconds=time.monotonic()-start)
persist()
print(json.dumps(summary),flush=True)
