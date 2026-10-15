"""Forward-only local saved-checkpoint KL diagnosis. No optimizer or simulator."""
import argparse,json,sys,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import torch
from checkpoint_state import verified_checkpoint
from composite_policy import CompositePolicy,adapter_digest,runtime_identity
from ppo_engine import write_kl_failure
from train_path_ppo import make_forward

def strict_json(value):
 if isinstance(value,dict):return {k:strict_json(v) for k,v in value.items()}
 if isinstance(value,list):return [strict_json(v) for v in value]
 if isinstance(value,float) and not math.isfinite(value):return str(value)
 return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-layers',type=int,default=16);p.add_argument('--all-samples',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);start=time.monotonic();cp=verified_checkpoint(a.checkpoint)
 state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False);m=state['metadata'];samples=m['samples'];step=state['step'];del state
 assert adapter_digest(a.prefix)==m['identity']['prefix_adapter_sha256'],'prefix mismatch'
 from qwen_backend import MODEL,AutoProcessor,Qwen3VLForConditionalGeneration
 from train_uav_flow_vla import official_stats
 from uav_flow_action_tokenizer import ActionTokenizer
 processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='sdpa')
 model.requires_grad_(False);del model.hf_device_map;policy=CompositePolicy.load(model,cp/'adapter/action',a.prefix,train_action=True)
 from accelerate import dispatch_model
 device_map={name:'cuda:0' for name in ['base_model.model.model.visual','base_model.model.model.language_model.embed_tokens','base_model.model.model.language_model.norm','base_model.model.model.language_model.rotary_emb','base_model.model.lm_head']}
 for i in range(model.config.text_config.num_hidden_layers):device_map[f'base_model.model.model.language_model.layers.{i}']='cuda:0' if i<a.gpu_layers else 'cpu'
 policy.model=dispatch_model(policy.model,device_map=device_map,main_device='cuda:0')
 class API:
  @staticmethod
  def cuda(batch):return {k:v.to('cuda:0') for k,v in batch.items()}
 runtime=runtime_identity(policy.model,processor,official_stats());diff={k:dict(saved=m['identity'].get(k),local=v) for k,v in runtime.items() if m['identity'].get(k)!=v}
 forward,_=make_forward(API,processor,ActionTokenizer(processor.tokenizer,official_stats()),policy,a.images,8)
 order=[i for group,w in m['orders'][m['cursor']] for i in group];all_order=order+([i for i in range(len(samples)) if i not in order] if a.all_samples else []);records=[];first=None
 print('REPLAY_ORDER',order,'DEVICE_MAP',device_map,flush=True)
 for i in all_order:
  with torch.no_grad():
   lp,_=forward([samples[i]]);old=torch.tensor([samples[i]['logps']],device=lp.device,dtype=lp.dtype);d=(lp-old).sum(-1);kl=(d.exp()-1-d).mean()
   value=float(kl);row=dict(index=i,task=samples[i]['task'],passage=samples[i]['passage'],call_index=samples[i]['call_index'],kl=value,log_ratio=float(d[0]),ratio=float(d.exp()[0]),old_logps=old.cpu().tolist(),new_logps=lp.cpu().tolist())
   if first is None and i in order and (not torch.isfinite(kl) or value>m['config']['hard_kl']):
    first=row
    write_kl_failure(a.out/'diagnostic_checkpoints',phase='pre_step',step=step,window=m['cursor'],indices=[i],samples=samples,new=lp,old=old,threshold=m['config']['hard_kl'],estimate=kl)
  records.append(row);avg=sum(x['kl'] for x in records)/len(records)
  report=dict(checkpoint_step=step,next_order=order,records=records,first_guard_trigger=first,mean_so_far=avg,saved_full_batch_kl=m['logs'][-1]['post_kl'],runtime_differences=diff,device_map=device_map,execution='BF16 CUDA inference with CPU weight offload, no_grad; not identical cloud hardware',updates_added=0,elapsed_seconds=time.monotonic()-start,complete=len(records)==len(all_order),full_batch_complete=len(records)==len(samples),scope='all_samples' if a.all_samples else 'next_update_window')
  if report['full_batch_complete']:report['full_batch_kl_error']=abs(avg-m['logs'][-1]['post_kl'])
  (a.out/'replay.json').write_text(json.dumps(strict_json(report),indent=2,allow_nan=False));print('REPLAY',i,value,'elapsed',round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':main()







