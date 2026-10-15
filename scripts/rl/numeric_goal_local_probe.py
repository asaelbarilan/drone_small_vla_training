"""D226 no-learning saved-observation action counterfactual. Local NF4 only."""
import argparse,json,math,sys,time,hashlib,faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(600, exit=True)
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig,LogitsProcessor,LogitsProcessorList
from numeric_goals import numeric_progress,numeric_action_instruction
from train_uav_flow_vla import encode,progress_text,OFFICIAL_PROMPT
from composite_policy import adapter_digest
from uav_flow_eval_server import qwen_loader,STATE

def main():
 p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 print('VERIFYING_ADAPTER',flush=True)
 digest=adapter_digest(a.adapter)
 assert digest=='75d55cf748d084c463ccb26842c4266abecbc7e038b29960ee94fea887d1a711',digest
 report=dict(adapter_sha256=digest,precision='nf4',scope='same saved image/box/state; sampled actions with seed7; no flight or learning; not BF16 equivalence',cases=[])
 def save(): (a.out/'actions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 STATE.update(goal=None,yaw_total=0.,last_yaw=None)
 print('LOADING_LOCAL_MODEL',flush=True)
 predictor=qwen_loader(str(a.adapter),8,'nf4',temperature=1.,progress=True,goal_memory=True,box=True,prefix_adapter=str(a.adapter),numeric_goals=True)
 print('MODEL_LOADED',flush=True)
 report['policy_provenance']=predictor.policy.provenance
 api,processor,tok=predictor.runtime;model=predictor.policy.activate('action');low=tok.vocab_size-256;high=tok.vocab_size
 class Mask(LogitsProcessor):
  def __call__(self,ids,scores):
   out=torch.full_like(scores,float('-inf'));out[:,low:high]=scores[:,low:high];return out
 config=GenerationConfig(do_sample=True,temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.,min_new_tokens=32,max_new_tokens=32,output_logits=True,output_scores=True,return_dict_in_generate=True,pad_token_id=processor.tokenizer.pad_token_id)
 effective,_=model._prepare_generation_config(config,False)
 assert (effective.temperature,effective.top_p,effective.top_k)==(1.,1.,0)
 report['effective_sampling']=dict(temperature=effective.temperature,top_p=effective.top_p,top_k=effective.top_k,use_model_defaults=False)
 base=ROOT/'reports/d224_reward_gated_ppo/evidence/attempts'
 selections=[('turn_start','15-41-24',0,None),('vertical_start','17-33-04',0,None),('vertical_0828_remaining','17-33-04',3,None),('ascent_start','17-35-01',0,None),('slope_descent_start','19-11-33',0,None),('vertical_at_goal_synthetic_pose','17-33-04',3,[0,0,-400,0])]
 start=time.monotonic()
 for label,part,index,override in selections:
  ep=json.loads(next((base/'episodes').glob('*'+part+'_r0.json')).read_text());s=ep['samples'][index]
  image=Image.open(base/'rollout_images'/s['image']).convert('RGB');pose=override or s['proprio'];metres=[v/100 for v in pose[:3]]+[pose[3]]
  used,meta=numeric_progress(s['instr'],metres,pose[3]);new=progress_text(used)
  row=dict(prompt=OFFICIAL_PROMPT.format(state=','.join(str(round(float(x),1)) for x in metres),instruction=s['instr']),box_text=s['box_text'])
  case=dict(label=label,task=ep['task'],image=s['image'],synthetic_pose=override is not None,state=pose,goal=meta,variants={})
  for variant,line in [('old_context',s['progress_text']),('fixed_context',new)]:
   action_instruction=s['instr'] if variant=='old_context' else numeric_action_instruction(s['instr'])
   row['prompt']=OFFICIAL_PROMPT.format(state=','.join(str(round(float(x),1)) for x in metres),instruction=action_instruction)
   row['progress_text']=line;batch=encode(processor,tok,row,8,with_answer=False,image=image)
   torch.manual_seed(7);torch.cuda.manual_seed_all(7)
   with torch.inference_mode():out=model.generate(**api.cuda(batch),generation_config=config,use_model_defaults=False,logits_processor=LogitsProcessorList([Mask()]))
   ids=out.sequences[0,batch['input_ids'].shape[1]:].tolist();actions=tok.decode(ids,8)
   logits=torch.stack(out.logits,dim=1)[0,:,low:high].float();lp=logits.log_softmax(-1);probs=lp.exp();entropy=-(probs*lp).sum(-1)
   sampled_scores=torch.stack(out.scores,dim=1)[0,:,low:high].float()
   sampler_error=(sampled_scores.log_softmax(-1)-lp).abs().max().item()
   assert sampler_error==0., sampler_error
   xyz=np.array(metres[:3]);yaw=math.radians(pose[3]);before=xyz.copy();y0=yaw
   for dx,dy,dz,dyaw in actions:
    c,ss=math.cos(yaw),math.sin(yaw);xyz+=np.array([c*dx-ss*dy,ss*dx+c*dy,dz]);yaw+=dyaw
   goal=np.array(meta['goal_start_frame'][:3]);still=all(max(abs(v[:3]))<.03 and abs(math.degrees(v[3]))<1 for v in actions)
   data=dict(sampler_logp_max_error=sampler_error,action_instruction=action_instruction,progress_text=line.strip(),actions_m_rad=actions.tolist(),chunk_delta_m=(xyz-before).tolist(),chunk_yaw_deg=math.degrees(yaw-y0),start_endpoint_error_m=float(np.linalg.norm(before-goal)),proposed_endpoint_error_m=float(np.linalg.norm(xyz-goal)),still_chunk=still,mean_entropy_nats=entropy.mean().item(),min_max_token_probability=probs.max(-1).values.min().item(),ids=ids)
   case['variants'][variant]=data
   del out,logits,lp,probs,entropy,batch
   print(label,variant,json.dumps({k:v for k,v in data.items() if k in ['progress_text','chunk_delta_m','chunk_yaw_deg','proposed_endpoint_error_m','still_chunk']}),flush=True)
  fixed=case['variants']['fixed_context']
  if label=='turn_start':ok=0<fixed['chunk_yaw_deg']<60 and max(abs(v) for v in fixed['chunk_delta_m'])<.1
  elif override is not None:ok=fixed['still_chunk']
  else:
   delta=np.array(fixed['chunk_delta_m']);desired=np.array(meta['goal_start_frame'][:3])-np.array(metres[:3])
   alignment=float(delta@desired/max(1e-12,np.linalg.norm(delta)*np.linalg.norm(desired)))
   case['direction_alignment_cosine']=alignment
   ok=fixed['proposed_endpoint_error_m']<fixed['start_endpoint_error_m'] and not fixed['still_chunk'] and alignment>.9 and abs(delta[1])<.1
  case['direction_or_stop_check']=bool(ok);report['cases'].append(case);report['elapsed_seconds']=time.monotonic()-start;save()
 report['all_action_checks_pass']=all(c['direction_or_stop_check'] for c in report['cases']);save()
 faulthandler.cancel_dump_traceback_later()
 print('DONE',report['all_action_checks_pass'],flush=True)
if __name__=='__main__':main()
