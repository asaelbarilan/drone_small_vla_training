"""D226 local-only wording ablation; fixed numeric goals and saved visual context."""
import faulthandler
faulthandler.enable();faulthandler.dump_traceback_later(600,exit=True)
import json,sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig,LogitsProcessor,LogitsProcessorList
from numeric_goals import numeric_progress
from train_uav_flow_vla import encode,progress_text,OFFICIAL_PROMPT
from uav_flow_eval_server import qwen_loader,STATE
from composite_policy import adapter_digest
ADAPTER=Path('D:/drone_vla_pilot/adapters/d207_s8000');OUT=ROOT/'reports/d226_numeric_goal_fix/wording.json'
assert adapter_digest(ADAPTER)=='75d55cf748d084c463ccb26842c4266abecbc7e038b29960ee94fea887d1a711'
STATE.update(goal=None,yaw_total=0.,last_yaw=None)
print('LOADING',flush=True)
pred=qwen_loader(str(ADAPTER),8,'nf4',temperature=1.,progress=True,goal_memory=True,box=True,prefix_adapter=str(ADAPTER),numeric_goals=True)
api,processor,tok=pred.runtime;model=pred.policy.activate('action');low=tok.vocab_size-256;high=tok.vocab_size
class Mask(LogitsProcessor):
 def __call__(self,ids,scores):
  out=torch.full_like(scores,float('-inf'));out[:,low:high]=scores[:,low:high];return out
config=GenerationConfig(do_sample=True,temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.,min_new_tokens=32,max_new_tokens=32,pad_token_id=processor.tokenizer.pad_token_id)
report=dict(scope='NF4 saved-input wording ablation, seed7, no learning or flight',cases=[])
folders=[(ROOT/'reports/d223_nonlanding_preflight/evidence/policy',0),(ROOT/'reports/d224_reward_gated_ppo/evidence/attempts',0),(ROOT/'reports/d224_reward_gated_ppo/evidence/attempts',1)]
for base,repeat in folders:
 e=json.loads((base/'episodes'/f'sim_2025-03-27_19-11-33_r{repeat}.json').read_text());s=e['samples'][0];image=Image.open(base/'rollout_images'/s['image']).convert('RGB')
 used,meta=numeric_progress(s['instr'],[0,0,0],0)
 for name,instr in [('raw',s['instr']),('components','Move 1.294 metres forward and 4.830 metres down simultaneously, then stop.'),('normalized','Descend 5 meters at 75 degrees')]:
  row=dict(prompt=OFFICIAL_PROMPT.format(state='0.0,0.0,0.0,0.0',instruction=instr),box_text=s['box_text'],progress_text=progress_text(used));batch=encode(processor,tok,row,8,with_answer=False,image=image)
  torch.manual_seed(7);torch.cuda.manual_seed_all(7)
  with torch.inference_mode():seq=model.generate(**api.cuda(batch),generation_config=config,use_model_defaults=False,logits_processor=LogitsProcessorList([Mask()]))
  ids=seq[0,batch['input_ids'].shape[1]:].tolist();actions=tok.decode(ids,8);xyz=np.zeros(3);yaw=0.
  for dx,dy,dz,dyaw in actions:
   c,ss=math.cos(yaw),math.sin(yaw);xyz+=np.array([c*dx-ss*dy,ss*dx+c*dy,dz]);yaw+=dyaw
  goal=np.array(meta['goal_start_frame'][:3]);alignment=float(xyz@goal/max(1e-12,np.linalg.norm(xyz)*np.linalg.norm(goal)));error=float(np.linalg.norm(xyz-goal))
  case=dict(input_seed=e['seed'],variant=name,instruction=instr,image_sha256=s['image_sha256'],box_text=s['box_text'],progress_text=row['progress_text'],chunk_delta_m=xyz.tolist(),endpoint_error_m=error,alignment=alignment,pass_direction=bool(alignment>.9 and abs(xyz[1])<.1 and error<np.linalg.norm(goal)),actions=actions.tolist())
  report['cases'].append(case);OUT.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in case.items() if k not in ['actions','box_text']}),flush=True)
faulthandler.cancel_dump_traceback_later();print('DONE',flush=True)
