"""Bounded local NF4 plumbing check. Synthetic rewards, no simulator flights.

Uses an available older local adapter when D208 is absent. Its results cannot
validate D208/BF16 or justify promoting a trained adapter.
"""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'scripts/rl'))
from train_path_ppo import make_forward
from ppo_engine import train_batch,PPOConfig
from checkpoint_state import verified_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--parity-only',action='store_true');p.add_argument('--chunk',type=int,default=8);p.add_argument('--precision',choices=['nf4','bf16'],default='nf4');p.add_argument('--max-seconds',type=int,default=180)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    report=dict(scope='saved-observation synthetic-reward PPO preflight',adapter=str(args.adapter),promotable=False,chunk=args.chunk,precision=args.precision,phases={})
    try:
        from uav_flow_eval_server import qwen_loader,STATE
        STATE.update(goal=None,yaw_total=0.,last_yaw=None)
        torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        torch.cuda.reset_peak_memory_stats()
        predictor=qwen_loader(str(args.adapter),args.chunk,args.precision,temperature=1.,progress=True,goal_memory=True,box=True,prefix_adapter=str(args.adapter))
        report['phases']['load_seconds']=time.monotonic()-start;print('MODEL_LOADED',flush=True)
        policy=predictor.policy;api,processor,tokenizer=predictor.runtime
        practice=ROOT/'reports/d210_dagger_audit/practice'
        image_path=practice/'rollout_images/ep00064_c000.png'
        task=json.loads((practice/'tasks/sim_2025-04-30_15-37-22.json').read_text())
        image=Image.open(image_path).convert('RGB');samples=[]
        for i in range(2):
            t=time.monotonic();predictor(image,[0,0,0,0],task['instruction'])
            sample=dict(STATE.pop('sample'));sample.update(task=f'synthetic_{i}',passage=0,call_index=0,
                image=image_path.name,image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                reward_v2=float(i),reward_spec_approved=True,terminated=True,truncated=False,discount_per_call=1.)
            samples.append(sample);report['phases'][f'sample_{i}_seconds']=time.monotonic()-t;print('SAMPLED',i,flush=True)
        policy.train_action=True;params=policy.action_parameters()
        cfg=policy.model.config;hidden=getattr(getattr(cfg,'text_config',cfg),'hidden_size')
        critic=torch.nn.Linear(hidden,1).cuda();torch.nn.init.zeros_(critic.weight);torch.nn.init.zeros_(critic.bias)
        optimizer=torch.optim.AdamW([dict(params=params,lr=1e-6),dict(params=critic.parameters(),lr=1e-4)])
        forward,bootstrap=make_forward(api,processor,tokenizer,policy,practice/'rollout_images',args.chunk)
        if args.parity_only:
            probe={}
            for mode in ('grad','no_grad','inference'):
                ctx=torch.enable_grad() if mode=='grad' else (torch.no_grad() if mode=='no_grad' else torch.inference_mode())
                with ctx:
                    lp,_=forward(samples[:1])
                    differences=(lp.detach().cpu()-torch.tensor([samples[0]['logps']]))[0]
                    probe[mode]=dict(max_token=float(differences.abs().max()),joint=float(differences.sum()),per_token=differences.tolist())
            from train_uav_flow_vla import encode
            from transformers import LogitsProcessor,LogitsProcessorList
            sample=samples[0]
            row={k:sample[k] for k in ('prompt','box_text','progress_text')}
            batch=encode(processor,tokenizer,row,args.chunk,with_answer=False,image=image)
            prefix_length=batch['input_ids'].shape[1]
            class ForceRecorded(LogitsProcessor):
                def __call__(self,ids,scores):
                    masked=torch.full_like(scores,float('-inf'))
                    masked[:,sample['ids'][ids.shape[1]-prefix_length]]=0
                    return masked
            low=tokenizer.vocab_size-tokenizer.n_bins
            for cache in (True,False):
                with torch.inference_mode():
                    out=policy.model.generate(**api.cuda(batch),do_sample=False,use_cache=cache,
                        min_new_tokens=4*args.chunk,max_new_tokens=4*args.chunk,output_logits=True,return_dict_in_generate=True,
                        logits_processor=LogitsProcessorList([ForceRecorded()]),pad_token_id=processor.tokenizer.pad_token_id)
                    logits=torch.stack(out.logits,dim=1)[0,:,low:tokenizer.vocab_size].float()
                    ids=torch.tensor(sample['ids'],device=logits.device)-low
                    lp=logits.log_softmax(-1).gather(1,ids[:,None]).flatten().cpu()
                    diff=lp-torch.tensor(sample['logps'])
                    probe['forced_generation_cache_'+str(cache)]=dict(max_token=float(diff.abs().max()),joint=float(diff.sum()))
            report['parity_probe']=probe
            report['status']='diagnostic_only'
            return 0
        def fingerprint(predicate):
            h=hashlib.sha256()
            for name,param in policy.model.named_parameters():
                if predicate(name,param):
                    h.update(name.encode());h.update(param.detach().cpu().view(torch.uint8).numpy().tobytes())
            return h.hexdigest()
        frozen=lambda n,p:'.prefix.' in n or (('visual' in n or 'vision_' in n) and 'lora_' in n)
        before_frozen=fingerprint(frozen)
        before_action=fingerprint(lambda n,p:p.requires_grad)
        config=PPOConfig(chunk=args.chunk,batch_size=1,accum=2,warmup_steps=1,max_updates=1,max_seconds=args.max_seconds)
        identity={'scope':'synthetic smoke only','precision':args.precision}
        (args.out/'samples.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
        report['update']=train_batch(policy,critic,optimizer,samples,forward,bootstrap,args.out/'checkpoints',
            config=config,identity=identity)
        report['action_changed']=before_action!=fingerprint(lambda n,p:p.requires_grad)
        report['frozen_adapters_unchanged']=before_frozen==fingerprint(frozen)
        if not report['action_changed'] or not report['frozen_adapters_unchanged']:
            raise ValueError('actor update/frozen parameter invariant failed')
        with torch.no_grad():expected=forward(samples[:1])[0].cpu()
        saved=verified_checkpoint(args.out/'checkpoints')/'adapter/action'
        policy.model.delete_adapter('action')
        policy.model.load_adapter(str(saved),adapter_name='action',is_trainable=True)
        policy.restore_action_weights_exact(saved)
        report['reload_action_weights_exact']=True
        optimizer=torch.optim.AdamW([dict(params=policy.action_parameters(),lr=1e-6),dict(params=critic.parameters(),lr=1e-4)])
        with torch.no_grad():critic.weight.fill_(99);critic.bias.fill_(99)
        report['reload']=train_batch(policy,critic,optimizer,samples,forward,bootstrap,args.out/'checkpoints',
            config=config,identity=identity,resume=True)
        with torch.no_grad():actual=forward(samples[:1])[0].cpu()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        report['reload_logps_exact']=True
        report['status']='passed'
    except Exception as e:
        report['status']='failed';report['error']=str(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
        (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
