"""D215 entry point for verified geometry-reward PPO artifacts.

Does not collect flights or launch cloud resources. Requires a complete approved
rollout artifact, exact policy hashes, bounded update/time limits and local model.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from composite_policy import CompositePolicy,adapter_digest,runtime_identity
from checkpoint_state import verified_checkpoint
from ppo_engine import PPOConfig,train_batch
from ppo_math import validate_ppo_samples


def make_forward(api,processor,tokenizer,policy,images,chunk):
    from train_uav_flow_vla import encode,collate
    low=tokenizer.vocab_size-tokenizer.n_bins;high=tokenizer.vocab_size
    def context(sample):
        image_path=(Path(images)/sample['image']).resolve()
        if not image_path.is_relative_to(Path(images).resolve()):raise ValueError('image path escapes root')
        if hashlib.sha256(image_path.read_bytes()).hexdigest()!=sample['image_sha256']:
            raise ValueError('rollout image checksum mismatch')
        image=Image.open(image_path).convert('RGB')
        row={k:sample[k] for k in ('prompt','box_text','progress_text')}
        batch=encode(processor,tokenizer,row,chunk,with_answer=False,image=image)
        if batch['input_ids'][0].tolist()!=sample['context_input_ids']:
            raise ValueError('exact rollout context mismatch')
        return batch
    def forward(samples):
        # Match the autoregressive cache path used for behavior sampling. A full
        # teacher-forced NF4 pass failed the measured probability parity gate.
        from model_replay import replay_cached_tokens
        policy.activate('action');probs=[];features=[]
        for sample in samples:
            batch=api.cuda(context(sample));ids=sample['ids']
            if len(ids)!=4*chunk:raise ValueError('invalid sampled action token count')
            lp,feature=replay_cached_tokens(policy.model.get_base_model(),batch,ids,
                low=low,high=high,temperature=sample['temperature'],
                pad_token_id=processor.tokenizer.pad_token_id)
            probs.append(lp);features.append(feature)
        return torch.stack(probs),torch.stack(features)
    def bootstrap(sample):
        policy.activate('action');b=context(sample['final_observation'])
        out=policy.model(**api.cuda(b),output_hidden_states=True,use_cache=False)
        return out.hidden_states[-1][:,-1]
    return forward,bootstrap



def load_initial_critic(critic, path):
    """D251: explicit linear value initialization, without actor or optimizer access."""
    state=torch.load(path,map_location='cpu',weights_only=True)
    current=critic.state_dict()
    if set(state)!=set(current) or any(state[k].shape!=current[k].shape for k in current):
        raise ValueError('initial critic architecture mismatch')
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite initial critic')
    critic.load_state_dict(state,strict=True)
    for k,v in critic.state_dict().items():
        if not torch.equal(v.detach().cpu(),state[k]):raise ValueError('initial critic changed on load')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--images',type=Path,required=True)
    p.add_argument('--action-adapter',type=Path,required=True);p.add_argument('--prefix-adapter',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--initial-critic',type=Path,help='Explicit D251 linear critic; disables warmup to preserve the verified fit')
    p.add_argument('--precision',choices=['bf16','nf4'],default='bf16')
    p.add_argument('--lr',type=float,default=1e-6);p.add_argument('--value-lr',type=float,default=1e-4)
    p.add_argument('--max-updates',type=int,required=True);p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accum',type=int,default=8)
    p.add_argument('--pause-after-updates',type=int)
    p.add_argument('--advantage-mode',choices=['centered','scale_only'],default='centered')
    args=p.parse_args()
    if args.resume and args.initial_critic:raise ValueError('resume must restore its own critic; cannot replace it')
    artifact=json.loads(args.rollouts.read_text(encoding='utf-8'))
    if artifact.get('schema')!='executed-path-ppo-v1' or artifact.get('preflight_passed') is not True:
        raise ValueError('unapproved or unknown rollout artifact')
    if args.lr<=0 or args.value_lr<=0:raise ValueError('positive learning rates required')
    samples=artifact['samples'];chunk=artifact['chunk'];validate_ppo_samples(samples,chunk,require_final_value=False)
    from sampling_contract import validate_sampling_contract
    validate_sampling_contract(samples)  # reject legacy warped-sampler batches before model loading
    expected=dict(action_adapter_sha256=adapter_digest(args.action_adapter),prefix_adapter_sha256=adapter_digest(args.prefix_adapter))
    for s in samples:
        provenance=s.get('policy_provenance',{})
        if any(provenance.get(k)!=v for k,v in expected.items()):raise ValueError('rollout policy identity mismatch')
        if provenance.get('precision')!=args.precision:raise ValueError('collection/training precision mismatch')
    from train_uav_flow_vla import setup,official_stats
    from uav_flow_action_tokenizer import ActionTokenizer
    identity=dict(rollouts_sha256=hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),precision=args.precision,**expected)
    api,processor,base,_=setup('qwen',args.precision)
    action=args.action_adapter
    if args.resume:
        saved=verified_checkpoint(args.out/'checkpoints')/'adapter'
        action=saved/'action' if (saved/'action').is_dir() else saved
    policy=CompositePolicy.load(base,action,args.prefix_adapter,train_action=True)
    runtime=runtime_identity(policy.model,processor,official_stats())
    for sample in samples:
        if any(sample['policy_provenance'].get(k)!=v for k,v in runtime.items()):
            raise ValueError('collection/training runtime or encoding identity mismatch')
    identity.update(runtime)
    tokenizer=ActionTokenizer(processor.tokenizer,official_stats())
    cfg=policy.model.config;hidden=getattr(getattr(cfg,'text_config',cfg),'hidden_size')
    critic=torch.nn.Linear(hidden,1).to(next(policy.model.parameters()).device)
    torch.nn.init.zeros_(critic.weight);torch.nn.init.zeros_(critic.bias)
    if args.initial_critic:identity['initial_critic_sha256']=load_initial_critic(critic,args.initial_critic)
    optimizer=torch.optim.AdamW([dict(params=policy.action_parameters(),lr=args.lr),dict(params=critic.parameters(),lr=args.value_lr)])
    forward,bootstrap=make_forward(api,processor,tokenizer,policy,args.images,chunk)
    config=PPOConfig(chunk=chunk,batch_size=args.batch_size,accum=args.accum,max_updates=args.max_updates,max_seconds=args.max_seconds,advantage_mode=args.advantage_mode,warmup_steps=0 if args.initial_critic else 8)
    args.out.mkdir(parents=True,exist_ok=True)
    def fingerprint(predicate):
        h=hashlib.sha256()
        for name,param in policy.model.named_parameters():
            if predicate(name,param):
                h.update(name.encode());h.update(param.detach().cpu().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    frozen=lambda n,p:'.prefix.' in n or (('visual' in n or 'vision_' in n) and 'lora_' in n)
    before_frozen=fingerprint(frozen);before_action=fingerprint(lambda n,p:p.requires_grad)
    report=train_batch(policy,critic,optimizer,samples,forward,bootstrap,args.out/'checkpoints',config=config,resume=args.resume,identity=identity,pause_after_updates=args.pause_after_updates)
    report['action_changed']=before_action!=fingerprint(lambda n,p:p.requires_grad)
    report['frozen_adapters_unchanged']=before_frozen==fingerprint(frozen)
    if not report['action_changed'] or not report['frozen_adapters_unchanged']:raise ValueError('actor/frozen invariant failed')
    with torch.no_grad():expected_logps=forward(samples[:1])[0].cpu()
    saved=verified_checkpoint(args.out/'checkpoints')/'adapter/action'
    policy.reload_action_exact(saved)
    with torch.no_grad():actual_logps=forward(samples[:1])[0].cpu()
    torch.testing.assert_close(actual_logps,expected_logps,rtol=0,atol=0)
    report['reload_logps_exact']=True
    report['policy_definition']=dict(frozen_prefix=str(args.prefix_adapter),initial_action=str(args.action_adapter),**expected)
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
