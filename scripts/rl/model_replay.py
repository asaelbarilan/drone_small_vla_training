"""Differentiable cached replay of recorded tokens under the current policy."""
import torch

def replay_cached_tokens(model,batch,ids,*,low,high,temperature,pad_token_id):
    from transformers import LogitsProcessor,LogitsProcessorList
    from transformers.generation.utils import GenerationMixin
    if batch['input_ids'].shape[0]!=1 or not ids or temperature<=0:
        raise ValueError('one nonempty sequence and positive temperature required')
    if any(i<low or i>=high for i in ids):raise ValueError('recorded token outside action support')
    n=batch['input_ids'].shape[1]
    class ForceRecorded(LogitsProcessor):
        def __call__(self,current_ids,scores):
            forced=torch.full_like(scores,float('-inf'))
            forced[:,ids[current_ids.shape[1]-n]]=0
            return forced
    out=GenerationMixin.generate.__wrapped__(model,**batch,do_sample=False,use_cache=True,
        min_new_tokens=len(ids),max_new_tokens=len(ids),output_logits=True,output_hidden_states=True,
        return_dict_in_generate=True,logits_processor=LogitsProcessorList([ForceRecorded()]),
        pad_token_id=pad_token_id)
    logits=torch.stack(out.logits,dim=1)[0,:,low:high].float()/temperature
    selected=torch.tensor(ids,device=logits.device)-low
    logps=logits.log_softmax(-1).gather(1,selected[:,None]).flatten()
    return logps,out.hidden_states[0][-1][0,-1].detach()
