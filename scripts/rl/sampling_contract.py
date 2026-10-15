"""D226 block PPO reuse of rollouts with ambiguous/overridden sampling."""
import math

CONTRACT='action-categorical-v2'

def validate_sampling_contract(samples):
    if not samples:raise ValueError('empty sampled batch')
    for s in samples:
        if s.get('sampling_contract')!=CONTRACT or s.get('policy_provenance',{}).get('sampling_contract')!=CONTRACT:
            raise ValueError('unverified sampling contract; collect fresh rollouts with action-categorical-v2')
        if s.get('use_model_defaults') is not False or s.get('top_k')!=0 or s.get('top_p')!=1.:
            raise ValueError('PPO requires untruncated categorical action sampling')
        t=s.get('temperature');err=s.get('sampler_logp_max_error')
        if type(t) not in (int,float) or not math.isfinite(t) or t<=0:
            raise ValueError('invalid sampling temperature')
        if type(err) not in (int,float) or not math.isfinite(err) or not 0<=err<=1e-6:
            raise ValueError('sampling/log-probability parity was not verified')
    return len(samples)
