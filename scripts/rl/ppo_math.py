"""D213 pure, CPU-testable PPO primitives. No model or environment imports."""
import numpy as np
import torch


def gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
    arrays = [np.asarray(x) for x in (rewards, values, next_values, terminated, truncated)]
    if any(x.ndim != 1 for x in arrays) or len({len(x) for x in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("nonempty equal-length vectors required")
    r, v, nv = [x.astype(float) for x in arrays[:3]]
    term, trunc = arrays[3:]
    if term.dtype != bool or trunc.dtype != bool or np.any(term & trunc):
        raise ValueError("explicit mutually exclusive boolean boundary flags required")
    if not np.isfinite([r, v, nv]).all() or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("finite targets and valid discounts required")
    if not (term[-1] or trunc[-1]):
        raise ValueError("last transition requires an explicit boundary")
    advantage = np.zeros_like(r)
    carry = 0.0
    for i in reversed(range(len(r))):
        delta = r[i] + gamma * (0 if term[i] else nv[i]) - v[i]
        carry = delta + (0 if term[i] or trunc[i] else gamma * lam * carry)
        advantage[i] = carry
    return advantage, advantage + v


def joint_ppo_loss(new_logps, old_logps, advantages, clip=0.05):
    if new_logps.ndim != 2 or new_logps.shape != old_logps.shape or min(new_logps.shape) == 0:
        raise ValueError("nonempty [calls,tokens] probabilities required")
    if advantages.shape != (len(new_logps),) or not 0 < clip < 1:
        raise ValueError("one advantage per call and valid clip required")
    if not all(torch.isfinite(x).all() for x in (new_logps, old_logps, advantages)):
        raise ValueError("nonfinite PPO inputs")
    log_ratio = (new_logps - old_logps.detach()).sum(-1)
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("nonfinite joint probability ratio")
    a = advantages.detach()
    loss = -torch.minimum(ratio*a, ratio.clamp(1-clip, 1+clip)*a).mean()
    return loss, ratio


def replay_parity(recorded, replayed, joint_tolerance=0.01, token_tolerance=0.01):
    old, new = np.asarray(recorded, float), np.asarray(replayed, float)
    if old.ndim != 2 or old.shape != new.shape or min(old.shape) == 0:
        raise ValueError("nonempty equal probability matrices required")
    if not np.isfinite([old,new]).all() or min(joint_tolerance,token_tolerance) <= 0:
        raise ValueError("finite probabilities and positive tolerances required")
    token = float(np.abs(new-old).max())
    joint = float(np.abs((new-old).sum(-1)).max())
    if token > token_tolerance or joint > joint_tolerance:
        raise ValueError(f"behavior replay mismatch: token={token}, joint={joint}")
    return dict(max_token_error=token,max_joint_error=joint)


def update_batches(order, batch_size, accum):
    """Yield microbatches and sample weights, including the final remainder."""
    if batch_size <= 0 or accum <= 0 or not len(order):
        raise ValueError("positive batch sizes and nonempty samples required")
    for start in range(0,len(order),batch_size*accum):
        window=order[start:start+batch_size*accum]
        yield [(window[j:j+batch_size],len(window[j:j+batch_size])/len(window))
               for j in range(0,len(window),batch_size)]


def validate_ppo_samples(samples, chunk, *, require_final_value=True):
    """Reject unusable episodes before loading a costly model.

    Approval here is a required upstream record, not a mechanism to approve a
    reward. Full model/image/version provenance remains a separate launch gate.
    """
    if not samples or chunk <= 0:
        raise ValueError("nonempty samples and positive chunk required")
    required={"task","passage","call_index","ids","logps","temperature",
              "context_input_ids","box_text","progress_text","reward_v2",
              "terminated","truncated","reward_spec_approved"}
    groups={}
    for s in samples:
        missing=required-s.keys()
        if missing or s.get('reward_spec_approved') is not True:
            raise ValueError(f"PPO preflight blocked: unapproved/incomplete rollout ({sorted(missing)})")
        if len(s['ids'])!=4*chunk or len(s['logps'])!=4*chunk or not s['context_input_ids']:
            raise ValueError("sampled action/context token counts invalid")
        if not isinstance(s['terminated'],bool) or not isinstance(s['truncated'],bool):
            raise ValueError("explicit boolean episode boundaries required")
        if s['terminated'] and s['truncated']:
            raise ValueError("contradictory episode boundaries")
        if not np.isfinite([s['reward_v2'],s['temperature'],*s['logps']]).all() or s['temperature']<=0:
            raise ValueError("finite rewards/logps and positive temperature required")
        groups.setdefault((s['task'],s['passage']),[]).append(s)
    for calls in groups.values():
        calls.sort(key=lambda s:s['call_index'])
        if [s['call_index'] for s in calls]!=list(range(len(calls))):
            raise ValueError("missing/duplicate call indices")
        if any(s['terminated'] or s['truncated'] for s in calls[:-1]):
            raise ValueError("calls after terminal boundary")
        last=calls[-1]
        if not(last['terminated'] or last['truncated']):
            raise ValueError("missing final boundary")
        if last['truncated'] and (not last.get('final_observation_id') or
                                  (require_final_value and ('final_value' not in last or not np.isfinite(last['final_value'])))):
            raise ValueError("truncation requires finite actual-final-state bootstrap")
    return len(groups)
