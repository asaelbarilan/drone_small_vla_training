"""D246 CPU-only stop-credit controls. No model, simulator or cloud calls."""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from audit_movement_reward import line
from movement_reward import movement_reward
from ppo_engine import prepare_advantages
from ppo_math import joint_ppo_loss

def run():
 rows=[]
 for name,kind,ref in [('ascent','Ascend/Descend',line(x=3,z=2)),('descent','Ascend/Descend',line(x=3,z=-2)),('right','Turn',line(yaw=26)),('left','Turn',line(yaw=-26))]:
  for mode in ['legacy_exp','rational_tolerant_heading']:
   score=lambda p,stop=True:movement_reward(p,ref,kind=kind,stopped=stop,altitude_cost_mode=mode)
   full=score(ref);assert full['complete']
   # Shared prefix cancels: compare stop now with finish then stop.
   for fraction in [0,.25,.5,.75,1.]:
    n=max(1,round(fraction*(len(ref)-1))+1)
    prefix=ref[:n];quiet=np.vstack([prefix,np.repeat(prefix[-1:],10,axis=0)])
    stopped=score(quiet);prefix_reward=score(quiet,False)['reward']
    qstop=stopped['reward']-prefix_reward
    if fraction<1:
     qcontinue=full['reward']-prefix_reward
     assert not stopped['complete'] and qcontinue>qstop
    else:
     # Continuing away from an already reached goal, then stopping.
     alternative=np.vstack([ref,ref[-2:len(ref)//2:-1]])
     qcontinue=score(alternative)['reward']-prefix_reward
     assert qstop>qcontinue
    assert not stopped['invalid_transitions']
    # Exact equal-probability critic control, not a fitted VLA critic.
    value=(qstop+qcontinue)/2
    raw=np.array([qstop-value,qcontinue-value])
    advantages=prepare_advantages(raw,'scale_only')
    logits=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    new=logits.log_softmax(0).reshape(2,1)
    old=new.detach().clone()
    loss,_=joint_ppo_loss(new,old,torch.tensor(advantages))
    loss.backward()
    updated=(logits-.05*logits.grad).softmax(0).detach().numpy()
    assert (updated[0]>.5)==(fraction==1)
    rows.append(dict(task=name,mode=mode,fraction=fraction,stop_return=qstop,continue_return=qcontinue,stop_advantage=float(advantages[0]),stop_probability_before=.5,stop_probability_after=float(updated[0]),complete=stopped['complete']))
 # Explicit counterexample: sign preservation cannot repair an underestimated critic.
 early=next(r for r in rows if r['task']=='ascent' and r['fraction']==.75)
 bad_value=min(early['stop_return'],early['continue_return'])-.1
 bad=prepare_advantages([early['stop_return']-bad_value,early['continue_return']-bad_value],'scale_only')
 assert bad[0]>0
 # Inspect all saved terminal chunks, not the advisory server stop flag.
 records=list(csv.DictReader((ROOT/'reports/d241_reward_advantage_audit/advantages.csv').read_text(encoding='utf-8-sig').splitlines()))
 raw=np.array([float(r['gae']) for r in records]);scaled=prepare_advantages(raw,'scale_only')
 saved=[]
 for p in sorted((ROOT/'reports/d238_two_task/evidence/round_1/attempts/episodes').glob('*_r?.json')):
  ep=json.loads(p.read_text());
  if ep['end_reason']!='model_stop':continue
  matches=[(i,r) for i,r in enumerate(records) if r['task']==ep['task'] and int(r['passage'])==ep['repeat']]
  i,r=max(matches,key=lambda x:int(x[1]['call']))
  assert raw[i]>0 and float(r['advantage'])<0 and scaled[i]>0
  saved.append(dict(task=ep['task'],passage=ep['repeat'],raw_gae=raw[i],centered=float(r['advantage']),scale_only=scaled[i]))
 assert len(saved)==3
 return dict(controls=rows,saved_terminal_chunks=saved,critic_counterexample=dict(early_stop_scale_only_advantage=float(bad[0]),meaning='A wrong critic can still reinforce early stopping; no unconditional guarantee.'),limitations=['Controls are constructed trajectories and exact-value toy categorical gradients, not new flights.','No empirical premature-stop episode in the four-flight training batch.','No direct urgency reward: waiting still at target before stopping has the same undiscounted return.','Server stop flag does not terminate execution: ten consecutive quiet readbacks do.','Existing opt-in scale_only preserves raw GAE sign; default and saved checkpoints remain unchanged.'])
if __name__=='__main__':
 result=run();out=ROOT/'reports/d246_stopping_credit';out.mkdir(exist_ok=True)
 result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(controls=len(result['controls']),saved_stops=result['saved_terminal_chunks'],critic_counterexample=result['critic_counterexample']),indent=2))