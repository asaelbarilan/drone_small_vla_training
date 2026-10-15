"""D253 reusable evaluator preserving the seven D251 criteria and thresholds."""
import numpy as np
from movement_reward import movement_reward

def family(kind):return 'Turn' if kind=='Turn/Rotate' else kind

def weighted_mse(flights,constant=None):
    return float(np.mean([np.mean([(s['mc_target']-(s['value_old'] if constant is None else constant))**2 for s in f['samples']]) for f in flights])) if flights else None

def numeric_criteria(flights,fit):
    terminal=[f for f in flights if f['end_reason']=='model_stop'];good=[f for f in terminal if f['complete']];bad=[f for f in terminal if not f['complete']]
    premature=[f for f in bad if f['reward']['endpoint_position_error_m']>.5 or f['reward']['endpoint_yaw_error_deg']>f['reward']['heading_tolerance_deg']]
    coverage=len(terminal)>=4 and len({f['task'] for f in terminal})==3 and len(good)>=2 and len(premature)>=1
    const=fit['development_target_mean'];mse=weighted_mse(terminal);zero=weighted_mse(terminal,0.);mean=weighted_mse(terminal,const)
    bykind={k:dict(mse=weighted_mse([f for f in terminal if family(f['kind'])==k]),zero=weighted_mse([f for f in terminal if family(f['kind'])==k],0.),mean=weighted_mse([f for f in terminal if family(f['kind'])==k],const)) for k in ['Turn','Ascend/Descend']}
    terms=[dict(flight=f['id'],kind=f['kind'],complete=f['complete'],target=f['samples'][-1]['mc_target'],value=f['samples'][-1]['value_old'],advantage=f['samples'][-1]['gae']) for f in terminal]
    terr=[abs(t['target']-t['value']) for t in terms];mae=float(np.mean(terr)) if terr else None;maxerror=max(terr) if terr else None
    calibration=bool(terminal) and mse<=.9*zero and mse<=.9*mean and all(z['mse'] is not None and z['mse']<=min(z['zero'],z['mean']) for z in bykind.values()) and mae<=.25 and maxerror<=.5
    stopcheck=bool(terms) and all(t['advantage']>=.01 if t['complete'] else t['advantage']<=-.01 for t in terms)
    states=[];continuation=[]
    for f in terminal:
     offset=0;poses=np.array(f['poses']);ref=f['reference'];kw=dict(kind=f['kind'],stopping_mode='accepted_goal')
     for i,s in enumerate(f['samples']):
      prefix=poses[:offset+1];hold=np.vstack([prefix,np.repeat(prefix[-1:],10,axis=0)])
      stop=movement_reward(hold,ref,stopped=True,**kw);qstop=stop['reward']-movement_reward(hold,ref,stopped=False,**kw)['reward'];a=qstop-s['value_old']
      states.append(dict(flight=f['id'],call=i,accepted=stop['complete'],value=s['value_old'],stop_return=qstop,stop_advantage=a,correct=bool(a>=.01 if stop['complete'] else a<=-.01)))
      end=offset+f['counts'][i];prefix2=poses[:end+1];hold2=np.vstack([prefix2,np.repeat(prefix2[-1:],10,axis=0)]);endq=movement_reward(hold2,ref,stopped=True,**kw)
      de=stop['endpoint_position_error_m']-endq['endpoint_position_error_m'];dy=stop['endpoint_yaw_error_deg']-endq['endpoint_yaw_error_deg']
      improving=(de>=.02 or dy>=.2) and de>=-.02 and dy>=-.2
      if f['complete'] and i<len(f['samples'])-1 and not stop['complete'] and improving:
       surplus=s['mc_target']-s['value_old'];gap=s['mc_target']-qstop
       continuation.append(dict(flight=f['id'],kind=f['kind'],call=i,remaining_return=s['mc_target'],value=s['value_old'],surplus=surplus,stop_return=qstop,gap=gap,correct=bool(surplus>=.01 and gap>=.10)))
      offset=end
    early=[a for a in states if not a['accepted']];arrived=[a for a in states if a['accepted']]
    rate=lambda rows:float(np.mean([s['correct'] for s in rows])) if rows else 0.
    count_by_kind={k:sum(family(a['kind'])==k for a in continuation) for k in ['Turn','Ascend/Descend']}
    statecheck=bool(early and arrived) and rate(early)>=.95 and rate(arrived)>=.95 and all(n>=2 for n in count_by_kind.values()) and rate(continuation)>=.90
    raw=np.array([s['gae'] for f in terminal for s in f['samples']]);devstd=fit['development_advantage_std'];teststd=float(raw.std()) if len(raw) else 0.;scaledmax=float(np.abs(raw/devstd).max()) if len(raw) and devstd>0 else None
    scalecheck=devstd>=.05 and .25*devstd<=teststd<=4*devstd and scaledmax is not None and scaledmax<=10

    return dict(checks=dict(C2_coverage=bool(coverage),C3_value=bool(calibration),C4_actual_stop=bool(stopcheck),C5_state_and_continuation=bool(statecheck),C6_scale=bool(scalecheck)),
        metrics=dict(terminal_flights=len(terminal),successes=len(good),failed_stops=len(bad),premature_stops=len(premature),mse=mse,zero_baseline_mse=zero,training_mean_baseline_mse=mean,per_kind=bykind,terminal_mae=mae,terminal_max_error=maxerror,terminals=terms,state_rates=dict(outside=rate(early),accepted=rate(arrived),continuation=rate(continuation),continuation_counts=count_by_kind),advantage_scale=dict(development_std=devstd,test_std=teststd,max_test_scaled_magnitude=scaledmax)),
        state_checks=dict(stops=states,continuation=continuation))
